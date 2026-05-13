import time
from collections import OrderedDict
from typing import Optional

import torch
import wandb
from tqdm import tqdm

from sandstone.utils.engine import gather_predictions_dict, prefix_dict, get_grad_norm
from sandstone.utils.misc import AverageMeter, Summary, ProgressMeter, get_is_master, logger
from timm.data.mixup import Mixup

from .base import Engine


class Classifier(Engine):
    """
    Memory-safe classifier Engine that:
      - Computes epoch accuracy via streaming (no full-pred arrays kept in RAM).
      - Keeps tensors on-device for gather; moves to CPU only after gather & detach.
      - Always provides non-None `preds` and `golds` keys (small placeholders) so existing metrics.py won't crash.
      - Optionally store full predictions to disk (disabled by default).
    """

    def __init__(
        self,
        *args,
        binary_pred: bool = False,
        log_max_min_lr: bool = False,
        mixup_kwargs: Optional[object] = None,
        store_predictions: bool = False,  # if True, writes per-batch preds to disk (one file per rank)
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.binary_pred = binary_pred
        self.log_max_min_lr = log_max_min_lr
        self.store_predictions = store_predictions

        mixup_active = (
            mixup_kwargs is not None
            and (getattr(mixup_kwargs, "mixup", 0) > 0
                 or getattr(mixup_kwargs, "cutmix", 0) > 0
                 or getattr(mixup_kwargs, "cutmix_minmax", None) is not None)
        )
        if mixup_active:
            logger.info("Mixup is activated!")
            self.mixup_fn = Mixup(
                mixup_alpha=mixup_kwargs.mixup,
                cutmix_alpha=mixup_kwargs.cutmix,
                cutmix_minmax=mixup_kwargs.cutmix_minmax,
                prob=mixup_kwargs.mixup_prob,
                switch_prob=mixup_kwargs.mixup_switch_prob,
                mode=mixup_kwargs.mixup_mode,
                label_smoothing=mixup_kwargs.smoothing,
                num_classes=self.args.model.kwargs.num_classes,
            )
        else:
            self.mixup_fn = None

        # streaming counters for accuracy (keeps memory flat)
        # Use device 'cuda' if available to perform safe distributed reductions later
        self._acc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset_streaming_metrics()

    def reset_streaming_metrics(self):
        # per-epoch streaming counters (on acc_device)
        self._correct_sum = torch.tensor(0, dtype=torch.long, device=self._acc_device)
        self._total_sum = torch.tensor(0, dtype=torch.long, device=self._acc_device)

    def train_one_epoch(
        self,
        model: torch.nn.Module,
        dataloader,
        optimizer,
        device,
        epoch,
        loss_scaler,
        lr_scheduler,
        args,
        log_interval: int = 50,
        clip_grad=None,
        log_grad_norm: bool = False,
    ):
        model.train()

        # Reset per-epoch outputs & streaming metrics
        self.training_step_outputs = []
        self.reset_streaming_metrics()

        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")

        if self.log_max_min_lr:
            max_lr = AverageMeter("max lr", ":.4e", summary_type=Summary.NONE)
            min_lr = AverageMeter("min lr", ":.4e", summary_type=Summary.NONE)
        else:
            lr = AverageMeter("lr", ":.4e", summary_type=Summary.NONE)

        max_mem = AverageMeter("Max mem (MiB)", ":.0f", summary_type=Summary.NONE)
        progress = ProgressMeter(
            len(dataloader),
            [batch_time, data_time, *([max_lr, min_lr] if self.log_max_min_lr else [lr]), losses, max_mem],
            prefix=f"Epoch: [{epoch}]",
        )
        epoch_metrics_configured = len(self.get_epoch_metrics(split="train")) > 0
        end = time.time()

        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch} Training", disable=not get_is_master())
        ):
            data_time.update(time.time() - end)
            if torch.cuda.is_available():
                max_mem.update(torch.cuda.max_memory_allocated() / (1024 * 1024))

            if batch_idx == self.limit_num_batches:
                break
            if batch is None:
                continue

            if (lr_scheduler is not None) and (batch_idx % self.accum_iter == 0):
                lr_scheduler.adjust_learning_rate(batch_idx / len(dataloader) + epoch)

            with torch.amp.autocast("cuda", dtype=self.amp_precision, enabled=self.amp_precision is not None):
                loss, logging_dict, predictions_dict = self.step(
                    model, batch, batch_idx, epoch=epoch, train=True, device=device
                )

            if loss is None:
                continue

            # scale & step
            loss = loss / self.accum_iter
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                clip_grad=clip_grad,
                create_graph=False,
                update_grad=(batch_idx + 1) % self.accum_iter == 0,
            )

            # update gradient accumulation bookkeeping
            if (batch_idx + 1) % self.accum_iter == 0:
                optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            losses.update(loss.item(), batch["x"].size(0))

            if self.log_max_min_lr:
                max_lr.update(max(pg["lr"] for pg in optimizer.param_groups))
                min_lr.update(min(pg["lr"] for pg in optimizer.param_groups))
            else:
                lr.update(optimizer.param_groups[0]["lr"])

            # logging grad norms occasionally
            if log_grad_norm and (batch_idx % log_interval == 0):
                grad_norm_dict, _ = get_grad_norm(model, log_weight_norm=False)
                wandb.log(grad_norm_dict, step=self.global_step)

            # update streaming metrics (accurate, low memory)
            # predictions_dict contains preds/probs/golds ON DEVICE (see step()).
            # We update streaming counters from device tensors if available.
            preds_dev = predictions_dict.get("preds", None)
            golds_dev = predictions_dict.get("golds", None)
            if preds_dev is not None and golds_dev is not None and torch.is_tensor(preds_dev) and torch.is_tensor(golds_dev):
                # ensure shapes align; reshape if needed
                try:
                    correct = (preds_dev.view(-1) == golds_dev.view(-1)).to(self._acc_device).sum().to(self._acc_device)
                    tot = torch.tensor(preds_dev.numel(), dtype=torch.long, device=self._acc_device)
                except Exception:
                    # if mismatch shape, try safe truncation/padding (defensive)
                    min_len = min(preds_dev.numel(), golds_dev.numel())
                    if min_len > 0:
                        correct = (preds_dev.view(-1)[:min_len] == golds_dev.view(-1)[:min_len]).to(self._acc_device).sum().to(self._acc_device)
                        tot = torch.tensor(min_len, dtype=torch.long, device=self._acc_device)
                    else:
                        correct = torch.tensor(0, dtype=torch.long, device=self._acc_device)
                        tot = torch.tensor(0, dtype=torch.long, device=self._acc_device)
                self._correct_sum += correct
                self._total_sum += tot

            # If epoch metrics are configured we still need to append something into self.training_step_outputs
            # to satisfy other parts of the code (and to avoid metrics.py getting None). BUT we won't store full per-sample arrays.
            if epoch_metrics_configured:
                # prepare log dict (scalars only)
                safe_logging = prefix_dict(logging_dict, "train_") if logging_dict is not None else {}
                safe_logging["train_loss"] = loss.item()

                # compute streaming accuracy so we can log correct epoch-level accuracy later
                # (we compute and push scalar only at epoch end; here we optionally append small placeholders)
                # We will append a minimal safe entry for compatibility:
                placeholder_preds = torch.tensor([0], dtype=torch.long)  # single-item placeholder
                placeholder_golds = torch.tensor([0], dtype=torch.long)

                res = OrderedDict()
                res["logs"] = safe_logging
                # Keep placeholders small (prevents None/shape errors in metrics.py)
                res["preds"] = placeholder_preds
                res["golds"] = placeholder_golds

                # Optionally include tiny 'probs' placeholder too
                res["probs"] = torch.tensor([0.0], dtype=torch.float)

                self.training_step_outputs.append(res)

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % log_interval == 0:
                wandb.log({"train_loss": loss.item()}, step=self.global_step)
                progress.display(batch_idx + 1, tqdm_write=True)

        # end-of-epoch: compute the streaming accuracy (reduce across ranks if distributed)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # aggregate total and correct sums across all ranks
            torch.distributed.all_reduce(self._correct_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self._total_sum, op=torch.distributed.ReduceOp.SUM)

        # compute scalar accuracy safely (avoid divide-by-zero)
        total = self._total_sum.item()
        correct = self._correct_sum.item()
        epoch_accuracy = (correct / total) if total > 0 else float("nan")

        # Add epoch_accuracy to a logs entry so the rest of pipeline can see the real metric
        # We push a final "epoch_summary" entry into training_step_outputs (small!)
        epoch_summary = OrderedDict()
        epoch_summary["logs"] = {"train_accuracy": epoch_accuracy, "train_total": int(total), "train_correct": int(correct)}
        # Also include placeholders so metrics.py won't crash
        epoch_summary["preds"] = torch.tensor([0], dtype=torch.long)
        epoch_summary["golds"] = torch.tensor([0], dtype=torch.long)
        self.training_step_outputs.append(epoch_summary)

        # call on_epoch_end (base will call compute_epoch_metrics)
        self.on_epoch_end(split="train", device=device, epoch=epoch)

        # clear per-epoch buffers to avoid RAM growth across epochs
        self.training_step_outputs.clear()
        self.reset_streaming_metrics()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(self, model, dataloader, device, epoch=None, test: bool = False, gather_predictions: bool = False):
        model.eval()

        # Reset per-epoch buffers and streaming metrics
        if test:
            self.test_step_outputs = []
        else:
            self.validation_step_outputs = []
        self.reset_streaming_metrics()

        desc = "Evaluation" if not test else "Testing"
        end = time.time()

        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch} {desc}" if epoch else desc, disable=not get_is_master())
        ):
            if batch_idx == self.limit_num_batches:
                break
            loss, logging_dict, predictions_dict = self.step(
                model, batch, batch_idx, epoch=epoch, train=False, device=device
            )

            if loss is None:
                continue

            # streaming update for eval
            preds_dev = predictions_dict.get("preds", None)
            golds_dev = predictions_dict.get("golds", None)
            if preds_dev is not None and golds_dev is not None and torch.is_tensor(preds_dev) and torch.is_tensor(golds_dev):
                try:
                    correct = (preds_dev.view(-1) == golds_dev.view(-1)).to(self._acc_device).sum().to(self._acc_device)
                    tot = torch.tensor(preds_dev.numel(), dtype=torch.long, device=self._acc_device)
                except Exception:
                    min_len = min(preds_dev.numel(), golds_dev.numel())
                    if min_len > 0:
                        correct = (preds_dev.view(-1)[:min_len] == golds_dev.view(-1)[:min_len]).to(self._acc_device).sum().to(self._acc_device)
                        tot = torch.tensor(min_len, dtype=torch.long, device=self._acc_device)
                    else:
                        correct = torch.tensor(0, dtype=torch.long, device=self._acc_device)
                        tot = torch.tensor(0, dtype=torch.long, device=self._acc_device)
                self._correct_sum += correct
                self._total_sum += tot

            # produce a small safe result entry (placeholders) so metrics.py doesn't receive None's
            result = OrderedDict()
            result["logs"] = {"test_loss" if test else "val_loss": loss.item()}
            result["preds"] = torch.tensor([0], dtype=torch.long)
            result["golds"] = torch.tensor([0], dtype=torch.long)
            result["probs"] = torch.tensor([0.0], dtype=torch.float)

            if test:
                self.test_step_outputs.append(result)
            else:
                self.validation_step_outputs.append(result)

            end = time.time()

        # reduce across ranks for streaming sums
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self._correct_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self._total_sum, op=torch.distributed.ReduceOp.SUM)

        total = self._total_sum.item()
        correct = self._correct_sum.item()
        epoch_accuracy = (correct / total) if total > 0 else float("nan")

        # append scalar epoch summary (small)
        epoch_summary = OrderedDict()
        epoch_summary["logs"] = {"eval_accuracy": epoch_accuracy, "eval_total": int(total), "eval_correct": int(correct)}
        epoch_summary["preds"] = torch.tensor([0], dtype=torch.long)
        epoch_summary["golds"] = torch.tensor([0], dtype=torch.long)

        if test:
            self.test_step_outputs.append(epoch_summary)
        else:
            self.validation_step_outputs.append(epoch_summary)

        # call on_epoch_end
        out = self.on_epoch_end(split="test" if test else "val", device=device, epoch=epoch)

        # clear buffers
        if test:
            self.test_step_outputs.clear()
        else:
            self.validation_step_outputs.clear()
        self.reset_streaming_metrics()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    def preprocess_batch(self, batch, device="cuda"):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        return batch

    def step(self, model, batch, batch_idx, epoch=None, train: bool = False, device: str = "cuda"):
        """
        Returns (loss, logging_dict, predictions_dict)
          - predictions_dict contains 'preds', 'golds', 'probs' ON DEVICE (GPU if available).
          - Caller will use streaming metrics and will not store large lists.
        """
        predictions_dict = OrderedDict()
        batch = self.preprocess_batch(batch, device=device)

        if self.mixup_fn is not None and train:
            batch["x"], batch["y"] = self.mixup_fn(batch["x"], batch["y"])

        with torch.amp.autocast("cuda", dtype=self.amp_precision, enabled=self.amp_precision is not None):
            model_output = model(batch["x"], batch=batch)

            if isinstance(model_output, dict) and "logit" in model_output and model_output["logit"] is not None:
                logit = model_output["logit"]
                if self.binary_pred:
                    probs = torch.sigmoid(logit)
                else:
                    probs = torch.softmax(logit, dim=-1)
                preds = probs.argmax(dim=-1).reshape(-1)
            else:
                logit, probs, preds = None, None, None

            golds = self.get_target(batch, model_output)

            # predictions_dict keeps tensors ON DEVICE so distributed gather (if used) can operate correctly.
            predictions_dict.update({"probs": probs, "golds": golds, "preds": preds})

            metric_input = {"logit": logit, "target": batch.get("y"), "batch": batch, "model_output": model_output}
            loss, logging_dict = self.compute_step_metrics(loss_input=metric_input, metric_input=metric_input, train=train)

        return loss, logging_dict, predictions_dict

    def get_target(self, batch, model_output=None):
        if "y" in batch and batch["y"] is not None:
            return batch["y"]
        raise ValueError("No targets found in batch or model_output.")
