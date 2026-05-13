import time
from collections import OrderedDict

import torch
import wandb
from tqdm import tqdm

from sandstone.utils.engine import gather_predictions_dict, prefix_dict, get_grad_norm
from sandstone.utils.misc import AverageMeter, Summary, ProgressMeter, get_is_master, logger
from timm.data.mixup import Mixup

from .base import Engine


class Classifier(Engine):
    """
    Memory-safe training/eval engine:
    - Avoids storing per-batch tensors on GPU/CPU.
    - Logs scalars only (loss.item()).
    - Optionally allows storing predictions (CPU) if absolutely required.
    """

    def __init__(
        self,
        *args,
        binary_pred: bool = False,
        log_max_min_lr: bool = False,
        mixup_kwargs=None,
        store_predictions: bool = False,   # << default OFF to prevent CPU RAM growth
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.binary_pred = binary_pred
        self.log_max_min_lr = log_max_min_lr
        self.store_predictions = store_predictions  # when True, stores CPU preds/golds (use with care!)

        # Mixup / CutMix setup
        mixup_active = (
            mixup_kwargs is not None
            and (
                mixup_kwargs.mixup > 0
                or mixup_kwargs.cutmix > 0.0
                or mixup_kwargs.cutmix_minmax is not None
            )
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

        # Reset CUDA peak stats for clean tracking this epoch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

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

        # If you *must* compute epoch metrics from predictions, you can enable store_predictions.
        # Otherwise, we won't accumulate per-batch outputs to avoid CPU RAM growth.
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

            # Per-iteration LR schedule
            if lr_scheduler is not None and batch_idx % self.accum_iter == 0:
                lr_scheduler.adjust_learning_rate(batch_idx / len(dataloader) + epoch)

            with torch.amp.autocast(
                "cuda", dtype=self.amp_precision, enabled=self.amp_precision is not None
            ):
                loss, logging_dict, predictions_dict = self.step(
                    model, batch, batch_idx, epoch=epoch, train=True, device=device
                )

            # Scale by accumulation
            loss = loss / self.accum_iter

            # Backward + (maybe) optimizer step
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                clip_grad=clip_grad,
                create_graph=False,
                update_grad=(batch_idx + 1) % self.accum_iter == 0,
            )

            if (batch_idx + 1) % self.accum_iter == 0:
                optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            # Scalars only
            cur_loss = loss.item()
            losses.update(cur_loss, batch["x"].size(0))

            if self.log_max_min_lr:
                max_lr_value = max(pg["lr"] for pg in optimizer.param_groups)
                min_lr_value = min(pg["lr"] for pg in optimizer.param_groups)
                max_lr.update(max_lr_value)
                min_lr.update(min_lr_value)
            else:
                lr_value = optimizer.param_groups[0]["lr"]
                lr.update(lr_value)

            # Optional grad norm logging
            if log_grad_norm and (batch_idx % log_interval == 0):
                grad_norm_dict, _ = get_grad_norm(model, log_weight_norm=False)
                # grad_norm_dict is scalars, safe to log
                wandb.log(grad_norm_dict, step=self.global_step)

            # ---- Minimal, memory-safe per-step logging ----
            if epoch_metrics_configured:
                # We’ll keep *only* lightweight logs; no predictions unless explicitly requested
                result = OrderedDict()
                log_dict = prefix_dict(logging_dict, "train_")
                log_dict["train_loss"] = cur_loss  # scalar
                result["logs"] = log_dict

                if self.store_predictions:
                    # VERY CAREFUL: this increases CPU memory usage.
                    # Still, we ensure CPU tensors only.
                    preds_cpu = {}
                    # If using multi-GPU, gather first, then move to CPU
                    if self.args.main.multi_gpu:
                        predictions_dict = gather_predictions_dict(predictions_dict)

                    for k, v in predictions_dict.items():
                        if torch.is_tensor(v):
                            preds_cpu[k] = v.detach().cpu()
                        else:
                            preds_cpu[k] = v
                    result.update(preds_cpu)

                    self.training_step_outputs.append(result)
                else:
                    # Logs only, tiny memory footprint
                    self.training_step_outputs.append(result)

            # Timing
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % log_interval == 0:
                wandb.log({"train_loss": cur_loss}, step=self.global_step)
                progress.display(batch_idx + 1, tqdm_write=True)

        # Epoch end hooks (may read training_step_outputs); immediately clear after
        self.on_epoch_end(split="train", device=device, epoch=epoch)
        self.training_step_outputs = []

    @torch.no_grad()
    def evaluate(self, model, dataloader, device, epoch=None, test=False, gather_predictions=False):
        model.eval()
        desc = "Evaluation" if not test else "Testing"

        # Reset peak stats for clean tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Streaming meters; avoid storing per-batch predictions
        losses = AverageMeter("Loss", ":.4e")
        progress = ProgressMeter(len(dataloader), [losses], prefix=f"Epoch: [{epoch}] {desc}" if epoch else desc)

        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch} {desc}" if epoch else desc, disable=not get_is_master())
        ):
            if batch_idx == self.limit_num_batches:
                break

            loss, logging_dict, predictions_dict = self.step(
                model, batch, batch_idx, epoch=epoch, train=False, device=device
            )

            # Streaming scalar loss
            cur_loss = float(loss) if not torch.is_tensor(loss) else loss.item()
            losses.update(cur_loss, batch["x"].size(0))

            # Minimal per-step record (logs only)
            result = OrderedDict()
            result["logs"] = {"test_loss" if test else "val_loss": cur_loss}

            if gather_predictions and self.store_predictions:
                # If you absolutely need full predictions, gather -> CPU, then store.
                if self.args.main.multi_gpu:
                    predictions_dict = gather_predictions_dict(predictions_dict)
                preds_cpu = {
                    k: (v.detach().cpu() if torch.is_tensor(v) else v)
                    for k, v in predictions_dict.items()
                }
                result.update(preds_cpu)

            if test:
                self.test_step_outputs.append(result)
            else:
                self.validation_step_outputs.append(result)

        # Log final loss (scalar)
        wandb.log({"test_loss" if test else "val_loss": losses.avg}, step=self.global_step)

        # Let the framework compute epoch metrics from what we kept (mostly logs)
        epoch_metrics = self.on_epoch_end(split="test" if test else "val", device=device, epoch=epoch)

        # Immediately clear buffers to release RAM
        if test:
            self.test_step_outputs = []
        else:
            self.validation_step_outputs = []

        return epoch_metrics

    def preprocess_batch(self, batch, device="cuda"):
        # Move tensors to device; leave non-tensors as-is
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
            else:
                batch[k] = v
        return batch

    def step(self, model, batch, batch_idx, epoch=None, train=False, device="cuda"):
        """
        - Returns (loss, logging_dict, predictions_dict)
        - Predictions are *not* detached here; we handle that only if we choose to store them.
        """
        predictions_dict = OrderedDict()
        batch = self.preprocess_batch(batch, device=device)

        # Mixup/CutMix only during training
        if self.mixup_fn is not None and train:
            batch["x"], batch["y"] = self.mixup_fn(batch["x"], batch["y"])

        with torch.amp.autocast("cuda", dtype=self.amp_precision, enabled=self.amp_precision is not None):
            model_output = model(batch["x"], batch=batch)

            # Collect logits/preds for metrics if needed
            if "logit" in model_output:
                logit = model_output["logit"]
                probs = torch.sigmoid(logit) if self.binary_pred else torch.softmax(logit, dim=-1)
                preds = probs.argmax(dim=-1).reshape(-1)
            else:
                logit, probs, preds = None, None, None

            golds = self.get_target(batch, model_output)

            # Prepare predictions dict (decision to store happens outside)
            if "time_at_event" in batch:
                predictions_dict["censors"] = batch["time_at_event"]
            predictions_dict["probs"] = probs
            predictions_dict["golds"] = golds
            predictions_dict["preds"] = preds

            metric_input = {
                "logit": logit,
                "target": batch["y"],
                "batch": batch,
                "model_output": model_output,
            }

            loss, logging_dict = self.compute_step_metrics(
                loss_input=metric_input, metric_input=metric_input, train=train
            )

        return loss, logging_dict, predictions_dict

    def get_target(self, batch, model_output=None):
        if "y" in batch:
            return batch["y"]
        raise ValueError("No targets found in batch or model_output.")
