
from einops import rearrange
import torch
import torch.nn as nn
from sandstone.models.msa import StackMSA


class Atlas(nn.Module):
    def __init__(
        self,
        args,
        kwargs_list,
        embed_dim=384,
        num_classes=100,
        fc_norm=nn.LayerNorm,
        pretrained=False,
        pretrained_path="",
        load_head=False,
        freeze_mode="none",
        freeze_stages=0,
        freeze_patterns=None,
        reset_head=True,
        **kwargs,
    ):
        super().__init__()
        assert kwargs_list is not None, "kwargs_list should be provided"
        self.args = args
        self.kwargs_list = kwargs_list
        self.num_classes = num_classes

        atlas_models = [StackMSA(args=args, **kwargs) for kwargs in kwargs_list]
        self.atlas_models = nn.ModuleList(atlas_models)

        self.embed_dim = embed_dim
        self.readout = "avg"

        self.readout_norm = nn.LayerNorm(self.embed_dim)

        self.fc_norm = nn.LayerNorm(self.embed_dim) if fc_norm else nn.Identity()
        self.fc = nn.Linear(self.embed_dim, num_classes)
        # # ----------- Pretrained loading -----------
        # pretrained = False
        # model_load_path = '/scratch/abircs/sudipta/atlas/logs/Atlas_Base_1024_ImageNet_21k/atlas_d2d10_base_r1024/epoch=14.ckpt'
        # if pretrained:
        #     assert model_load_path != "", "Need to provide model_load_path for pretrained model"
        #     print(f"Loading pretrained weights from {model_load_path}")
        
        #     ckpt = torch.load(model_load_path, map_location="cpu")
        #     state_dict = ckpt["model"] if "model" in ckpt else ckpt

        #     # Remove classifier weights from checkpoint
        #     for k in list(state_dict.keys()):
        #         if "head" in k or "fc.weight" in k or "fc.bias" in k:
        #             del state_dict[k]

        #     #print("\n===== Model Parameters =====")
        #     #for name, param in self.state_dict().items():
        #         #print(f"{name} : {tuple(param.shape)}")

        #     #print("\n===== Checkpoint Parameters =====")
        #     #for name, param in state_dict.items():
        #         #print(f"{name} : {tuple(param.shape)}")

        #     # Load weights
        #     msg = self.load_state_dict(state_dict, strict=False)
            
        #     # Figure out successfully loaded keys
        #     #all_model_keys = set(self.state_dict().keys())
        #     #ckpt_keys = set(state_dict.keys())
        #     #loaded_keys = all_model_keys.intersection(ckpt_keys) - set(msg.missing_keys)

        #     #print("\n===== Loading Report =====")
        #     #print("Successfully loaded keys:", sorted(list(loaded_keys)))
        #     print("Missing keys (not loaded):", msg.missing_keys)
        #     print("Unexpected keys (in checkpoint but not in model):", msg.unexpected_keys)
        #     #print("Finished loading pretrained weights (ignoring head).")

        #     #exit(0)

        if pretrained:
            self._load_pretrained(pretrained_path=pretrained_path, load_head=load_head)
        
        if reset_head:
            self.fc = nn.Linear(self.embed_dim, num_classes)

        self.apply_freezing(
            freeze_mode=freeze_mode,
            freeze_stages=freeze_stages,
            freeze_patterns=freeze_patterns,
        )
        self.print_trainable_parameters()
    
    #-----------------------------------Added----------------------------------------#
    def _strip_module_prefix(self, state_dict):
        if len(state_dict) > 0 and list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        return state_dict

    def _load_pretrained(self, pretrained_path, load_head=False):
        assert pretrained_path, "Need to provide pretrained_path when pretrained=True"
        print(f"Loading pretrained weights from {pretrained_path}")

        ckpt = torch.load(pretrained_path, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        state_dict = self._strip_module_prefix(state_dict)

        if not load_head:
            for k in list(state_dict.keys()):
                if k.startswith("fc.") or "head" in k:
                    del state_dict[k]

        msg = self.load_state_dict(state_dict, strict=False)
        print("Missing keys (not loaded):", msg.missing_keys)
        print("Unexpected keys (in checkpoint but not in model):", msg.unexpected_keys)

    def _freeze_module(self, module):
        for p in module.parameters():
            p.requires_grad = False

    def _unfreeze_module(self, module):
        for p in module.parameters():
            p.requires_grad = True

    def apply_freezing(self, freeze_mode="none", freeze_stages=0, freeze_patterns=None):
        freeze_patterns = freeze_patterns or []
        for p in self.parameters():
            p.requires_grad = True
        if freeze_mode == "none":
            return
        elif freeze_mode == "head_only":
            for p in self.parameters():
                p.requires_grad = False
            self._unfreeze_module(self.fc)
            self._unfreeze_module(self.fc_norm)
            self._unfreeze_module(self.readout_norm)
            return
        elif freeze_mode == "backbone":
            for atlas_model in self.atlas_models:
                self._freeze_module(atlas_model)
            self._unfreeze_module(self.fc)
            self._unfreeze_module(self.fc_norm)
            self._unfreeze_module(self.readout_norm)
            return
        elif freeze_mode == "first_n_stages":
            for i, atlas_model in enumerate(self.atlas_models):
                if i < freeze_stages:
                    self._freeze_module(atlas_model)
            return
        elif freeze_mode == "except_last_stage":
            for i, atlas_model in enumerate(self.atlas_models):
                if i < len(self.atlas_models) - 1:
                    self._freeze_module(atlas_model)
            return
        elif freeze_mode == "by_name":
            for name, param in self.named_parameters():
                for pattern in freeze_patterns:
                    if pattern in name:
                        param.requires_grad = False
                        break
            return
        else:
            raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_parameters(self):
        total, trainable = 0, 0
        for _, p in self.named_parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
        print(f"Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / total:.2f}%)")
    #-----------------------------------Added----------------------------------------#

    def forward_head(self, x, pre_logits: bool = False, scales=None):
        # collect features from relevant scales
        if scales is None:
            scales = range(len(x))

        feats = [x[i] for i in scales]
        feats_BNC = torch.cat(feats, dim=1)

        # normalize the inputs
        if isinstance(self.readout_norm, nn.LayerNorm):
            x_BNC = self.readout_norm(feats_BNC)
        else:
            x_BNC = feats_BNC

        if self.readout == "avg":
            x_BC = x_BNC.mean(dim=1)
        else:
            raise NotImplementedError
        x_BC = self.fc_norm(x_BC)
        return x_BC if pre_logits else self.fc(x_BC)


    def forward(self, x, batch=None):
        base_model = self.atlas_models[0]

        bsz = x.shape[0]
        x = base_model.forward_raw(x)
        layouts = base_model.multiscale_layout
        num_stages = len(self.atlas_models)

        for level, atlas_model in enumerate(self.atlas_models):
            is_last = len(x) == 1 or level == num_stages - 1
            grid_sizes = [layout["grid_size"] for layout in layouts]
            x = atlas_model.forward_features(
                x, grid_sizes=grid_sizes, process_readout=False
            )
            if not is_last:
                layouts = layouts[1:]
                x = x[1:]

        feats = [rearrange(scale, "(b nw) k c -> b (nw k) c", b=bsz) for scale in x]
        x = self.forward_head(feats)
        return {"logit": x}
