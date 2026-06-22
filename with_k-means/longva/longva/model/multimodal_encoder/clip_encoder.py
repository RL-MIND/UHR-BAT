import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

from longva.utils import rank0_print

try:
    from s2wrapper import forward as multiscale_forward
except:
    pass


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print("The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print(
                "The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`."
            )
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def _infer_dtype(self):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float16

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        vision_dtype = self._infer_dtype()
        try:
            self.vision_tower = CLIPVisionModel.from_pretrained(
                self.vision_tower_name,
                device_map=device_map,
                attn_implementation="flash_attention_2",
                torch_dtype=vision_dtype,
            )
        except (ValueError, ImportError):
            self.vision_tower = CLIPVisionModel.from_pretrained(
                self.vision_tower_name,
                device_map=device_map,
                attn_implementation="sdpa",
                torch_dtype=vision_dtype,
            )
        except TypeError:
            # Older transformers versions may not accept attn_implementation.
            self.vision_tower = CLIPVisionModel.from_pretrained(
                self.vision_tower_name,
                device_map=device_map,
                torch_dtype=vision_dtype,
            )
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs, select_feature_type=None):
        if select_feature_type is None:
            select_feature_type = self.select_feature

        if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
            select_every_k_layer = len(image_forward_outs.hidden_states) // 4
            image_features = torch.cat(
                [
                    image_forward_outs.hidden_states[i]
                    for i in range(
                        select_every_k_layer + self.select_layer,
                        len(image_forward_outs.hidden_states),
                        select_every_k_layer,
                    )
                ],
                dim=-1,
            )
            select_feature_type = select_feature_type.replace("slicefour_", "")
        elif self.select_feature in ["slice_m25811_f6_patch", "slice_m25811_f6_cls_patch"]:
            select_layers = [-2, -5, -8, -11, 6]
            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in select_layers], dim=-1)
            select_feature_type = select_feature_type.replace("slice_m25811_f6_", "")
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]

        if select_feature_type == "patch":
            image_features = image_features[:, 1:]
        elif select_feature_type == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {select_feature_type}")
        return image_features

    def _interp_pos_embed(self, pos_embed: torch.Tensor, new_grid: int) -> torch.Tensor:
        """
        Interpolate CLIP positional embeddings to a new grid size (keep cls token).
        """
        squeeze_back = False
        if pos_embed.dim() == 2:
            pos_embed = pos_embed.unsqueeze(0)
            squeeze_back = True
        num_extra_tokens = 1  # cls token
        extra_tokens = pos_embed[:, :num_extra_tokens]
        pos_tokens = pos_embed[:, num_extra_tokens:]
        dim = pos_tokens.shape[-1]
        grid_size = int(pos_tokens.shape[1] ** 0.5)
        pos_tokens = pos_tokens.reshape(1, grid_size, grid_size, dim).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(pos_tokens, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)
        out = torch.cat([extra_tokens, pos_tokens], dim=1)
        if squeeze_back:
            out = out.squeeze(0)
        return out

    def forward(self, images, select_feature_type=None):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True
                )
                image_feature = self.feature_select(image_forward_out, select_feature_type=select_feature_type).to(
                    image.dtype
                )
                image_features.append(image_feature)
        else:
            pixel_values = images.to(device=self.device, dtype=self.dtype)
            new_grid = pixel_values.shape[-1] // self.config.patch_size
            orig_grid = self.config.image_size // self.config.patch_size
            pos_embed_backup = None

            # CLIPVisionModel wraps a .vision_model which holds embeddings
            backbone = getattr(self.vision_tower, "vision_model", None)
            pos_module = None
            pos_ids_backup = None
            pos_module = None
            pos_buffer = None
            if backbone is not None and hasattr(backbone, "embeddings"):
                pos_module = backbone.embeddings.position_embedding
                pos_buffer = getattr(backbone.embeddings, "position_ids", None)

            if new_grid != orig_grid and pos_module is not None:
                pos_embed_backup = pos_module.weight.data.clone()
                new_pos = self._interp_pos_embed(pos_embed_backup, new_grid)
                pos_module.weight.data = new_pos.to(pos_embed_backup.device)
                if pos_buffer is not None:
                    pos_ids_backup = pos_buffer.data.clone()
                    new_seq = new_pos.shape[0]
                    new_pos_ids = torch.arange(new_seq, device=pos_buffer.device).unsqueeze(0)
                    backbone.embeddings.position_ids = new_pos_ids

            image_forward_outs = self.vision_tower(pixel_values, output_hidden_states=True)

            if pos_embed_backup is not None and pos_module is not None:
                pos_module.weight.data = pos_embed_backup
            if pos_ids_backup is not None and backbone is not None and hasattr(backbone, "embeddings"):
                backbone.embeddings.position_ids = pos_ids_backup

            image_features = self.feature_select(image_forward_outs, select_feature_type=select_feature_type).to(
                images.dtype
            )

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        _hidden_size = self.config.hidden_size
        if "slicefour" in self.select_feature:
            _hidden_size *= 4
        if "slice_m25811_f6" in self.select_feature:
            _hidden_size *= 5
        return _hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        _num_patches = (self.config.image_size // self.config.patch_size) ** 2
        if "cls_patch" in self.select_feature:
            _num_patches += 1
        return _num_patches

    @property
    def image_size(self):
        return self.config.image_size


class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        self.s2_scales = getattr(args, "s2_scales", "336,672,1008")
        self.s2_scales = list(map(int, self.s2_scales.split(",")))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        super().__init__(vision_tower, args, delay_load)

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, "unfreeze_mm_vision_tower", False):
            self.image_processor.size["shortest_edge"] = self.s2_image_size
            self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        vision_dtype = self._infer_dtype()
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name,
            device_map=device_map,
            attn_implementation="flash_attention_2",
            torch_dtype=vision_dtype,
        )
        self.vision_tower.requires_grad_(False)

        self.image_processor.size["shortest_edge"] = self.s2_image_size
        self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype), output_hidden_states=True
        )
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = multiscale_forward(
                    self.forward_feature,
                    image.unsqueeze(0),
                    img_sizes=self.s2_scales,
                    max_split_size=self.s2_split_size,
                    split_forward=True,
                )
                image_features.append(image_feature)
        else:
            image_features = multiscale_forward(
                self.forward_feature,
                images,
                img_sizes=self.s2_scales,
                max_split_size=self.s2_split_size,
                split_forward=True,
            )

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
