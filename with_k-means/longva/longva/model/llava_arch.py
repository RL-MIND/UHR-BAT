#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import math
import os
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from longva.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from longva.mm_utils import get_anyres_image_grid_shape, process_multiscale_remote_image
from longva.utils import rank0_print

try:
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
except Exception:
    apply_rotary_pos_emb = None

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .multimodal_resampler.builder import build_vision_resampler


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

        if not hasattr(self.config, "multiscale_target_sizes"):
            self.config.multiscale_target_sizes = [672, 1344, 2688, 4032]
        if not hasattr(self.config, "scale_residual_base"):
            self.config.scale_residual_base = 64
        self.scale_pos_residuals = nn.ParameterDict()
        self.tile_pos_residuals = nn.ParameterDict()
        self._scale_residual_initialized = False
        # mirror residual containers onto base model like PS3 separates per-scale pos emb on trunk
        base_model = getattr(self, "model", None)
        if base_model is not None and not hasattr(base_model, "scale_pos_residuals"):
            base_model.scale_pos_residuals = self.scale_pos_residuals
            base_model.tile_pos_residuals = self.tile_pos_residuals
            base_model._scale_residual_initialized = self._scale_residual_initialized
        self.multiscale_token_mlp = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(
                f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}"
            )
            incompatible_keys = self.vision_resampler.load_state_dict(
                get_w(mm_projector_weights, "vision_resampler"), strict=False
            )
            rank0_print(
                f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}"
            )


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    @property
    def device(self):
        return next(self.parameters()).device

    def get_2dPool(self, image_feature):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        image_features = self.get_model().vision_resampler(image_features, images=images)
        return image_features

    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        # v2
        # _videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        # rank0_print('line 198: _videos_or_images_features: ', _videos_or_images_features.shape)
        # chunked forward to avoid OOM
        # Disable HF gradient checkpointing to avoid ZeRO double-backward
        vision_tower = self.get_model().get_vision_tower()
        CHUNK_SIZE = 128
        feats = []
        for chunk in torch.split(videos_or_images, CHUNK_SIZE, dim=0):
            feats.append(vision_tower(chunk))
        videos_or_images_features = torch.cat(feats, dim=0)  # [577, 576, 1024]
        # rank0_print("line 207: videosc_or_images_features: ", videos_or_images_features.shape)
        # assert _videos_or_images_features.shape == videos_or_images_features.shape, 'Shape mismatch'
        # assert torch.allclose(_videos_or_images_features, videos_or_images_features), 'Feature mismatch'
        per_videos_or_images_features = torch.split(
            videos_or_images_features, split_sizes, dim=0
        )  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []

        for idx, feat in enumerate(per_videos_or_images_features):
            # feat1 = self.get_model().mm_projector(feat)
            _feat = []
            for chunk in torch.split(feat, CHUNK_SIZE, dim=0):
                _feat.append(self.get_model().mm_projector(chunk))
            feat = torch.cat(_feat, dim=0)
            # assert torch.allclose(feat1, feat2), "Feature mismatch"
            # Post pooling
            if idx in video_idx_in_batch:
                feat = self.get_2dPool(feat)
            all_videos_or_images_features.append(feat)
        return all_videos_or_images_features

    def _compute_token_keep_mask(
        self,
        num_patches: int,
        tokens_per_patch: int,
        pruning_info: Optional[Dict[str, List[int]]],
        keep_target: int,
        prune_ratio: float,
        device: torch.device,
    ) -> torch.Tensor:
        total_tokens = max(0, num_patches * max(tokens_per_patch, 0))
        keep_mask = torch.ones(total_tokens, dtype=torch.bool, device=device)
        if total_tokens == 0:
            return keep_mask

        keep_ratio = getattr(self.config, "sam_keep_ratio", None)
        if keep_ratio is not None:
            keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
            target_keep = int(round(total_tokens * keep_ratio))
        else:
            target_keep = max(0, int(keep_target))

        target_keep = min(target_keep, total_tokens)
        if target_keep <= 0:
            return keep_mask

        target_prune = max(0, total_tokens - target_keep)
        if target_prune == 0 or pruning_info is None:
            return keep_mask

        token_classes = pruning_info.get("token_classes")
        mask_order = pruning_info.get("mask_order") or []
        class_to_indices: Dict[int, List[int]] = {}

        use_token_level = token_classes is not None and len(token_classes) == total_tokens

        rand_seed = int.from_bytes(os.urandom(8), byteorder="big", signed=False)
        rand_gen = torch.Generator(device=device)
        rand_gen.manual_seed(rand_seed)

        if use_token_level:
            for token_idx, cls_id in enumerate(token_classes):
                class_id = cls_id or 0
                class_to_indices.setdefault(class_id, []).append(token_idx)
        else:
            patch_classes = pruning_info.get("patch_classes") or []
            for patch_idx in range(num_patches):
                class_id = patch_classes[patch_idx] if patch_idx < len(patch_classes) else 0
                class_id = class_id or 0
                indices = class_to_indices.setdefault(class_id, [])
                base = patch_idx * tokens_per_patch
                indices.extend(range(base, base + tokens_per_patch))

        prune_sequence = list(mask_order)
        if 0 not in prune_sequence:
            prune_sequence.append(0)

        target_tokens = min(target_prune, total_tokens)
        removed: List[int] = []
        removed_count = 0

        for class_id in prune_sequence:
            if removed_count >= target_tokens:
                break
            token_indices = class_to_indices.get(class_id, [])
            if not token_indices:
                continue
            remove_cap = int(len(token_indices) * prune_ratio)
            if remove_cap <= 0:
                continue
            allowance = min(remove_cap, target_tokens - removed_count)
            if allowance <= 0:
                break
            indices_tensor = torch.tensor(token_indices, device=device, dtype=torch.long)
            perm = torch.randperm(indices_tensor.shape[0], generator=rand_gen, device=device)
            selected = indices_tensor[perm[:allowance]].tolist()
            removed.extend(selected)
            removed_count += allowance

        if removed_count < target_tokens:
            ordered_classes = [cid for cid in prune_sequence if cid in class_to_indices]
            for class_id in ordered_classes:
                if removed_count >= target_tokens:
                    break
                token_indices = class_to_indices[class_id]
                remaining = [idx for idx in token_indices if idx not in removed]
                if not remaining:
                    continue
                allowance = min(target_tokens - removed_count, len(remaining))
                if allowance <= 0:
                    break
                remaining_tensor = torch.tensor(remaining, device=device, dtype=torch.long)
                perm = torch.randperm(remaining_tensor.shape[0], generator=rand_gen, device=device)
                selected = remaining_tensor[perm[:allowance]].tolist()
                removed.extend(selected)
                removed_count += allowance

        if removed:
            remove_idx = torch.tensor(sorted(set(removed)), device=device, dtype=torch.long)
            keep_mask[remove_idx] = False

        return keep_mask

    def _mask_prune_tokens(
        self,
        patch_tokens: torch.Tensor,
        cls_tokens: torch.Tensor,
        pruning_info: Optional[Dict[str, List[int]]],
        keep_target: int,
        prune_ratio: float,
        precomputed_keep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_patches, tokens_per_patch, embed_dim = patch_tokens.shape
        flat_tokens = patch_tokens.reshape(-1, embed_dim)
        total_tokens = flat_tokens.shape[0]
        if precomputed_keep_mask is not None:
            keep_mask = precomputed_keep_mask.to(device=flat_tokens.device, dtype=torch.bool)
            if keep_mask.numel() != total_tokens:
                raise ValueError(
                    f"Precomputed keep mask has length {keep_mask.numel()}, expected {total_tokens} tokens."
                )
        else:
            keep_mask = self._compute_token_keep_mask(
                num_patches,
                tokens_per_patch,
                pruning_info,
                keep_target,
                prune_ratio,
                device=flat_tokens.device,
                )
        kept_tokens = flat_tokens[keep_mask]

        # 对被剪掉的同一掩码类 token 取均值后补回
        aggregated_tokens: List[torch.Tensor] = []
        if pruning_info is not None and keep_mask.numel() == total_tokens:
            removed_mask = ~keep_mask
            if bool(removed_mask.any()):
                class_ids = torch.zeros(total_tokens, device=flat_tokens.device, dtype=torch.long)
                token_classes = pruning_info.get("token_classes")
                patch_classes = pruning_info.get("patch_classes")
                if token_classes is not None and len(token_classes) == total_tokens:
                    class_ids = torch.as_tensor(token_classes, device=flat_tokens.device, dtype=torch.long)
                elif patch_classes is not None and len(patch_classes) == num_patches and tokens_per_patch > 0:
                    per_patch = torch.as_tensor(patch_classes, device=flat_tokens.device, dtype=torch.long)
                    class_ids = per_patch.repeat_interleave(tokens_per_patch)
                removed_classes = class_ids[removed_mask]
                removed_tokens = flat_tokens[removed_mask]
                for cls_id in torch.unique(removed_classes):
                    cls_mask = removed_classes == cls_id
                    if bool(cls_mask.any()):
                        aggregated_tokens.append(removed_tokens[cls_mask].mean(dim=0, keepdim=True))

        if getattr(self.config, "debug_print_token_counts", False):
            rank_str = ""
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank_str = f"[Rank {torch.distributed.get_rank()}] "
            print(
                f"{rank_str}[SAM Prune] kept {kept_tokens.shape[0]} / {total_tokens} tokens "
                f"({kept_tokens.shape[0] / max(1, total_tokens):.2%})"
            )

        cls_flat = cls_tokens.reshape(-1, embed_dim)
        if aggregated_tokens:
            kept_tokens = torch.cat([kept_tokens] + aggregated_tokens, dim=0)
        return torch.cat([cls_flat, kept_tokens], dim=0)

    def _residual_host(self):
        """
        Some callers are the outer CausalLM wrapper (no direct scale_pos_residuals),
        while the actual parameters live on the base model. Route to the first module
        that has the attributes, creating them if needed.
        """
        if hasattr(self, "scale_pos_residuals"):
            return self
        base = getattr(self, "get_model", lambda: None)()
        return base if base is not None else self

    @staticmethod
    def _build_simple_causal_mask(attention_mask: Optional[torch.Tensor], seq_len: int, dtype, device):
        """Minimal causal mask builder when HF helper is unavailable."""
        causal = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        causal = torch.triu(causal, diagonal=1)
        causal = causal.unsqueeze(0).unsqueeze(0)  # [1,1,L,L]
        if attention_mask is not None:
            # attention_mask: [B,L] with 1 for valid
            attn = attention_mask.to(device=device, dtype=dtype)
            attn = (1.0 - attn) * torch.finfo(dtype).min
            attn = attn.unsqueeze(1).unsqueeze(1)  # [B,1,1,L]
            causal = causal + attn
        return causal

    def _init_scale_residuals(self, patch_size: Optional[int]):
        host = self._residual_host()
        if getattr(host, "_scale_residual_initialized", False):
            return
        # 若 load_state_dict 已经填充了 scale_pos_residuals（形状与 checkpoint 一致），
        # 直接使用，避免此处重新零初始化覆盖训练好的权重。
        if hasattr(host, "scale_pos_residuals") and len(host.scale_pos_residuals) > 0:
            host._scale_residual_initialized = True
            return
        if patch_size is None:
            return
        for scale in getattr(self.config, "multiscale_target_sizes", [672, 1344, 2688, 4032]):
            grid_side = max(1, min(int(scale // patch_size), int(getattr(self.config, "scale_residual_base", 64))))
            # fresh random init for new runs; checkpoints will overwrite via load_state_dict
            std = 1.0 / math.sqrt(self.config.hidden_size)
            param = nn.Parameter(torch.randn(1, grid_side * grid_side, self.config.hidden_size) * std)
            host.scale_pos_residuals[str(int(scale))] = param
        host._scale_residual_initialized = True

    def _get_scale_pos_residual(self, scale: int, tokens_per_side: int, device: torch.device, dtype: torch.dtype):
        self._init_scale_residuals(getattr(self.get_vision_tower().config, "patch_size", None))
        host = self._residual_host()
        param = host.scale_pos_residuals.get(str(int(scale)))
        if param is None:
            return None
        grid_side = int(round(math.sqrt(param.shape[1])))
        residual = param
        residual = residual.view(1, grid_side, grid_side, -1).permute(0, 3, 1, 2)
        if tokens_per_side != grid_side:
            residual = F.interpolate(
                residual,
                size=(tokens_per_side, tokens_per_side),
                mode="bilinear",
                align_corners=False,
            )
        residual = residual.permute(0, 2, 3, 1).reshape(tokens_per_side * tokens_per_side, -1)
        return residual.to(device=device, dtype=dtype)

    def _get_tile_pos_residual(self, scale: int, tile_index: int, grid_side: int, device, dtype):
        host = self._residual_host()
        key = f"{int(scale)}_{grid_side}"
        if key not in host.tile_pos_residuals:
            host.tile_pos_residuals[key] = nn.Parameter(torch.zeros(grid_side * grid_side, self.config.hidden_size))
        param = host.tile_pos_residuals[key]
        tile_index = int(tile_index)
        if tile_index >= param.shape[0]:
            tile_index = param.shape[0] - 1
        return param[tile_index].to(device=device, dtype=dtype)

    def encode_multiscale_tiles(
        self,
        tiles_by_scale: Dict[int, Dict],
        masks_by_scale: Optional[Dict[int, Dict]] = None,
        tile_size: int = 336,
        add_scale_residual: bool = True,
        add_tile_residual: bool = False,
    ):
        """
        Encode multiscale tiles (already padded/resized and cut) and align masks.
        tiles_by_scale: scale -> {"tiles": Tensor (T,C,H,W), "coords": list[(r,c)]}
        masks_by_scale: scale -> {"masks": Tensor (T, ts, ts), "coords": list[(r,c)]}
        Returns embeddings_by_scale: scale -> Tensor (T, tokens, hidden)
                masks_flat_by_scale: scale -> Tensor (T*tokens)
                mapping: list of (scale, tile_idx, tokens_per_tile)
        """
        vision_tower = self.get_vision_tower()
        mm_projector = self.get_model().mm_projector
        vision_resampler = self.get_model().vision_resampler
        patch_size = getattr(vision_tower.config, "patch_size", 14)
        tokens_per_side = tile_size // patch_size
        # Normalize potential list input (e.g., per-image list)
        if isinstance(tiles_by_scale, list):
            tiles_by_scale = tiles_by_scale[0] if len(tiles_by_scale) > 0 else {}
        if not isinstance(tiles_by_scale, dict):
            raise ValueError(f"tiles_by_scale must be dict or list of dict, got {type(tiles_by_scale)}")

        embeddings_by_scale = {}
        masks_flat_by_scale = {}
        coords_by_scale = {}
        for scale in sorted(tiles_by_scale.keys()):
            entry = tiles_by_scale[scale]
            pixels = entry["tiles"]
            coords = entry.get("coords", [])
            # If coords missing, try to reuse coords from masks (they are generated together in preprocessing)
            if (not coords or len(coords) != pixels.shape[0]) and masks_by_scale and int(scale) in masks_by_scale:
                mask_coords = masks_by_scale[int(scale)].get("coords", [])
                if len(mask_coords) == pixels.shape[0]:
                    coords = mask_coords
            if not coords or len(coords) != pixels.shape[0]:
                # Fallback to a dense grid when coords are missing or misaligned
                grid_side = int(max(1, round(math.sqrt(pixels.shape[0]))))
                coords = [(i // grid_side, i % grid_side) for i in range(pixels.shape[0])]
                if getattr(self.config, "_log_vision_tokens", False):
                    try:
                        rank0_print(
                            f"[Warn] Missing/short coords for scale {scale}: tiles={pixels.shape[0]}, "
                            f"coords={len(entry.get('coords', []))}; reconstructed dense grid {grid_side}x{grid_side}"
                        )
                    except Exception:
                        pass
            pixels = pixels.to(device=vision_tower.device, dtype=vision_tower.dtype)
            feats = vision_tower(pixels)
            projector_dtype = next(mm_projector.parameters()).dtype
            feats = feats.to(dtype=projector_dtype)
            feats = mm_projector(feats)
            feats = vision_resampler(feats, images=pixels)
            # add residuals
            scale_res = (
                self._get_scale_pos_residual(scale, tokens_per_side, feats.device, feats.dtype)
                if add_scale_residual
                else None
            )
            grid_side = int(scale // tile_size)
            for t in range(feats.shape[0]):
                tile_res = (
                    self._get_tile_pos_residual(scale, t, grid_side, feats.device, feats.dtype)
                    if add_tile_residual
                    else None
                )
                if scale_res is not None:
                    feats[t, : scale_res.shape[0], :] = feats[t, : scale_res.shape[0], :] + scale_res
                if tile_res is not None:
                    feats[t] = feats[t] + tile_res
            embeddings_by_scale[int(scale)] = feats
            if masks_by_scale and int(scale) in masks_by_scale:
                mk = masks_by_scale[int(scale)]["masks"]
                if mk.shape[0] != feats.shape[0]:
                    mk = mk[: feats.shape[0]]
                masks_flat_by_scale[int(scale)] = mk.reshape(mk.shape[0], -1).to(device=feats.device)
            coords_by_scale[int(scale)] = coords
        return embeddings_by_scale, masks_flat_by_scale, coords_by_scale

    def encode_multiscale_pixels(self, image, target_sizes: Optional[Sequence[int]] = None):
        vision_tower = self.get_vision_tower()
        if vision_tower is None:
            raise ValueError("Vision tower is not initialized.")
        if target_sizes is None:
            target_sizes = getattr(self.config, "multiscale_target_sizes", [672, 1344, 2688, 4032])
        target_sizes = [int(s) for s in target_sizes]

        if isinstance(image, dict):
            multiscale_pixels = {int(k): v for k, v in image.items()}
        elif isinstance(image, torch.Tensor):
            inferred_size = image.shape[-1]
            multiscale_pixels = {int(inferred_size): image}
        else:
            processor = getattr(vision_tower, "image_processor", None)
            if processor is None:
                raise ValueError("Vision tower is missing image_processor; cannot preprocess image.")
            multiscale_pixels = process_multiscale_remote_image(image, processor, target_sizes=target_sizes)

        outputs: Dict[int, torch.Tensor] = {}
        for scale in target_sizes:
            pixels = multiscale_pixels.get(int(scale))
            if pixels is None:
                continue
            if pixels.ndim == 3:
                pixels = pixels.unsqueeze(0)
            pixels = pixels.to(device=vision_tower.device, dtype=vision_tower.dtype)
            feats = vision_tower(pixels)
            feats = self.get_model().mm_projector(feats)
            feats = self.get_model().vision_resampler(feats, images=pixels)
            outputs[int(scale)] = feats
        return outputs

    def select_tokens_with_masks(
        self,
        attn_by_scale: Dict[int, torch.Tensor],
        token_embeds_by_scale: Dict[int, torch.Tensor],
        masks_by_scale: Dict[int, torch.Tensor],
        topk_per_scale: Sequence[int],
        target_sizes: Optional[Sequence[int]] = None,
        add_scale_residual: bool = True,
    ):
        """
        Select tokens per scale using attention-driven K-Means clusters with positional awareness (segmentation masks are ignored).
        Clusters are scored by their mean attention; tokens whose attention is above the cluster mean are kept
        individually, while the remaining tokens within that cluster are averaged into a single token. The existing
        top-k per-scale budget and downstream pruning flow remain unchanged.
        """
        if target_sizes is None:
            target_sizes = sorted(attn_by_scale.keys())
        target_sizes = [int(s) for s in target_sizes]
        topk_list = [int(x) for x in topk_per_scale]
        if len(topk_list) < len(target_sizes):
            last = topk_list[-1] if topk_list else 0
            topk_list = topk_list + [last] * (len(target_sizes) - len(topk_list))

        batch_size = None
        per_batch_embeds: List[List[torch.Tensor]] = []
        per_batch_indices: List[List[Dict]] = []

        def _run_kmeans(features: torch.Tensor, num_clusters: int, iters: int = 10, normalize: bool = True) -> torch.Tensor:
            """
            Lightweight K-Means clustering on token features. Returns cluster labels for each token.
            """
            if num_clusters <= 1 or features.shape[0] <= 1:
                return torch.zeros(features.shape[0], device=features.device, dtype=torch.long)
            feats = F.normalize(features.detach(), dim=1) if normalize else features.detach()
            # Init centers with random tokens to avoid extra dependencies.
            init_idx = torch.randperm(feats.shape[0], device=feats.device)[:num_clusters]
            centers = feats[init_idx]
            for _ in range(max(1, iters)):
                distances = torch.cdist(feats, centers, p=2)
                labels = distances.argmin(dim=1)
                new_centers = []
                for cid in range(num_clusters):
                    mask = labels == cid
                    if mask.any():
                        new_centers.append(feats[mask].mean(dim=0))
                    else:
                        # Keep previous center when a cluster is empty
                        new_centers.append(centers[cid])
                new_centers = torch.stack(new_centers, dim=0)
                if torch.allclose(new_centers, centers):
                    break
                centers = new_centers
            return labels

        for scale_idx, scale in enumerate(target_sizes):
            attn_map = attn_by_scale.get(int(scale))
            token_embeds = token_embeds_by_scale.get(int(scale))
            if attn_map is None or token_embeds is None:
                continue
            # attn_map: (tiles, ts, ts); token_embeds: (tiles, tokens, hidden)
            k_limit = max(0, int(topk_list[scale_idx]))
            if k_limit <= 0:
                continue
            tiles = attn_map.shape[0]
            tokens_per_tile = token_embeds.shape[1] if token_embeds.dim() >= 2 else 0
            tokens_per_side = int(round(math.sqrt(tokens_per_tile))) if tokens_per_tile > 0 else None
            ts = attn_map.shape[1] if attn_map.dim() >= 3 else None
            if tokens_per_side is None or tokens_per_side * tokens_per_side != tokens_per_tile or ts is None or ts != tokens_per_side:
                raise RuntimeError(
                    f"[Error] Multiscale alignment失败: scale={scale}, attn_map={tuple(attn_map.shape)}, token_embeds={tuple(token_embeds.shape)}, "
                    f"tokens_per_side={tokens_per_side}, tokens_per_tile={tokens_per_tile}"
                )
            if batch_size is None:
                batch_size = 1
                per_batch_embeds = [[] for _ in range(batch_size)]
                per_batch_indices = [[] for _ in range(batch_size)]

            if getattr(self.config, "_log_vision_tokens", False):
                try:
                    rank0_print(
                        f"[Debug] Scale {scale}: attn_shape={tuple(attn_map.shape)}, tokens_per_tile={tokens_per_tile}, "
                        f"tiles={tiles}, k_limit={k_limit}"
                    )
                except Exception:
                    pass

            residual_map = None
            if add_scale_residual and tokens_per_side is not None:
                residual_map = self._get_scale_pos_residual(scale, tokens_per_side, token_embeds.device, token_embeds.dtype)

            flat_attn = attn_map.reshape(-1)
            embeds = token_embeds.reshape(-1, token_embeds.shape[-1])
            if residual_map is not None:
                embeds = embeds + residual_map.repeat(attn_map.shape[0], 1)

            # Build positional coordinates for clustering (flattened to match attn/embeds order).
            coord_scale = float(getattr(self.config, "kmeans_coord_scale", 0.05))
            feat_scale = float(getattr(self.config, "kmeans_feat_scale", 1.0))
            kmeans_max_iters = int(getattr(self.config, "kmeans_max_iters", 10))
            cfg_num_clusters = getattr(self.config, "kmeans_num_clusters", None)
            yy, xx = torch.meshgrid(
                torch.arange(ts, device=attn_map.device),
                torch.arange(ts, device=attn_map.device),
                indexing="ij",
            )
            xx = xx.float() / (max(1, ts - 1))
            yy = yy.float() / (max(1, ts - 1))
            coords_flat = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
            coords_repeated = coords_flat.repeat(tiles, 1)
            # Prepare clustering features: normalized embeds with optional scaling + coords
            cluster_feats = torch.cat(
                [feat_scale * F.normalize(embeds, dim=1), coord_scale * coords_repeated],
                dim=1,
            )

            # 防止索引越界：确保注意力/掩码长度与 token 数一致
            if flat_attn.numel() != embeds.shape[0]:
                raise ValueError(
                    f"[Error] scale {scale}: attn tokens {flat_attn.numel()} != embed tokens {embeds.shape[0]}"
                )
            num_tokens = embeds.shape[0]
            num_clusters_cfg = None
            try:
                num_clusters_cfg = int(cfg_num_clusters) if cfg_num_clusters is not None else None
            except Exception:
                num_clusters_cfg = None
            if num_clusters_cfg is not None and num_clusters_cfg > 0:
                num_clusters = min(max(1, num_clusters_cfg), num_tokens)
            else:
                num_clusters = min(max(1, k_limit), num_tokens)
            cluster_labels = _run_kmeans(cluster_feats, num_clusters, iters=kmeans_max_iters, normalize=False)
            cluster_ids = torch.unique(cluster_labels)
            # Order clusters by mean attention descending
            cluster_scores = []
            for cid in cluster_ids.tolist():
                mask = cluster_labels == cid
                score = float(flat_attn[mask].mean().item()) if mask.any() else float("-inf")
                cluster_scores.append((cid, score))
            cluster_scores.sort(key=lambda x: x[1], reverse=True)

            remaining = k_limit
            selected_embeds: List[torch.Tensor] = []
            selected_meta: List[Dict] = []

            for cid, score in cluster_scores:
                if remaining <= 0:
                    break
                pos_idx = torch.nonzero(cluster_labels == cid, as_tuple=False).squeeze(-1)
                if pos_idx.numel() == 0:
                    continue
                class_attn = flat_attn[pos_idx]
                class_mean = float(class_attn.mean().item())
                high_mask = class_attn > class_mean
                high_pos = pos_idx[high_mask]
                high_scores = class_attn[high_mask]
                if high_pos.numel() > 0 and remaining > 0:
                    order = torch.argsort(high_scores, descending=True)
                    for idx in order.tolist():
                        if remaining <= 0:
                            break
                        sel_pos = int(high_pos[idx].item())
                        sel_score = float(high_scores[idx].item())
                        selected_embeds.append(embeds[sel_pos].unsqueeze(0))
                        selected_meta.append({"scale": int(scale), "pos": [sel_pos], "score": sel_score})
                        remaining -= 1
                if remaining <= 0:
                    break
                low_mask = ~high_mask
                if low_mask.any() and remaining > 0:
                    low_pos = pos_idx[low_mask]
                    mean_embed = embeds[low_pos].mean(dim=0, keepdim=True)
                    mean_score = float(class_attn[low_mask].mean().item())
                    selected_embeds.append(mean_embed)
                    selected_meta.append(
                        {
                            "scale": int(scale),
                            "pos": low_pos.tolist(),
                            "score": mean_score,
                            "aggregated": True,
                            "cluster": int(cid),
                        }
                    )
                    remaining -= 1

            if selected_embeds:
                per_batch_embeds[0].append(torch.cat(selected_embeds, dim=0))
                per_batch_indices[0].append({"scale": int(scale), "tokens": selected_meta})
                if getattr(self.config, "_log_vision_tokens", False):
                    try:
                        sel_count = per_batch_embeds[0][-1].shape[0]
                        rank0_print(f"[Debug] Scale {scale}: selected {sel_count} tokens (k_limit={k_limit})")
                    except Exception:
                        pass

        if batch_size is None:
            return {"selected_embeddings": [], "selected_indices": []}

        combined = []
        for embeds_list in per_batch_embeds:
            if embeds_list:
                combined.append(torch.cat(embeds_list, dim=0))
            else:
                combined.append(
                    torch.zeros(
                        0,
                        self.config.hidden_size,
                        device=self.device,
                        dtype=next(self.parameters()).dtype,
                    )
                )

        mlp = getattr(self, "multiscale_token_mlp", None)
        if mlp is None:
            base_model = getattr(self, "get_model", lambda: None)()
            mlp = getattr(base_model, "multiscale_token_mlp", None)
        if mlp is None:
            raise AttributeError("[Error] multiscale_token_mlp 未初始化，无法处理多尺度选中 token。")

        transformed = [mlp(t) if t.numel() > 0 else t for t in combined]

        return {"selected_embeddings": transformed, "selected_indices": per_batch_indices}

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        modalities=["image"],
        image_sizes=None,
        vision_pruning=None,
        return_mm_input_ids: bool = False,
        image_features_override: Optional[List[torch.Tensor]] = None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or (images is None and image_features_override is None) or input_ids.shape[1] == 1:
            if return_mm_input_ids:
                return input_ids, position_ids, attention_mask, past_key_values, None, labels, input_ids
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if image_features_override is not None:
            image_features = image_features_override
        elif type(images) is list or images.ndim == 5 or images.ndim == 4:
            if type(images) is not list and images.ndim == 4:
                images = images.unsqueeze(1)
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            if vision_pruning is not None:
                pruning_list = list(vision_pruning)
            else:
                pruning_list = [None] * len(images_list)
            if len(pruning_list) != len(images_list):
                pruning_list = [None] * len(images_list)

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            # rank0_print("line 246: concat_images: ", concat_images.shape)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # _image_features = self.encode_multimodals_(concat_images, video_idx_in_batch, split_sizes)
            # assert torch.allclose(image_features[0], _image_features[0]), "Feature mismatch"
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            base_keep_target = max(0, int(getattr(self.config, "sam_prune_target", 0)))
            num_images = max(1, len(images_list))
            if base_keep_target > 0:
                per_image_target = max(1, math.ceil(base_keep_target / num_images))
            else:
                per_image_target = 0
            prune_ratio = float(getattr(self.config, "sam_prune_ratio", 0.9))

            if mm_patch_merge_type.startswith("sam_mask"):
                new_image_features = []
                mm_projector = self.get_model().mm_projector
                patch_chunk = 64
                for image_idx, _image in enumerate(images_list):
                    pruning_info = pruning_list[image_idx] if image_idx < len(pruning_list) else None
                    if image_idx in video_idx_in_batch:
                        image_feature = self.encode_multimodals(_image, video_idx_in_batch, split_sizes)
                        image_feature = image_feature.flatten(0, 1).to(memory_format=torch.channels_last)
                        new_image_features.append(image_feature)
                        continue
                    if _image.size(0) <= 1:
                        image_feature = self.encode_multimodals(_image, video_idx_in_batch, split_sizes)[0]
                        new_image_features.append(image_feature)
                        continue
                    patch_tensor = _image[1:]
                    token_keep_mask = None
                    tokens_per_patch_meta = None
                    if pruning_info is not None and patch_tensor.size(0) > 0:
                        tokens_per_patch_meta = pruning_info.get("tokens_per_patch")
                        if tokens_per_patch_meta is None:
                            tokens_per_side = pruning_info.get("tokens_per_side")
                            if tokens_per_side is not None:
                                tokens_per_patch_meta = int(tokens_per_side) * int(tokens_per_side)
                    if (
                        pruning_info is not None
                        and per_image_target > 0
                        and tokens_per_patch_meta is not None
                        and tokens_per_patch_meta > 0
                    ):
                        keep_mask_full = self._compute_token_keep_mask(
                            patch_tensor.size(0),
                            tokens_per_patch_meta,
                            pruning_info,
                            per_image_target,
                            prune_ratio,
                            device=torch.device("cpu"),
                        )
                        expected_tokens = patch_tensor.size(0) * tokens_per_patch_meta
                        if keep_mask_full.numel() == expected_tokens:
                            keep_mask_full = keep_mask_full.view(patch_tensor.size(0), tokens_per_patch_meta)
                            patch_keep_mask = keep_mask_full.any(dim=1)
                            if not bool(patch_keep_mask.any()):
                                patch_keep_mask = torch.zeros_like(patch_keep_mask, dtype=torch.bool)
                                patch_keep_mask[0] = True
                                keep_mask_full = torch.ones_like(keep_mask_full, dtype=torch.bool)
                            patch_tensor = patch_tensor[patch_keep_mask]
                            token_keep_mask = keep_mask_full[patch_keep_mask].reshape(-1).to(torch.bool)
                            pruning_info = None
                    encoded_chunks = []
                    for chunk in torch.split(patch_tensor, patch_chunk, dim=0):
                        chunk_feats = vision_tower(chunk, select_feature_type="cls_patch")
                        chunk_feats = chunk_feats.contiguous()
                        encoded_chunks.append(chunk_feats)
                    patch_features = torch.cat(encoded_chunks, dim=0)
                    num_patches, tokens_total, hidden_dim = patch_features.shape
                    flat_features = patch_features.reshape(-1, hidden_dim)
                    projector_chunk = max(1, patch_chunk * tokens_total)
                    projected_chunks = []
                    for proj_chunk in torch.split(flat_features, projector_chunk, dim=0):
                        projected_chunks.append(mm_projector(proj_chunk))
                    projected = torch.cat(projected_chunks, dim=0).view(num_patches, tokens_total, -1)
                    cls_tokens = projected[:, :1, :]
                    patch_tokens = projected[:, 1:, :]
                    image_feature = self._mask_prune_tokens(
                        patch_tokens,
                        cls_tokens,
                        pruning_info,
                        per_image_target,
                        prune_ratio,
                        precomputed_keep_mask=token_keep_mask,
                    )
                    new_image_features.append(image_feature)
                image_features = new_image_features

            elif mm_patch_merge_type == "flat":
                base_image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
                image_features = [x.flatten(0, 1) for x in base_image_features]

            elif mm_patch_merge_type == "unires" or "unires" in mm_patch_merge_type:
                new_image_features = []
                mm_projector = self.get_model().mm_projector
                patch_chunk = 64
                max_tokens = None
                if "max" in mm_patch_merge_type:
                    match = re.search(r"max(\d+)", mm_patch_merge_type)
                    if match:
                        max_tokens = int(match.group(1))
                for image_idx, _image in enumerate(images_list):
                    pruning_info = pruning_list[image_idx] if image_idx < len(pruning_list) else None
                    if image_idx in video_idx_in_batch:
                        image_feature = self.encode_multimodals(_image, video_idx_in_batch, split_sizes)
                        image_feature = image_feature.flatten(0, 1).to(memory_format=torch.channels_last)
                        new_image_features.append(image_feature)
                        continue
                    elif _image.size(0) > 1:
                        patch_tensor = _image[1:]
                        encoded_chunks = []
                        for chunk in torch.split(patch_tensor, patch_chunk, dim=0):
                            chunk_feats = vision_tower(chunk, select_feature_type="cls_patch")
                            encoded_chunks.append(chunk_feats.contiguous())
                        if not encoded_chunks:
                            image_feature = self.encode_multimodals(_image, video_idx_in_batch, split_sizes)[0]
                            new_image_features.append(image_feature)
                            continue
                        patch_features = torch.cat(encoded_chunks, dim=0)
                        num_patches, tokens_total, hidden_dim = patch_features.shape
                        flat_features = patch_features.reshape(-1, hidden_dim)
                        projector_chunk = max(1, patch_chunk * tokens_total)
                        projected_chunks = []
                        for proj_chunk in torch.split(flat_features, projector_chunk, dim=0):
                            projected_chunks.append(mm_projector(proj_chunk))
                        projected = torch.cat(projected_chunks, dim=0).view(num_patches, tokens_total, -1)
                        cls_tokens = projected[:, :1, :]
                        patch_tokens = projected[:, 1:, :]
                        total_tokens = patch_tokens.reshape(-1, patch_tokens.shape[-1]).shape[0]
                        target_keep = per_image_target if per_image_target > 0 else total_tokens
                        if max_tokens is not None:
                            target_keep = min(target_keep, max_tokens)
                        image_feature = self._mask_prune_tokens(
                            patch_tokens,
                            cls_tokens,
                            pruning_info,
                            target_keep,
                            prune_ratio,
                        )
                    else:
                        image_feature = self.encode_multimodals(_image, video_idx_in_batch, split_sizes)[0]
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            error_message = """
            Something is wrong with the input shape. Most likely, you did not wrap the video input in a list:
            This is correct:
                model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
            This is wrong:
                model.generate(input_ids, images=video_tensor,  modalities=["video"], **gen_kwargs)
            """
            raise ValueError(error_message)
            # image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        pad_token_id = getattr(self.config, "pad_token_id", 0)
        if pad_token_id is None:
            pad_token_id = 0
        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        mm_input_ids = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                if return_mm_input_ids:
                    mm_input_ids.append(cur_input_ids)
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_mm_ids = [] if return_mm_input_ids else None

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if return_mm_input_ids:
                    cur_new_mm_ids.append(cur_input_ids_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    # Normalize image features to 2D (tokens, hidden)
                    if isinstance(cur_image_features, torch.Tensor):
                        if cur_image_features.dim() > 2:
                            cur_image_features = cur_image_features.view(-1, cur_image_features.shape[-1])
                    elif isinstance(cur_image_features, (list, tuple)):
                        flat_parts = []
                        for part in cur_image_features:
                            if part is None:
                                continue
                            if isinstance(part, torch.Tensor):
                                if part.dim() > 2:
                                    part = part.view(-1, part.shape[-1])
                                flat_parts.append(part)
                        if flat_parts:
                            cur_image_features = torch.cat(flat_parts, dim=0)
                        else:
                            # create empty placeholder
                            hidden = cur_input_embeds_no_im[i].shape[-1] if cur_input_embeds_no_im else self.config.hidden_size
                            cur_image_features = torch.zeros(0, hidden, device=cur_input_embeds_no_im[i].device if cur_input_embeds_no_im else self.device, dtype=self.dtype)
                    else:
                        hidden = cur_input_embeds_no_im[i].shape[-1] if cur_input_embeds_no_im else self.config.hidden_size
                        cur_image_features = torch.zeros(0, hidden, device=cur_input_embeds_no_im[i].device if cur_input_embeds_no_im else self.device, dtype=self.dtype)
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
                    if return_mm_input_ids:
                        cur_new_mm_ids.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                getattr(self.config, "image_token_index", IMAGE_TOKEN_INDEX),
                                device=cur_input_ids.device,
                                dtype=cur_input_ids.dtype,
                            )
                        )
            # rank0_print("line 464: cur_new_input_embeds", [x.shape for x in cur_new_input_embeds])
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            if return_mm_input_ids:
                cur_new_mm_ids = [x.to(self.device) for x in cur_new_mm_ids]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            if return_mm_input_ids and cur_new_mm_ids is not None:
                cur_new_mm_ids = torch.cat(cur_new_mm_ids)

            # Debug: report vision token count per sample
            if hasattr(self, "config") and getattr(self.config, "_log_vision_tokens", False):
                try:
                    # text tokens are the concatenation of all non-image pieces
                    text_tokens = sum(split_sizes)
                    vision_tokens = cur_new_input_embeds.shape[0] - text_tokens
                    rank0_print(f"[Debug] Sample {batch_idx}: vision tokens inserted = {vision_tokens}")
                except Exception:
                    pass

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            if return_mm_input_ids and cur_new_mm_ids is not None:
                mm_input_ids.append(cur_new_mm_ids)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        if return_mm_input_ids:
            mm_input_ids = [x[:tokenizer_model_max_length] for x, modality in zip(mm_input_ids, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        mm_input_ids_padded = None
        if return_mm_input_ids:
            mm_input_ids_padded = torch.full(
                (batch_size, max_len),
                pad_token_id,
                dtype=_input_ids.dtype if hasattr(_input_ids, "dtype") else torch.long,
                device=new_input_embeds[0].device,
            )

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            cur_mm_ids = mm_input_ids[i] if return_mm_input_ids else None
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
                    if return_mm_input_ids and cur_mm_ids is not None:
                        mm_input_ids_padded[i, -cur_len:] = cur_mm_ids
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
                    if return_mm_input_ids and cur_mm_ids is not None:
                        mm_input_ids_padded[i, :cur_len] = cur_mm_ids

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
        if return_mm_input_ids:
            mm_input_ids_out = mm_input_ids_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = (
                torch.arange(new_input_embeds.size(1), device=new_input_embeds.device)
                .unsqueeze(0)
                .to(new_input_embeds.device)
            )
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        if return_mm_input_ids:
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, mm_input_ids_out
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    @torch.no_grad()
    def compute_multiscale_image_attention(
        self,
        image,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_sizes: Sequence[int] = (672, 1344, 2688, 4032),
        position_ids: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = None,
        num_images: Optional[int] = None,
        multiscale_masks: Optional[Dict[int, Dict]] = None,
        base_size: Optional[int] = None,
    ) -> Dict:
        """
        Compute last-layer attention from the final text token to vision tokens using a 1344x1344 base image.

        Steps:
        1) Pad and resize the image to the provided target_sizes (default: 672/1344/2688/4032) and encode each with
           the vision tower to record token grid sizes.
        2) Run the 1344x1344 view through the vision MLP and language model encoder to obtain hidden states.
        3) Manually build Q/K for the final decoder layer, compute attention from the last text token to all image
           tokens, average over heads, and interpolate that map to every requested scale.
        """

        vision_tower = self.get_vision_tower()
        if vision_tower is None:
            raise ValueError("Vision tower is not initialized; cannot compute image attention.")
        if modalities is None:
            modalities = ["image"]

        base_size = base_size or getattr(self.config, "multiscale_base_attention_size", 672)
        if target_sizes is None:
            target_sizes = getattr(self.config, "multiscale_target_sizes", (672, 1344, 2688, 4032))
        target_sizes = list(dict.fromkeys(int(s) for s in target_sizes))
        if base_size not in target_sizes:
            target_sizes.append(base_size)

        base_model = self.get_model()

        with torch.no_grad():
            # Build per-scale tiles and embeddings
            # Normalize possible list inputs (e.g., coming from batch with multiple images)
            if isinstance(image, list) and len(image) > 0:
                image = image[0]

            if isinstance(image, dict) and "tiles" in next(iter(image.values())):
                tiles_by_scale = {int(k): v for k, v in image.items()}
            else:
                if isinstance(image, dict):
                    multiscale_pixels = {int(k): v for k, v in image.items()}
                elif isinstance(image, torch.Tensor):
                    inferred = int(image.shape[-1])
                    multiscale_pixels = {inferred: image}
                else:
                    processor = getattr(vision_tower, "image_processor", None)
                    if processor is None:
                        raise ValueError("Vision tower is missing image_processor; cannot preprocess image.")
                    multiscale_pixels = process_multiscale_remote_image(image, processor, target_sizes=target_sizes)
                tiles_by_scale = {
                    k: {"tiles": (v if v.ndim == 4 else v.unsqueeze(0)), "coords": []}
                    for k, v in multiscale_pixels.items()
                }
            # If caller only passes tile tensors but not coords, use mask coords (if present) to avoid misalignment
            if multiscale_masks:
                for k, entry in tiles_by_scale.items():
                    if entry.get("coords"):
                        continue
                    mask_coords = multiscale_masks.get(int(k), {}).get("coords", [])
                    if mask_coords and len(mask_coords) == entry["tiles"].shape[0]:
                        entry["coords"] = mask_coords
            embeddings_by_scale, _, coords_by_scale = self.encode_multiscale_tiles(
                tiles_by_scale, masks_by_scale=multiscale_masks, add_scale_residual=True, add_tile_residual=False
            )
            if base_size not in embeddings_by_scale:
                raise ValueError(f"Base scale {base_size} not available for attention.")
            base_feats = embeddings_by_scale[base_size]
            token_counts_by_scale: Dict[int, Dict[str, int]] = {}
            for scale, feats in embeddings_by_scale.items():
                tokens_per_side = int(round(math.sqrt(max(feats.shape[1], 1))))
                if tokens_per_side * tokens_per_side != feats.shape[1] and feats.shape[1] > 1:
                    alt_side = int(round(math.sqrt(max(feats.shape[1] - 1, 1))))
                    if alt_side * alt_side == feats.shape[1] - 1:
                        tokens_per_side = alt_side
                token_counts_by_scale[scale] = {
                    "num_tokens": int(feats.shape[1]),
                    "tokens_per_side": tokens_per_side,
                }
            flat_cat = base_feats.reshape(1, -1, base_feats.shape[-1])

            # Ensure batched inputs
            if input_ids is not None and input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            if position_ids is not None and position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)

            (
                _,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                mm_input_ids,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                images=None,
                modalities=modalities,
                image_sizes=None,
                vision_pruning=None,
                return_mm_input_ids=True,
                image_features_override=[flat_cat.squeeze(0)],
            )

            base_model = self.get_model()

            # First pass: get hidden_states without storing attentions
            base_outputs = base_model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=False,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states_all = base_outputs.hidden_states
            if hidden_states_all is None or len(hidden_states_all) < 2:
                raise ValueError("Expected hidden_states to extract last layer input.")
            hidden_before_last = hidden_states_all[-2]

            cache_position = torch.arange(
                hidden_before_last.shape[1], device=hidden_before_last.device, dtype=torch.long
            )
            pos_ids = position_ids if position_ids is not None else cache_position.unsqueeze(0)
            rotary = getattr(base_model, "rotary_emb", None)
            if rotary is None and hasattr(base_model, "model"):
                rotary = getattr(base_model.model, "rotary_emb", None)
            if rotary is None:
                # Fall back to layer-held rotary (common in HF Qwen2 implementations)
                try:
                    rotary = getattr(base_model.layers[0].self_attn, "rotary_emb", None)
                except Exception:
                    rotary = None
            if rotary is None and hasattr(base_model, "model"):
                try:
                    rotary = getattr(base_model.model.layers[0].self_attn, "rotary_emb", None)
                except Exception:
                    rotary = None
            if rotary is None:
                raise AttributeError("Base model missing rotary_emb for positional encoding.")
            position_embeddings = rotary(hidden_before_last, seq_len=hidden_before_last.shape[1])
            causal_mask = None
            try:
                from transformers.models.qwen2.modeling_qwen2 import create_causal_mask as _hf_create_causal_mask

                causal_mask = _hf_create_causal_mask(
                    config=getattr(base_model, "config", None) or getattr(getattr(base_model, "model", None), "config"),
                    input_embeds=hidden_before_last,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=None,
                    position_ids=pos_ids,
                )
                if isinstance(causal_mask, dict):
                    causal_mask = causal_mask.get("full_attention")
            except Exception:
                causal_mask = self._build_simple_causal_mask(
                    attention_mask=attention_mask,
                    seq_len=hidden_before_last.shape[1],
                    dtype=hidden_before_last.dtype,
                    device=hidden_before_last.device,
                )

            layers = getattr(base_model, "layers", None)
            if layers is None and hasattr(base_model, "model"):
                layers = getattr(base_model.model, "layers", None)
            if layers is None:
                raise AttributeError("Base model missing transformer layers for attention extraction.")
            last_layer = layers[-1]
            self_attn = getattr(last_layer, "self_attn", None)
            if self_attn is None:
                raise AttributeError("Last layer missing self_attn module for attention extraction.")
            normed = last_layer.input_layernorm(hidden_before_last)
            hidden_shape = (*normed.shape[:2], -1, self_attn.head_dim)
            query_states = self_attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
            key_states = self_attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            if apply_rotary_pos_emb is None:
                raise RuntimeError("apply_rotary_pos_emb is required to compute attention scores.")
            # HF Qwen2 apply_rotary_pos_emb expects position_ids
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids=pos_ids
            )
            # Align KV heads with Q heads for MQA/GQA models (e.g., Qwen2)
            num_q_heads = query_states.shape[1]
            num_k_heads = key_states.shape[1]
            if num_k_heads != num_q_heads:
                if num_q_heads % num_k_heads != 0:
                    raise RuntimeError(
                        f"Head mismatch: q_heads={num_q_heads}, k_heads={num_k_heads}, cannot broadcast."
                    )
                repeat_factor = num_q_heads // num_k_heads
                key_states = key_states.repeat_interleave(repeat_factor, dim=1)
            attn_scaling = getattr(self_attn, "scaling", None)
            if attn_scaling is None:
                attn_scaling = 1.0 / math.sqrt(getattr(self_attn, "head_dim", query_states.shape[-1]))

        tokens_per_tile_raw = base_feats.shape[1]
        num_tiles_base = base_feats.shape[0]
        tile_tokens_side = int(round(math.sqrt(max(tokens_per_tile_raw, 1))))
        cls_tokens_per_tile = 0
        if tile_tokens_side * tile_tokens_side != tokens_per_tile_raw and tokens_per_tile_raw > 1:
            alt_side = int(round(math.sqrt(max(tokens_per_tile_raw - 1, 1))))
            if alt_side * alt_side == tokens_per_tile_raw - 1:
                cls_tokens_per_tile = tokens_per_tile_raw - alt_side * alt_side
                tile_tokens_side = alt_side
            else:
                raise RuntimeError(
                    f"Base tile tokens are not square (tokens_per_tile={tokens_per_tile_raw}); cannot map attention grid."
                )
        patch_tokens_per_tile = tile_tokens_side * tile_tokens_side
        expected_img_tokens_raw = num_tiles_base * tokens_per_tile_raw
        expected_img_tokens = num_tiles_base * patch_tokens_per_tile
        image_token_idx = getattr(self.config, "image_token_index", IMAGE_TOKEN_INDEX)
        image_token_mask = mm_input_ids == image_token_idx
        pad_token_id = getattr(self.config, "pad_token_id", 0)
        valid_text_mask = (mm_input_ids != image_token_idx) & (mm_input_ids != pad_token_id)
        if attention_mask is not None:
            valid_text_mask = valid_text_mask & attention_mask.to(dtype=torch.bool, device=mm_input_ids.device)
        seq_positions = torch.arange(mm_input_ids.size(1), device=mm_input_ids.device).unsqueeze(0).expand_as(
            mm_input_ids
        )
        text_positions = seq_positions.masked_fill(~valid_text_mask, -1)
        last_text_idx = text_positions.max(dim=1).values
        if (last_text_idx < 0).any():
            raise ValueError("Failed to locate a valid text token for attention extraction.")
        image_positions = []
        for b in range(mm_input_ids.size(0)):
            img_pos = torch.nonzero(image_token_mask[b].reshape(-1), as_tuple=False).squeeze(-1)
            if img_pos.numel() < expected_img_tokens_raw:
                raise RuntimeError(
                    f"Image token count too small: got {img_pos.numel()}, expected {expected_img_tokens_raw}"
                )
            if img_pos.numel() > expected_img_tokens_raw:
                if getattr(self.config, "_log_vision_tokens", False):
                    rank0_print(
                        f"[Warn] image tokens exceed expected: got {img_pos.numel()}, expected {expected_img_tokens_raw}; truncating"
                    )
                img_pos = img_pos[:expected_img_tokens_raw]
            if img_pos.numel() % tokens_per_tile_raw != 0:
                raise RuntimeError(
                    f"Image token layout mismatch: got {img_pos.numel()} tokens, not divisible by tokens_per_tile={tokens_per_tile_raw}"
                )
            tiles_in_seq = img_pos.numel() // tokens_per_tile_raw
            if tiles_in_seq != num_tiles_base:
                raise RuntimeError(f"Derived image tiles mismatch: got {tiles_in_seq}, expected {num_tiles_base}")
            if cls_tokens_per_tile > 0:
                img_pos = img_pos.view(tiles_in_seq, tokens_per_tile_raw)[:, cls_tokens_per_tile:]
                img_pos = img_pos.reshape(-1)
            if img_pos.numel() != expected_img_tokens:
                raise RuntimeError(
                    f"Derived image positions mismatch for batch {b}: got {img_pos.numel()}, expected {expected_img_tokens}"
                )
            image_positions.append(img_pos)
        if getattr(self.config, "_log_vision_tokens", False):
            try:
                rank0_print(
                    f"[AttnDebug] base_scale={base_size}, tiles={base_feats.shape[0]}, tokens_per_tile={base_feats.shape[1]}, "
                    f"expected_img_tokens={expected_img_tokens}, seq_len={mm_input_ids.shape[1]}"
                )
            except Exception:
                pass

        attn_vectors = []
        for b in range(mm_input_ids.size(0)):
            img_pos = image_positions[b]
            if img_pos.numel() != expected_img_tokens:
                raise RuntimeError(
                    f"Derived image positions mismatch for batch {b}: got {img_pos.numel()}, expected {expected_img_tokens}"
                )
            q_last = query_states[b, :, last_text_idx[b].item(), :]
            k_all = key_states[b]
            scores = torch.matmul(
                q_last.unsqueeze(1).to(torch.float32), k_all.transpose(-2, -1).to(torch.float32)
            ).squeeze(1)
            scores = scores * float(attn_scaling)
            if causal_mask is not None:
                mask_row = causal_mask[b, 0, last_text_idx[b].item(), :]
                scores = scores + mask_row.to(dtype=scores.dtype)
            probs = torch.softmax(scores, dim=-1)
            attn_vectors.append(probs[:, img_pos].mean(dim=0))

        # Map base attention back to base tiles, then interpolate to other scales
        base_tokens_per_tile = patch_tokens_per_tile
        base_tile_coords = coords_by_scale.get(base_size, [(i, 0) for i in range(base_feats.shape[0])])
        if base_tile_coords:
            max_row = max(r for r, _ in base_tile_coords)
            max_col = max(c for _, c in base_tile_coords)
            base_grid_side_tiles = max(max_row, max_col) + 1
        else:
            base_grid_side_tiles = int(math.sqrt(base_feats.shape[0]))
        expected_tokens = base_tokens_per_tile * len(base_tile_coords)
        if attn_vectors[0].numel() != expected_tokens:
            if attn_vectors[0].numel() > expected_tokens:
                if getattr(self.config, "_log_vision_tokens", False):
                    rank0_print(
                        f"[Warn] Truncating attention tokens: got {attn_vectors[0].numel()}, "
                        f"expected {expected_tokens} (tiles={len(base_tile_coords)}, tokens_per_tile={base_tokens_per_tile})"
                    )
                attn_vectors[0] = attn_vectors[0][:expected_tokens]
            else:
                raise RuntimeError(
                    f"Attention/token mismatch at base scale: attn={attn_vectors[0].numel()}, "
                    f"expected={expected_tokens} ({len(base_tile_coords)} tiles x {base_tokens_per_tile} tokens)"
                )
        base_grid_tokens = torch.zeros(
            tile_tokens_side * base_grid_side_tiles,
            tile_tokens_side * base_grid_side_tiles,
            device=attn_vectors[0].device,
            dtype=attn_vectors[0].dtype,
        )
        ptr = 0
        for idx, (r, c) in enumerate(base_tile_coords):
            tile_attn = attn_vectors[0][ptr : ptr + base_tokens_per_tile]
            ptr += base_tokens_per_tile
            tile_map = tile_attn.reshape(tile_tokens_side, tile_tokens_side)
            rs = r * tile_tokens_side
            cs = c * tile_tokens_side
            base_grid_tokens[rs : rs + tile_tokens_side, cs : cs + tile_tokens_side] = tile_map

        attention_by_scale: Dict[int, torch.Tensor] = {}
        patch_size = getattr(vision_tower.config, "patch_size", None)
        tile_tokens = tile_tokens_side
        for scale in sorted(embeddings_by_scale.keys()):
            coords = coords_by_scale.get(scale, [])
            tiles_in_feat = embeddings_by_scale[scale].shape[0]
            if not coords or len(coords) != tiles_in_feat:
                grid_tile_side = int(math.sqrt(max(1, tiles_in_feat)))
                coords = [(i // grid_tile_side, i % grid_tile_side) for i in range(tiles_in_feat)]
            else:
                grid_tile_side = max(max(r for r, _ in coords), max(c for _, c in coords)) + 1
            target_side = tile_tokens * grid_tile_side
            if target_side <= 0:
                raise ValueError(f"Invalid target_side computed for scale {scale}")
            resized = F.interpolate(
                base_grid_tokens.unsqueeze(0).unsqueeze(0),
                size=(target_side, target_side),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            # Gather tiles following the order of coords to align with embeddings
            tiles = []
            for r, c in coords:
                rs = r * tile_tokens
                cs = c * tile_tokens
                if rs + tile_tokens > resized.shape[0] or cs + tile_tokens > resized.shape[1]:
                    raise RuntimeError(
                        f"Tile coord ({r},{c}) out of resized attention grid for scale {scale}: "
                        f"grid_side={grid_tile_side}, tokens_side={tile_tokens}, resized={tuple(resized.shape)}"
                    )
                tiles.append(resized[rs : rs + tile_tokens, cs : cs + tile_tokens])
            attention_by_scale[int(scale)] = torch.stack(tiles, dim=0)

        base_attention = torch.stack(attn_vectors, dim=0) if len(attn_vectors) > 1 else attn_vectors[0]

        return {
            "attn_by_scale": attention_by_scale,
            "base_scale_attention": base_attention,
            "last_text_index": last_text_idx,
            "image_token_positions": image_positions,
            "mm_input_ids": mm_input_ids,
            "token_counts_by_scale": token_counts_by_scale,
            "tile_indices": coords_by_scale,
        }

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )

        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
