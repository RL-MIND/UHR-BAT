#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate XLRS(-lite) with multiscale mask + attention pruning.
Outputs JSONL (one line per sample) and supports resume + multi-GPU sharding.

Example:
  python -u longva/scripts/eval_xlrs_multiscale_to_json.py \
    --ckpt /path/to/checkpoint_dir \
    --data_json /path/to/xlrs_or_mme_style_dataset.json \
    --image_root /path/to/image_root \
    --output_json /path/to/output_results.jsonl \
    --multiscale_topk 80,320,600,2000 \
    --max_samples 0 \
    --kmeans_num_clusters 600 \
    --devices cuda:0,cuda:1,cuda:2,cuda:3 \
    --kmeans_max_iters 100
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm
import traceback

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None

# Allow huge remote-sensing images
Image.MAX_IMAGE_PIXELS = None

# Add repo root to PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)

# Force rank0_print to behave like print for easier logging in multi-process eval.
# Must run after repo root is injected, otherwise another installed "longva"
# package may be imported first.
try:
    import longva.utils as _lv_utils

    _lv_utils.rank0_print = print
except Exception:
    pass

from longva.conversation import conv_templates  # noqa: E402
from longva.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from longva.mm_utils import (  # noqa: E402
    KeywordsStoppingCriteria,
    load_multiscale_tile_masks,
    process_images,
    split_image_to_multiscale_tiles,
    tokenizer_image_token,
)
from longva.model.builder import load_pretrained_model  # noqa: E402


def _sanitize_cache_name(image_rel_path: str) -> str:
    normalized = os.path.normpath(image_rel_path.strip())
    return normalized.replace("\\", "__").replace("/", "__")


_LEADING_IMAGE_TOKEN_PATTERN = re.compile(r"^((?:\s*<image>\s*)+)", re.IGNORECASE)
_SPECIAL_TOKEN_CLEANER = re.compile(
    r"<\|[^>]+?\|>(?: ?assistant)?|<image>|</?s>|<pad>|<eos>|<unk>",
    flags=re.IGNORECASE,
)


def _strip_leading_image_tokens(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    match = _LEADING_IMAGE_TOKEN_PATTERN.match(text)
    if not match:
        return text
    remainder = text[match.end() :]
    remainder = remainder.lstrip("\r\n")
    removed = match.group(1)
    if "\n" in removed and remainder and not remainder.startswith("\n"):
        remainder = "\n" + remainder
    return remainder


def _stable_json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _result_content_key(images: Any, question: Any, ground_truth: Any) -> Optional[str]:
    if images is None or question is None:
        return None
    rel_list = [images] if isinstance(images, str) else list(images or [])
    if not rel_list:
        return None
    return "content:" + _stable_json_key([rel_list, str(question), ground_truth])


def _sample_resume_keys(sample: Dict, index: int) -> List[str]:
    keys = [f"idx:{index}"]
    try:
        rel_list, q_text, gt, _ = extract_question_and_gt(sample)
        content_key = _result_content_key(rel_list, q_text, gt)
        if content_key:
            keys.append(content_key)
    except Exception:
        pass
    sid = sample.get("id")
    if sid is not None:
        keys.append("id:" + _stable_json_key(sid))
    return keys


def _load_completed_keys(out_path: Path) -> set:
    keys = set()
    if not out_path.exists():
        return keys
    try:
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sample_index = obj.get("sample_index")
                if isinstance(sample_index, int):
                    keys.add(f"idx:{sample_index}")
                content_key = _result_content_key(
                    obj.get("images") or obj.get("image"),
                    obj.get("question"),
                    obj.get("ground_truth"),
                )
                if content_key:
                    keys.add(content_key)
                elif sample_index is None:
                    sid = obj.get("id")
                    if sid is not None:
                        keys.add("id:" + _stable_json_key(sid))
    except Exception:
        pass
    return keys


def _append_line(lock: Any, out_path: Path, line: str) -> None:
    if lock:
        try:
            lock.acquire()
        except Exception:
            lock = None
    try:
        with out_path.open("a", encoding="utf-8") as f:
            f.write(line)
    finally:
        if lock:
            try:
                lock.release()
            except Exception:
                pass


def decode_chatml_response(tokenizer, sequences: torch.LongTensor) -> str:
    decoded = _safe_batch_decode(tokenizer, sequences, skip_special_tokens=False)[0]
    # Qwen/ChatML 有的模板不带换行（<|im_start|>assistant），而 conv.roles 里也可能省略 \n
    start_tags = ["<|im_start|>assistant\n", "<|im_start|>assistant"]
    end_tag = "<|im_end|>"
    resp = decoded
    for st in start_tags:
        if st in decoded:
            resp = decoded.split(st)[-1]
            break
    if "<|im_end|>assistant" in resp:
        resp = resp.split("<|im_end|>assistant")[-1]
    if end_tag in resp:
        resp = resp.split(end_tag)[0]
    for cutter in ["\n###", "###\n", "###"]:
        if cutter in resp:
            resp = resp.split(cutter)[0]
            break
    return resp.strip()


def _first_nonempty_line(text: str) -> str:
    if not isinstance(text, str):
        return text
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


def _clean_generated_text(text: Optional[str]) -> str:
    """
    Remove special tokens (e.g., <|endoftext|>) and normalize whitespace.
    """
    if text is None:
        return ""
    cleaned = _SPECIAL_TOKEN_CLEANER.sub(" ", str(text))
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


def _extract_single_choice_letter(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    text = _clean_generated_text(text)
    for line in (l.strip() for l in text.splitlines() if l.strip()):
        m = re.match(r"^[^A-Fa-f]*([ABCDEF])(?:\\b|[^A-Za-z])", line)
        if m:
            return m.group(1).upper()
        break
    seen = []
    for ch in text.upper():
        if ch in {"A", "B", "C", "D", "E", "F"} and ch not in seen:
            seen.append(ch)
    if len(seen) == 1:
        return seen[0]
    return None


def _extract_choice_set(text: Optional[str]) -> List[str]:
    """
    Extract sorted, de-duplicated choices (A-F) from text.
    Used to handle multi-select where all options must be correct.
    """
    if not isinstance(text, str):
        return []
    text = _clean_generated_text(text)
    letters: List[str] = []
    for ch in text.upper():
        if ch in {"A", "B", "C", "D", "E", "F"} and ch not in letters:
            letters.append(ch)
    return sorted(letters)


def _format_options(options: Any) -> str:
    if not options:
        return ""
    if isinstance(options, dict):
        lines = []
        for key in sorted(options.keys(), key=lambda x: str(x)):
            label = str(key).strip().upper()
            value = str(options[key]).strip()
            if label and value:
                lines.append(f"{label}. {value}")
        return "\n".join(lines)
    if isinstance(options, (list, tuple)):
        lines = []
        for idx, value in enumerate(options):
            label = chr(ord("A") + idx)
            lines.append(f"{label}. {str(value).strip()}")
        return "\n".join(lines)
    return str(options).strip()


def extract_question_and_gt(sample: Dict) -> Tuple[List[str], str, Optional[str], List[str]]:
    """
    Support both legacy XLRS/MME-style samples and public XHRBench samples.

    Legacy samples use `image` + `conversations`; XHRBench uses
    `images` + `question` + `options` + `answer`.
    """
    image_field = sample.get("image", None)
    if image_field is None:
        image_field = sample.get("images", None)
    rel_list = [image_field] if isinstance(image_field, str) else list(image_field or [])
    if not rel_list:
        raise ValueError("Sample missing image path.")

    conversations = sample.get("conversations") or []
    q_text = None
    gt = None
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
        if msg.get("from") == "human" and q_text is None:
            q_text = msg.get("value", "")
        if msg.get("from") == "gpt" and gt is None:
            gt = str(msg.get("value", "")).strip()

    if q_text is None and sample.get("question") is not None:
        q_text = str(sample.get("question", "")).strip()
        option_text = _format_options(sample.get("options"))
        if option_text:
            q_text = (
                f"{q_text}\n{option_text}\n"
                "Select the best answer based on the image. Answer with the option letter only."
            )
    if gt is None and sample.get("answer") is not None:
        answer = sample.get("answer")
        if isinstance(answer, (list, tuple)):
            gt = "".join(str(x).strip() for x in answer)
        else:
            gt = str(answer).strip()

    gt_choices = _extract_choice_set(gt)
    if q_text is None:
        raise ValueError("Missing question text. Expected conversations or question/options fields.")
    if len(gt_choices) > 1 and isinstance(q_text, str):
        marker = "Select the best answer based on the image."
        if marker in q_text and "You should choose multiple answers" not in q_text:
            q_text = q_text.replace(
                marker,
                marker + " You should choose multiple answers.",
            )
    return rel_list, q_text, gt, gt_choices


def _safe_batch_decode(tokenizer, sequences, **decode_kwargs) -> List[str]:
    """
    Decode while replacing special image token ids (negative) with a safe placeholder to avoid None -> str errors.
    """
    try:
        seq_tensor = torch.as_tensor(sequences, dtype=torch.long)
    except Exception:
        return tokenizer.batch_decode(sequences, **decode_kwargs)
    replace_id = tokenizer.pad_token_id
    if replace_id is None:
        replace_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else 0
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    # mask both the configured image token id and any remaining negative ids
    seq_tensor = seq_tensor.clone()
    seq_tensor = seq_tensor.masked_fill(seq_tensor == IMAGE_TOKEN_INDEX, replace_id)
    seq_tensor = seq_tensor.masked_fill(seq_tensor < 0, replace_id)
    if vocab_size:
        seq_tensor = seq_tensor.masked_fill(seq_tensor >= vocab_size, replace_id)
    return tokenizer.batch_decode(seq_tensor, **decode_kwargs)


def _now_sync() -> float:
    """
    Wall-clock timestamp with a best-effort CUDA sync to measure GPU latency.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    return time.perf_counter()


def _start_profiler(enable: bool):
    if not enable:
        return None
    try:
        import torch.profiler as profiler
    except Exception as exc:
        print(f"[Warn] torch.profiler unavailable, skip FLOPs: {exc}")
        return None
    activities = [profiler.ProfilerActivity.CPU]
    try:
        if torch.cuda.is_available():
            activities.append(profiler.ProfilerActivity.CUDA)
    except Exception:
        pass
    try:
        return profiler.profile(
            activities=activities,
            with_flops=True,
            profile_memory=False,
            record_shapes=False,
        )
    except Exception as exc:
        print(f"[Warn] failed to start profiler, skip FLOPs: {exc}")
        return None


def _extract_flops(prof) -> Optional[float]:
    if prof is None:
        return None
    try:
        events = prof.key_averages()
    except Exception:
        return None
    total = 0.0
    has = False
    for evt in events:
        fl = getattr(evt, "flops", None)
        if fl is None:
            continue
        has = True
        total += float(fl)
    return total if has else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate XLRS with multiscale attention pruning (masks optional).")
    parser.add_argument("--ckpt", required=True, help="Checkpoint directory.")
    parser.add_argument("--model_name", default="llava_qwen", help="Model name hint for loader.")
    parser.add_argument(
        "--vision_tower",
        default=None,
        help="Optional local vision tower path to override ckpt config (useful in offline mode).",
    )
    parser.add_argument("--data_json", required=True, help="Path to JSON list with XLRS samples.")
    parser.add_argument("--image_root", required=True, help="Root folder that contains images referenced by data_json.")
    parser.add_argument(
        "--mask_root",
        required=False,
        default=None,
        help="Directory of multiscale tile masks (.pt from build_multiscale_tile_masks.py). Optional with K-Means selection.",
    )
    parser.add_argument("--output_json", default="xlrs_multiscale_results.jsonl", help="Output JSONL path.")
    parser.add_argument("--devices", default=None, help="Comma-separated devices for parallel eval, e.g., cuda:0,cuda:1.")
    parser.add_argument("--device", default="cuda:0", help="Device when --devices is not set.")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit number of samples (0 for all).")
    parser.add_argument("--no_resume", action="store_true", help="Disable resume; overwrite output file.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy).")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p for sampling.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate.")
    parser.add_argument("--multiscale_topk", default="80,320,600,2000", help="Comma-separated top-k per scale.")
    parser.add_argument(
        "--multiscale_target_sizes",
        default="672,1344,2688,4032",
        help="Comma-separated square sizes.",
    )
    parser.add_argument(
        "--kmeans_num_clusters",
        type=int,
        default=None,
        help="Optional fixed number of K-Means clusters per scale (overrides top-k for clustering only).",
    )
    parser.add_argument(
        "--kmeans_max_iters",
        type=int,
        default=10,
        help="Max K-Means iterations shared across scales.",
    )
    parser.add_argument("--tile_size", type=int, default=336, help="Tile size used in preprocessing.")
    parser.add_argument("--patch_size", type=int, default=14, help="Vision patch size inside a tile.")
    parser.add_argument(
        "--allow_missing_mask",
        action="store_true",
        help="If set, continue with multiscale tiles even when mask files are missing (empty masks).",
    )
    parser.add_argument("--prompt_version", default="qwen_1_5", help="Conversation template key.")
    parser.add_argument("--log_latency", action="store_true", help="Record per-sample latency to a metrics JSONL.")
    parser.add_argument(
        "--profile_samples",
        type=int,
        default=0,
        help="Number of samples per worker to profile for FLOPs (uses torch.profiler; heavy).",
    )
    parser.add_argument(
        "--metrics_json",
        default=None,
        help="Optional metrics JSONL path. Defaults to <output_json>.metrics.jsonl when logging is enabled.",
    )
    return parser.parse_args()


def _load_mm_extras_if_missing(model, ckpt_dir: str):
    """
    Ensure multiscale_token_mlp / text_to_vision_proj / scale_pos_residuals are loaded
    even if from_pretrained skipped them (common when using merged HF checkpoints).
    """
    if safe_load_file is None:
        return
    index_path = Path(ckpt_dir) / "model.safetensors.index.json"
    if not index_path.exists():
        return
    data = json.loads(index_path.read_text())
    weight_map: Dict[str, str] = data.get("weight_map", {})
    target_prefixes = ("model.multiscale_token_mlp", "model.text_to_vision_proj", "model.scale_pos_residuals")
    target_keys = [k for k in weight_map.keys() if k.startswith(target_prefixes)]
    if not target_keys:
        return

    shard_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    sub_state: Dict[str, torch.Tensor] = {}
    for k in target_keys:
        shard_file = weight_map[k]
        shard_path = Path(ckpt_dir) / shard_file
        if shard_file not in shard_cache:
            shard_cache[shard_file] = safe_load_file(str(shard_path))
        shard = shard_cache[shard_file]
        if k in shard:
            sub_state[k] = shard[k]
    if sub_state:
        model.load_state_dict(sub_state, strict=False)


def _parse_int_list(val: str) -> List[int]:
    return [int(x) for x in str(val).split(",") if str(x).strip()]


def prepare_multiscale_inputs(
    image_paths: Sequence[str],
    image_processor,
    image_root: str,
    mask_root: Optional[str],
    target_sizes: Sequence[int],
    tile_size: int,
    patch_size: int,
    allow_missing_mask: bool = False,
) -> Tuple[List[Dict[int, Dict]], List[Optional[Dict[int, Dict]]], List[Tuple[int, int]]]:
    multiscale_pixels: List[Dict[int, Dict]] = []
    multiscale_masks: List[Optional[Dict[int, Dict]]] = []
    image_sizes: List[Tuple[int, int]] = []
    for rel in image_paths:
        abs_path = os.path.join(os.path.abspath(image_root), rel)
        img = Image.open(abs_path).convert("RGB")
        image_sizes.append(img.size)
        tiles = split_image_to_multiscale_tiles(
            img,
            image_processor,
            target_sizes=target_sizes,
            tile_size=tile_size,
        )
        masks = {}
        if mask_root:
            mask_path = os.path.join(mask_root, f"{_sanitize_cache_name(rel)}.pt")
            if os.path.isfile(mask_path):
                try:
                    mask_entry = torch.load(mask_path, map_location="cpu")
                    masks = load_multiscale_tile_masks(
                        mask_entry,
                        scales=target_sizes,
                        tile_size=tile_size,
                        patch_size=patch_size,
                    )
                except Exception as exc:
                    raise RuntimeError(f"Failed to load mask for {rel}: {exc}") from exc
                if not masks:
                    if not allow_missing_mask:
                        raise FileNotFoundError(f"Mask empty or invalid for {rel}: {mask_path}")
                    masks = {}
            elif not allow_missing_mask:
                raise FileNotFoundError(f"Mask not found for {rel}: {mask_path}")
        multiscale_pixels.append(tiles)
        multiscale_masks.append(masks)
    return multiscale_pixels, multiscale_masks, image_sizes


def build_prompt_and_inputs(tokenizer, conv, question: str, image_tokens: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    conv.messages = []
    cleaned_question = _strip_leading_image_tokens(question) if image_tokens else question
    # 确保有 <image> 占位符，否则 prepare_inputs_labels_for_multimodal 会认为没有图像
    if image_tokens and "<image>" not in cleaned_question:
        cleaned_question = "<image>\n" + cleaned_question
    human_message = (cleaned_question, image_tokens) if image_tokens else cleaned_question
    conv.append_message(conv.roles[0], human_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return input_ids, attention_mask


def run_worker(
    rank: int,
    indices: List[int],
    args: argparse.Namespace,
    out_path: Path,
    lock: Any,
) -> None:
    device = args.devices_list[rank]
    metrics_path = getattr(args, "metrics_path", None)
    profile_remaining = int(getattr(args, "profile_samples", 0) or 0)
    log_latency = bool(getattr(args, "log_latency", False) or profile_remaining > 0)

    gen_kwargs = dict(
        do_sample=bool(args.temperature and args.temperature > 1e-8) or (args.top_p is not None and args.top_p < 1.0),
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=1,
        use_cache=True,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1.0,
    )
    gen_kwargs["min_new_tokens"] = max(1, gen_kwargs.get("min_new_tokens", 1))
    if not gen_kwargs["do_sample"]:
        gen_kwargs.pop("temperature", None)
        gen_kwargs.pop("top_p", None)

    overwrite_config = None
    if args.vision_tower:
        vt = str(Path(args.vision_tower).expanduser().resolve())
        overwrite_config = {
            "mm_vision_tower": vt,
            "vision_tower": vt,
        }
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.ckpt,
        None,
        args.model_name,
        device_map=device,
        overwrite_config=overwrite_config,
    )
    model.config._log_vision_tokens = True
    # Propagate K-Means config to model (shared across scales).
    def _set_cfg(target, name, value):
        if target is None:
            return
        cfg = getattr(target, "config", None)
        if cfg is not None:
            try:
                setattr(cfg, name, value)
            except Exception:
                pass

    if args.kmeans_num_clusters is not None:
        _set_cfg(model, "kmeans_num_clusters", args.kmeans_num_clusters)
        _set_cfg(getattr(model, "get_model", lambda: None)(), "kmeans_num_clusters", args.kmeans_num_clusters)
    if args.kmeans_max_iters is not None:
        _set_cfg(model, "kmeans_max_iters", args.kmeans_max_iters)
        _set_cfg(getattr(model, "get_model", lambda: None)(), "kmeans_max_iters", args.kmeans_max_iters)

    # from_pretrained may drop multiscale extras; reload them explicitly
    try:
        _load_mm_extras_if_missing(model, args.ckpt)
    except Exception as exc:
        print(f"[Warn] failed to load multiscale extras: {exc}")
    try:
        model.generation_config.repetition_penalty = 1.0
    except Exception:
        pass
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Enable vision-token logging similar to training
    try:
        model.config._log_vision_tokens = False
    except Exception:
        pass
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        # Ensure BOS is a valid scalar
        bos_id = getattr(model.generation_config, "bos_token_id", None)
        if isinstance(bos_id, (list, tuple)) and bos_id:
            bos_id = bos_id[0]
        if bos_id is None:
            bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        model.generation_config.bos_token_id = int(bos_id)
        if not gen_kwargs["do_sample"]:
            model.generation_config.top_k = 0
            if hasattr(model.generation_config, "temperature"):
                model.generation_config.temperature = 1.0
            if hasattr(model.generation_config, "top_p"):
                model.generation_config.top_p = 1.0
    # Mirror pad/bos on model.config to be safe
    if getattr(model, "config", None) is not None:
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if getattr(model.config, "bos_token_id", None) is None:
            model.config.bos_token_id = bos_id

    data = json.loads(Path(args.data_json).read_text(encoding="utf-8"))
    conv = conv_templates[args.prompt_version].copy()
    target_sizes = _parse_int_list(args.multiscale_target_sizes)
    topk = _parse_int_list(args.multiscale_topk)
    model_dtype = next(model.parameters()).dtype

    for idx in indices:
        sample = data[idx]
        try:
            start_time = _now_sync() if log_latency else None
            rel_list, q_text, gt, gt_choices = extract_question_and_gt(sample)
            abs_list = [os.path.join(args.image_root, rel) for rel in rel_list]
            for p in abs_list:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Image missing: {p}")

            # Always build plain images first for fallback and base sizes
            pil_list = [Image.open(p).convert("RGB") for p in abs_list]
            image_sizes = [img.size for img in pil_list]
            images_payload = process_images(pil_list, image_processor, model.config)
            if isinstance(images_payload, torch.Tensor):
                images_payload = images_payload.to(model.device, dtype=model_dtype)
            elif isinstance(images_payload, list):
                try:
                    if len(images_payload) == 1:
                        images_payload = images_payload[0].unsqueeze(0)
                    elif all(x.shape == images_payload[0].shape for x in images_payload):
                        images_payload = torch.stack(images_payload)
                    else:
                        # Fallback: let processor handle padding/resize to a common size
                        images_payload = image_processor(pil_list, return_tensors="pt")["pixel_values"]
                    images_payload = images_payload.to(model.device, dtype=model_dtype)
                except Exception:
                    # last resort: keep list on device
                    images_payload = [im.to(model.device, dtype=model_dtype) for im in images_payload]
            else:
                # unexpected type; rebuild with processor
                images_payload = image_processor(pil_list, return_tensors="pt")["pixel_values"].to(
                    model.device, dtype=model_dtype
                )

            # 准备多尺度 tiles/masks（训练同路径）
            ms_pixels, ms_masks, image_sizes = prepare_multiscale_inputs(
                rel_list,
                image_processor,
                image_root=args.image_root,
                mask_root=args.mask_root,
                target_sizes=target_sizes,
                tile_size=args.tile_size,
                patch_size=args.patch_size,
                allow_missing_mask=args.allow_missing_mask,
            )
            # With K-Means selection, masks are optional; fall back to empty dicts when missing.
            if ms_masks is None or all((not entry) for entry in ms_masks):
                ms_masks = [{} for _ in ms_pixels]
            try:
                scale_keys = sorted({int(k) for entry in ms_masks for k in (entry or {}).keys()})
                # print(f"[Debug] Using multiscale masks at scales {scale_keys} with topk {topk}")
            except Exception:
                pass

            input_ids, attention_mask = build_prompt_and_inputs(tokenizer, conv, q_text, rel_list)
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            # Guard against OOV ids to avoid device-side asserts（但保留图像占位符）
            vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
            image_token_id = getattr(model.config, "image_token_index", IMAGE_TOKEN_INDEX)
            if vocab_size is not None:
                # 仅屏蔽除图像占位符以外的非法 token，避免把 <image>=-200 覆写掉
                oov_mask = (input_ids >= vocab_size) | ((input_ids < 0) & (input_ids != image_token_id))
                if oov_mask.any():
                    input_ids = input_ids.masked_fill(oov_mask, tokenizer.eos_token_id)
            # 如果缺少图像占位符，强制补上一个 IMAGE_TOKEN_INDEX
            if (input_ids == image_token_id).sum() == 0 and images_payload is not None:
                image_tok = torch.tensor([[image_token_id]], device=input_ids.device, dtype=input_ids.dtype)
                input_ids = torch.cat([image_tok, input_ids], dim=1)
                pad = torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([pad, attention_mask], dim=1)

            # 直接调用模型 generate，传入多尺度相关参数，保持与训练时逻辑一致
            gen_start = _now_sync() if log_latency else None
            prof = _start_profiler(profile_remaining > 0)
            if prof is None and profile_remaining > 0:
                profile_remaining = 0
            if prof:
                try:
                    prof.__enter__()
                except Exception:
                    prof = None
                    profile_remaining = 0
            flops_total = None
            with torch.inference_mode():
                stop_strings = [conv.sep]
                if getattr(conv, "sep2", None):
                    stop_strings.append(conv.sep2)
                stop_strings.extend(["\n###", "###", "<|im_start|>", "<|im_end|>"])
                stopping = KeywordsStoppingCriteria(stop_strings, tokenizer, input_ids)
                output = model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=images_payload,
                    image_sizes=image_sizes,
                    modalities=["image"] * len(rel_list),
                    multiscale_pixels=ms_pixels,
                    multiscale_masks=ms_masks,
                    multiscale_topk=topk,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    stopping_criteria=[stopping],
                    return_dict_in_generate=True,
                    output_scores=True,
                    **gen_kwargs,
                )
            gen_end = _now_sync() if log_latency else None
            if prof:
                try:
                    prof.__exit__(None, None, None)
                    flops_total = _extract_flops(prof)
                except Exception:
                    flops_total = None
                profile_remaining = max(0, profile_remaining - 1)

            sequences = output.sequences
            gen_steps = len(output.scores) if getattr(output, "scores", None) is not None else 0
            prompt_len = sequences.shape[1] - gen_steps if gen_steps > 0 else input_ids.shape[1]
            gen_tokens = sequences[:, prompt_len:]
            raw_full = _safe_batch_decode(tokenizer, sequences, skip_special_tokens=False)[0]
            raw_gen = _safe_batch_decode(tokenizer, gen_tokens, skip_special_tokens=False)[0]
            clean_tail = _clean_generated_text(raw_gen)
            pred_seq = decode_chatml_response(tokenizer, sequences)
            pred_gen = decode_chatml_response(tokenizer, gen_tokens)
            pred_text = clean_tail or _clean_generated_text(pred_gen) or _clean_generated_text(pred_seq)
            raw = pred_text if pred_text else (pred_gen if pred_gen else raw_gen)
            pred_choice = _extract_single_choice_letter(pred_text)
            pred_choices = _extract_choice_set(pred_text)
            used_logit_fallback = False
            clean_output = None
            if pred_choices:
                clean_output = "".join(sorted(pred_choices))
            elif pred_choice:
                clean_output = pred_choice
            else:
                clean_output = _first_nonempty_line(pred_text)
            if pred_choices:
                clean_output = "".join(sorted(pred_choices))
            elif clean_output is None:
                clean_output = _first_nonempty_line(pred_text)

            if metrics_path and log_latency:
                metrics_obj = {
                    "id": sample.get("id", idx),
                    "device": device,
                    "preprocess_s": (gen_start - start_time) if (gen_start is not None and start_time is not None) else None,
                    "generate_s": (gen_end - gen_start) if (gen_end is not None and gen_start is not None) else None,
                    "total_s": (gen_end - start_time) if (gen_end is not None and start_time is not None) else None,
                    "gen_tokens": gen_steps,
                    "flops": flops_total,
                }
                _append_line(lock, metrics_path, json.dumps(metrics_obj, ensure_ascii=False) + "\n")

            result = {
                "sample_index": idx,
                "id": sample.get("id"),
                "images": rel_list,
                "category": sample.get("category"),
                "question": q_text,
                "ground_truth": gt,
                "model_output": clean_output,
                # "raw_output": raw,
                "raw_tail": raw_gen,
                "image_sizes": image_sizes,
                "logit_fallback": used_logit_fallback if used_logit_fallback else False,
            }
            gt_choices = _extract_choice_set(gt)
            if gt_choices:
                # 多选需全中且不多选
                result["correct"] = sorted(pred_choices) == sorted(gt_choices)

            line = json.dumps(result, ensure_ascii=False) + "\n"
            _append_line(lock, out_path, line)
        except Exception as exc:
            err_obj = {
                "sample_index": idx,
                "id": sample.get("id"),
                "images": sample.get("image") or sample.get("images"),
                "error": str(exc),
                "trace": traceback.format_exc(),
            }
            _append_line(lock, out_path, json.dumps(err_obj, ensure_ascii=False) + "\n")


def main() -> None:
    global args
    args = parse_args()

    data = json.loads(Path(args.data_json).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"data_json must be a list, got {type(data)}")
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.no_resume and out_path.exists():
        out_path.unlink()
    metrics_path: Optional[Path] = None
    log_metrics = bool(args.log_latency or (args.profile_samples and args.profile_samples > 0))
    if log_metrics:
        if not args.metrics_json:
            args.metrics_json = f"{args.output_json}.metrics.jsonl"
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if args.no_resume and metrics_path.exists():
            metrics_path.unlink()
    args.metrics_path = metrics_path

    resume_enabled = not args.no_resume
    completed_keys = _load_completed_keys(out_path) if resume_enabled else set()
    indices_all: List[int] = []
    for i, sample in enumerate(data):
        if resume_enabled and any(key in completed_keys for key in _sample_resume_keys(sample, i)):
            continue
        indices_all.append(i)
    if args.max_samples and args.max_samples > 0:
        indices_all = indices_all[: args.max_samples]

    if args.devices:
        devices_list = [d.strip() for d in args.devices.split(",") if d.strip()]
    else:
        if "," in str(args.device):
            devices_list = [d.strip() for d in str(args.device).split(",") if d.strip()]
        else:
            devices_list = [args.device]
    args.devices_list = devices_list

    if len(devices_list) == 1:
        lock = None
        with tqdm(total=len(indices_all), desc="Eval", unit="item") as pbar:
            run_worker(0, indices_all, args, out_path, lock)
            pbar.update(len(indices_all))
        print(f"Done. Wrote {len(indices_all)} results to {out_path}")
        return

    world_size = len(devices_list)
    shards: List[List[int]] = [[] for _ in range(world_size)]
    for i, idx in enumerate(indices_all):
        shards[i % world_size].append(idx)

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    lock = manager.Lock()

    procs = []
    for rank in range(world_size):
        p = mp.Process(
            target=run_worker,
            args=(rank, shards[rank], args, out_path, lock),
            daemon=True,
        )
        p.start()
        procs.append(p)

    with tqdm(total=len(indices_all), desc="Eval", unit="item") as pbar:
        while any(p.is_alive() for p in procs):
            done = 0
            if out_path.exists():
                try:
                    with out_path.open("r", encoding="utf-8") as f:
                        done = sum(1 for _ in f)
                except Exception:
                    pass
            pbar.n = done
            pbar.refresh()
            time.sleep(0.2)
        try:
            with out_path.open("r", encoding="utf-8") as f:
                pbar.n = sum(1 for _ in f)
                pbar.refresh()
        except Exception:
            pass

    for p in procs:
        p.join()
    failed = [p.exitcode for p in procs if p.exitcode and p.exitcode != 0]
    if failed:
        print(f"[WARN] {len(failed)} worker(s) exited with errors: {failed}. Partial results may be written.")
    print(f"Done. Wrote up to {len(indices_all)} results to {out_path}")


if __name__ == "__main__":
    main()
