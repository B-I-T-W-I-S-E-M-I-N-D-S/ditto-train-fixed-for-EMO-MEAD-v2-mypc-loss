# """
# extract_audio_emotion_feat_by_Emotion2vec.py
# --------------------------------------------
# Extracts continuous Emotion2vec audio features for each training clip and
# saves them as per-frame .npy arrays at 25 fps.

# Output shape: [N, 1024]  (N = number of video frames, 1024 = Emotion2vec hidden dim)

# Usage:
#     python prepare_data/scripts/extract_audio_emotion_feat_by_Emotion2vec.py \\
#         -i example/data_info.json \\
#         --emotion2vec_model iic/emotion2vec_plus_large

# The input JSON must contain:
#     {
#         "wav_list":         [...],   # absolute paths to .wav files (16 kHz)
#         "emo2vec_npy_list": [...],   # desired output .npy paths
#         "frame_num_list":   [...],   # number of video frames per clip (int)
#     }

# If "frame_num_list" is absent, the feature length is derived from the audio
# duration and FPS=25.
# """

# from __future__ import annotations

# import math
# import os
# import sys
# import traceback
# from dataclasses import dataclass
# from io import BytesIO
# from typing import Optional

# import numpy as np
# import torch
# import torch.nn.functional as F
# import torchaudio
# import tyro
# from tqdm.contrib import tzip
# from typing_extensions import Annotated

# # ── patch torch.load for PyTorch 2.6+ ──────────────────────────────────────
# _orig_torch_load = torch.load
# def _patched_torch_load(*args, **kwargs):
#     kwargs.setdefault("weights_only", False)
#     return _orig_torch_load(*args, **kwargs)
# torch.load = _patched_torch_load
# # ───────────────────────────────────────────────────────────────────────────

# CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(CUR_DIR))

# from utils.utils import load_json

# FPS = 25
# SAMPLE_RATE = 16_000
# EMOTION2VEC_DIM = 1024   # hidden dim of Emotion2vec backbone
# WINDOW_SAMPLES = SAMPLE_RATE * 3   # 3-second sliding window for extraction


# def _linear_interpolate(arr: np.ndarray, target_len: int) -> np.ndarray:
#     """Interpolate [T, D] → [target_len, D] along the time axis."""
#     if arr.shape[0] == target_len:
#         return arr
#     t_src = np.linspace(0, 1, arr.shape[0])
#     t_dst = np.linspace(0, 1, target_len)
#     result = np.zeros((target_len, arr.shape[1]), dtype=arr.shape.dtype
#                       if hasattr(arr, 'dtype') else np.float32)
#     for d in range(arr.shape[1]):
#         result[:, d] = np.interp(t_dst, t_src, arr[:, d])
#     return result.astype(np.float32)


# def load_emotion2vec(emotion2vec_model: str, device: str):
#     """Load Emotion2vec model via FunASR hub downloader."""
#     from funasr.download.download_from_hub import download_model
#     from funasr.models.emotion2vec.model import Emotion2vec

#     kwargs = download_model(model=emotion2vec_model)
#     kwargs["tokenizer"] = None
#     kwargs["input_size"] = None
#     kwargs["frontend"] = None
#     model = Emotion2vec(**kwargs, vocab_size=-1).to(device)

#     init_param = kwargs.get("init_param", None)
#     if init_param:
#         _load_state(model, init_param,
#                     ignore_mismatch=kwargs.get("ignore_init_mismatch", True),
#                     scope_map=kwargs.get("scope_map", []))
#     model.eval()
#     return model


# def _load_state(model, path, ignore_mismatch=True, scope_map=()):
#     """Mirrors MEMO's load_emotion2vec_model helper."""
#     dst = model.state_dict()
#     src = torch.load(path, map_location="cpu")
#     src = src.get("state_dict", src)
#     src = src.get("model_state_dict", src)
#     src = src.get("model", src)

#     if isinstance(scope_map, str):
#         scope_map = scope_map.split(",")
#     scope_map = list(scope_map) + ["module.", "None"]

#     for k in dst.keys():
#         k_src = k
#         for i in range(0, len(scope_map), 2):
#             sp = scope_map[i] if scope_map[i].lower() != "none" else ""
#             dp = scope_map[i + 1] if scope_map[i + 1].lower() != "none" else ""
#             if dp == "" and (sp + k) in src:
#                 k_src = sp + k
#             elif k.startswith(dp) and k.replace(dp, sp, 1) in src:
#                 k_src = k.replace(dp, sp, 1)
#         if k_src in src:
#             if ignore_mismatch and dst[k].shape != src[k_src].shape:
#                 pass
#             else:
#                 dst[k] = src[k_src]
#     model.load_state_dict(dst, strict=True)


# @torch.no_grad()
# def extract_emotion2vec_features(
#     wav_path: str,
#     emotion2vec_model,
#     frame_num: int,
#     device: str = "cuda",
# ) -> np.ndarray:
#     """
#     Extract per-frame continuous Emotion2vec features.

#     Returns:
#         np.ndarray of shape [frame_num, EMOTION2VEC_DIM]
#     """
#     wav, sr = torchaudio.load(wav_path)
#     if sr != SAMPLE_RATE:
#         wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
#     wav = wav[0] if wav.dim() == 2 else wav   # mono, shape [N_samples]

#     # ── Sliding window feature extraction ──────────────────────────────────
#     segment_feats = []
#     step = SAMPLE_RATE  # 1-second stride
#     total_samples = wav.shape[0]

#     # Pad so we have at least one window
#     if total_samples < WINDOW_SAMPLES:
#         wav = F.pad(wav, (0, WINDOW_SAMPLES - total_samples))
#         total_samples = wav.shape[0]

#     positions = list(range(0, total_samples - WINDOW_SAMPLES + 1, step))
#     if not positions:
#         positions = [0]

#     for start in positions:
#         segment = wav[start: start + WINDOW_SAMPLES].to(device)
#         segment = F.layer_norm(segment, segment.shape).view(1, -1)
#         feats = emotion2vec_model.extract_features(segment)
#         # feats["x"]: [1, T_feat, 1024] — mean-pool over time
#         feat_vec = feats["x"].mean(dim=1).squeeze(0).cpu().numpy()   # [1024]
#         segment_feats.append(feat_vec)

#     # Stack → [num_segments, 1024]
#     feat_arr = np.stack(segment_feats, axis=0).astype(np.float32)

#     # Interpolate to video frame count
#     feat_interp = _linear_interpolate(feat_arr, frame_num)   # [frame_num, 1024]
#     return feat_interp


# def process_data_list(
#     wav_list: list,
#     emo2vec_npy_list: list,
#     frame_num_list: Optional[list],
#     emotion2vec_model_name: str,
#     device: str,
# ):
#     print(f"[Emotion2vec] Loading model: {emotion2vec_model_name}")
#     emo_model = load_emotion2vec(emotion2vec_model_name, device)

#     for idx, (wav, npy) in enumerate(tzip(wav_list, emo2vec_npy_list)):
#         try:
#             if os.path.isfile(npy):
#                 continue

#             if frame_num_list is not None:
#                 frame_num = int(frame_num_list[idx])
#             else:
#                 # estimate from audio duration
#                 info = torchaudio.info(wav)
#                 duration_sec = info.num_frames / info.sample_rate
#                 frame_num = math.ceil(duration_sec * FPS)

#             feat = extract_emotion2vec_features(wav, emo_model, frame_num, device)

#             os.makedirs(os.path.dirname(npy), exist_ok=True)
#             np.save(npy, feat)

#         except Exception:
#             traceback.print_exc()


# @dataclass
# class Options:
#     # Path to data_info.json containing wav_list, emo2vec_npy_list, frame_num_list
#     input_data_json: Annotated[str, tyro.conf.arg(aliases=["-i"])] = ""
#     # FunASR model id or local path for Emotion2vec
#     emotion2vec_model: str = "iic/emotion2vec_plus_large"
#     device: str = "cuda"


# def main():
#     tyro.extras.set_accent_color("bright_cyan")
#     opt: Options = tyro.cli(Options)
#     assert opt.input_data_json, "Must supply --input_data_json / -i"

#     data_info = load_json(opt.input_data_json)
#     wav_list = data_info["wav_list"]
#     emo2vec_npy_list = data_info["emo2vec_npy_list"]
#     frame_num_list = data_info.get("frame_num_list", None)

#     process_data_list(
#         wav_list=wav_list,
#         emo2vec_npy_list=emo2vec_npy_list,
#         frame_num_list=frame_num_list,
#         emotion2vec_model_name=opt.emotion2vec_model,
#         device=opt.device,
#     )
#     print("[Emotion2vec] Done.")


# if __name__ == "__main__":
#     main()







"""
extract_audio_emotion_feat_by_Emotion2vec.py
--------------------------------------------
Extracts continuous Emotion2vec audio features for each training clip and
saves them as per-frame .npy arrays at 25 fps.

Output shape: [N, 1024]  (N = number of video frames, 1024 = Emotion2vec hidden dim)

Usage:
    python prepare_data/scripts/extract_audio_emotion_feat_by_Emotion2vec.py \\
        -i example/data_info.json \\
        --emotion2vec_model iic/emotion2vec_plus_large

The input JSON must contain:
    {
        "wav_list":         [...],   # absolute paths to .wav files (16 kHz)
        "emo2vec_npy_list": [...],   # desired output .npy paths
        "frame_num_list":   [...],   # number of video frames per clip (int)
    }

If "frame_num_list" is absent, the feature length is derived from the audio
duration and FPS=25.
"""

from __future__ import annotations

import math
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torchaudio
import tyro
from tqdm.contrib import tzip
from typing_extensions import Annotated

# ── patch torch.load for PyTorch 2.6+ ──────────────────────────────────────
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
# ───────────────────────────────────────────────────────────────────────────

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR_DIR))

from utils.utils import load_json

FPS = 25
SAMPLE_RATE = 16_000
EMOTION2VEC_DIM = 1024   # hidden dim of Emotion2vec backbone


def _linear_interpolate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Interpolate [T, D] → [target_len, D] along the time axis."""
    if arr.shape[0] == target_len:
        return arr
    t_src = np.linspace(0, 1, arr.shape[0])
    t_dst = np.linspace(0, 1, target_len)
    result = np.zeros((target_len, arr.shape[1]), dtype=arr.dtype)
    for d in range(arr.shape[1]):
        result[:, d] = np.interp(t_dst, t_src, arr[:, d])
    return result.astype(np.float32)


def load_emotion2vec(emotion2vec_model: str, device: str):
    """Load Emotion2vec model via the modern FunASR AutoModel API.

    Works with funasr >= 1.1.0 which removed the internal
    funasr.download.download_from_hub module.
    """
    from funasr import AutoModel

    model = AutoModel(
        model=emotion2vec_model,
        device=device,
    )
    # AutoModel wraps the underlying nn.Module; expose it so callers can call
    # .extract_features() directly when needed.
    return model


def extract_emotion2vec_features(
    wav_path: str,
    emotion2vec_model,
    frame_num: int,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract per-frame continuous Emotion2vec hidden-state features using the
    modern FunASR AutoModel API.

    Uses ``granularity="frame"`` so the model returns a hidden-state vector
    for every encoder frame (~50 ms), then linearly interpolates to the target
    video frame count.

    Returns:
        np.ndarray of shape [frame_num, EMOTION2VEC_DIM]
    """
    # Load audio as a flat numpy array (float32, 16 kHz, mono)
    wav, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.dim() == 2:
        wav = wav[0]                           # stereo → mono
    wav_np = wav.cpu().numpy().astype(np.float32)

    # AutoModel.generate() with extract_embedding=True returns a list of dicts.
    # Each dict has an "feats" key: np.ndarray [T_enc, hidden_dim].
    res = emotion2vec_model.generate(
        wav_np,
        output_dir=None,
        granularity="frame",
        extract_embedding=True,
    )
    # res is a list with one element per input utterance
    feat_arr = res[0]["feats"]                 # [T_enc, 1024]
    if feat_arr.ndim == 1:
        feat_arr = feat_arr[np.newaxis, :]     # safety: ensure 2-D

    feat_interp = _linear_interpolate(feat_arr, frame_num)   # [frame_num, 1024]
    return feat_interp


def process_data_list(
    wav_list: list,
    emo2vec_npy_list: list,
    frame_num_list: Optional[list],
    emotion2vec_model_name: str,
    device: str,
):
    print(f"[Emotion2vec] Loading model: {emotion2vec_model_name}")
    emo_model = load_emotion2vec(emotion2vec_model_name, device)

    for idx, (wav, npy) in enumerate(tzip(wav_list, emo2vec_npy_list)):
        try:
            if os.path.isfile(npy):
                continue

            if frame_num_list is not None:
                frame_num = int(frame_num_list[idx])
            else:
                # estimate from audio duration
                info = torchaudio.info(wav)
                duration_sec = info.num_frames / info.sample_rate
                frame_num = math.ceil(duration_sec * FPS)

            feat = extract_emotion2vec_features(wav, emo_model, frame_num, device)

            os.makedirs(os.path.dirname(npy), exist_ok=True)
            np.save(npy, feat)

        except Exception:
            traceback.print_exc()


@dataclass
class Options:
    # Path to data_info.json containing wav_list, emo2vec_npy_list, frame_num_list
    input_data_json: Annotated[str, tyro.conf.arg(aliases=["-i"])] = ""
    # FunASR model id or local path for Emotion2vec
    emotion2vec_model: str = "iic/emotion2vec_plus_large"
    device: str = "cuda"


def main():
    tyro.extras.set_accent_color("bright_cyan")
    opt: Options = tyro.cli(Options)
    assert opt.input_data_json, "Must supply --input_data_json / -i"

    data_info = load_json(opt.input_data_json)
    wav_list = data_info["wav_list"]
    emo2vec_npy_list = data_info["emo2vec_npy_list"]
    frame_num_list = data_info.get("frame_num_list", None)

    process_data_list(
        wav_list=wav_list,
        emo2vec_npy_list=emo2vec_npy_list,
        frame_num_list=frame_num_list,
        emotion2vec_model_name=opt.emotion2vec_model,
        device=opt.device,
    )
    print("[Emotion2vec] Done.")


if __name__ == "__main__":
    main()
