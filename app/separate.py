"""乐器分离：Demucs htdemucs_6s（鼓/贝斯/吉他/键盘/人声/其他），鼓轨再用 htdemucs_ft 的鼓专用模型提纯。"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

STEMS = ["drums", "bass", "guitar", "piano", "vocals", "other"]

_models: dict[str, object] = {}


def _get(name: str):
    if name not in _models:
        from demucs.pretrained import get_model

        _models[name] = get_model(name)
    return _models[name]


def _apply(model, wav: torch.Tensor, device: str, progress_cb=None) -> torch.Tensor:
    """wav: [2, T] -> [S, 2, T]。按块处理以便汇报进度并控制显存。"""
    from demucs.apply import apply_model

    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    x = ((wav - mean) / std)[None]
    with torch.no_grad():
        out = apply_model(model, x, device=device, split=True, overlap=0.25, shifts=0, progress=False)[0]
    return out * std + mean


def separate(
    audio: np.ndarray,
    device: str = "cuda",
    hq_drums: bool = True,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, np.ndarray]:
    """audio: [2, T] @44.1k。返回 {stem: [2, T]}。"""
    progress = progress or (lambda p, m: None)
    wav = torch.from_numpy(audio)

    progress(0.0, "分离乐器（6 轨）")
    six = _get("htdemucs_6s")
    six.to(device).eval()
    out = _apply(six, wav, device).cpu()
    six.to("cpu")
    stems = {name: out[i].numpy() for i, name in enumerate(six.sources)}

    if hq_drums:
        progress(0.6, "提纯鼓轨")
        try:
            bag = _get("htdemucs_ft")
            src = list(bag.sources)
            di = src.index("drums")
            # 包内 4 个模型各自针对一个声部微调，挑出权重落在鼓上的那个
            weights = bag.weights if bag.weights is not None else [[1.0] * len(src)] * len(bag.models)
            mi = int(np.argmax([w[di] for w in weights]))
            dm = bag.models[mi]
            dm.to(device).eval()
            d = _apply(dm, wav, device).cpu()[di].numpy()
            dm.to("cpu")
            # 其余声部整体扣除新旧鼓轨之差，保持各轨之和 = 原曲
            stems["other"] = stems["other"] + (stems["drums"] - d)
            stems["drums"] = d
        except Exception as e:  # 下载失败等情况退回 6 轨模型的鼓轨
            print("[separate] 鼓轨提纯失败，使用 6 轨模型结果：", e)

    if device == "cuda":
        torch.cuda.empty_cache()
    progress(1.0, "分离完成")
    return {k: stems[k] for k in STEMS}
