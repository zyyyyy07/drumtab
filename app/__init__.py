"""鼓谱：自动扒谱工具。

所有模型权重都放在项目文件夹的 models/ 里，不往用户目录（~/.cache）写任何东西。
这段必须在 torch / huggingface 被导入之前执行。
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
(MODELS / "torch").mkdir(parents=True, exist_ok=True)
(MODELS / "huggingface").mkdir(parents=True, exist_ok=True)

os.environ["TORCH_HOME"] = str(MODELS / "torch")
os.environ["HF_HOME"] = str(MODELS / "huggingface")
os.environ["HF_HUB_CACHE"] = str(MODELS / "huggingface" / "hub")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
