from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


class ChatModel:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.backend = cfg["backend"]
        if self.backend == "openai_compatible":
            self.model = cfg["model_name"]
        elif self.backend == "azure_openai":
            self.model = cfg["azure_deployment"]
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")
        self.gen = cfg.get("generation", {}) or {}
        self.multimodal = bool(cfg.get("multimodal", False))
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def _make_client(self):
        if self.backend == "openai_compatible":
            from openai import OpenAI

            return OpenAI(
                base_url=self.cfg["base_url"],
                api_key=self.cfg.get("api_key") or os.environ.get("OPENAI_API_KEY") or "EMPTY",
            )
        from openai import AzureOpenAI

        return AzureOpenAI(api_version=self.cfg["api_version"])

    def chat(self, messages: list[dict], **gen_overrides) -> str:
        gen = {**self.gen, **gen_overrides}
        resp = self.client.chat.completions.create(model=self.model, messages=messages, **gen)
        return resp.choices[0].message.content

    def vqa(self, image, question: str, **gen_overrides) -> str:
        if not self.multimodal:
            raise ValueError(f"Model {self.model!r} is not configured as multimodal")
        url = _image_to_data_url(image)
        return self.chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            **gen_overrides,
        )


def load_model(name: str, config_path: str | Path | None = None) -> ChatModel:
    cfg = load_config(config_path)
    if name not in cfg["models"]:
        available = ", ".join(sorted(cfg["models"]))
        raise KeyError(f"Model {name!r} not in config; available: {available}")
    return ChatModel(cfg["models"][name])


def load_default_planner(config_path: str | Path | None = None) -> ChatModel:
    cfg = load_config(config_path)
    return load_model(cfg["default_planner"], config_path=config_path)


def load_default_vlm(config_path: str | Path | None = None) -> ChatModel:
    cfg = load_config(config_path)
    model = load_model(cfg["default_vlm"], config_path=config_path)
    if not model.multimodal:
        raise ValueError(f"default_vlm={cfg['default_vlm']!r} is not configured as multimodal")
    return model


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open() as f:
        return yaml.safe_load(f)


_load_config = load_config  # backwards-compat alias


def _image_to_data_url(image) -> str:
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image).__name__}")
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
