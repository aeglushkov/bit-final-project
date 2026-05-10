from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def _paper_cwd(paper_code_dir: Path):
    paper_code_dir = paper_code_dir.resolve()
    orig = os.getcwd()
    if str(paper_code_dir) not in sys.path:
        sys.path.insert(0, str(paper_code_dir))
    try:
        os.chdir(paper_code_dir)
        yield
    finally:
        os.chdir(orig)


def build_clip_text_encoder(
    paper_code_dir: str | Path,
    model_path: str = "data/model_weights/CLIP/ViT-L-14-336px.pt",
    device: str = "cuda",
):
    """Build a callable `text -> np.ndarray` using the same CLIP weights the
    paper's `ObjectMemory` used to compute OBJ Feat / CTX Feat. The returned
    encoder must produce vectors in the same embedding space as those features
    or retrieval will be garbage."""
    import clip
    import torch

    paper_code_dir = Path(paper_code_dir)
    with _paper_cwd(paper_code_dir):
        model, _ = clip.load(model_path, device=device)
    model.eval()

    @torch.no_grad()
    def encode(text: str) -> np.ndarray:
        tokens = clip.tokenize([text]).to(device)
        emb = model.encode_text(tokens)
        return emb.detach().cpu().numpy().squeeze(0).astype(np.float32)

    return encode
