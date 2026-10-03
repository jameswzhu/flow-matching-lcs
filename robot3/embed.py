"""
Embed every prompt (training, held-out, paraphrases) once with a frozen text encoder
and save the vectors; the policy trains on these, never on the encoder itself.

    python embed.py clip       # CLIP ViT-B/32 text tower (transformers), 512-D
    python embed.py st         # sentence-transformers all-MiniLM-L6-v2, 384-D
    python embed.py oracle     # control: multi-hot [red, blue, square, circle] of the
                               # intended match set (perfect attribute binding)

Writes emb/<name>.npz with prompts [P] and vectors [P, E] (L2-normalised).
"""
import os, sys
import numpy as np
from env3 import ALL_PROMPTS, KINDS

os.makedirs("emb", exist_ok=True)
PROMPTS = list(ALL_PROMPTS)


def embed_clip(prompts, name="openai/clip-vit-base-patch32"):
    import torch
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection
    tok = CLIPTokenizer.from_pretrained(name)
    model = CLIPTextModelWithProjection.from_pretrained(name).eval()
    with torch.no_grad():
        e = model(**tok(prompts, padding=True, return_tensors="pt")).text_embeds
    return e.numpy().astype(np.float64)


def embed_st(prompts, name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device="cpu").encode(prompts).astype(np.float64)


def embed_oracle(prompts):
    out = []
    for p in prompts:
        m = ALL_PROMPTS[p]
        out.append([any(KINDS[k][0] == "red" for k in m), any(KINDS[k][0] == "blue" for k in m),
                    any(KINDS[k][1] == "square" for k in m), any(KINDS[k][1] == "circle" for k in m)])
    return np.array(out, float)


if __name__ == "__main__":
    for enc in sys.argv[1:] or ["oracle"]:
        e = {"clip": embed_clip, "st": embed_st, "oracle": embed_oracle}[enc](PROMPTS)
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        np.savez(f"emb/{enc}.npz", prompts=np.array(PROMPTS), vectors=e)
        cos = e @ e.T
        iu = np.triu_indices(len(PROMPTS), 1)
        print(f"{enc}: {e.shape}, pairwise cos min/median/max "
              f"{cos[iu].min():.2f}/{np.median(cos[iu]):.2f}/{cos[iu].max():.2f}")
        for a, b in [("blue square", "red square"), ("blue square", "blue circle"),
                     ("square", "box"), ("blue circle", "navy circle"), ("shape", "object")]:
            print(f"   cos({a!r}, {b!r}) = {cos[PROMPTS.index(a), PROMPTS.index(b)]:.3f}")
