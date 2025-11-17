# embedding_service.py
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import torch

print("🔹 Loading MiniLM embedding model at startup...")

EMBED_MODEL = SentenceTransformer(
    "./models/all-MiniLM-L6-v2",
    device="cpu"
)

# Force full materialization (prevents meta tensors)
for _, param in EMBED_MODEL._first_module().named_parameters():
    param.data = param.data.to(torch.device("cpu"))

def get_embed_model():
    return EMBED_MODEL

@lru_cache(maxsize=8192)
def get_embedding(text: str):
    return EMBED_MODEL.encode(text, convert_to_tensor=False)

def embed_text(text: str) -> list:
    return EMBED_MODEL.encode(text, convert_to_tensor=False).tolist()
