import ast
import re

from sentence_transformers import SentenceTransformer
import numpy as np


model = SentenceTransformer("all-MiniLM-L6-v2")
TOKEN_PATTERN = re.compile(r"[^\w']+", re.UNICODE)


def normalize_text(value):
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        parsed = value

    if isinstance(parsed, (list, tuple)):
        text = " ".join(str(part) for part in parsed)
    else:
        text = str(parsed)

    return " ".join(
        token.lower()
        for token in TOKEN_PATTERN.split(text)
        if token
    )


def encode(texts):
    return model.encode(
        [normalize_text(text) for text in texts],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")