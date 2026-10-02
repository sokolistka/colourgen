import ast

from embedding import encode
from build_index import build_faiss_index


class PaletteRetriever:

    def __init__(self, df):

        self.df = df

        self.index, self.embeddings = build_faiss_index(df)

    def search(self, query, k=3):

        query_embedding = encode([query])

        similarities, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for similarity, idx in zip(similarities[0], indices[0]):

            if idx == -1:
                continue

            row = self.df.iloc[idx]

            # putting togather to avoid gru problems
            try:
                words = ast.literal_eval(row["text_input"])
                description = " ".join(words)
            except (ValueError, SyntaxError):
                description = str(row["text_input"])

            # string to list
            try:
                palette = ast.literal_eval(row["palette"])
            except (ValueError, SyntaxError):
                palette = row["palette"]

            # lab string to list
            try:
                palette_lab = ast.literal_eval(row["palette_lab"])
            except (ValueError, SyntaxError):
                palette_lab = row["palette_lab"]

            results.append({
                "description": description,
                "palette": palette,
                "palette_lab": palette_lab,
                "similarity": float(similarity)
            })

        return results