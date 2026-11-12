from embedding import encode
import faiss


def build_faiss_index(df):

    descriptions = df["text_input"].apply(
        lambda x: x.replace("[", "")
                  .replace("]", "")
                  .replace("'", "")
                  .replace('"', "")
    ).tolist()

    embeddings = encode(descriptions)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index, embeddings