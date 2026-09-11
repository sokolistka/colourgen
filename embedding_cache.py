from embedding import encode


def build_cache(df):

    descriptions = df["text_input"].apply(
        lambda x: x.replace("[", "")
                  .replace("]", "")
                  .replace("'", "")
                  .replace('"', "")
    ).tolist()

    embeddings = encode(descriptions)

    return embeddings