import pandas as pd
import tensorflow as tf


def preprocess(train_df: pd.DataFrame, model_schema) -> pd.DataFrame:
    # Get the list of input features for the candidate model from the model schema
    input_model_schema = model_schema["input_schema"]["columnar_schema"]
    candidate_features = [feat["name"] for feat in input_model_schema]

    # Select the candidate features from the training DataFrame
    item_df = train_df[candidate_features]

    # Drop duplicate rows based on the 'article_id' column to get unique candidate items
    item_df.drop_duplicates(subset="article_id", inplace=True)

    return item_df


def embed(df: pd.DataFrame, candidate_model) -> pd.DataFrame:
    ds = tf.data.Dataset.from_tensor_slices({col: df[col] for col in df})

    candidate_embeddings = ds.batch(2048).map(
        lambda x: (x["article_id"], candidate_model(x))
    )

    all_article_ids = tf.concat([batch[0] for batch in candidate_embeddings], axis=0)
    all_embeddings = tf.concat([batch[1] for batch in candidate_embeddings], axis=0)

    all_article_ids = all_article_ids.numpy().astype(int).tolist()
    all_embeddings = all_embeddings.numpy().tolist()

    embeddings_df = pd.DataFrame(
        {
            "article_id": all_article_ids,
            "embeddings": all_embeddings,
        }
    )

    return embeddings_df
