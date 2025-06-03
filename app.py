from dotenv import dotenv_values
import json
from openai import OpenAI
import os
import numpy as np
import pandas as pd
import tiktoken
from tenacity import retry, wait_random_exponential, stop_after_attempt
import pickle
from nomic import atlas

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def distance_from_embeddings(query_vec, matrix_vecs):
    query_norm = np.linalg.norm(query_vec)
    matrix_norms = np.linalg.norm(matrix_vecs, axis=1)
    dot_products = np.dot(matrix_vecs, query_vec)
    cosine_similarities = dot_products / (matrix_norms * query_norm)
    distances = 1 - cosine_similarities  # cosine distance
    return distances


def indices_of_nearest_neighbors_from_distances(distances, count):
    return np.argsort(distances)[:count]

config = dotenv_values(".env")
client = OpenAI(api_key=config["APIKEY"])

df = pd.read_csv("./movie_plots.csv")
movies = df[df["Origin/Ethnicity"]=="Tamil"].sort_values("Release Year",ascending=False).head(1500)
plots = movies["Plot"].values

enc = tiktoken.encoding_for_model("text-embedding-3-small")
total_tokens = sum([len(enc.encode(plot)) for plot in plots])
cost = total_tokens*0.02/1000000
#print(f"Estimated Cost: ${cost: 2f}")


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, model):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


ecp = "movie_embeddings.pkl"

if os.path.exists(ecp) and os.path.getsize(ecp) > 0:
    embcache = pd.read_pickle(ecp)
else:
    embcache = {} 
    with open(ecp, "wb") as ecf:
        pickle.dump(embcache, ecf)


def emb_from_str(
    string,
    model="text-embedding-3-small",
    embedding_cache=embcache 
):
    if (string, model) not in embedding_cache:
        embedding_cache[(string, model)] = get_embedding(string, model)
        print(f"GOT EMBEDDING FOR: {string[:20]}")
        with open(ecp, "wb") as ecf:
            pickle.dump(embedding_cache, ecf)
    return embedding_cache[(string, model)]

data=movies[["Title","Genre","Plot","Director"]].to_dict("records")
plot_embeddings = [emb_from_str(plot, model ="text-embedding-3-small") for plot in plots]
atlas.map_data(
    embeddings=np.array(plot_embeddings),
    data=data
)

def get_similar(
        strings,
        index_of_src,
        count=6,
        model="text-embedding-3-small"
):
    #get all embeddings
    embeddings=[emb_from_str(string) for string in strings]
    #get embedding for our specific query
    query_embedding = embeddings[index_of_src]
    #get indices of nearest neighbours
    distances = distance_from_embeddings(query_embedding,embeddings)
    indices = indices_of_nearest_neighbors_from_distances(distances, count=6)
    return indices

def print_recommendations(indices, movies_df):
    ref_title = movies_df.iloc[indices[0]]["Title"]
    ref_genre = movies_df.iloc[indices[0]]["Genre"]
    print(f"\nReference Movie: {ref_title} ({ref_genre})\n")

    for i, idx in enumerate(indices[1:], start=1):
        title = movies_df.iloc[idx]["Title"]
        genre = movies_df.iloc[idx]["Genre"]
        print(f"{i}. Title: {title:<25} Genre: {genre}")

indices = get_similar(plots,198)
print_recommendations(indices,movies)