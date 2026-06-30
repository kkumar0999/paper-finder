import math
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3" # Name of the model used for the word embedding

@st.cache_resource
def load_embedding_model():
    """
    Provides the LLM model used for the word embedding.
    """
    return SentenceTransformer(MODEL_NAME)

def calc_similarity_scores(query_vec, paper_matrix):
    """
    Calculates the similarity scores between a query vector and paper vectors.

    Parameters:
    - query_vec: A 1D numpy array of shape (N,) representing the user query
    - paper_matrix: A 2D numpy array of shape (M, N) where each row represents a paper

    Preconditions:
    - query_vec.shape[0] == paper_matrix.shape[1]
    """
    assert query_vec.shape[0] == paper_matrix.shape[1], (
        f"Dimension mismatch! Paper database columns ({paper_matrix.shape[1]}) must match query vector "
        f"length ({query_vec.shape[0]})"
    )

    similarity_scores = np.dot(paper_matrix, query_vec) # Calculates the dot products
    return similarity_scores