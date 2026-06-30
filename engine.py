import numpy as np
import streamlit as st
import requests
import time
import feedparser
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3" # Name of the model used for the word embedding
URL = "https://export.arxiv.org/api/query" # Database URL

def find_papers(query):
    #model = load_embedding_model(MODEL_NAME)
    papers = get_papers(query)
    print(papers)

def countdown(fmt, seconds, end="\n"):
    for time_left in range(seconds, 0, -1):
        print(f"\r{fmt.format(time_left)}", end="")
        time.sleep(1)
    print(end=end)

def get_papers(query, count=500, batch_limit=50, cooldown=4, rate_cooldown=15):
    """
    Returns a list of requested papers from the database.

    Parameters:
    - query: The user query
    - count: The number of papers to be requested
    - batch_limit: The number of papers to be requested per batch
    - cooldown: The cooldown to wait between batches
    - rate_cooldown: The cooldown to wait if a rate limit is hit
    """
    papers = []
    papers_left = count

    # Pulls papers for query
    print(f"Pulling {count} papers from {URL} for the query: {query}")
    page = 1
    while papers_left > 0:
        offset = count - papers_left
        if papers_left < batch_limit:
            limit = papers_left
        else:
            limit = batch_limit
        
        print(f"Fetching page {page} (papers {offset+1} to {offset+limit})")
        
        # Parameters for database query
        params = {
            "search_query": f"all:{query}",
            "max_results": limit,
            "start": offset,
        }

        response = requests.get(URL, params=params) # Gets response from database
        success = False # Boolean flag for if the request succeeded or not

        if response.status_code == 200: # All good
            feed = feedparser.parse(response.text)
            for entry in feed.entries:
                paper_data = {
                    "paperId": entry.id.split("/abs/")[-1],
                    "title": entry.title,
                    "abstract": entry.summary
                }
                papers.append(paper_data)

            success = True
        elif response.status_code == 429: # Rate limited
            countdown("Rate limited, retrying after {} seconds...", rate_cooldown)
            time.sleep(rate_cooldown)

            # Retries query
            response = requests.get(URL, params=params)
            if response.status_code == 200: # All good
                feed = feedparser.parse(response.text)
                for entry in feed.entries:
                    paper_data = {
                        "paperId": entry.id.split("/abs/")[-1],
                        "title": entry.title,
                        "abstract": entry.summary
                    }
                    papers.append(paper_data)

                success = True
            else:
                print("Retry failed, will retry this request in the next loop.")
        else:   # Any other error; exits the loop
            print(f"Error {response.status_code}: {response.text}")
            break

        if success:
            page += 1
            papers_left -= limit

        countdown("Cooldown: {} second(s) left...", cooldown) # Waits to prevent rate limiting
    
    return papers

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

find_papers("Autonomous vehicles")