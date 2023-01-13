from cmath import isnan
import csv
from math import nan
import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken
import icecream as ic

openai.api_key = "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"

COMPLETIONS_MODEL = "text-davinci-002"


# print(results)

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
df = pd.read_csv('./output.csv')
df.set_index(["title", "content"])
print(f"{len(df)} rows in the data.")
# print(df.sample(5))

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    print(text)
    print("-=-=-=-=-=-<>?><?><?><?><><?><?><?><?><?><?><?><?><")
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]


def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }


document_embeddings = compute_doc_embeddings(df)
# print(type(document_embeddings))

example_entry = list(document_embeddings.items())[0]
print(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

# print(order_document_sections_by_query_similarity("How many holidays are full time RN’s required to work?", document_embeddings)[:5])


MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002
encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

print(f"Context separator contains {separator_len} tokens")

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"



COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}


def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")


print(answer_query_with_context("How many holidays are full time RN’s required to work?", df, document_embeddings))
