import os

from lib.search_utils import load_movies, RRF_K, DEFAULT_SEARCH_LIMIT, SEARCH_MULTIPLIER
from lib.hybrid_search import HybridSearch
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def generate_answer(search_results, query, limit=DEFAULT_SEARCH_LIMIT) -> str:
    context = ""

    for result in search_results[:limit]:
        context += f"{result["title"]}: {result["document"]}\n\n"

    prompt = prompt = f"""Hoopla is a streaming service for movies. You are a RAG agent that provides a human answer
to the user's query based on the documents that were retrieved during search. Provide a comprehensive
answer that addresses the user's query.
a

Query: {query}

Documents:
{context}
"""
    
    resp = client.models.generate_content(model=model, contents=prompt)
    return (resp.text or "").strip()


def multi_document_summary(query: str, search_results: list[dict], limit: int = DEFAULT_SEARCH_LIMIT) -> str:
    docs_text = ""

    for i, res in enumerate(search_results[:limit], 1):
        docs_text += f"Document {i}: {res["title"]}; {res["document"]}\n\n"

    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{docs_text}
Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:
"""
    
    resp = client.models.generate_content(model=model, contents=prompt)
    return (resp.text or "").strip()

def multi_document_summary_citations(query: str, search_results: list[dict], limit: int=DEFAULT_SEARCH_LIMIT) -> str:
    docs_text = ""

    for i, res in enumerate(search_results[:limit], 1):
        docs_text += f"{i}. {res["title"]}: {res["document"]}\n\n"
    
    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{docs_text}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    
    resp = client.models.generate_content(model=model, contents=prompt)
    return (resp.text or "").strip()

def multi_document_answer(question: str, search_results: list[dict], limit: int=DEFAULT_SEARCH_LIMIT) -> str:
    docs_text = ""

    for i, res in enumerate(search_results[:limit], 1):
        docs_text += f"{i}. {res["title"]}; {res["document"]}\n\n"
    
    prompt = prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {question}

Documents:
{docs_text}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
    
    resp = client.models.generate_content(model=model, contents=prompt)
    return (resp.text or "").strip()


def rag(query: str, limit=DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)
    search_results = searcher.rrf_search(query, RRF_K, limit * SEARCH_MULTIPLIER)
    
    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found"
        }
    
    answer = generate_answer(search_results, query, DEFAULT_SEARCH_LIMIT)

    return {
        "query": query,
        "search_results": search_results,
        "answer": answer,
    }

def rag_command(query):
    return rag(query)

def summarize_command(query: str, limit: int=DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)
    search_results = searcher.rrf_search(query, RRF_K, limit * SEARCH_MULTIPLIER)

    if not search_results:
        return {"query": query, "error": "No results found"}

    summary = multi_document_summary(query, search_results, limit)

    return {
        "query": query,
        "search_results": search_results[:limit],
        "summary": summary,
    }

def citations_command(query: str, limit: int=DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    search_results = searcher.rrf_search(query, RRF_K, limit * SEARCH_MULTIPLIER)

    if not search_results:
        return {"query": query, "error": "No results found"}

    summary_wt_citations = multi_document_summary_citations(query, search_results, limit)

    return {
        "query": query,
        "search_results": search_results[:limit],
        "summary": summary_wt_citations,
    }

def question_command(question: str, limit: int=DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    search_results = searcher.rrf_search(question, RRF_K, limit * SEARCH_MULTIPLIER)

    if not search_results:
        return {"question": question, "error": "No results found"}

    answer = multi_document_answer(question, search_results, limit)

    return {
        "question": question,
        "search_results": search_results[:limit],
        "answer": answer
    }


