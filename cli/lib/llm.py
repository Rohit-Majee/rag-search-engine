from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)


class MovieQuery(BaseModel):
    enhanced_query: str = Field(
        description="The corrected movie search query with spelling errors fixed. No quotes or filler text."
    )

class RewrittenQuery(BaseModel):
    enhanced_query: str = Field(
        description="The rewritten, highly specific search query optimized for movie retrieval. No quotes or filler."
    )

class ExpandedQuery(BaseModel):
    expanded_terms: str = Field(
        description="Synonyms, related concepts, and genre terms to expand the search. No quotes or conversational text. A single string of space-separated keywords. No brackets, no quotes, no JSON."
    )

class ReRankScore(BaseModel):
    rerank_score: int

def enhance_query_spell(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
            Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
            Preserve punctuation and capitalization unless a change is required for a typo fix.
            If there are no spelling errors, or if you're unsure, output the original query unchanged.
            Output only the final query text, nothing else.
            User query: "{query}"
            """

    try:
        response = client.beta.chat.completions.parse(
            model=os.getenv("MODEL"),
            messages=[{"role": "user", "content": prompt}],
            response_format=MovieQuery,
        )

        result = response.choices[0].message.parsed
        return result.enhanced_query


    except Exception as e:
        print(f"\n[Warning] Structured LLM enhancement failed: {e}")
        return query
    

def enhance_query_rewrite(query: str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep the rewritten query concise (under 10 words)
                - It should be a Google-style search query, specific enough to yield relevant results
                - Don't use boolean logic

                Examples:
                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                If you cannot improve the query, output the original unchanged.
                Output only the rewritten query text, nothing else.

                User query: "{query}"
            """
    try:
        response = client.beta.chat.completions.parse(
            model=os.getenv("MODEL"),
            messages=[{"role": "user", "content": prompt}],
            response_format=RewrittenQuery,
        )

        result = response.choices[0].message.parsed
        return result.enhanced_query


    except Exception as e:
        print(f"\n[Warning] Structured LLM enhancement failed: {e}")
        return query
    

def enhance_query_expand(query: str) -> str:
    prompt = f"""Expand the user-provided movie search query below with related terms.

                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                Output only the additional terms; they will be appended to the original query.

                # TODO: Add a strict negative constraint right here!
                # Tell the model exactly what syntax is forbidden. 
                # Example: "CRITICAL: DO NOT output a JSON array, python list, commas, or quotes. Output ONLY a continuous space-separated string of words."

                Examples:
                - "scary bear movie" -> "frightening horror grizzly terrifying film"
                - "action movie with bear" -> "thriller chase fight adventure"
                - "comedy with bear" -> "funny humor lighthearted"

                User query: "{query}"
            """
    try:
        response = client.beta.chat.completions.parse(
            model=os.getenv("MODEL"),
            messages=[{"role": "user", "content": prompt}],
            response_format=ExpandedQuery,
        )

        result = response.choices[0].message.parsed
        return result.expanded_terms

    except Exception as e:
        print(f"\n[Warning] Structured LLM expansion failed: {e}")
        return query

def individual_rerank(query: str, title,document) -> ReRankScore:
    prompt = f"""You are an expert search engine evaluator scoring the relevance of a movie to a user's search query.

                User Query: "{query}"
                Movie Title: "{title}"
                Movie Synopsis: "{document}"

                Evaluate relevance strictly using this rubric:
                9-10: Perfect match. Directly satisfies the exact user intent, genre, and plot details.
                7-8: Highly relevant. Fits the main themes and intent, but might miss a minor peripheral detail.
                4-6: Partially relevant. Shares some keywords or tangential concepts, but is clearly not what the user is looking for.
                1-3: Barely relevant. Only matches on generic terms (e.g., just the word "movie").
                0: Completely irrelevant.

                Output ONLY the raw integer number. Do not include quotes, explanations, or the word 'Score'.
                """

    completion = client.beta.chat.completions.parse(
        model=os.getenv("MODEL"),
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format=ReRankScore,
    )
    result = completion.choices[0].message.parsed
    return result.rerank_score


def batch_rerank(query:str, doc_list_str:str):
    prompt = f"""Rank the movies listed below by relevance to the following search query.
    
            Query: "{query}"
    
            Movies:{doc_list_str}
    
            Return the movie IDs in order of relevance, best match first.
    
            Your response must be a raw JSON array of integers.
            Do not wrap the JSON in Markdown. Do not use a ```json code block.
            Do not include any explanatory text.
    
            For example:
            [75, 12, 34, 2, 1]
    
            Ranking:"""
    
    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.choices[0].message.content.strip()
    ranked_ids = json.loads(raw_text)
    return ranked_ids

    
def llm_judge(query, formatted_results):
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

                Query: "{query}"

                Results:{chr(10).join(formatted_results)}

                Scale:
                - 3: Highly relevant
                - 2: Relevant
                - 1: Marginally relevant
                - 0: Not relevant

                Do NOT give any numbers other than 0, 1, 2, or 3.

                Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

                [2, 0, 3, 2, 0, 1]"""

    

    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=[{"role": "user", "content": prompt}],
    )
    
    raw_text = response.choices[0].message.content.strip()
    scores = json.loads(raw_text)

    return scores