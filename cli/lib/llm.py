from openai import OpenAI
import os
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