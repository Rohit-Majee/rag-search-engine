import argparse
import base64
import mimetypes
from openai import OpenAI

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)


def main():
    parser = argparse.ArgumentParser(description="Multimodal Query Rewriting")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--query", required=True, help="Text query to rewrite")
    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    try:
        with open(args.image, "rb") as f:
            img = f.read()
    except FileNotFoundError:
        print(f"[Error] Could not find image at {args.image}")
        return

    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"

    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
                    - Synthesize visual and textual information
                    - Focus on movie-specific details (actors, scenes, style, etc.)
                    - Return only the rewritten query, without any additional commentary"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            model=os.getenv("IMAGE_MODEL"), 
            messages=messages,
        )
        
        content = response.choices[0].message.content
        print(f"Rewritten query: {content.strip()}")
        
        if response.usage is not None:
            print(f"Total tokens:    {response.usage.total_tokens}")
            
    except Exception as e:
        print(f"[Error] Vision model failed: {e}")

if __name__ == "__main__":
    main()