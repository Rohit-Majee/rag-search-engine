import argparse
from lib.semantic_search import verify_model,embed_text,verify_embeddings,embed_query_text,search_command
def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="verify the embedding model loads properly")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate and print text embeddings")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="verify all the documents are loaded properly")

    embed_query_parser = subparsers.add_parser("embed_query", help="Convert a search query string into a vector")
    embed_query_parser.add_argument("query", type=str, help="The search query text to embed")

    search_parser = subparsers.add_parser("search", help="Search for movies by semantic meaning")
    search_parser.add_argument("query", type=str, help="The search query text")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return (default: 5)")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()