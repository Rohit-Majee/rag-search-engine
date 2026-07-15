import argparse
from lib.semantic_search import verify_model,embed_text,verify_embeddings,embed_query_text,search_command,chunk_text,semantic_chunk_text,embed_chunks,search_chunked
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

    chunk_parser = subparsers.add_parser("chunk", help="helps to chunk a text")
    chunk_parser.add_argument("text", type=str, help="The text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words per chunk (default: 200)")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words between chunks (default: 0)")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="helps to chunk a text semantically")
    semantic_chunk_parser.add_argument("text", type=str, help="The text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Number of words per chunk (default: 4)")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words between chunks (default: 0)")
    
    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Generate and cache embeddings for chunked documents")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search using semantic chunked embeddings")
    search_chunked_parser.add_argument("query", type=str, help="The search query text")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")  
    
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
        case "chunk":
            chunk_text(args.text, args.chunk_size,args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()