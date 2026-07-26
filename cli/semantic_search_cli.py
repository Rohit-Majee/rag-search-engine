import argparse
from lib.semantic_search import verify_model,embed_text,verify_embeddings,embed_query_text,search_command,chunk_text,semantic_chunk_text,embed_chunks,search_chunked
def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search Engine CLI - Generate embeddings, chunk text, and perform vector search")
    subparsers = parser.add_subparsers(dest="command", help="Available semantic search and embedding commands")

    verify_parser = subparsers.add_parser("verify", help="Verify that the sentence transformer model loads and initializes correctly")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate and output the vector embeddings for a given input string")
    embed_text_parser.add_argument("text", type=str, help="The input text string to convert into vector embeddings")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Generate embeddings for all documents in the dataset and verify caching")

    embed_query_parser = subparsers.add_parser("embed_query", help="Convert a user search query into a vector representation for similarity comparison")
    embed_query_parser.add_argument("query", type=str, help="The search query string to be embedded")

    search_parser = subparsers.add_parser("search", help="Search the movie database using semantic vector similarity (cosine similarity)")
    search_parser.add_argument("query", type=str, help="The natural language query to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of search results to return (default: 5)")

    chunk_parser = subparsers.add_parser("chunk", help="Split input text into fixed-size chunks based on word count")
    chunk_parser.add_argument("text", type=str, help="The long text string to be divided into chunks")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Maximum number of words contained in each chunk (default: 200)")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of words to overlap between sequential chunks to preserve context (default: 0)")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Split input text into semantic chunks to better preserve logical context")
    semantic_chunk_parser.add_argument("text", type=str, help="The long text string to be divided into semantic chunks")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximum allowed units per semantic chunk (default: 4)")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping units between sequential semantic chunks (default: 0)")
    
    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Process the dataset, apply chunking, and generate vector embeddings for all document chunks")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search the movie database against the chunked document embeddings for higher precision")
    search_chunked_parser.add_argument("query", type=str, help="The natural language query to search for")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Maximum number of matched chunk results to return (default: 5)")  
    
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