import argparse
from lib.keyword_search import search_command, build_command, tf_command, idf_command, tf_idf_command, bm25_idf_command, bm25_tf_command,bm25_search_command
from lib.search_utils import BM25_K1,BM25_B


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search Engine CLI - Index and search movie documents")
    subparsers = parser.add_subparsers(dest="command", help="Available search, indexing, and scoring commands")

    # 1. Search Command
    search_parser = subparsers.add_parser("search", help="Search the movie database using basic keyword matching")
    search_parser.add_argument("query", type=str, help="The text query or keywords to search for")

    # 2. Build Command
    build_parser = subparsers.add_parser("build", help="Parse the dataset and build the inverted index required for searching")

    # 3. TF Command
    tf_parser = subparsers.add_parser("tf", help="Calculate the standard Term Frequency (TF) of a specific word within a given document")
    tf_parser.add_argument("doc_id", type=int, help="The unique integer ID of the document (movie)") 
    tf_parser.add_argument("term", type=str, help="The specific word to analyze")

    # 4 IDF Command
    idf_parser = subparsers.add_parser("idf", help="Calculate the Inverse Document Frequency (IDF) of a word across the entire dataset")
    idf_parser.add_argument("term", type=str, help="The specific word to analyze")

    # 5 TF-IDF Command
    tf_idf_parser = subparsers.add_parser("tfidf", help="Calculate the combined TF-IDF score for a specific word in a given document")
    tf_idf_parser.add_argument("doc_id", type=int, help="The unique integer ID of the document (movie)") 
    tf_idf_parser.add_argument("term", type=str, help="The specific word to analyze")

    # 6 BM25 IDF Command
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Calculate the BM25-optimized Inverse Document Frequency (IDF) for a word")
    bm25_idf_parser.add_argument("term", type=str, help="The specific word to get the BM25 IDF score for")

    # 7 BM25 TF command
    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Calculate the BM25-optimized Term Frequency (TF) for a word in a given document"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="The unique integer ID of the document (movie)")
    bm25_tf_parser.add_argument("term", type=str, help="The specific word to get the BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 k1 parameter (controls term saturation threshold)")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter (controls document length normalization)")

    # 8 BM25 Search Command
    bm25search_parser = subparsers.add_parser("bm25search", help="Search the movie database using the advanced BM25 ranking algorithm")
    bm25search_parser.add_argument("query", type=str, help="The text query or keywords to search for")
    bm25search_parser.add_argument("limit", type=int, nargs='?', default=5, help="Maximum number of search results to return (default: 5)")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for : {args.query}")
            results = search_command(args.query)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}")

        case "bm25search":
            results = bm25_search_command(args.query, args.limit)
            
            for i, (movie, score) in enumerate(results, start=1):
                print(f"{i}. ({movie['id']}) {movie['title']} - Score: {score:.2f}")
                
        case "build":
            print("Building the Inverse Indexing...")
            build_command()
            
        case "tf":
            tf_command(args.doc_id, args.term) 
        
        case "idf":
            idf = idf_command(args.term) 
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            tf_idf = tf_idf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()