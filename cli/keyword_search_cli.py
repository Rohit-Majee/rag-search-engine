import argparse
from lib.keyword_search import search_command, build_command, tf_command, idf_command, tf_idf_command, bm25_idf_command, bm25_tf_command,bm25_search_command
from lib.search_utils import BM25_K1,BM25_B


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 1. Search Command
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    # 2. Build Command
    build_parser = subparsers.add_parser("build", help="Builds the inverse indexing")

    # 3. TF Command
    tf_parser = subparsers.add_parser("tf", help="Gives the term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID") 
    tf_parser.add_argument("term", type=str, help="Term")

    # 4 IDF Command
    idf_parser = subparsers.add_parser("idf", help="Gives the inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term")

    # 5 TF-IDF Command
    tf_idf_parser = subparsers.add_parser("tfidf", help="Gives the TF-IDF score")
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID") 
    tf_idf_parser.add_argument("term", type=str, help="Term")

    # 6 BM25 IDF Command
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    # 7 BM25 TF command
    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    # 8 BM25 Search Command
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type=int, nargs='?', default=5, help="Limit of search items")

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