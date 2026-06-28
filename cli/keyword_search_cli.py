import argparse
from lib.keyword_search import search_command, build_command, tf_command, idf_command, tf_idf_command

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

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for : {args.query}")
            results = search_command(args.query)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}")
                
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

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()