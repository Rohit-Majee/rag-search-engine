import argparse
from lib.hybrid_search import normalize_scores,weighted_search_command,rrf_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")


    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="A list of raw scores separated by spaces")

    ws_parser = subparsers.add_parser("weighted-search", help="A hybrid search with weighted average combined.")
    ws_parser.add_argument("query", type=str, help="Search query")
    ws_parser.add_argument("--alpha", type=float, nargs="?",default=0.5, help="Percentage of weight for BM25.")
    ws_parser.add_argument("--limit", type=int, nargs="?",default=5, help="limit of results to return.")

    rrf_parser = subparsers.add_parser("rrf-search", help="A hybrid search with RRF scores.")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("--k", type=int, nargs="?",default=60, help="controls how much more weight we give to higher-ranked results vs. lower-ranked ones.")
    rrf_parser.add_argument("--limit", type=int, nargs="?",default=5, help="limit of results to return.")
    
    args = parser.parse_args()

    match args.command:
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "normalize":
            scores = normalize_scores(args.scores) 
            for score in scores:
                print(f"* {score:.4f}") 
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()