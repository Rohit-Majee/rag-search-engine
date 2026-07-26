import argparse
from lib.hybrid_search import normalize_scores,weighted_search_command,rrf_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search Engine CLI - Combine keyword and semantic search results")
    subparsers = parser.add_subparsers(dest="command", help="Available hybrid search, normalization, and ranking commands")


    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of raw scores to a 0.0 - 1.0 scale using Min-Max scaling")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="A space-separated list of raw numeric scores to normalize")

    ws_parser = subparsers.add_parser("weighted-search", help="Perform a hybrid search using a weighted average of normalized keyword and semantic scores")
    ws_parser.add_argument("query", type=str, help="The natural language query to search for")
    ws_parser.add_argument("--alpha", type=float, nargs="?",default=0.5, help="Weight coefficient (0.0 to 1.0) for the BM25 keyword score. Semantic score gets (1 - alpha). (default: 0.5)")
    ws_parser.add_argument("--limit", type=int, nargs="?",default=5, help="Maximum number of search results to return (default: 5)")

    rrf_parser = subparsers.add_parser("rrf-search", help="Perform a hybrid search using Reciprocal Rank Fusion (RRF) to combine results")
    rrf_parser.add_argument("query", type=str, help="The natural language query to search for")
    rrf_parser.add_argument("--k", type=int, nargs="?",default=60, help="The RRF constant 'k' that dampens the impact of extreme high rankings (default: 60)")
    rrf_parser.add_argument("--limit", type=int, nargs="?",default=5, help="Maximum number of search results to return (default: 5)")
    rrf_parser.add_argument("--enhance",type=str,choices=["spell","rewrite","expand"],help="Apply an LLM-based query enhancement technique before executing the search")
    rrf_parser.add_argument("--rerank-method",type=str,choices=["individual","batch","cross_encoder"],help="Apply a secondary re-ranking model to sort the final retrieved results")
    rrf_parser.add_argument("--evaluate", action="store_true", help="Automatically evaluate the relevance of the retrieved search results using an LLM judge")

    args = parser.parse_args()

    match args.command:
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit, args.enhance, args.rerank_method,args.evaluate)
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