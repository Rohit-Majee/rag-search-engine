import argparse
from lib.evaluation import evaluate

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI - Measure and benchmark search engine performance using standardized metrics")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of top results to evaluate per query (represents 'k' in metrics like Precision@k and Recall@k) (default: 5)")

    args = parser.parse_args()
    limit = args.limit

    evaluate(limit)

if __name__ == "__main__":
    main()