import argparse
from lib.augmented_generation import rag,summarize,citations,ask_question

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize search results for a query")
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Number of results to summarize")

    citations_parser = subparsers.add_parser("citations", help="Answer query with citations")
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument("--limit", type=int, default=5, help="Number of results to retrieve")

    question_parser = subparsers.add_parser("question", help="Conversational Q&A based on search results")
    question_parser.add_argument("query", type=str, help="Question to ask")
    question_parser.add_argument("--limit", type=int, default=5, help="Number of results to retrieve")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag(query)
        case "summarize":
            summarize(args.query, args.limit)
        case "citations":
            citations(args.query, args.limit)
        case "question":
            ask_question(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()