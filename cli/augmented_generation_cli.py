import argparse
from lib.augmented_generation import rag,summarize,citations,ask_question

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval-Augmented Generation (RAG) CLI - Generate AI answers grounded in your search data")
    subparsers = parser.add_subparsers(dest="command", help="Available generative AI commands")

    rag_parser = subparsers.add_parser("rag", help="Execute a full RAG pipeline (retrieve relevant documents and generate an AI answer)")
    rag_parser.add_argument("query", type=str, help="The natural language question or prompt to answer using RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Retrieve documents matching the query and generate a concise AI summary of the results")
    summarize_parser.add_argument("query", type=str, help="The natural language query used to find documents to summarize")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Maximum number of retrieved documents to include in the summary (default: 5)")

    citations_parser = subparsers.add_parser("citations", help="Generate an AI answer to the query with inline citations referencing the retrieved source documents")
    citations_parser.add_argument("query", type=str, help="The natural language question to answer with cited sources")
    citations_parser.add_argument("--limit", type=int, default=5, help="Maximum number of source documents to retrieve for citation (default: 5)")

    question_parser = subparsers.add_parser("question", help="Engage in an interactive conversational Q&A session grounded in the retrieved search results")
    question_parser.add_argument("query", type=str, help="The initial question to start the conversational Q&A session")
    question_parser.add_argument("--limit", type=int, default=5, help="Maximum number of background documents to retrieve for context (default: 5)")

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