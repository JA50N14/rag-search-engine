import argparse

from lib.rag_search import rag_command, summarize_command, citations_command, question_command
from lib.search_utils import DEFAULT_SEARCH_LIMIT

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarizes search results")
    summarize_parser.add_argument("query", type=str, help="Search query for Summarize")
    summarize_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of search results to return")

    citations_parser = subparsers.add_parser("citations", help="Provides citations in search results")
    citations_parser.add_argument("query", type=str, help="Search query for Citations")
    citations_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of search results to return")

    questions_parser = subparsers.add_parser("question", help="Provides answer to a question")
    questions_parser.add_argument("question", type=str, help="Question you want to ask")
    questions_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of search results to return")

    args = parser.parse_args()

    match args.command:
        case "rag":
            results = rag_command(args.query)
            print("Search Results:")
            for res in results["search_results"]:
                print(f"   - {res["title"]}")
            print()

            print("RAG Response:")
            print({results["answer"]})
        case "summarize":
            results = summarize_command(args.query, args.limit)
            
            print("Search Results:")
            for res in results["search_results"]:
                print(f"   - {res["title"]}")
            print()

            print("LLM Summary:")
            print(f"{results["summary"]}")
        case "citations":
            results = citations_command(args.query, args.limit)

            print("Search Results:")
            for res in results["search_results"]:
                print(f"   - {res["title"]}")
            print()
            
            print("LLM Answer:")
            print(f"{results["summary"]}")
        case "question":
            results = question_command(args.question, args.limit)

            print("Search Results:")
            for res in results["search_results"]:
                print(f"   - {res["title"]}")
            print()

            print("Answer:")
            print(f"{results["answer"]}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()