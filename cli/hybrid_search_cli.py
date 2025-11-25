import argparse

from lib.hybrid_search import (
    normalize_scores,
    weighted_search_command,
    rrf_search_command,
)

from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="List of scores to be normalized")

    weighted_parser = subparser.add_parser("weighted-search", help="Perform weighted hybrid search")
    weighted_parser.add_argument("query", type=str, help="search query")
    weighted_parser.add_argument("--alpha", type=float, nargs="?", default=0.5, help="Weight for BM25 vs semantic (0=all semantic, 1=all BM25, default=0.5)")
    weighted_parser.add_argument("--limit", type=int, nargs="?", default=DEFAULT_SEARCH_LIMIT, help="Number of results to return (default=5)")

    rrf_parser = subparser.add_parser("rrf-search", help="Perform Reciprocal Rank Fusion hybrid search")
    rrf_parser.add_argument("query", type=str, help="search query")
    rrf_parser.add_argument("--k", type=int, default=60, help="RRF k parameter controlling weight distribution - higher-ranked results vs lower-ranked ones (1=high rank distribution, 100=low rank distribution) - default=60")
    rrf_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Reranking method")
    rrf_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to return (default=5)")
    

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            results = weighted_search_command(args.query, args.alpha, args.limit)
            print(f"Weighted Hybrid Search Results for '{results["query"]}' (alpha={results["alpha"]}):")
            print(f"   Alpha {results["alpha"]}: {int(results["alpha"] * 100)}% Keyword, {int((1 - results["alpha"]) * 100)}% Semantic")
            for i, res in enumerate(results["results"], 1):
                print(f"{i}. {res["title"]}")
                print(f"   Hybrid Score: {res.get("score", 0):.3f}")
                metadata = res.get("metadata", {})
                if "bm25_score" in metadata and "semantic_score" in metadata:
                    print(
                        f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
                    )
                print(f"   {res['document'][:100]}...")
                print()
        case "rrf-search":
            results = rrf_search_command(args.query, args.k, args.enhance, args.rerank_method, args.limit)
            
            if results["enhanced_query"]:
                print(f"Enhnaced query ({results["enhance_method"]}): '{results["original_query"]}' -> '{results["enhanced_query"]}'\n")

            if results["reranked"]:
                print(
                    f"Reranking top {len(results['results'])} results using {results['rerank_method']} method...\n"
                )

            print(f"Reciprocal Rank Fusion Results for '{results["query"]}' (k={results["k"]}):")

            for i, res in enumerate(results["results"], 1):
                print(f"{i}. {res["title"]}")
                
                if "individual_score" in res:
                    print(f"   Rerank Score: {res.get("individual_score", 0):.3f}/10")
                if "batch_rank" in res:
                    print(f"   Rerank Rank: {res.get("batch_rank", 0)}")
                if "cross_encoder_score" in res:
                    print(f"   Cross-Encoder Score: {res["crossencoder_score"]:.3f}")
                print(f"   RRF Score: {res["score"]:.3f}")
                metadata = res.get("metadata", {})
                print(f"   BM25 Rank: {metadata["bm25_rank"]}, Semantic Rank: {metadata["semantic_rank"]}")
                print(f"   {res["document"][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
