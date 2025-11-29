import argparse

from lib.multimodal_search import verify_image_embedding, image_search_command

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding", help="Image Path")
    verify_parser.add_argument("image", type=str, help="Image path")

    image_search_parser = subparsers.add_parser("image_search", help="Image Search")
    image_search_parser.add_argument("image", type=str, help="Image path")
    
    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            results = image_search_command(args.image)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res["title"]} (similarity: {res["similarity_score"]:.3f})")
                print(f"{res["description"][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()