import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Generate and check the shape of an image embedding",
    )
    verify_parser.add_argument("image_path", type=str, help="Path to the image file")

    search_parser = subparsers.add_parser(
        "image_search",
        help="Search movies using an image query",
    )
    search_parser.add_argument("image_path", type=str, help="Path to the image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)

        case "image_search":
            results = image_search_command(args.image_path)
            for i, res in enumerate(results, start=1):
                desc = res["description"]
                if len(desc) > 100:
                    desc = desc[:100] + "..."
                print(f"{i}. {res['title']} (similarity: {res['score']:.3f})")
                print(f"   {desc}\n")


if __name__ == "__main__":
    main()