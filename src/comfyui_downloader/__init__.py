import argparse
from .downloader import download_models_from_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download models listed in a JSON file."
    )
    parser.add_argument(
        "json_file", help="Path to the JSON file containing model URLs."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="models",
        help="Base directory where files will be saved.",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show which files would be downloaded and where, without downloading.",
    )
    args = parser.parse_args()

    download_models_from_json(args.json_file, args.output_dir, args.dry_run)
