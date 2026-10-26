"""Convert PDF pages to PNG files based on metadata.json."""

import argparse
import json
from pathlib import Path

import fitz
from tqdm import tqdm


def convert(pdf_path: str, metadata_path: str, output_dir: str, scale: int = 3) -> None:
    with Path(metadata_path).open(encoding="utf-8") as f:
        metadata = json.load(f)

    doc = fitz.open(pdf_path)
    matrix = fitz.Matrix(2.0 * scale, 2.0 * scale)

    for entry in tqdm(metadata, desc="Converting pages"):
        drawing_number = entry["DrawingNumber"]
        page_index = entry["DrawingPage"] - 1  # 1-based → 0-based

        page_dir = Path(output_dir) / drawing_number
        page_dir.mkdir(parents=True, exist_ok=True)

        out_path = page_dir / f"{drawing_number}.png"
        if out_path.exists():
            continue

        pix = doc[page_index].get_pixmap(matrix=matrix)
        pix.save(str(out_path))

    doc.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDF pages to PNG using metadata.json")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--scale", type=int, default=3, help="Scale factor (default: 3)")
    args = parser.parse_args()

    convert(args.pdf, args.metadata, args.output, args.scale)


if __name__ == "__main__":
    main()
