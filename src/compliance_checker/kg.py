"""
Build the Neo4j knowledge graph from engineering drawing PDF books.

Indexes every book under data/drawings/ that has a metadata.json (see BOOKS).
"""

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import openai
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from loguru import logger
from neo4j import GraphDatabase
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# Each drawing book is a directory under DRAWINGS_ROOT with a metadata.json
# (list of {"DrawingTitle", "DrawingNumber", "DrawingPage"} entries).
DRAWINGS_ROOT = Path("data/drawings")
BOOKS = [
    {"filename": d.name, "metadata_path": str(d / "metadata.json")}
    for d in sorted(DRAWINGS_ROOT.glob("*/"))
    if (d / "metadata.json").exists()
]


def encode_text(client: OpenAI, text: str) -> list[float] | None:
    """Embed *text* with text-embedding-3-small."""
    try:
        response = client.embeddings.create(input=[text], model="text-embedding-3-small")
        return response.data[0].embedding if response.data else None
    except openai.OpenAIError:
        logger.exception("Embedding request failed for text: {!r}", text)
        return None


def add_drawing_file(kg: Neo4jGraph, filename: str) -> None:
    """Ensure a File node exists for *filename*."""
    cypher = "MERGE (f:File {filename: $filename})"
    kg.query(cypher, params={"filename": filename})


def add_page(  # noqa: PLR0913
    kg: Neo4jGraph,
    filename: str,
    page_index: int,
    page_title: str,
    page_number: str,
    text_embedding: list[float],
) -> None:
    """Upsert a Page node and link it to its parent File."""
    cypher = """
        MERGE (p:Page {title: $page_title})
        ON CREATE
            SET p.index = $page_index,
                p.number = $page_number,
                p.textEmbedding = $text_embedding
        MERGE (f:File {filename: $filename})
        MERGE (p)-[:EXCERPTED_FROM]->(f)
    """
    params = {
        "filename": filename,
        "page_index": page_index,
        "page_title": page_title,
        "page_number": page_number,
        "text_embedding": text_embedding,
    }
    kg.query(cypher, params=params)


def create_vector(kg: Neo4jGraph) -> None:
    """Create the shared cosine vector index (idempotent)."""
    kg.query(
        """
        CREATE VECTOR INDEX `references` IF NOT EXISTS
        FOR (p:Page) ON (p.textEmbedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }
        }
        """,
    )


def main() -> None:
    parser = ArgumentParser(description="Build Neo4j KG from engineering drawing books.")
    parser.add_argument(
        "--neo4j_uri",
        default=os.getenv("NEO4J_URI", ""),
        help="Neo4j connection URI",
    )
    parser.add_argument(
        "--neo4j_username",
        default=os.getenv("NEO4J_USERNAME", "neo4j"),
        help="Neo4j username",
    )
    parser.add_argument(
        "--neo4j_password",
        default=os.getenv("NEO4J_PASSWORD", ""),
        help="Neo4j password",
    )
    parser.add_argument(
        "--neo4j_database",
        default=os.getenv("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name",
    )
    parser.add_argument(
        "--openai_api_key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI API key",
    )
    args = parser.parse_args()

    auth = (args.neo4j_username, args.neo4j_password)
    with GraphDatabase.driver(args.neo4j_uri, auth=auth) as driver:
        driver.verify_connectivity()
    logger.info("Neo4j connection OK")

    kg = Neo4jGraph(
        url=args.neo4j_uri,
        username=args.neo4j_username,
        password=args.neo4j_password,
        database=args.neo4j_database,
    )
    client = OpenAI(api_key=args.openai_api_key)

    for book in BOOKS:
        filename = book["filename"]
        metadata_path = Path(book["metadata_path"])
        logger.info("Processing: {}", filename)

        with metadata_path.open(encoding="utf-8") as fh:
            metadata = json.load(fh)

        add_drawing_file(kg, filename)

        for entry in tqdm(metadata, desc=filename[:30]):
            page_index = entry["DrawingPage"]
            page_title = entry["DrawingTitle"]
            page_number = entry["DrawingNumber"]
            embedding = encode_text(client, page_title)
            if embedding is None:
                logger.warning("Skipping {}: embedding failed", page_number)
                continue
            add_page(kg, filename, page_index, page_title, page_number, embedding)

    create_vector(kg)
    logger.info("Vector index 'references' created (or already exists).")


if __name__ == "__main__":
    main()
