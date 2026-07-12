import os
import sys
import uuid
import json
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_texts, get_embedding_dim
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import chunk_text

logfire.configure(service_name="enterprise-ingestion-service")

# Local folder where parsed + chunked JSON metadata is saved(replaces GCS processed bucket)
PROCESSED_DATA_DIR = "processed_data"

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url = settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)

def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest

def process_file(file_path: str, filename: str, source_type: str):
    """Parse -> chunk -> save locally -> embed -> index in Qdrant."""
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".",1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return 

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} - skipping.")
                return 
            
            # 2. Chunk text
            chunks = chunk_text(full_text)
            if not chunks:
                return 
            
            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data -> {local_path}")

            # 4. Embed and index in Qdrant
            with logfire.span("Vectorizing & Indexing"):
                embeddings = embed_texts(chunks)
                points = [
                    models.PointStruct(
                        id = str(uuid.uuid4()),
                        vector = vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                        },
                    )
                    for chunk, vector in zip(chunks, embeddings)
                ]

                qdrant_client.upsert(
                    collection_name = settings.QDRANT_COLLECTION,
                    points = points,
                )
                logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}.")
        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")

def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)

def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        # wip collection if requested
        if wipe:
            with logfire.span("Wipping Collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")

        # Recreate collection - dimension resolve at runtime after embedding model probe
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            dim = get_embedding_dim()
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION}' "
                f"({dim}-dim, Cosine)."
            )
        
        # Route to sub-folders or treat the whole dir as one souce
        # Get all folders inside base_dir
        subdirs = [
            d for d in os.listdir(base_dir)

            # Keep only directories
            if os.path.isdir(os.path.join(base_dir,d))
        ]

        # No subfolders found
        if not subdirs:
            # if the user explicitly provided a source type
            if explicit_source_type:
                # Use that source type
                source_type = explicit_source_type
            else:
                # Get folder name
                base_name = os.path.basename(
                    os.path.normalpath(base_dir)
                ).lower()

                # Automatically determine source type
                source_type = (
                    "true"
                    if "true" in base_name
                    else "noisy"
                    if "noisy" in base_name
                    else "general"
                )

            # Log selected source type
            logfire.info(
                f"No sub-folders found - processing "
                f"'{base_dir}' as '{source_type}'."
            )

            # Process every file inside this directory
            process_directory(
                base_dir,
                source_type,
            )
        
        # Process each subfolder separately
        else:
            # Iterate over every subdirectory
            for subdir in subdirs:
                # Automatically determine source type
                source_type = (
                    "true"
                    if "true" in subdir.lower()
                    else "noisy"
                    if "noisy" in subdir.lower()
                    else subdir
                )

                # Process all files inside this subdirectory
                process_directory(
                    os.path.join(base_dir,subdir),
                    source_type,
                )

if __name__ == "__main__":

    # Example ways to run this script
    # python -m app.ingestion.processor DATA --wipe
    # process the DATA folder after deleting the old Qdrant collection.
    # python -m app.ingestion.processor DATA/true_data true
    # process only the "true_data" folder and explicitly set source_type = "true"

    # check whether "--wipe" flag was provided
    # returns True if present, otherwise False
    wipe_requested = "--wipe" in sys.argv

    # Remove "--wipe" from the command-line arguments
    # This leaves only the useful positional arguments
    clean_args = [
        a for a in sys.argv if a != "--wipe"
    ]

    # Get the target directory from the command-line arguments
    # clean_args[0] = processor.py
    # clean_args[1] = DATA
    # If no directory is provided, use "DATA" as the default
    target_dir = (
        clean_args[1]
        if len(clean_args) > 1
        else "DATA"
    )

    # Get the optional source type
    # Example:
    # python processor.py DATA true
    # explicit_type = "true"
    # If not provided, keep it as None
    explicit_type = (
        clean_args[2]
        if len(clean_args) > 2
        else None 
    )

    # Check whether the target directory actually exists
    if not os.path.exists(target_dir):
        # Print an error message
        print(
            f"Error: path '{target_dir}' does not exist."
        )
        # Exit the program with status code 1(failure)
        sys.exit(1)

    # Start the complete ingestion pipeline
    run_universal_ingestion(
        # Folder to process
        target_dir,
        # Optional source_type
        explicit_source_type=explicit_type,
        # Whether to wipe the existing Qdrant collection
        wipe=wipe_requested,
    )

    # Log successful completion
    logfire.info("Ingestion job completed.")