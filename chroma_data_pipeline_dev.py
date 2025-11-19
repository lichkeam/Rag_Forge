import sys
import io
import chromadb
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os
from groq import Groq
from dotenv import load_dotenv
from util import read_data, download_data

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def chunk_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks (original function).

    Args:
        text: Input text to split
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def chunk_text_semantic(
    text: str,
    target_chunk_size: int = 500,
    api_key: Optional[str] = None,
    model: str = "llama-3.1-8b-instant",
    rate_limit_delay: float = 3.0
) -> List[str]:
    """
    Split text into semantic chunks using LLM with rate limiting.

    Args:
        text: Input text to split
        target_chunk_size: Target size for each chunk (approximate)
        api_key: Groq API key (if None, will try to use environment variable)
        model: Llama model to use for semantic splitting
        rate_limit_delay: Delay in seconds between API calls (default: 60s)

    Returns:
        List of semantically coherent text chunks
    """
    if not text.strip():
        return []

    # Initialize Groq client
    try:
        client = Groq(api_key=api_key) if api_key else Groq()
        print(f"Groq client initialized")
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        print("Falling back to fixed-size chunking...")
        return chunk_text(text, chunk_size=target_chunk_size, overlap=50)

    # Estimated number of tokens（1 token ≈ 4 characters）
    estimated_tokens = len(text) // 4
    print(f"Estimated input tokens: ~{estimated_tokens}")

    # Create prompt for LLM
    prompt = f"""You are a text segmentation expert. Your task is to split the following text into semantically coherent chunks.

Rules:
1. Each chunk should contain complete thoughts or paragraphs
2. Target chunk size is approximately {target_chunk_size} characters (can vary between {target_chunk_size-100} to {target_chunk_size+100})
3. Split at natural boundaries (sentences, paragraphs, topic changes)
4. Preserve the original text exactly - do not summarize or modify
5. Output ONLY the chunks separated by "===CHUNK_SEPARATOR===" marker
6. Do not add any explanations or numbering

Text to split:
{text}

Output the chunks now:"""

    try:
        print(
            f"Calling LLM API at {datetime.now().strftime('%H:%M:%S')}...",
            flush=True)

        # Call LLM
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=8000
        )

        response = completion.choices[0].message.content

        # Actual usage of token
        if hasattr(completion, 'usage'):
            print(f"API call successful")
            print(f"  - Input tokens: {completion.usage.prompt_tokens}")
            print(f"  - Output tokens: {completion.usage.completion_tokens}")
            print(f"  - Total tokens: {completion.usage.total_tokens}")

        # Parse the response
        chunks = [chunk.strip()
                  for chunk in response.split("===CHUNK_SEPARATOR===")]
        chunks = [chunk for chunk in chunks if chunk]

        # Validate chunks
        if not chunks:
            print("Warning: LLM returned no chunks, falling back to fixed-size chunking")
            return chunk_text(text, chunk_size=target_chunk_size, overlap=50)

        # Check if total length is preserved
        original_length = len(text)
        total_chunk_length = sum(len(chunk) for chunk in chunks)

        if abs(total_chunk_length - original_length) > original_length * 0.1:
            print(
                f"Warning: Chunk length mismatch (original: {original_length}, chunks: {total_chunk_length})")
            print("Falling back to fixed-size chunking...")
            return chunk_text(text, chunk_size=target_chunk_size, overlap=50)

        print(
            f"Semantic chunking created {len(chunks)} chunks (avg size: {total_chunk_length//len(chunks)} chars)")

        # Rate limiting:
        if rate_limit_delay > 0:
            print(
                f"Rate limiting: waiting {rate_limit_delay} seconds before next API call...")
            time.sleep(rate_limit_delay)

        return chunks

    except Exception as e:
        print(f"Error during semantic chunking: {e}")
        print("Falling back to fixed-size chunking...")
        return chunk_text(text, chunk_size=target_chunk_size, overlap=50)


def process_dataset_for_rag(
    df: pd.DataFrame,
    content_column: str,
    id_column: Optional[str] = None,
    metadata_columns: Optional[List[str]] = None,
    chunk_size: int = 500,
    overlap: int = 50,
    use_semantic_chunking: bool = False,
    groq_api_key: Optional[str] = None,
    llm_model: str = "llama-3.3-70b-versatile"
) -> Tuple[List[str], List[Dict]]:
    """
    Process dataset into chunks suitable for RAG (Retrieval-Augmented Generation).

    Args:
        df: Input DataFrame
        content_column: Name of column containing main content
        id_column: Name of column containing unique identifiers (optional)
        metadata_columns: List of columns to include as metadata (optional)
        chunk_size: Size of each text chunk (for fixed chunking) or target size (for semantic)
        overlap: Overlap between consecutive chunks (only for fixed chunking)
        use_semantic_chunking: If True, use LLM-based semantic chunking; if False, use fixed-size chunking
        groq_api_key: API key for Groq (only needed if use_semantic_chunking=True)
        llm_model: Llama model to use for semantic chunking

    Returns:
        Tuple of (list of text chunks, list of metadata dictionaries)
    """
    all_chunks = []
    all_metadata = []

    chunking_method = "SEMANTIC (LLM-based)" if use_semantic_chunking else "FIXED-SIZE"
    print(
        f"\nProcessing {len(df)} records using {chunking_method} chunking...")
    print(f"Content column: '{content_column}'")

    if metadata_columns:
        print(f"Metadata columns: {metadata_columns}")
    else:
        print("Warning: No metadata assigned.")

    for idx, row in df.iterrows():
        content = str(row[content_column])

        # Choose chunking method
        if use_semantic_chunking:
            chunks = chunk_text_semantic(
                content,
                target_chunk_size=chunk_size,
                api_key=groq_api_key,
                model=llm_model
            )
        else:
            chunks = chunk_text(content, chunk_size, overlap)

        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)

            # Create metadata dictionary
            metadata = {
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "source_row": int(idx),
                "chunking_method": chunking_method  # 記錄使用的切割方法
            }

            # Add ID if specified
            if id_column and id_column in row:
                metadata["source_id"] = str(
                    row[id_column]) if pd.notna(
                    row[id_column]) else ""

            # Add other metadata columns
            if metadata_columns:
                for col in metadata_columns:
                    if col in row:
                        value = row[col]
                        if pd.isna(value):
                            metadata[col] = ""
                        elif isinstance(value, (str, int, float, bool)):
                            metadata[col] = value
                        else:
                            metadata[col] = str(value)

            all_metadata.append(metadata)

        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} records...")

    print(f"\nGenerated {len(all_chunks)} text chunks total")
    return all_chunks, all_metadata


def create_or_get_collection(
    client: chromadb.PersistentClient,
    db_name: str
) -> chromadb.Collection:
    """
    Get existing collection or create new one if it doesn't exist.

    Args:
        client: ChromaDB client instance
        db_name: Name of the collection

    Returns:
        ChromaDB Collection object
    """
    try:
        collection = client.get_collection(name=db_name)
        print(f"Database exists with {collection.count()} chunks")
        print("Skipping data insertion")
    except Exception:
        print("Creating new database...")
        collection = client.create_collection(name=db_name)

    return collection


def add_chunks_to_db(
    collection: chromadb.Collection,
    chunks: List[str],
    metadata: List[Dict],
    batch_size: int = 5000
) -> None:
    """
    Add text chunks to ChromaDB collection in batches.

    Args:
        collection: ChromaDB collection to add to
        chunks: List of text chunks
        metadata: List of metadata dictionaries
        batch_size: Number of chunks to add per batch
    """
    total_chunks = len(chunks)

    for i in range(0, total_chunks, batch_size):
        batch_end = min(i + batch_size, total_chunks)
        batch_chunks = chunks[i:batch_end]
        batch_metadata = metadata[i:batch_end]
        batch_ids = [f"chunk_{j}" for j in range(i, batch_end)]

        collection.add(
            documents=batch_chunks,
            metadatas=batch_metadata,
            ids=batch_ids
        )

        print(f"Added {batch_end} / {total_chunks} chunks")

    print(f"\nComplete! Database contains {collection.count()} chunks")


def main(
    download: bool = False,
    create_db: bool = False,
    debugging_process_dataset: bool = True,
    source: str = "jrobischon/wikipedia-movie-plots",
    metadata_columns: List[str] = ['category', 'filename', 'title'],
    content_column: str = "content",
    datapath: str = "_.csv",
    db_name: str = "test",
    readdf_sep: str = '\t',
    db_path: str = "./chroma_db",
    use_semantic_chunking: bool = False,
    groq_api_key: Optional[str] = None,
    llm_model: str = "llama-3.3-70b-versatile"
):
    """
    Main function to orchestrate data processing and database creation.

    Args:
        download: Whether to download dataset from Kaggle
        create_db: Whether to create ChromaDB database
        debugging_process_dataset: Whether to print sample chunks for debugging
        source: Kaggle dataset identifier
        metadata_columns: Columns to include as metadata
        content_column: Column containing main text content
        datapath: Path to local CSV file
        db_name: Name for ChromaDB collection
        readdf_sep: CSV delimiter character
        db_path: Path to store ChromaDB files
        use_semantic_chunking: If True, use LLM semantic chunking; if False, use fixed-size
        groq_api_key: Groq API key (required if use_semantic_chunking=True)
        llm_model: Llama model for semantic chunking
    """
    if metadata_columns is None:
        raise ValueError("metadata_columns cannot be None")

    # Debug mode
    if debugging_process_dataset:
        print("\n=== DEBUGGING MODE ===")
        df = read_data(datapath=datapath, sep=readdf_sep)
        all_chunks, all_metadata = process_dataset_for_rag(
            df,
            content_column=content_column,
            id_column=None,
            metadata_columns=metadata_columns,
            chunk_size=500,
            overlap=50,
            use_semantic_chunking=use_semantic_chunking,
            groq_api_key=groq_api_key,
            llm_model=llm_model
        )

        print("\n--- Sample Chunks (first 5) ---")
        for idx, (chunk, meta) in enumerate(zip(all_chunks, all_metadata)):
            if idx >= 5:
                break
            print(f"\n[Chunk {idx}]")
            print(f"Text: {chunk[:100]}...")
            print(f"Metadata: {meta}")

    # Download dataset
    if download:
        print("\n=== DOWNLOADING DATASET ===")
        download_data(source=source)

    # Create database
    if create_db:
        print("\n=== CREATING DATABASE ===")
        df = read_data(datapath=datapath, sep=readdf_sep)
        all_chunks, all_metadata = process_dataset_for_rag(
            df,
            content_column=content_column,
            id_column=None,
            metadata_columns=metadata_columns,
            chunk_size=500,
            overlap=50,
            use_semantic_chunking=use_semantic_chunking,
            groq_api_key=groq_api_key,
            llm_model=llm_model
        )

        client = chromadb.PersistentClient(path=db_path)
        collection = create_or_get_collection(client, db_name)

        if collection.count() > 0:
            print("Database already populated, skipping insertion")
            return

        add_chunks_to_db(collection, all_chunks, all_metadata, batch_size=5000)


if __name__ == "__main__":

    load_dotenv()
    groq_apikey = os.getenv("GROQ_API_KEY")

    main(
        use_semantic_chunking=False,
        debugging_process_dataset=True,
        download=False,
        create_db=False,
        groq_api_key=groq_apikey,
        metadata_columns=['category', 'filename', 'title'],
        content_column="content",
        datapath="bbc-news.csv",
        db_name="bbc_news_semantic",
        readdf_sep='\t',
        db_path="./chroma_db",
        llm_model="llama-3.1-8b-instant"
    )
