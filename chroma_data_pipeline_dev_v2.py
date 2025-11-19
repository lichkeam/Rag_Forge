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
from abc import ABC, abstractmethod

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class TextChunker(ABC):

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass



class FixedSizeChunker(TextChunker):
    """Fixed-size chunking with overlap"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Args:
            chunk_size: Size of each chunk in characters
            overlap: Number of overlapping characters between chunks
        """
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")
        
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        
        return chunks


class SemanticChunker(TextChunker):
    """LLM-based semantic chunking"""
    
    def __init__(
        self, 
        target_chunk_size: int = 500,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        rate_limit_delay: float = 3.0
    ):
        """
        Args:
            target_chunk_size: Target size for each chunk (approximate)
            api_key: Groq API key
            model: Llama model to use
            rate_limit_delay: Delay between API calls in seconds
        """
        self.target_chunk_size = target_chunk_size
        self.model = model
        self.rate_limit_delay = rate_limit_delay
        
        try:
            self.client = Groq(api_key=api_key) if api_key else Groq()
            print(f"Groq client initialized")
        except Exception as e:
            print(f"Error initializing Groq client: {e}")
            self.client = None
    
    def chunk(self, text: str) -> List[str]:
        """Split text into semantic chunks using LLM"""
        if not text.strip():
            return []
        
        # Fallback if client not initialized
        if self.client is None:
            print("Groq client not available, using fallback chunker")
            fallback = FixedSizeChunker(self.target_chunk_size, 50)
            return fallback.chunk(text)
        
        estimated_tokens = len(text) // 4
        print(f"Estimated input tokens: ~{estimated_tokens}")
        
        prompt = self._build_prompt(text)
        
        try:
            print(f"Calling LLM API at {datetime.now().strftime('%H:%M:%S')}...", flush=True)
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=8000
            )
            
            response = completion.choices[0].message.content
            
            # Log token usage
            if hasattr(completion, 'usage'):
                print(f"API call successful")
                print(f"  - Input tokens: {completion.usage.prompt_tokens}")
                print(f"  - Output tokens: {completion.usage.completion_tokens}")
                print(f"  - Total tokens: {completion.usage.total_tokens}")
            
            chunks = self._parse_response(response, text)
            
            # Rate limiting
            if self.rate_limit_delay > 0:
                print(f"Rate limiting: waiting {self.rate_limit_delay} seconds...")
                time.sleep(self.rate_limit_delay)
            
            return chunks
            
        except Exception as e:
            print(f"Error during semantic chunking: {e}")
            print("Falling back to fixed-size chunking...")
            fallback = FixedSizeChunker(self.target_chunk_size, 50)
            return fallback.chunk(text)
    
    def _build_prompt(self, text: str) -> str:
        """Build prompt for LLM"""
        return f"""You are a text segmentation expert. Your task is to split the following text into semantically coherent chunks.

Rules:
1. Each chunk should contain complete thoughts or paragraphs
2. Target chunk size is approximately {self.target_chunk_size} characters (can vary between {self.target_chunk_size-100} to {self.target_chunk_size+100})
3. Split at natural boundaries (sentences, paragraphs, topic changes)
4. Preserve the original text exactly - do not summarize or modify
5. Output ONLY the chunks separated by "===CHUNK_SEPARATOR===" marker
6. Do not add any explanations or numbering

Text to split:
{text}

Output the chunks now:"""
    
    def _parse_response(self, response: str, original_text: str) -> List[str]:
        """Parse and validate LLM response"""
        chunks = [chunk.strip() for chunk in response.split("===CHUNK_SEPARATOR===")]
        chunks = [chunk for chunk in chunks if chunk]
        
        if not chunks:
            print("Warning: LLM returned no chunks, falling back")
            fallback = FixedSizeChunker(self.target_chunk_size, 50)
            return fallback.chunk(original_text)
        
        # Validate chunk length
        original_length = len(original_text)
        total_chunk_length = sum(len(chunk) for chunk in chunks)
        
        if abs(total_chunk_length - original_length) > original_length * 0.1:
            print(f"Warning: Chunk length mismatch (original: {original_length}, chunks: {total_chunk_length})")
            fallback = FixedSizeChunker(self.target_chunk_size, 50)
            return fallback.chunk(original_text)
        
        print(f"Semantic chunking created {len(chunks)} chunks (avg size: {total_chunk_length//len(chunks)} chars)")
        return chunks


class DatasetProcessor:
    """Process dataset into chunks for RAG"""
    
    def __init__(
        self,
        chunker: TextChunker,
        content_column: str,
        id_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None
    ):
        """
        Args:
            chunker: TextChunker instance (strategy pattern)
            content_column: Column containing main content
            id_column: Column with unique identifiers
            metadata_columns: Columns to include as metadata
        """
        self.chunker = chunker
        self.content_column = content_column
        self.id_column = id_column
        self.metadata_columns = metadata_columns or []
    
    def process(self, df: pd.DataFrame) -> Tuple[List[str], List[Dict]]:
        """
        Process DataFrame into chunks and metadata
        
        Returns:
            Tuple of (text_chunks, metadata_list)
        """
        all_chunks = []
        all_metadata = []
        
        chunking_method = self.chunker.__class__.__name__
        print(f"\nProcessing {len(df)} records using {chunking_method}...")
        print(f"Content column: '{self.content_column}'")
        
        if self.metadata_columns:
            print(f"Metadata columns: {self.metadata_columns}")
        else:
            print("Warning: No metadata assigned.")
        
        for idx, row in df.iterrows():
            content = str(row[self.content_column])
            chunks = self.chunker.chunk(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata = self._create_metadata(row, idx, chunk_idx, len(chunks), chunking_method)
                all_metadata.append(metadata)
            
            # Progress indicator
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(df)} records...")
        
        print(f"\nGenerated {len(all_chunks)} text chunks total")
        return all_chunks, all_metadata
    
    def _create_metadata(
        self, 
        row: pd.Series, 
        row_idx: int, 
        chunk_idx: int, 
        total_chunks: int,
        chunking_method: str
    ) -> Dict:
        """Create metadata dictionary for a chunk"""
        metadata = {
            "chunk_index": chunk_idx,
            "total_chunks": total_chunks,
            "source_row": int(row_idx),
            "chunking_method": chunking_method
        }
        
        # Add ID if specified
        if self.id_column and self.id_column in row:
            metadata["source_id"] = str(row[self.id_column]) if pd.notna(row[self.id_column]) else ""
        
        # Add other metadata columns
        for col in self.metadata_columns:
            if col in row:
                value = row[col]
                if pd.isna(value):
                    metadata[col] = ""
                elif isinstance(value, (str, int, float, bool)):
                    metadata[col] = value
                else:
                    metadata[col] = str(value)
        
        return metadata


class ChromaDBManager:
    """Manage ChromaDB operations"""
    
    def __init__(self, db_path: str = "./chroma_db"):
        """
        Args:
            db_path: Path to store ChromaDB files
        """
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
    
    def get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """Get existing collection or create new one"""
        try:
            collection = self.client.get_collection(name=collection_name)
            print(f"Database exists with {collection.count()} chunks")
            print("Skipping data insertion")
        except Exception:
            print("Creating new database...")
            collection = self.client.create_collection(name=collection_name)
        
        return collection
    
    def add_chunks(
        self,
        collection: chromadb.Collection,
        chunks: List[str],
        metadata: List[Dict],
        batch_size: int = 5000
    ) -> None:
        """Add chunks to collection in batches"""
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


class RAGPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(
        self,
        processor: DatasetProcessor,
        db_manager: ChromaDBManager,
        data_loader: callable
    ):
        """
        Args:
            processor: DatasetProcessor instance
            db_manager: ChromaDBManager instance
            data_loader: Function to load data (e.g., read_data)
        """
        self.processor = processor
        self.db_manager = db_manager
        self.data_loader = data_loader
    
    def debug_mode(self, datapath: str, sep: str = '\t', num_samples: int = 5):
        """Run in debug mode to inspect chunks"""
        print("\n=== DEBUGGING MODE ===")
        df = self.data_loader(datapath=datapath, sep=sep)
        all_chunks, all_metadata = self.processor.process(df)
        
        print(f"\n--- Sample Chunks (first {num_samples}) ---")
        for idx, (chunk, meta) in enumerate(zip(all_chunks, all_metadata)):
            if idx >= num_samples:
                break
            print(f"\n[Chunk {idx}]")
            print(f"Text: {chunk[:100]}...")
            print(f"Metadata: {meta}")
    
    def create_database(
        self,
        datapath: str,
        collection_name: str,
        sep: str = '\t',
        batch_size: int = 5000
    ):
        """Create and populate ChromaDB"""
        print("\n=== CREATING DATABASE ===")
        df = self.data_loader(datapath=datapath, sep=sep)
        all_chunks, all_metadata = self.processor.process(df)
        
        collection = self.db_manager.get_or_create_collection(collection_name)
        
        if collection.count() > 0:
            print("Database already populated, skipping insertion")
            return
        
        self.db_manager.add_chunks(collection, all_chunks, all_metadata, batch_size)


# Example usage
if __name__ == "__main__":
    from util import read_data, download_data
    
    load_dotenv()
    groq_apikey = os.getenv("GROQ_API_KEY")
    
    # Choose chunking strategy
    chunker = FixedSizeChunker(chunk_size=500, overlap=50)
    # chunker = SemanticChunker(
    #     target_chunk_size=500,
    #     api_key=groq_apikey,
    #     model="llama-3.1-8b-instant",
    #     rate_limit_delay=3.0
    # )
    
    # Create processor
    processor = DatasetProcessor(
        chunker=chunker,
        content_column="content",
        id_column=None,
        metadata_columns=['category', 'filename', 'title']
    )
    
    # Create DB manager
    db_manager = ChromaDBManager(db_path="./chroma_db")
    
    # Create pipeline
    pipeline = RAGPipeline(
        processor=processor,
        db_manager=db_manager,
        data_loader=read_data
    )
    
    # Run in debug mode
    pipeline.debug_mode(
        datapath="bbc-news.csv",
        sep='\t',
        num_samples=5
    )
    
    # Or create database
    # pipeline.create_database(
    #     datapath="bbc-news.csv",
    #     collection_name="bbc_news_semantic",
    #     sep='\t',
    #     batch_size=5000
    # )