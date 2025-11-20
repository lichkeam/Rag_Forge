# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Import your existing classes
from chroma_data_pipeline_dev_v2 import (
    FixedSizeChunker,
    SemanticChunker,
    DatasetProcessor,
    ChromaDBManager,
    RAGPipeline,
)
from util import read_data

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PORT = int(os.getenv("PORT", 8080))

app = FastAPI(
    title="RAG Pipeline API",
    description="BBC News RAG system with ChromaDB",
    version="1.0.0",
)

# Global variables - initialized at startup
db_manager = None
collection = None
chunker = None

# === Pydantic Models ===


class ChunkerConfig(BaseModel):
    type: str = "fixed"  # "fixed" or "semantic"
    chunk_size: int = 500
    overlap: int = 50
    rate_limit_delay: float = 3.0


class CreateDBRequest(BaseModel):
    datapath: str = "bbc-news.csv"
    collection_name: str = "bbc_news"
    sep: str = "\t"
    chunker_config: ChunkerConfig = ChunkerConfig()
    batch_size: int = 5000


class QueryRequest(BaseModel):
    query: str
    n_results: int = 3


class ChunkTextRequest(BaseModel):
    text: str
    chunker_config: ChunkerConfig = ChunkerConfig()


# === Startup Event ===


@app.on_event("startup")
async def startup_event():
    """Initialize ChromaDB connection"""
    global db_manager, collection

    print("Initializing ChromaDB manager...")
    db_manager = ChromaDBManager(db_path="./chroma_db")

    try:
        # Try to load existing collection
        collection = db_manager.client.get_collection(name="bbc_news")
        print(f"Loaded existing collection with {collection.count()} chunks")
    except BaseException:
        print("No existing collection found. Use /create_database to build one.")
        collection = None


# === API Endpoints ===


@app.get("/")
def root():
    """Health check endpoint"""
    status = {
        "status": "running",
        "message": "RAG Pipeline API is ready",
        "chromadb_initialized": db_manager is not None,
        "collection_loaded": collection is not None,
    }

    if collection:
        status["total_chunks"] = collection.count()

    return status


@app.get("/stats")
def get_stats():
    """Get database statistics"""
    if not collection:
        raise HTTPException(status_code=404, detail="No collection loaded")

    return {
        "collection_name": collection.name,
        "total_chunks": collection.count(),
        "db_path": db_manager.db_path,
    }


@app.post("/chunk_text")
def chunk_text(request: ChunkTextRequest):
    """Test text chunking functionality"""
    try:
        # Create chunker based on config
        if request.chunker_config.type == "semantic":
            chunker = SemanticChunker(
                target_chunk_size=request.chunker_config.chunk_size,
                api_key=GROQ_API_KEY,
                rate_limit_delay=request.chunker_config.rate_limit_delay,
            )
        else:
            chunker = FixedSizeChunker(
                chunk_size=request.chunker_config.chunk_size,
                overlap=request.chunker_config.overlap,
            )

        chunks = chunker.chunk(request.text)

        return {
            "chunker_type": request.chunker_config.type,
            "num_chunks": len(chunks),
            "chunks": chunks,
            "avg_chunk_size": sum(len(c) for c in chunks) // len(chunks)
            if chunks
            else 0,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create_database")
def create_database(request: CreateDBRequest):
    """Create or rebuild ChromaDB"""
    global collection

    try:
        print(f"\n=== Creating database: {request.collection_name} ===")

        # Create chunker
        if request.chunker_config.type == "semantic":
            chunker = SemanticChunker(
                target_chunk_size=request.chunker_config.chunk_size,
                api_key=GROQ_API_KEY,
                rate_limit_delay=request.chunker_config.rate_limit_delay,
            )
        else:
            chunker = FixedSizeChunker(
                chunk_size=request.chunker_config.chunk_size,
                overlap=request.chunker_config.overlap,
            )

        # Create processor
        processor = DatasetProcessor(
            chunker=chunker,
            content_column="content",
            id_column=None,
            metadata_columns=["category", "filename", "title"],
        )

        # Create pipeline
        pipeline = RAGPipeline(
            processor=processor, db_manager=db_manager, data_loader=read_data
        )

        # Execute creation
        pipeline.create_database(
            datapath=request.datapath,
            collection_name=request.collection_name,
            sep=request.sep,
            batch_size=request.batch_size,
        )

        # Reload collection
        collection = db_manager.client.get_collection(name=request.collection_name)

        return {
            "status": "success",
            "collection_name": request.collection_name,
            "total_chunks": collection.count(),
            "chunker_type": request.chunker_config.type,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error creating database: {str(e)}"
        )


@app.post("/query")
def query_rag(request: QueryRequest):
    """Query the RAG system"""
    if not collection:
        raise HTTPException(
            status_code=404, detail="No collection loaded. Use /create_database first."
        )

    try:
        # Use ChromaDB's native query functionality
        results = collection.query(
            query_texts=[request.query], n_results=request.n_results
        )

        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append(
                {
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                    if results["metadatas"]
                    else {},
                    "distance": results["distances"][0][i]
                    if results["distances"]
                    else None,
                }
            )

        return {
            "query": request.query,
            "n_results": len(formatted_results),
            "results": formatted_results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collection/{collection_name}")
def delete_collection(collection_name: str):
    """Delete a collection"""
    global collection

    try:
        db_manager.client.delete_collection(name=collection_name)
        collection = None
        return {
            "status": "success",
            "message": f"Collection '{collection_name}' deleted",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections")
def list_collections():
    """List all collections"""
    try:
        collections = db_manager.client.list_collections()
        return {
            "collections": [col.name for col in collections],
            "count": len(collections),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Start Server ===


if __name__ == "__main__":
    import uvicorn

    print(f"Starting server on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
