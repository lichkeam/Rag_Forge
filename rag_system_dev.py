"""
Simple RAG Query System - Simplified Version
A streamlined system for querying vector databases and generating answers

Filename: rag_system.py
"""

import sys
import io
import os
import chromadb
from dotenv import load_dotenv
import requests
from typing import Optional, Dict, Any
from Similarity_Search import custom_similarity_search, compare_search_results
from util import SimpleMetadataFormatter

# Set output encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# ============================================================================
# RAG Core Functions
# ============================================================================


def create_rag_prompt(
    query: str,
    search_results: Dict[str, Any],
    metadata_formatter: Optional[callable] = None,
    context_intro: str = "Here is the relevant content:",
    custom_instruction: Optional[str] = None,
    separator: str = "-" * 50,
    debug: bool = False,
) -> str:
    """
    Generate RAG prompt

    Args:
        query: User's query
        search_results: ChromaDB search results
        metadata_formatter: Metadata formatting function
        context_intro: Context introduction text
        custom_instruction: Custom instruction for LLM
        separator: Separator between documents
        debug: Whether to show debug information

    Returns:
        str: Complete prompt
    """

    # Check search results
    if not search_results or "documents" not in search_results:
        raise ValueError("Invalid search_results format: missing 'documents' field")

    if not search_results["documents"] or not search_results["documents"][0]:
        print("Warning: No documents found")
        return f"{context_intro}\n\n[No relevant data found]\n\nQuestion: {query}\n\nCannot answer this question."

    documents = search_results["documents"][0]
    metadatas = search_results["metadatas"][0]

    # Debug: Show search overview
    if debug:
        print("=" * 80)
        print("[DEBUG] Search Results Overview")
        print("=" * 80)
        print(f"Query: {query}")
        print(f"Documents found: {len(documents)}")

        if "distances" in search_results and search_results["distances"]:
            distances = search_results["distances"][0]
            print("\nSimilarity scores (lower = more similar):")
            for i, dist in enumerate(distances, 1):
                max(0, (1 - dist) * 100)
                print(f"  Doc {i}: {dist:.4f}")
            print()

    # Build context
    context = f"{context_intro}\n\n"

    for i, (doc, metadata) in enumerate(zip(documents, metadatas), 1):
        # Debug: Show document details
        if debug:
            print(f"[DEBUG] Document {i} Details")
            print("-" * 80)
            print("Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            print(f"Content length: {len(doc)} chars")
            # print(f"Content preview: {doc[:100]}...")
            print(f"Content preview: {doc}")
            print()

        # Build document section
        context += f"[Document {i}]\n"

        # Use formatter if provided
        if metadata_formatter:
            try:
                context += metadata_formatter(metadata) + "\n"
            except Exception as e:
                print(f"Warning: Formatter failed: {e}")
                # Fallback: show title if available
                context += f"Title: {metadata.get('Title', 'N/A')}\n"
        else:
            # Default: show all fields except internal ones
            exclude = {"chunk_index", "total_chunks", "source_row"}
            for key, value in metadata.items():
                if key not in exclude and value:
                    name = key.replace("_", " ").title()
                    context += f"{name}: {value}\n"

        context += f"Content: {doc}\n"
        context += separator + "\n\n"

    # Add instruction
    if custom_instruction is None:
        instruction = f"""###Instruction: Based on the content provided above, please answer the following question:
{query}

Please answer based only on the provided content. Make sure to reference each relevant document as [Document X] in your answer. If the content does not contain relevant information, say "The provided data does not contain relevant information"."""
    else:
        instruction = custom_instruction.format(query=query)

    prompt = f"{context}\n{instruction}"

    # Debug: Show prompt preview
    if debug:
        print("=" * 80)
        print("User Prompt:", query)
        print("=" * 80)
        print("[DEBUG] Generated Prompt Preview")
        print("=" * 80)
        print(f"Prompt length: {len(prompt)} chars")
        print("SHOW Prompt, Show first 700 digits")
        print("=" * 80)
        print(prompt[:700])
        print("\n omit intermediate part....")
        print(prompt[-100:])
        print("=" * 80)
        print("Prompt END")
        print()

    return prompt


def call_groq_llm(
    prompt: str,
    api_key: str,
    model: str = "llama-3.1-8b-instant",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    debug: bool = False,
) -> str:
    """
    Call Groq LLM API

    Args:
        prompt: Prompt to send
        api_key: Groq API key
        model: Model name
        max_tokens: Maximum output tokens
        temperature: Sampling temperature
        debug: Whether to show debug information

    Returns:
        str: LLM response

    Raises:
        Exception: If API call fails
    """

    # Debug: Show API call info
    if debug:
        print("=" * 80)
        print("[DEBUG] Calling LLM API")
        print("=" * 80)
        print(f"Model: {model}")
        print(f"Prompt length: {len(prompt)} chars")
        print(f"Estimated tokens: ~{len(prompt) // 4}")
        print()

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=data,
            headers=headers,
            timeout=30,
        )

        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]

            # Debug: Show response info
            if debug:
                usage = response.json().get("usage", {})
                print("[DEBUG] API Response Success")
                print("-" * 80)
                print(f"Response length: {len(answer)} chars")
                print(f"Token usage: {usage}")
                print(f"Response preview (first 200 chars):")
                print(answer[:200])
                if len(answer) > 200:
                    print("...")
                print("=" * 80)
                print()

            return answer

        else:
            # API error - raise exception
            error_msg = f"API Error {response.status_code}: {response.text}"
            raise Exception(error_msg)

    except requests.exceptions.Timeout:
        raise Exception("API request timeout (30 seconds)")

    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")


def rag_query(
    query: str,
    groq_api_key: str,
    collection: chromadb.Collection,
    n_results: int = 5,
    metadata_formatter: Optional[callable] = None,
    model: str = "llama-3.1-8b-instant",
    context_intro: str = "###Content:",
    custom_instruction: Optional[str] = None,
    use_custom_search: bool = False,
    similarity_metric: str = "cosine",
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Complete RAG query pipeline

    Args:
        query: User query
        groq_api_key: Groq API key
        collection: ChromaDB collection
        n_results: Number of documents to retrieve
        metadata_formatter: Metadata formatting function
        model: LLM model name
        context_intro: Context introduction text
        custom_instruction: Custom instruction for LLM
        use_custom_search: If True, use custom similarity search; if False, use ChromaDB built-in
        similarity_metric: Distance metric for custom search ("cosine" or "euclidean")
        debug: Whether to show debug information

    Returns:
        dict: {
            'success': bool,
            'answer': str or None,
            'search_results': dict or None,
            'error': str or None,
            'stats': dict or None,
            'search_method': str
        }
    """

    try:
        # Step 1: Retrieve documents
        # Customized search
        if use_custom_search:
            if debug:
                print(
                    f"[DEBUG] Using CUSTOM similarity search (metric={similarity_metric})..."
                )

            results = custom_similarity_search(
                query=query,
                collection=collection,
                n_results=n_results,
                metric=similarity_metric,
                debug=debug,
            )
            search_method = f"Custom ({similarity_metric})"

        # If not customized, Built-in search
        else:
            if debug:
                print(f"[DEBUG] Using BUILT-IN ChromaDB query...")

            results = collection.query(query_texts=[query], n_results=n_results)
            search_method = "Built-in"

        # Check if results found
        if not results["documents"][0]:
            return {
                "success": False,
                "answer": None,
                "search_results": results,
                "error": "No relevant documents found",
                "search_method": search_method,
            }

        # Step 2: Generate prompt
        if debug:
            print(f"[DEBUG] Generating prompt...")

        prompt = create_rag_prompt(
            query=query,
            search_results=results,
            metadata_formatter=metadata_formatter,
            context_intro=context_intro,
            custom_instruction=custom_instruction,
            separator="=" * 60,
            debug=debug,
        )

        # Step 3: Get LLM answer
        if debug:
            print(f"[DEBUG] Calling LLM...")

        answer = call_groq_llm(prompt, groq_api_key, model, debug=debug)

        return {
            "success": True,
            "answer": answer,
            "search_results": results,
            "error": None,
            "stats": {
                "documents_found": len(results["documents"][0]),
                "prompt_length": len(prompt),
                "answer_length": len(answer),
            },
            "search_method": search_method,
        }

    except Exception as e:
        import traceback

        error_msg = f"RAG query failed: {str(e)}"

        if debug:
            print(f"\n[DEBUG] Error details:")
            print(traceback.format_exc())

        return {
            "success": False,
            "answer": None,
            "search_results": None,
            "error": error_msg,
            "traceback": traceback.format_exc() if debug else None,
            "search_method": search_method
            if "search_method" in locals()
            else "Unknown",
        }


# ============================================================================
# Predefined Formatters
# ============================================================================


# # Default formatter (works with any dataset)
DEFAULT_FORMATTER = SimpleMetadataFormatter()

# ============================================================================
# Main Function
# ============================================================================


def main(
    # Database settings
    db_path="./chroma_db",
    collection_name="bbc_news",
    # Model settings
    model="llama-3.3-70b-versatile",
    # Query settings
    query="Give me some economy news",
    n_results=5,
    # Prompt settings
    context_intro="Here are the relevant news articles:",
    custom_instruction="Based on the news above, please answer: {query}",
    # Search settings
    similarity_metric="cosine",
    # Debug settings
    debug=True,
):
    """
    Main function - Tests built-in vs custom similarity search

    Args:
        db_path: Path to ChromaDB database
        collection_name: Name of collection to use
        model: LLM model ("llama-3.1-8b-instant" or "llama-3.3-70b-versatile")
        query: Query string
        n_results: Number of results to retrieve
        context_intro: Context introduction text
        custom_instruction: Custom instruction template
        similarity_metric: Distance metric ("cosine" or "euclidean")
        debug: Enable debug output
    """

    # Load environment variables
    load_dotenv()
    groq_apikey = os.getenv("GROQ_API_KEY")

    if not groq_apikey:
        print("Error: GROQ_API_KEY not found")
        print("Please ensure GROQ_API_KEY is set in .env file")
        return

    # Connect to database
    try:
        print("Connecting to vector database...")
        client = chromadb.PersistentClient(path=db_path)

        # List available collections
        collections = client.list_collections()
        print(f"Found {len(collections)} collection(s):")
        for col in collections:
            print(f"  - {col.name}")

        # Select collection
        collection = client.get_collection(name=collection_name)
        print(f"Using collection: {collection_name}")
        print(f"Document count: {collection.count()}")

    except Exception as e:
        print(f"Database connection failed: {e}")
        return

    print("\n" + "=" * 80)
    print("Starting RAG Query Comparison")
    print("=" * 80)
    print()

    # Setup query
    print(f"User query: {query}\n")

    # ========================================================================
    # Test 1: Built-in ChromaDB query
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Built-in ChromaDB Query")
    print("=" * 80)

    result_builtin = rag_query(
        query=query,
        groq_api_key=groq_apikey,
        collection=collection,
        n_results=n_results,
        model=model,
        metadata_formatter=DEFAULT_FORMATTER,
        context_intro=context_intro,
        custom_instruction=custom_instruction,
        use_custom_search=False,  # Use built-in
        debug=debug,
    )

    # ========================================================================
    # Test 2: Custom similarity search
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Custom Similarity Search")
    print("=" * 80)

    result_custom = rag_query(
        query=query,
        groq_api_key=groq_apikey,
        collection=collection,
        n_results=n_results,
        model=model,
        metadata_formatter=DEFAULT_FORMATTER,
        context_intro=context_intro,
        custom_instruction=custom_instruction,
        use_custom_search=True,  # Use custom
        similarity_metric=similarity_metric,
        debug=debug,
    )

    # ========================================================================
    # Compare Results
    # ========================================================================
    if result_builtin["success"] and result_custom["success"]:
        compare_search_results(
            builtin_results=result_builtin["search_results"],
            custom_results=result_custom["search_results"],
            top_n=n_results,
        )

    # ========================================================================
    # Display Final Answers
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL ANSWERS COMPARISON")
    print("=" * 80)

    print("\n--- Built-in Query Answer ---")
    if result_builtin["success"]:
        print(result_builtin["answer"])
    else:
        print(f"Failed: {result_builtin['error']}")

    print("\n--- Custom Search Answer ---")
    if result_custom["success"]:
        print(result_custom["answer"])
    else:
        print(f"Failed: {result_custom['error']}")

    print("\n" + "=" * 80)
    print("RAG Query Comparison Complete")
    print("=" * 80)


if __name__ == "__main__":
    # ===========================================================================
    # CONFIGURATION - Modify all parameters here
    # ===========================================================================
    questions = [
        "What are the challenges is European Commission facing?",
        "What were the main challenges facing the airline industry according to these articles?",
        "Compare the economic policies of different countries during 2004-2005.",
        "Summarize the term of computer technology in 2000 to 2020",
        "What common themes appear across multiple business articles about China?",
        "Show me news about the 2025 election in Japan.",
    ]

    main(
        # Database settings
        db_path="./chroma_db",
        collection_name="bbc_news",
        # Model settings
        model="llama-3.3-70b-versatile",  # or "llama-3.1-8b-instant"
        # Query settings
        query=questions[-1],
        n_results=5,
        # Prompt settings
        context_intro="Here are the relevant news articles:",
        custom_instruction=None,
        # Search settings
        similarity_metric="cosine",  # or "euclidean"
        # Debug settings
        debug=True,
    )
