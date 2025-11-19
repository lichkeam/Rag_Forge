import chromadb
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

# ============================================================================
# Custom Similarity Search
# ============================================================================

def compute_embedding(
    text: str,
    embedding_function: Any
) -> np.ndarray:
    """
    Compute embedding for a single text using ChromaDB's embedding function
    # Adopted LLM model: "all-MiniLM-L6-v2"
    Args:
        text: Input text
        embedding_function: ChromaDB embedding function
    
    Returns:
        np.ndarray: Embedding vector
    """
    embeddings = embedding_function([text])
    return np.array(embeddings[0])


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        float: Cosine similarity (-1 to 1, higher = more similar)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        float: Euclidean distance (lower = more similar)
    """
    return np.linalg.norm(vec1 - vec2)


def custom_similarity_search(
    query: str,
    collection: chromadb.Collection,
    n_results: int = 5,
    metric: str = "cosine",
    debug: bool = False
) -> Dict[str, Any]:
    """
    Custom similarity search implementation to verify against ChromaDB's built-in query
    
    Args:
        query: Query text
        collection: ChromaDB collection
        n_results: Number of top results to return
        metric: Distance metric ("cosine" or "euclidean")
        debug: Whether to print debug information
    
    Returns:
        dict: Search results in same format as ChromaDB query
            {
                'documents': [[doc1, doc2, ...]],
                'metadatas': [[meta1, meta2, ...]],
                'distances': [[dist1, dist2, ...]],
                'ids': [[id1, id2, ...]]
            }
    """
    
    if debug:
        print(f"[DEBUG] Custom similarity search")
        print(f"  Metric: {metric}")
        print(f"  n_results: {n_results}")
    
    # Get embedding function from collection
    # all-MiniLM-L6-v2
    embedding_function = collection._embedding_function
    
    # Step 1: Compute query embedding
    if debug:
        print(f"[DEBUG] Computing query embedding...")
    
    query_embedding = compute_embedding(query, embedding_function)
    
    if debug:
        print(f"  Query embedding shape: {query_embedding.shape}")
        print(f"  Query embedding preview: {query_embedding[:8]}")
    
    # Step 2: Get all documents from collection
    if debug:
        print(f"[DEBUG] Retrieving all documents from collection...")
    
    all_data = collection.get(
        include=['embeddings', 'documents', 'metadatas']
    )
    
    total_docs = len(all_data['ids'])
    
    if debug:
        print(f"  Total documents in collection: {total_docs}")
    
    if total_docs == 0:
        return {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]],
            'ids': [[]]
        }
    
    # Step 3: Compute similarities for all documents
    if debug:
        print(f"[DEBUG] Computing similarities...")
    
    similarities = []
    
    for i, doc_embedding in enumerate(all_data['embeddings']):
        doc_embedding_array = np.array(doc_embedding)
        
        if metric == "cosine":
            # ChromaDB uses cosine distance = 1 - cosine_similarity, but the 'distance' output is 2*(1 - cos_similarity)
            # So we compute cosine similarity and convert to distance
            sim = cosine_similarity(query_embedding, doc_embedding_array)
            # For normalized vectors: ||norm(a) - norm(b)||² = 2 - 2*dot(norm(a), norm(b)) = 2*(1 - cos_sim). Note: ||norm(a)|| = 1
            distance = 2*(1 - sim)  # Convert similarity to distance
        
        elif metric == "euclidean":
            distance = euclidean_distance(query_embedding, doc_embedding_array)
        
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'")
        
        similarities.append({
            'distance': distance,
            'index': i
        })
    
    # Step 4: Sort by distance (lower = more similar)
    similarities.sort(key=lambda x: x['distance'])
    #sort() Small to Large
    
    # Step 5: Get top-k results
    top_k = similarities[:n_results]
    
    if debug:
        print(f"[DEBUG] Top {n_results} results:")
        for rank, item in enumerate(top_k, 1):
            print(f"  Rank {rank}: distance={item['distance']:.4f}, index={item['index']}")
    
    # Step 6: Format results to match ChromaDB query format
    result_docs = []
    result_metas = []
    result_distances = []
    result_ids = []
    
    for item in top_k:
        idx = item['index']
        result_docs.append(all_data['documents'][idx])
        result_metas.append(all_data['metadatas'][idx])
        result_distances.append(item['distance'])
        result_ids.append(all_data['ids'][idx])
    
    return {
        'documents': [result_docs],
        'metadatas': [result_metas],
        'distances': [result_distances],
        'ids': [result_ids]
    }


def compare_search_results(
    builtin_results: Dict[str, Any],
    custom_results: Dict[str, Any],
    top_n: int = 5
) -> None:
    """
    Compare built-in ChromaDB query results with custom similarity search results
    
    Args:
        builtin_results: Results from collection.query()
        custom_results: Results from custom_similarity_search()
        top_n: Number of top results to compare
    """
    
    print("\n" + "=" * 80)
    print("COMPARISON: Built-in vs Custom Similarity Search")
    print("=" * 80)
    
    builtin_ids = builtin_results['ids'][0][:top_n]
    custom_ids = custom_results['ids'][0][:top_n]
    
    builtin_distances = builtin_results['distances'][0][:top_n]
    custom_distances = custom_results['distances'][0][:top_n]
    
    print(f"\nTop {top_n} Results Comparison:\n")
    print(f"{'Rank':<6} {'Built-in ID':<20} {'Distance':<12} | {'Custom ID':<20} {'Distance':<12} {'Match':<8}")
    print("-" * 90)
    
    for i in range(min(top_n, len(builtin_ids), len(custom_ids))):
        builtin_id = builtin_ids[i]
        custom_id = custom_ids[i]
        builtin_dist = builtin_distances[i]
        custom_dist = custom_distances[i]
        
        # match = "V" if builtin_id == custom_id else "X"
        # dist_diff = abs(builtin_dist - custom_dist)
        
        # print(f"{i+1:<6} {builtin_id:<20} {builtin_dist:<12.6f} | {custom_id:<20} {custom_dist:<12.6f} {match:<8}")
        print(f"{i+1:<6} {builtin_id:<20} {builtin_dist:<12.6f} | {custom_id:<20} {custom_dist:<12.6f}")
        
        # if dist_diff > 1e-6:
        #     print(f"       Distance difference: {dist_diff:.6e}")
    
    # Calculate metrics
    # matching_ids = sum(1 for i in range(min(len(builtin_ids), len(custom_ids))) if builtin_ids[i] == custom_ids[i])
    # match_rate = matching_ids / min(len(builtin_ids), len(custom_ids)) * 100 if builtin_ids else 0
    
    # avg_dist_diff = np.mean([abs(builtin_distances[i] - custom_distances[i]) 
    #                           for i in range(min(len(builtin_distances), len(custom_distances)))])
    
    print("\n" + "-" * 90)
    # print(f"Match Rate: {matching_ids}/{min(len(builtin_ids), len(custom_ids))} ({match_rate:.1f}%)")
    # print(f"Match Rate: {matching_ids}/{min(len(builtin_ids), len(custom_ids))}")
    # print(f"Average Distance Difference: {avg_dist_diff:.6e}")
    
    # if match_rate == 100 and avg_dist_diff < 1e-6:
    #     print("Results are IDENTICAL!")
    # elif match_rate > 90:
    #     print("Results are VERY SIMILAR but not identical")
    # else:
    #     print("Results are DIFFERENT")
    
    # print("=" * 80 + "\n")


def debug_distance_calculation(
    query: str,
    collection: chromadb.Collection,
    doc_index: int = 0
) -> None:
    """
    Debug distance calculation step by step
    """
    print("\n" + "=" * 80)
    print("STEP-BY-STEP DISTANCE CALCULATION DEBUG")
    print("=" * 80)
    
    # Get embedding function
    embedding_function = collection._embedding_function
    
    # Get query embedding
    query_embedding = compute_embedding(query, embedding_function)
    
    # Get document embedding
    all_data = collection.get(include=['embeddings', 'documents', 'ids'])
    doc_embedding = np.array(all_data['embeddings'][doc_index])
    doc_id = all_data['ids'][doc_index]
    
    print(f"Query: {query[:50]}...")
    print(f"Document ID: {doc_id}")
    print(f"Document preview: {all_data['documents'][doc_index][:50]}...")
    
    # Print embedding info
    print(f"\n--- Embedding Info ---")
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Query embedding norm: {np.linalg.norm(query_embedding):.6f}")
    print(f"Doc embedding shape: {doc_embedding.shape}")
    print(f"Doc embedding norm: {np.linalg.norm(doc_embedding):.6f}")
    
    # Calculate step by step
    print(f"\n--- Step-by-Step Calculation ---")
    
    # Step 1: Dot product
    dot_prod = np.dot(query_embedding, doc_embedding)
    print(f"1. Dot product: {dot_prod:.6f}")
    
    # Step 2: Norms
    norm_q = np.linalg.norm(query_embedding)
    norm_d = np.linalg.norm(doc_embedding)
    print(f"2. Query norm: {norm_q:.6f}")
    print(f"3. Doc norm: {norm_d:.6f}")
    
    # Step 3: Cosine similarity
    cos_sim = dot_prod / (norm_q * norm_d)
    print(f"4. Cosine similarity: {cos_sim:.6f}")
    
    # Step 4: Different distance formulas
    print(f"\n--- Different Distance Formulas ---")
    dist_method_1 = 1 - cos_sim
    print(f"Method 1: 1 - cos_sim = {dist_method_1:.6f}")
    
    dist_method_2 = (1 - cos_sim) ** 2
    print(f"Method 2: (1 - cos_sim)² = {dist_method_2:.6f}")
    
    # Squared L2 distance
    squared_l2 = np.sum((query_embedding - doc_embedding) ** 2)
    print(f"Method 3: ||a - b||² = {squared_l2:.6f}")
    
    # L2 distance (not squared)
    l2_dist = np.linalg.norm(query_embedding - doc_embedding)
    print(f"Method 4: ||a - b|| = {l2_dist:.6f}")
    
    # Normalized vectors method
    query_norm_vec = query_embedding / norm_q
    doc_norm_vec = doc_embedding / norm_d
    norm_dot = np.dot(query_norm_vec, doc_norm_vec)
    dist_method_5 = 1 - norm_dot
    print(f"Method 5: 1 - dot(norm(a), norm(b)) = {dist_method_5:.6f}")
    
    # Method 6: Using the formula: ||a - b||² = ||a||² + ||b||² - 2*dot(a,b)
    # For normalized vectors: ||norm(a) - norm(b)||² = 2 - 2*dot(norm(a), norm(b)) = 2*(1 - cos_sim)
    dist_method_6 = 2 * (1 - cos_sim)
    print(f"Method 6: 2*(1 - cos_sim) = {dist_method_6:.6f}")
    
    # Get ChromaDB's actual distance
    print(f"\n--- ChromaDB Built-in Query ---")
    chromadb_result = collection.query(query_texts=[query], n_results=50)
    
    # Find the matching document
    try:
        matching_idx = chromadb_result['ids'][0].index(doc_id)
        chromadb_dist = chromadb_result['distances'][0][matching_idx]
        print(f"ChromaDB distance: {chromadb_dist:.6f}")
        
        print(f"\n--- Ratio Comparisons (Which equals 1.0?) ---")
        print(f"Method 1 / ChromaDB = {dist_method_1 / chromadb_dist:.6f}x")
        print(f"Method 2 / ChromaDB = {dist_method_2 / chromadb_dist:.6f}x")
        print(f"Method 3 / ChromaDB = {squared_l2 / chromadb_dist:.6f}x")
        print(f"Method 4 / ChromaDB = {l2_dist / chromadb_dist:.6f}x")
        print(f"Method 5 / ChromaDB = {dist_method_5 / chromadb_dist:.6f}x")
        print(f"Method 6 / ChromaDB = {dist_method_6 / chromadb_dist:.6f}x")
        
        # Find which one matches
        methods = [
            ("Method 1: 1 - cos_sim", dist_method_1),
            ("Method 2: (1 - cos_sim)²", dist_method_2),
            ("Method 3: ||a - b||²", squared_l2),
            ("Method 4: ||a - b||", l2_dist),
            ("Method 5: 1 - dot(norm)", dist_method_5),
            ("Method 6: 2*(1 - cos_sim)", dist_method_6),
        ]
        
        print(f"\n--- Match Results ---")
        for name, dist in methods:
            diff = abs(dist - chromadb_dist)
            if diff < 1e-5:
                print(f"✓ {name} MATCHES! (diff: {diff:.2e})")
            else:
                print(f"✗ {name} (diff: {diff:.6f})")
        
    except ValueError:
        print(f"Document {doc_id} not found in query results")
    
    print("=" * 80 + "\n")