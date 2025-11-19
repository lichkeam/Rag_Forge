import chromadb


def list_chromadb_collections(db_path="./chroma_db"):
    """
    List all collections in a ChromaDB database and preview first 3 items.
    """
    try:
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()

        if not collections:
            print(f"No collections found in {db_path}")
            return

        print(f"Found {len(collections)} collection(s) in {db_path}:\n")

        for idx, collection in enumerate(collections, 1):
            print(f"{idx}. Collection: '{collection.name}'")
            count = collection.count()
            print(f"   Documents: {count}")

            if count > 0:
                sample = collection.peek(limit=1)
                if sample["metadatas"]:
                    print(f"   Metadata keys: {list(sample['metadatas'][0].keys())}")

                # --- Show 3 examples ---
                print("\n   Preview first 3 items:")
                preview = collection.peek(limit=3)

                for i in range(len(preview["ids"])):
                    print(f"     [{i+1}] id: {preview['ids'][i]}")

                    metadata = preview["metadatas"][i]
                    if metadata:
                        max_len = max(len(k) for k in metadata.keys())
                        for k, v in sorted(metadata.items()):
                            print(f"      {k.ljust(max_len)} : {v}")
                        # print(f"         metadata: {metadata}")

                    doc = preview["documents"][i]
                    if doc:
                        # short_doc = doc[:120] + "..." if len(doc) > 120 else doc
                        print(f"         document: {doc}")
                print()

    except Exception as e:
        print(f"Error accessing database: {e}")


if __name__ == "__main__":
    list_chromadb_collections(db_path="./chroma_db")
