import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from main import split_text_into_chunks, extract_text_from_pdf  # Import functions from main.py

def store_embeddings_in_qdrant(pdf_path: str, arxiv_id: str, collection_name: str = "arxiv_embeddings"):
    """
    Extracts text from the given PDF, splits it into chunks, converts them into embeddings,
    and stores the embeddings in the Qdrant vector database.
    """
    # Initialize Qdrant client (assuming Qdrant is running locally on default port 6333)
    client = QdrantClient("localhost", port=6333)
    
    # Ensure the collection exists
    if collection_name not in [col.name for col in client.get_collections().collections]:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Adjust size as per embedding model
        )
    
    # Extract and split text into chunks
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
    
    # Initialize FastEmbed model
    embedder = TextEmbedding()
    
    # Convert chunks into embeddings
    embeddings = list(embedder.embed(chunks))  # Convert generator to list
    
    # Store embeddings in Qdrant
    points = [
        PointStruct(id=i, vector=embeddings[i], payload={"text": chunks[i], "arxiv_id":arxiv_id})
        for i in range(len(chunks))
    ]
    
    client.upsert(collection_name=collection_name, points=points)
    print(f"Stored {len(points)} embeddings in Qdrant collection: {collection_name}")

# Example usage (assuming PDF path is passed dynamically)
if __name__ == "__main__":
    pdf_file = "2402.00159.pdf"  # Example PDF
    arxiv_id = "2402.00159"  # Example ArXiv ID
    store_embeddings_in_qdrant(pdf_file,arxiv_id)