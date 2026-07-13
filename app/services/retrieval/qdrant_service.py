import logfire

# Import Qdrant client to communicate with the Qdrant vector database
from qdrant_client import QdrantClient 

# Import Qdrant models (not used in this file, but useful for filters and advanced queries)
from qdrant_client.http import models

# Import application configuration (Qdrant URL, API key, collection name, etc.)
from app.config import settings

# Import function that converts a user query into a embedding vector
from app.services.retrieval.embedding import embed_query

# Initialize Qdrant Client
# Create a connection to the Qdrant vector database
client = QdrantClient(
    # URL of the Qdrant server
    url = settings.QDRANT_URL,

    # API key used for authentication
    api_key=settings.QDRANT_API_KEY
)

# Search Function
# Search the enterprise knowledge base using semantic similarity
def search_enterprise_knowledge(
        query: str,
        limit: int = 8,
):
    """
    Perfroms a high-precision semantic search
    in the enterprise knowledge base.
    """
    try:
        # Convert the user's query into an embedding vector
        query_vector = embed_query(query)

        # Search the Qdrant collection using the query vector
        response = client.query_points(
            # Name of the Qdrant collection to search
            collection_name = settings.QDRANT_COLLECTION,

            # Vector representation of the user's query
            query = query_vector,

            # Maximum number of matching documents to return 
            limit = limit,

            # Return payload (stored metadata) along with vectors
            with_payload = True
        )
        # Create an empty list to store formatted search results
        results = []

        # Iterate over every retrieved point
        for res in response.points:
            # Store only the required information
            results.append({
                # Retrieved document text
                "content": res.payload.get(
                    "text",
                    "",
                ),

                # Original filename of the retrieved document
                "source": res.payload.get(
                    "source",
                    "Unknown",
                ),

                # Similarity score assigned by Qdrant
                "score": res.score
            })
        # Return the formatted search results
        return results
    except Exception as e:
        # Log the error if the search fails
        logfire.error(
            f"❌ Qdrant Search Failed: {e}"
        )
        # Return an empty list instead of crashing the application
        return []