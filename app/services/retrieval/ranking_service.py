import time
import logfire

# Ranker -> Loads the reraning model
# RerankRequest -> Represents the input request for reranking
from flashrank import Ranker, RerankRequest

# Lazy initialization - Ranker is loaded on first use to logfire.configure() has run
# Global variable to store the FlashRank model instance.
_ranker = None 

# Return the FlashRank model, creating it only on the first call.
def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily.
    FlashRank uses a local ONNX model (ms-macro-MiniLM-L-6-v2) for ultra-fast reranking.
    """
    global _ranker
    if _ranker is None:
        logfire.info("🧠 Initializing FlashRank Model (TinyBERT) locally...")
        try:
            # Load FlashRank using a custom cache directory.
            # We use a specific cache directory to avoid permission issues in production
            _ranker = Ranker(cache_dir="/tmp/flashrank")
        except Exception:
            # If custom cache directory fails,
            # load FlashRank using its default settings.
            _ranker = Ranker() 
    # Return the initialized model
    return _ranker 

# Rerank Retrieved Documents using cross-encoder model
def rerank_documents(
        query: str,
        documents: list[str],
        top_n: int = 5,
) -> list[str]:
    """
    Improve retrieval quality by re-ranking documents according to their semantic relevance to the user's query.
    """

    if not documents:
        return []
    start_time = time.time()
    logfire.info(
        f"[Reranker] Sending {len(documents)} docs"
        f"to FlashRank Cross-Encoder..."
    )
    try:
        ranker = _get_ranker()
        passages = [
            {
                "id": i,
                "text": doc,
            }
            for i, doc in enumerate(documents)
        ]
        request = RerankRequest(
            query = query,
            passages= passages,
        )
        results = ranker.rerank(request)
        reranked_docs = []
        for res in results[:top_n]:
            reranked_docs.append(
                res["text"]
            )
        duration = time.time() - start_time 
        top_score = (
            results[0]["score"]
            if results
            else "N/A"
        )
        logfire.info(
            f"[Successful] Reranking done in {duration:.2f}s. "
            f"Top semantic score: {top_score}"
        )
        return reranked_docs 
    except Exception as e:
        logfire.error(
            f"[Failure] Semantic Reranking Failed: {e}"
        )
        # if FlashRank fails, return the original Qdrant retrieval order so application still produces an answer
        return documents[:top_n]
