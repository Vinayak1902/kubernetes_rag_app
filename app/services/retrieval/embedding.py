# import time module for adding delays during retries
import time

# import logfire for logging, tracing and monitoring
import logfire

# import Google's Gemini embedding model from LangChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import application settings (contains API keys and configuration)
from app.config import settings

# number of texts to embed in one batch
BATCH_SIZE = 50

# Dimension of Gemini embedding vectors
_GEMINI_DIM = 3072

# Dimension of fallback sentence-transformer embeddings
_FALLBACK_DIM = 768 # all-mpnet-base-v2

# stores the currently active embedding model (Gemini or fallback)
_active_model = None

# stores which model is active ("gemini" or "fallback") 
_model_type: str | None = None

# -----------------------------
# Model Initialization
# -----------------------------

# Try to initialize Gemini embedding model
def _probe_gemini():
    """
    Try one embed call to verify Gemini is reachable.
    Returns the model if successful, otherwise None.
    """
    try:
        # Create Gemini embedding model
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )

        # Perform a small test embedding to verify API connectivity
        model.embed_query("probe")

        # Log successful initialization 
        logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")

        # return the initialized model
        return model 

    except Exception as e:

        # Log warning if Gemini cannot be used
        logfire.warning(
            f"Gemini probe failed: {e}. Will use sentence-transformers fallback."
        )

        # Return None so fallback model can be loaded
        return None
    
# Load the local sentence-transformer model
def _load_fallback():

    # Import only when needed 
    from sentence_transformers import SentenceTransformer

    # Log loading of fallback model
    logfire.info(
        "Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim)."
    )

    # Load and return the pretrained model
    return SentenceTransformer("all-mpnet-base-v2")

# Initialize the embedding model only once
def _init():
    """
    Initialse embedding model once per process.
    Called lazily on first use.
    """

    # Access global variables
    global _active_model, _model_type

    # If already initialized, do nothing
    if _active_model is not None:
        return
    
    # Try using Gemini first
    gemini = _probe_gemini()

    if gemini:

        # Gemini is available
        _active_model = gemini
        _model_type = "gemini"

    else:

        # otherwise load local fallback model
        _active_model = _load_fallback()
        _model_type = "fallback"

# -------------------------------------
# Public Helper Function
# -------------------------------------

# Return embedding vector dimension
def get_embedding_dim() -> int:
    """
    Return the vector dimension for the active model.
    """

    # Ensure model is initialized
    _init()

    # Return appropriate vector dimension
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


# ------------------------------------
# Batch Embedding with Retry Logic
# ------------------------------------

# Generate embeddings for one batch of text
def _embed_batch(batch: list[str]) -> list[list[float]]:

    # If Gemini is active
    if _model_type == "gemini":

        # Retry up to 4 times using exponential backoff

        for attempt in range(4):

            try:
                # Generate embeddings for the batch
                return _active_model.embed_documents(batch)
            
            except Exception as e:

                # Convert exception to lowercase string
                err = str(e).lower()

                # Check if the error is related to rate limiting
                is_rate_limit = any(
                    x in err
                    for x in (
                        "429",
                        "rate",
                        "quota",
                        "resource_exhausted",
                    )
                )

                # Retry only if rate-limited and attempts remain
                if is_rate_limit and attempt < 3:

                    # Wait time doubles each retry(1,2,4,8 seconds)
                    wait = 2 ** attempt

                    # Log retry information
                    logfire.warning(
                        f"Gemini rate limit hit - retrying in {wait}s "
                        f"(attempt {attempt + 1}/4)."
                    )

                    # Pause before retrying 
                    time.sleep(wait)
                
                else:

                    # Log any non-recoverable error
                    logfire.error(f"Gemini embedding failed: {e}")

                    # Raise the exception
                    raise 
        
        # Raise error if all retries failed
        raise RuntimeError(
            "Gemini rate limit persisted after 4 attempts."
        )
    
    else:

        # Use local sentence-transformer model 
        return _active_model.encode(
            batch,
            show_progress_bar=False,
        ).tolist()

# -------------------------
# Public API 
# -------------------------

# Generate embedding for a single query 
def embed_query(query: str) -> list[float]:

    # Ensure model is initialized 
    _init()

    # Use Gemini if available
    if _model_type == "gemini":
        return _active_model.embed_query(query)
    
    # Otherwise use local model
    return _active_model.encode([query])[0].tolist()

# Generate embeddings for multiple texts
def embed_texts(texts: list[str]) -> list[list[float]]:

    # Ensure model is initialized 
    _init()

    # Store embeddings from all batches
    all_embeddings: list[list[float]] = []

    # Process texts in batches
    for i in range(0, len(texts), BATCH_SIZE):

        # Select current batch
        batch = texts[i: i+ BATCH_SIZE]

        # Create a tracing span for this embedding batch
        with logfire.span(
            "Embed batch",
            model=_model_type,
            start=i,
            size=len(batch),
        ):
            
            # Generate embeddings and append to final list
            all_embeddings.extend(_embed_batch(batch))

    # Return embeddings for all texts
    return all_embeddings

"""
embed_texts(texts)
        │
        ▼
      _init()
        │
        ▼
 Is model already loaded?
        │
   ┌────┴────┐
   │         │
  Yes        No
   │         │
   │     Try Gemini
   │         │
   │    ┌────┴────┐
   │    │         │
   │ Success    Failed
   │    │         │
   │    ▼         ▼
   │ Gemini   Load SentenceTransformer
   │
   ▼
Split texts into batches of 50
        │
        ▼
For each batch
        │
        ▼
Generate embeddings
        │
        ▼
If Gemini → Retry on rate limit
        │
        ▼
Collect all vectors
        │
        ▼
Return embeddings
"""