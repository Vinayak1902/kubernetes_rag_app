import logfire
# portkey acts as a middleman between application and llm(groq), if some failure then it will handle
from portkey_ai import (
    Portkey,
    createHeaders,
    PORTKEY_GATEWAY_URL
)
from langchain_openai import ChatOpenAI
from app.config import settings

# how the portkey should behave
GATEWAY_CONFIG = {
    #strategy used when multiple models are available
    "strategy": {
        "mode": "fallback"
    },
    # configure response caching
    "cache": {
        "mode": "simple"
    },

    "retry": {
        "attempts": 2,
        "on_status_codes": [
            429,   # rate limit
            503    # service unavailable 
        ]
    },

    # List of models in fallback order
    "targets": [
        # Primary model
        {
            "override_params": {
                "model":
                f"@{settings.GROQ_SLUG}/llama-3.3-70b-versatile"
            }
        },
        # Backup model if the first one fails
        {
            "override_params": {
                "model":
                f"@{settings.GROQ_SLUG_2}/llama-3.1-8b-instant"
            }
        },
    ]
}

portkey_client = Portkey(
    api_key = settings.PORTKEY_API_KEY,
    config=GATEWAY_CONFIG
)

# returns a langchain-compatible LLM object
def get_langchain_llm(
        feature: str = "rag",
) -> ChatOpenAI:
    """
    Returns a ChatOpenAI object that actually sends requests through Portkey instead of OpenAI.
    Portkey handles: fallback, retry, cache, monitoring
    """

    # create and return LLM object
    return ChatOpenAI(
        api_key = settings.PORTKEY_API_KEY,

        # send requests to Portkey Gateway
        base_url = PORTKEY_GATEWAY_URL,

        # Primary model to use
        model = 
        f"@{settings.GROQ_SLUG}/llama-3.3-70b-versatile",

        # Make outputs deterministic
        temperature = 0,

        # Add portkey specific headers
        default_headers = createHeaders(
            api_key = settings.PORTKEY_API_KEY,
            config = GATEWAY_CONFIG,
            metadata = {
                # which feature is making request
                "feature": feature,
                # User identifier
                "_user": "rag-system",
                # environment information
                "environment": "production"
            }
        )
    )

# Extract Cache Status
# Read the cache status from the Portkey response headers
def extract_cache_status(
        response,
) -> str:
    """
    Returns whether the response came from cache.
    Possible values:
    HIT 
    MISS
    """
    # check multiple possible response objects
    for attr in (
        "_raw_response",
        "_response",
        "_http_response",
    ):
        # try to get the response object
        raw = getattr(
            response,
            attr,
            None,
        )

        # If the response object exists
        if raw is not None:
            # Read the cache-status header
            status = getattr(
                raw,
                "headers",
                {},
            ).get(
                "x-portkey-cache-status",
                "",
            )
            # If found, return it in uppercase
            if status:
                return status.upper()
    # if no cache header exists, assume the response was not cached
    return "MISS"
# 
# Application
#       │
#       ▼
# get_langchain_llm()
#       │
#       ▼
# ChatOpenAI
# (base_url = Portkey)
#       │
#       ▼
# Portkey Gateway
#       │
#       ├──────────────► Check Cache
#       │                     │
#       │               Cache HIT?
#       │               │        │
#       │              Yes      No
#       │               │        │
#       │               ▼        ▼
#       │          Return     Send Request
#       │                        │
#       ▼                        ▼
# Primary Groq Model      Retry (429/503)?
#       │                        │
#       │                  Retry 2 Times
#       │                        │
#       ▼                        ▼
# Success?──────────────►No──────────────►Fallback Groq Model
#       │                                    │
#       └────────────────Yes─────────────────┘
#                      │
#                      ▼
#              Return Response