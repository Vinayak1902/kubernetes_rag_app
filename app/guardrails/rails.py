# from app.guardrails.rails import initialize_rails, guard 
import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS

_rails: LLMRails | None = None 

def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses llama-3.1-8b-instant for fast intent classification at the gate -
    the heavier llama-3.3-70b-versatile is reserved for the RAG pipeline.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT 
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("Nemo Guardrails initialised (llama-3.1-8b-instant).")

def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the Nemo rails gate.

    Returns:
        (True, rail_response) - a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)         - message is clean; proceed to LangGraph. 
    """
    if _rails is None:
        logfire.warning("[WARNING] Guardrails not initialized, skipping gate.")
        return False, None 
    with logfire.span("Guardrails Check"):
        result = _rails.generate(message=[{"role": "user", "content": message}])

        # Nemo returns {'role': 'assistant', 'content': '...'} - extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("Guardrails passed.")
        return False, None