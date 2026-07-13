# Import TypedDict for creating a dictionary with predefined keys and value types
from typing import TypedDict, List, Annotated

# Import the add operator (used to merge/apply addition to values)
import operator

# Define the structure of the shared state used by the agents
class AgentState(TypedDict):
    # Conversation history exchanged between agents
    # Annotated + operator.add tells LangGraph that when multiple nodes updated "messages", the new messages should be APPENDED to the existing list instead of replacing it.
    messages: Annotated[List[dict], operator.add]

    # Stores the current user query being processed
    current_query: str 

    # Stores the retrieved documents from the vector database
    documents: List[str]

    # Stotes the current workflow status
    # Example: "planning", "retrieving", "completed"
    status: str 

    # Stores the final answer generated for the user
    final_answer: str
