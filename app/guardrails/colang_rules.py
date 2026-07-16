# Colang Rules
# Store all Colang guardrail rules as a multi-line string
COLANG_CONTENT = """
define user ask off topic
  "tell me a joke"
  "what is the capital of france"
  "write me a poem"
  "what is 2 plus 2"
  "What should I eat for dinner"
  "Who won the game yesterday"
  "Recommend a movie"
  "What is the weather today"
  "Can you help me with math homework"
  "Tell me about world history"
  "What is the best restaurant near me"

define bot refuse off topic
    "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. I can't help with that - but ask me anything technical!"

# Create a flow that maps off-topic user input to the  refusal response
"""