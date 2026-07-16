# RAG‑Kubernetes Demo

A minimal example project that demonstrates how to use the *rag‑kubernetes* repository to load models, build a graph, and run queries.  The source is deliberately small so you can copy it into another repo with minimal effort.

## Project Layout

```
├─ main.py          # Entry point that pulls everything together
├─ graph.py         # Utility that generates a toy dependency graph
├─ rails.py         # Small helper for loading RAG kernels
├─ colang_rules.py
├─ colang_rules.py
└─ README.md
```

## Prerequisites

- Python 3.11 or newer
- `pip install -r requirements.txt`  (the repo ships a minimal list)

## Running the Demo

```bash
# Start the demo
python main.py
```

The script will:
1. Load a synthetic embedding model from `rails.py`.
2. Construct a lightweight knowledge graph with `graph.py`.
3. Execute a few example queries printed to the console.

## Extending the Demo

Feel free to replace the toy graph in `graph.py` with a real knowledge base (e.g., a Neo4j DB).  The RAG kernel in `rails.py` is already set up to accept any vector store.

## Contributing

- Fork the repo.
- Make your changes in a feature branch.
- Run `pytest` to ensure quality.
- Open a PR when ready.

## License

MIT © 2026.  All rights reserved.
