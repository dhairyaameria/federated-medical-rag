# Notebooks Guide

## Available Notebooks

### 01_data_exploration.ipynb
Interactive exploration of the PubMedQA dataset including:
- Data loading and visualization
- Statistical analysis of questions and answers
- Medical concept extraction
- Quality checks and data distribution

**To use**: Open in Jupyter and run cells sequentially.

### 02_baseline_rag.ipynb
Implementation of a single-node RAG system as a baseline:
- Basic retrieval setup
- Performance metrics
- Comparison benchmark
- Evaluation on test queries

**To use**: Ensure Qdrant is running and `.env` is configured.

### 03_federated_rag.ipynb
Demonstration of the federated RAG architecture:
- Multi-client retrieval
- Reciprocal Rank Fusion visualization
- Privacy-preserving comparisons
- Federated vs single-node analysis

**To use**: Requires all clients to be set up and running.

## Running Notebooks

```bash
# Start Jupyter
source venv/bin/activate
jupyter notebook

# Or use JupyterLab
jupyter lab
```

## Creating New Notebooks

To create these notebooks interactively:

1. Start Jupyter: `jupyter notebook`
2. Create new notebooks with appropriate names
3. Use the helper files in `src/` for imports
4. Follow the structure in `FILE_PURPOSES.md`

## Quick Start

The system has already been set up! You can:
1. Run `python main.py --setup` (already done)
2. Run `python main.py --query "your question"` 
3. Start exploring with notebooks

See `FILE_PURPOSES.md` for detailed explanations of each file.


