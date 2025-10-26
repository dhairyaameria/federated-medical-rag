# Federated Medical RAG - Starter Code Template

## Installation Guide

### Prerequisites
```bash
# Python 3.9+
python --version

# Create virtual environment
python -m venv fedrag_env
source fedrag_env/bin/activate  # On Windows: fedrag_env\Scripts\activate
```

### Install Dependencies
```bash
# Core dependencies
pip install flwr[simulation]==1.11.1
pip install fed-rag[qdrant,llamaindex]==0.0.27
pip install torch torchvision torchaudio
pip install transformers sentence-transformers
pip install llama-index llama-index-vector-stores-qdrant
pip install langchain langchain-openai langchain-community
pip install qdrant-client
pip install datasets
pip install openai anthropic

# For development
pip install jupyter notebook
pip install matplotlib seaborn
pip install pytest black flake8
```

### Download PubMedQA Dataset
```python
from datasets import load_dataset

# Load PubMedQA dataset
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

# Save locally
dataset.save_to_disk("./data/pubmedqa")
```

---

## Project Structure

```
federated-medical-rag/
├── data/
│   ├── pubmedqa/              # Downloaded dataset
│   ├── hospital_a/            # Client 1 data
│   ├── hospital_b/            # Client 2 data
│   └── research_center/       # Client 3 data
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessing.py       # Text preprocessing
│   ├── embeddings.py          # Embedding models
│   ├── vector_store.py        # Vector database operations
│   ├── retrieval.py           # Retrieval logic
│   ├── server.py              # Flower server
│   ├── client.py              # Flower client
│   └── llm_generator.py       # LLM response generation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_rag.ipynb
│   └── 03_federated_rag.ipynb
├── tests/
│   └── test_retrieval.py
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Usage

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup clients (split and index data)
python main.py --setup

# Start Qdrant (in separate terminal)
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Run Queries
```bash
# Set API key
export OPENAI_API_KEY="your-key-here"

# Run single query
python main.py --query "What are the side effects of Metformin?"

# Run demo queries
python main.py
```

### 3. Docker Deployment
```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f
```

---

## Next Steps

1. **Enhance Retrieval**: Add reranking, hybrid search (BM25 + Dense)
2. **Add Authentication**: Implement mTLS between clients
3. **Monitoring**: Add Prometheus metrics and Grafana dashboards
4. **Web Interface**: Build Gradio/Streamlit UI
5. **Fine-tuning**: Use FedRAG to fine-tune retrieval models
6. **Multi-modal**: Add support for medical images and tables

---

## Resources

- [Flower Documentation](https://flower.ai/docs/)
- [FedRAG GitHub](https://github.com/VectorInstitute/fed-rag)
- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [PubMedQA Dataset](https://pubmedqa.github.io/)
- [BioBERT Paper](https://arxiv.org/abs/1901.08746)
