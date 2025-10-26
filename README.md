# Federated Medical Literature Q&A System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flower](https://img.shields.io/badge/Flower-1.11.1-green.svg)](https://flower.ai/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A privacy-preserving federated learning system for medical literature question answering using Flower, FedRAG, LlamaIndex, and LangChain.

## 🎯 Overview

This project implements a **Federated Retrieval-Augmented Generation (RAG)** system that enables multiple healthcare institutions to collaboratively answer medical queries without sharing their private data. Each institution maintains its own local knowledge base while contributing to a unified, intelligent medical assistant.

### Key Features

- ✅ **Privacy-Preserving**: Data never leaves the local institution
- ✅ **HIPAA-Compliant**: Designed for healthcare data regulations
- ✅ **Scalable**: Supports multiple federated clients
- ✅ **Real-time**: Fast query processing with distributed retrieval
- ✅ **Accurate**: Uses medical-specific embeddings (BioBERT) and LLMs
- ✅ **Production-Ready**: Docker deployment with monitoring

## 🏗️ Architecture

```
┌─────────────┐
│   User UI   │
└──────┬──────┘
       │
┌──────▼───────────┐
│  LangChain       │  Query Processing
│  Orchestrator    │
└──────┬───────────┘
       │
┌──────▼───────────┐
│  Flower Server   │  Federated Coordination
│  + LLM Engine    │
└──┬───┬────┬──────┘
   │   │    │
┌──▼─┐┌▼──┐┌▼────┐
│C1  ││C2 ││C3   │  Local Knowledge Bases
│ ⚕️ ││🏥  ││🔬   │  + Retrieval Engines
└────┘└───┘└─────┘
```

## 📊 Use Case

**Scenario**: Three healthcare institutions want to build a collaborative medical Q&A system:
- **Hospital A**: Oncology research papers
- **Hospital B**: Clinical treatment guidelines  
- **Research Center**: Drug trial data

**Query**: "What are the side effects of Metformin in elderly patients?"

**Result**: System retrieves relevant information from all three institutions and generates a comprehensive answer with source citations, all while keeping each institution's data private.

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (for containerized deployment)
- 8GB+ RAM recommended
- NVIDIA GPU (optional, for faster embedding generation)

### Installation

#### Option 1: Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/federated-medical-rag.git
cd federated-medical-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

#### Option 2: Docker (Recommended for Production)

```bash
# Clone the repository
git clone https://github.com/yourusername/federated-medical-rag.git
cd federated-medical-rag

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### Dataset Setup

```bash
# Download and prepare PubMedQA dataset
python main.py --setup

# This will:
# 1. Download PubMedQA from HuggingFace
# 2. Split into 3 client datasets
# 3. Generate embeddings
# 4. Index in Qdrant vector stores
```

### Running Queries

#### Command Line

```bash
# Single query
python main.py --query "What are the side effects of Metformin?"

# Interactive mode
python main.py --interactive

# Batch queries from file
python main.py --batch queries.txt
```

#### Web Interface

```bash
# Start Gradio UI
python app.py

# Open browser to http://localhost:7860
```

#### API

```bash
# Start FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000

# Query via curl
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the side effects of Metformin?"}'
```

## 📁 Project Structure

```
federated-medical-rag/
├── data/                       # Data directory
│   ├── pubmedqa/              # Raw PubMedQA dataset
│   ├── hospital_a/            # Client 1 data
│   ├── hospital_b/            # Client 2 data
│   └── research_center/       # Client 3 data
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── data_loader.py         # Dataset loading utilities
│   ├── preprocessing.py       # Text preprocessing
│   ├── embeddings.py          # Embedding models (BioBERT)
│   ├── vector_store.py        # Qdrant operations
│   ├── retrieval.py           # Retrieval logic
│   ├── client.py              # Flower client implementation
│   ├── server.py              # Flower server implementation
│   └── llm_generator.py       # LLM answer generation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_rag.ipynb
│   └── 03_federated_rag.ipynb
├── tests/
│   ├── test_retrieval.py
│   ├── test_embeddings.py
│   └── test_aggregation.py
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboards/
├── Dockerfile.server          # Server container
├── Dockerfile.client          # Client container
├── Dockerfile.api             # API container
├── docker-compose.yml         # Multi-container orchestration
├── requirements.txt           # Python dependencies
├── main.py                    # Main execution script
├── api.py                     # FastAPI application
├── app.py                     # Gradio web interface
└── README.md                  # This file
```

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Model Configuration
embedding_model = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
llm_model = "gpt-4"  # or "gpt-3.5-turbo", "claude-3-opus", etc.

# Retrieval Configuration
top_k = 10                    # Documents per client
similarity_threshold = 0.7    # Minimum similarity score
chunk_size = 512              # Tokens per chunk
chunk_overlap = 50            # Overlap between chunks

# Federated Configuration
num_clients = 3               # Number of federated clients
k_rrf = 60                    # Reciprocal Rank Fusion parameter

# Vector Store Configuration
vector_store_type = "qdrant"
host = "localhost"
port = 6333
```

## 📊 Available Datasets

### Included
- **PubMedQA**: 1K expert-annotated + 211K generated medical Q&A pairs

### Compatible (can be added)
- **BioASQ**: Biomedical semantic Q&A
- **MIMIC-IV**: Clinical notes (requires credentialing)
- **PubMed Abstracts**: 36M+ biomedical articles
- **StatPearls**: Medical textbooks
- **Clinical Practice Guidelines**

### Adding Custom Datasets

```python
# In src/data_loader.py
class CustomDataLoader:
    def load_custom_data(self, path: str):
        # Load your documents
        documents = []
        # Process and return
        return documents

# Use in client
client = MedicalRAGClient(
    client_id=0,
    data_path="./data/custom",
    loader=CustomDataLoader()
)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_retrieval.py

# Run with coverage
pytest --cov=src tests/

# Run integration tests
pytest tests/integration/
```

## 📈 Monitoring & Evaluation

### Access Monitoring Dashboards

```bash
# Prometheus (metrics)
http://localhost:9090

# Grafana (visualization)
http://localhost:3000
# Username: admin
# Password: admin
```

### Evaluation Metrics

The system tracks:

**Retrieval Metrics**:
- Precision@k
- Recall@k  
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)

**Generation Metrics**:
- Faithfulness (answer grounded in sources)
- Relevance (answers the question)
- Citation accuracy

**System Metrics**:
- Query latency
- Throughput (queries/sec)
- Client contribution balance

Run evaluation:
```bash
python evaluate.py --test-set data/test_queries.json
```

## 🔒 Security & Privacy

### Data Protection
- ✅ All data encrypted at rest (AES-256)
- ✅ TLS 1.3 for client-server communication
- ✅ mTLS available for production deployment
- ✅ PHI anonymization before indexing
- ✅ Audit logging for all queries

### Compliance
- ✅ HIPAA-ready architecture
- ✅ GDPR-compliant (data minimization, right to deletion)
- ✅ Configurable data retention policies

### Setup Production Security

```bash
# Generate TLS certificates
./scripts/generate_certs.sh

# Configure mTLS
docker-compose -f docker-compose.prod.yml up -d

# Enable audit logging
export ENABLE_AUDIT_LOG=true
```

## 🚀 Deployment

### Development
```bash
docker-compose up -d
```

### Production

#### AWS
```bash
# Using ECS Fargate
./scripts/deploy_aws.sh

# Or Kubernetes (EKS)
kubectl apply -f k8s/
```

#### Azure
```bash
# Using Azure Container Instances
./scripts/deploy_azure.sh

# Or AKS
az aks create --name fedrag-cluster --resource-group fedrag-rg
```

#### On-Premises
```bash
# Using Kubernetes
helm install fedrag ./helm/fedrag-chart
```

## 🔄 Advanced Features

### Multi-Turn Conversations

```python
from src.conversation import ConversationManager

manager = ConversationManager()
conversation_id = manager.start_conversation()

# Ask follow-up questions
response1 = manager.query(conversation_id, "What is diabetes?")
response2 = manager.query(conversation_id, "What are the treatments?")
```

### Query Expansion

```python
from src.query_expansion import MedicalQueryExpander

expander = MedicalQueryExpander()
expanded = expander.expand("diabetes treatment")
# Returns: ["diabetes treatment", "diabetes mellitus therapy", 
#           "diabetic care", "glycemic control"]
```

### Custom Reranking

```python
from src.reranking import CrossEncoderReranker

reranker = CrossEncoderReranker(
    model="cross-encoder/ms-marco-MedMiniLM-L-6-v2"
)
reranked_docs = reranker.rerank(query, documents, top_k=5)
```

### Fine-tuning with FedRAG

```python
from fed_rag import FedRAGTrainer

trainer = FedRAGTrainer(
    clients=clients,
    train_data=train_queries
)
trainer.federated_fine_tune(epochs=3)
```

## 📚 Documentation

- [Architecture Deep Dive](docs/architecture.md)
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Fork the repo
git checkout -b feature/your-feature
git commit -am 'Add new feature'
git push origin feature/your-feature
# Create Pull Request
```

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{federated_medical_rag_2025,
  title = {Federated Medical Literature Q&A System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/federated-medical-rag}
}
```

Also cite the underlying frameworks:

```bibtex
@article{beutel2020flower,
  title={Flower: A Friendly Federated Learning Framework},
  author={Beutel, Daniel J and Topal, Taner and others},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}

@software{Fajardo_fed-rag_2025,
  author = {Fajardo, Andrei and Emerson, David},
  title = {{fed-rag}},
  year = {2025},
  url = {https://github.com/VectorInstitute/fed-rag}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Flower](https://flower.ai/) - Federated learning framework
- [Vector Institute](https://vectorinstitute.ai/) - FedRAG framework
- [LlamaIndex](https://www.llamaindex.ai/) - RAG framework
- [HuggingFace](https://huggingface.co/) - Model hub and datasets
- [PubMedQA](https://pubmedqa.github.io/) - Dataset

## 📧 Contact

- Project Lead: [Your Name](mailto:your.email@example.com)
- Issues: [GitHub Issues](https://github.com/yourusername/federated-medical-rag/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/federated-medical-rag/discussions)

## 🗺️ Roadmap

### Q1 2025
- ✅ Core federated RAG implementation
- ✅ PubMedQA integration
- ✅ Docker deployment

### Q2 2025
- ⬜ Multi-modal support (medical images)
- ⬜ Advanced reranking strategies
- ⬜ Mobile app (iOS/Android)

### Q3 2025
- ⬜ Real-time continuous learning
- ⬜ Blockchain-based audit trails
- ⬜ Edge device deployment

### Q4 2025
- ⬜ Clinical trial integration
- ⬜ EHR system connectors
- ⬜ Regulatory approval (FDA, CE)

---

**Built with ❤️ for healthcare professionals worldwide**
