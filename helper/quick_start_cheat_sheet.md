# Federated Medical RAG - Quick Start Cheat Sheet

## 🚀 SUPER QUICK START (15 minutes)

```bash
# 1. Clone and setup
git clone <your-repo>
cd federated-medical-rag
python -m venv venv && source venv/bin/activate

# 2. Install
pip install -r requirements.txt

# 3. Set API key
export OPENAI_API_KEY="sk-..."

# 4. Prepare data
python main.py --setup

# 5. Run your first query!
python main.py --query "What are the side effects of aspirin?"
```

---

## 📋 ESSENTIAL COMMANDS

### Data Setup
```bash
# Download and split dataset
python main.py --setup

# Load custom data
python src/data_loader.py --path ./data/custom --client-id 0
```

### Running Queries
```bash
# Single query
python main.py --query "your question here"

# Interactive mode
python main.py --interactive

# Batch mode
python main.py --batch queries.txt
```

### Docker Deployment
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f flower-server

# Stop all
docker-compose down
```

### API Usage
```bash
# Start API server
uvicorn api:app --reload

# Query via curl
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is diabetes?"}'

# Health check
curl http://localhost:8000/health
```

---

## 🔧 CONFIGURATION QUICK REFERENCE

### Key Settings in `src/config.py`

```python
# Models
embedding_model = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
llm_model = "gpt-4"  # or "gpt-3.5-turbo"

# Retrieval
top_k = 10                    # docs per client
chunk_size = 512              # tokens per chunk
similarity_threshold = 0.7    # minimum score

# Federated
num_clients = 3
k_rrf = 60  # RRF parameter
```

### Environment Variables
```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional
export ANTHROPIC_API_KEY="sk-ant-..."
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
```

---

## 📊 KEY ARCHITECTURE COMPONENTS

### Data Flow
```
Query → LangChain → Flower Server → [Client 1, Client 2, Client 3]
                                          ↓
                    Results ← Aggregation (RRF) ← [Results, Results, Results]
                       ↓
                 LLM Generation → Final Answer
```

### Components Map
| Component | File | Purpose |
|-----------|------|---------|
| Client    | `src/client.py` | Local retrieval |
| Server    | `src/server.py` | Coordination & aggregation |
| Embeddings | `src/embeddings.py` | BioBERT encoding |
| Vector Store | `src/vector_store.py` | Qdrant operations |
| LLM       | `src/llm_generator.py` | Answer generation |
| Config | `src/config.py` | Settings |

---

## 🐛 COMMON ISSUES & FIXES

### Issue: "Qdrant connection refused"
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Or check if running
docker ps | grep qdrant
```

### Issue: "OpenAI API key not found"
```bash
# Set in current terminal
export OPENAI_API_KEY="your-key"

# Or add to .env file
echo "OPENAI_API_KEY=your-key" > .env
```

### Issue: "Out of memory during embedding"
```python
# In src/embeddings.py, reduce batch size
embeddings = self.model.encode(
    texts,
    batch_size=8,  # Reduce from 32
    ...
)
```

### Issue: "Slow query responses"
```python
# Enable caching in config
cache_enabled = True
cache_ttl = 3600  # seconds

# Reduce retrieval
top_k = 5  # instead of 10
```

---

## 📈 EVALUATION COMMANDS

```bash
# Run evaluation
python evaluate.py --test-set data/test.json

# Generate metrics report
python evaluate.py --report --output results/

# Compare with baseline
python evaluate.py --baseline --compare
```

---

## 🔍 DEBUGGING

### Check Client Status
```python
# In Python console
from src.client import MedicalRAGClient

client = MedicalRAGClient(0, "./data/hospital_a")
results = client.retrieve("test query")
print(f"Found {len(results)} documents")
```

### Test Vector Store
```python
from src.vector_store import QdrantVectorStore

store = QdrantVectorStore()
collections = store.client.get_collections()
print(collections)
```

### Test Embeddings
```python
from src.embeddings import BioBERTEmbedder

embedder = BioBERTEmbedder()
emb = embedder.encode_query("diabetes")
print(f"Embedding shape: {emb.shape}")
```

---

## 📊 MONITORING

### Access Dashboards
```
Prometheus: http://localhost:9090
Grafana:    http://localhost:3000 (admin/admin)
API Docs:   http://localhost:8000/docs
Web UI:     http://localhost:7860
```

### Key Metrics to Watch
```
Query Latency:       <5s target
Retrieval Accuracy:  >80% target
Cache Hit Rate:      >40% good
Error Rate:          <1% target
```

---

## 🎯 EXAMPLE QUERIES

### Medical Conditions
```
"What are the symptoms of Type 2 Diabetes?"
"How is hypertension diagnosed?"
"What causes rheumatoid arthritis?"
```

### Treatments
```
"What are the treatment options for depression?"
"How effective is chemotherapy for lung cancer?"
"What medications treat high cholesterol?"
```

### Drug Information
```
"What are the side effects of Metformin?"
"Can aspirin prevent heart attacks?"
"What is the recommended dosage of insulin?"
```

### Research Questions
```
"What is the latest research on Alzheimer's disease?"
"Are probiotics effective for digestive health?"
"What are the risks of vitamin D deficiency?"
```

---

## 💾 DATA MANAGEMENT

### Add Custom Dataset
```python
# Create custom loader
from src.data_loader import PubMedQALoader

loader = PubMedQALoader()
# Add your documents
client_data = {
    'documents': ['doc1', 'doc2', ...],
    'metadata': [{'id': 1}, {'id': 2}, ...]
}
loader.save_client_data([client_data])
```

### Backup Vector Store
```bash
# Using Qdrant snapshot
curl -X POST "http://localhost:6333/collections/medical_docs/snapshots"

# Download snapshot
curl "http://localhost:6333/collections/medical_docs/snapshots/{snapshot_name}" \
  --output backup.snapshot
```

---

## 🔒 SECURITY CHECKLIST

- [ ] Set strong API keys
- [ ] Enable TLS/SSL in production
- [ ] Set up authentication
- [ ] Configure firewall rules
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Data encryption at rest
- [ ] Anonymize PHI data

---

## 📚 HELPFUL LINKS

### Official Documentation
- Flower: https://flower.ai/docs/
- FedRAG: https://vectorinstitute.github.io/fed-rag/
- LlamaIndex: https://docs.llamaindex.ai/
- Qdrant: https://qdrant.tech/documentation/

### Datasets
- PubMedQA: https://pubmedqa.github.io/
- BioASQ: http://bioasq.org/
- HuggingFace: https://huggingface.co/datasets

### Community
- Flower Slack: flower.ai/join
- GitHub Issues: <your-repo>/issues

---

## 🎓 LEARNING RESOURCES

### Tutorials
1. Flower Quickstart: https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
2. LlamaIndex Guide: https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html
3. RAG Course: https://www.deeplearning.ai/short-courses/building-applications-vector-databases/

### Papers
1. Flower: https://arxiv.org/abs/2007.14390
2. FedRAG: https://arxiv.org/abs/2410.13272
3. BioBERT: https://arxiv.org/abs/1901.08746
4. PubMedQA: https://arxiv.org/abs/1909.06146

---

## 🚀 PERFORMANCE TIPS

### Speed Up Retrieval
```python
# Use faster embedding model
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# Reduce chunk size
chunk_size = 256

# Enable caching
cache_enabled = True
```

### Reduce Costs
```python
# Use cheaper LLM for simple queries
if is_simple_query(query):
    llm_model = "gpt-3.5-turbo"  # $0.001/1K vs $0.03/1K

# Implement query deduplication
# Add response caching
```

### Improve Accuracy
```python
# Use reranking
rerank = True

# Increase retrieval
top_k = 20

# Use query expansion
expand_query = True
```

---

## 🎯 PROJECT MILESTONES

### Week 1: ✅ Setup & Baseline
- Environment configured
- Dataset downloaded
- Basic RAG working

### Week 3: ✅ Federated
- 3 clients running
- Retrieval working
- Results aggregating

### Week 5: ✅ LLM Integration
- Answer generation
- Citations working
- Quality validated

### Week 8: ✅ Production
- Docker deployed
- API live
- Monitoring active

---

## 🆘 GET HELP

```bash
# Check system health
python -m src.health_check

# Run diagnostics
python -m src.diagnostics --verbose

# Generate debug report
python -m src.debug_report --output debug.txt
```

### Still Stuck?
1. Check the logs: `docker-compose logs -f`
2. Read the full docs: `/docs/`
3. Search issues: GitHub Issues
4. Ask community: Flower Slack

---

**Remember**: Start simple, test often, iterate quickly! 🚀

**Good luck building your Federated Medical RAG system!** 🏥✨
