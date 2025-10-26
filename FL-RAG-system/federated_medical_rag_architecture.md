# Federated Medical Literature Q&A System - Technical Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE LAYER                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Web Dashboard / API / Slack Bot / Mobile App                     │  │
│  │  - Query Input Interface                                          │  │
│  │  - Results Display with Citations                                 │  │
│  │  - Authentication & Authorization                                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATION LAYER                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  LangChain Orchestrator                                           │  │
│  │  - Query Routing & Preprocessing                                  │  │
│  │  - Multi-step Reasoning Chain                                     │  │
│  │  - Response Aggregation & Post-processing                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    FEDERATED LEARNING LAYER (Flower)                     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      Flower Server                                │  │
│  │  ┌─────────────────────────────────────────────────────────────┐ │  │
│  │  │  FedRAG Coordinator                                          │ │  │
│  │  │  - Query Distribution to Clients                            │ │  │
│  │  │  - Reciprocal Rank Fusion (RRF) Aggregation                 │ │  │
│  │  │  - Result Merging & Ranking                                 │ │  │
│  │  │  - Client Health Monitoring                                 │ │  │
│  │  └─────────────────────────────────────────────────────────────┘ │  │
│  │                                                                    │  │
│  │  ┌─────────────────────────────────────────────────────────────┐ │  │
│  │  │  LLM Generation Engine                                       │ │  │
│  │  │  - Context Assembly from Retrieved Docs                     │ │  │
│  │  │  - Prompt Engineering & Chain-of-Thought                    │ │  │
│  │  │  - Answer Generation with Citations                         │ │  │
│  │  └─────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                    ↓                ↓                ↓
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  FLOWER CLIENT 1     │  │  FLOWER CLIENT 2     │  │  FLOWER CLIENT 3     │
│  (Hospital A)        │  │  (Hospital B)        │  │  (Research Center)   │
├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤
│ LOCAL KNOWLEDGE BASE │  │ LOCAL KNOWLEDGE BASE │  │ LOCAL KNOWLEDGE BASE │
│                      │  │                      │  │                      │
│ ┌──────────────────┐ │  │ ┌──────────────────┐ │  │ ┌──────────────────┐ │
│ │ Document Store   │ │  │ │ Document Store   │ │  │ │ Document Store   │ │
│ │ - PubMed Papers  │ │  │ │ - Clinical Notes │ │  │ │ - Research Data  │ │
│ │ - Clinical Trials│ │  │ │ - Drug Info      │ │  │ │ - Trials         │ │
│ │ - Protocols      │ │  │ │ - Guidelines     │ │  │ │ - Case Studies   │ │
│ └──────────────────┘ │  │ └──────────────────┘ │  │ └──────────────────┘ │
│         ↓            │  │         ↓            │  │         ↓            │
│ ┌──────────────────┐ │  │ ┌──────────────────┐ │  │ ┌──────────────────┐ │
│ │ Preprocessing    │ │  │ │ Preprocessing    │ │  │ │ Preprocessing    │ │
│ │ - Text Cleaning  │ │  │ │ - Anonymization  │ │  │ │ - Chunking       │ │
│ │ - Chunking       │ │  │ │ - Segmentation   │ │  │ │ - Metadata       │ │
│ └──────────────────┘ │  │ └──────────────────┘ │  │ └──────────────────┘ │
│         ↓            │  │         ↓            │  │         ↓            │
│ ┌──────────────────┐ │  │ ┌──────────────────┐ │  │ ┌──────────────────┐ │
│ │ Embedding Model  │ │  │ │ Embedding Model  │ │  │ │ Embedding Model  │ │
│ │ - BioBERT        │ │  │ │ - BioBERT        │ │  │ │ - BioBERT        │ │
│ │ - PubMedBERT     │ │  │ │ - PubMedBERT     │ │  │ │ - PubMedBERT     │ │
│ └──────────────────┘ │  │ └──────────────────┘ │  │ └──────────────────┘ │
│         ↓            │  │         ↓            │  │         ↓            │
│ ┌──────────────────┐ │  │ ┌──────────────────┐ │  │ ┌──────────────────┐ │
│ │ Vector Store     │ │  │ │ Vector Store     │ │  │ │ Vector Store     │ │
│ │ - Qdrant/Chroma  │ │  │ │ - Qdrant/Chroma  │ │  │ │ - Qdrant/Chroma  │ │
│ │ - FAISS          │ │  │ │ - FAISS          │ │  │ │ - FAISS          │ │
│ └──────────────────┘ │  │ └──────────────────┘ │  │ └──────────────────┘ │
│         ↓            │  │         ↓            │  │         ↓            │
│ ┌──────────────────┐ │  │ ┌──────────────────┐ │  │ ┌──────────────────┐ │
│ │ Retrieval Engine │ │  │ │ Retrieval Engine │ │  │ │ Retrieval Engine │ │
│ │ - LlamaIndex     │ │  │ │ - LlamaIndex     │ │  │ │ - LlamaIndex     │ │
│ │ - Semantic Search│ │  │ │ - Hybrid Search  │ │  │ │ - Dense Retrieval│ │
│ │ - Reranking      │ │  │ │ - MMR            │ │  │ │ - BM25           │ │
│ └──────────────────┘ │  │ └──────────────────┘ │  │ └──────────────────┘ │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

## Component Details

### 1. **User Interface Layer**
- **Technology**: Gradio / Streamlit / React + FastAPI
- **Features**:
  - Natural language query input
  - Results with source citations
  - Confidence scores
  - Multi-turn conversation support

### 2. **Orchestration Layer (LangChain)**
```python
Components:
- Query Preprocessor
- Intent Classifier (medical domain routing)
- Multi-step Reasoning Chain
- Response Synthesizer
- Citation Manager
```

### 3. **Federated Learning Layer (Flower + FedRAG)**

#### **Flower Server**
```python
Strategy: FedAvg with Custom RAG Aggregation
Key Functions:
- distribute_query(): Send query to all clients
- aggregate_results(): Merge retrieved documents using RRF
- rank_documents(): Rerank based on relevance scores
- generate_answer(): Use LLM with aggregated context
```

#### **LLM Generation Options**
- **Cloud APIs**: OpenAI GPT-4, Anthropic Claude
- **Medical-Specific**: BioGPT, MedPaLM, ClinicalBERT
- **Local Options**: Llama-3-70B-Medical, Mistral-Medical

### 4. **Flower Clients (Each Hospital/Institution)**

#### **a. Document Store**
```
Datasets per client:
- PubMedQA subset (split by specialty)
- PubMed articles (filtered by institution focus)
- Internal clinical guidelines (anonymized)
- Local research papers
```

#### **b. Preprocessing Pipeline**
```python
Steps:
1. Text extraction (PDF, HTML, plain text)
2. Medical entity recognition (NER)
3. Anonymization (PHI removal)
4. Section segmentation (Abstract, Methods, Results)
5. Chunking (512-1024 tokens per chunk)
6. Metadata extraction (authors, date, journal)
```

#### **c. Embedding Models**
```
Options:
- microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
- allenai/scibert_scivocab_uncased
- pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb
```

#### **d. Vector Stores**
```
Recommended: Qdrant (best for production)
Alternatives:
- Chroma (simplest setup)
- FAISS (fastest for local)
- Weaviate (advanced features)
- Pinecone (managed service)
```

#### **e. Retrieval Engine (LlamaIndex)**
```python
Retrieval Strategy:
- Initial retrieval: k=10-20 documents per client
- Reranking: Cross-encoder reranking
- Diversity: MMR (Maximal Marginal Relevance)
- Hybrid: BM25 + Dense retrieval fusion
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ QUERY: "What are the side effects of Metformin?"           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: LangChain Preprocesses Query                       │
│ - Extract medical entities: "Metformin" (drug)             │
│ - Classify intent: Side effects query                      │
│ - Expand query: Add synonyms (Glucophage)                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Flower Server Distributes to Clients               │
│ - Send embedded query to all active clients                │
│ - Include retrieval parameters (k=10, threshold=0.7)        │
└─────────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ CLIENT 1     │  │ CLIENT 2     │  │ CLIENT 3     │
│ Retrieves:   │  │ Retrieves:   │  │ Retrieves:   │
│ - 10 docs    │  │ - 10 docs    │  │ - 10 docs    │
│ - Scores     │  │ - Scores     │  │ - Scores     │
└──────────────┘  └──────────────┘  └──────────────┘
        ↓                  ↓                  ↓
        └──────────────────┼──────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Flower Server Aggregates Results (RRF)             │
│ - Merge 30 documents (10 from each client)                 │
│ - Apply Reciprocal Rank Fusion                             │
│ - Deduplicate similar documents                            │
│ - Rerank top 5-10 documents                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: LLM Generates Answer                               │
│ - Input: Top 5 documents as context                        │
│ - Prompt: Medical QA with citations                        │
│ - Output: Answer with source references                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ RESPONSE:                                                   │
│ "Common side effects of Metformin include:                 │
│ 1. Gastrointestinal issues [1,2]                          │
│ 2. Vitamin B12 deficiency [3]                             │
│ 3. Lactic acidosis (rare) [2,4]                           │
│                                                             │
│ Sources:                                                    │
│ [1] Hospital A - Clinical Guidelines 2024                  │
│ [2] Research Center - NEJM Study                           │
│ [3] Hospital B - Internal Protocol                         │
│ [4] PubMed - Systematic Review"                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### **Core Framework**
```yaml
Federated Learning: Flower 1.x
RAG Fine-tuning: FedRAG 0.0.27
Vector Store: Qdrant 1.x
Embedding: LlamaIndex + HuggingFace
LLM Orchestration: LangChain
```

### **Machine Learning**
```yaml
Embeddings:
  - microsoft/BiomedNLP-BiomedBERT-base
  - sentence-transformers/all-mpnet-base-v2

Reranking:
  - cross-encoder/ms-marco-MiniLM-L-6-v2
  - BAAI/bge-reranker-large

LLMs:
  - OpenAI GPT-4 (via API)
  - Meta Llama-3-70B-Medical
  - BioGPT
```

### **Infrastructure**
```yaml
Backend: FastAPI / Flask
Database: PostgreSQL (metadata), Redis (cache)
Message Queue: RabbitMQ / Kafka
Monitoring: Prometheus + Grafana
Logging: ELK Stack
```

### **Data Processing**
```yaml
PDF Extraction: PyMuPDF, pdfplumber
Text Processing: spaCy, ScispaCy
Chunking: LangChain TextSplitters
Anonymization: Presidio
```

---

## Deployment Architecture

### **Option 1: Docker Compose (Development)**
```yaml
services:
  flower-server:
    image: flower-server:latest
    ports: ["8080:8080"]
  
  flower-client-1:
    image: flower-client:latest
    volumes: ["./data/hospital_a:/data"]
  
  flower-client-2:
    image: flower-client:latest
    volumes: ["./data/hospital_b:/data"]
  
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
  
  api-server:
    image: api-server:latest
    ports: ["8000:8000"]
```

### **Option 2: Kubernetes (Production)**
```yaml
Components:
- Flower Server: Deployment + Service
- Flower Clients: StatefulSet (one per institution)
- Qdrant Cluster: Operator-managed
- API Gateway: Ingress + LoadBalancer
- Monitoring: Prometheus Operator
```

### **Option 3: Cloud Services**
```
AWS:
- Flower Server: ECS Fargate
- Clients: EC2 in private VPC per institution
- Vector DB: Qdrant Cloud or Amazon OpenSearch
- LLM: SageMaker Endpoints

Azure:
- Container Instances / AKS
- Azure OpenAI Service
- Cosmos DB for metadata
```

---

## Security & Privacy

### **Data Protection**
```
1. Encryption at Rest: AES-256
2. Encryption in Transit: TLS 1.3
3. Access Control: RBAC + mTLS between clients
4. Audit Logging: All queries and responses logged
5. Anonymization: PHI scrubbing before indexing
```

### **Compliance**
```
- HIPAA: BAA agreements, encrypted backups
- GDPR: Right to deletion, data portability
- FDA 21 CFR Part 11: Audit trails, e-signatures
```

---

## Evaluation Metrics

### **Retrieval Quality**
```
- Precision@k: Relevance of top-k documents
- Recall@k: Coverage of relevant documents
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)
```

### **Generation Quality**
```
- Faithfulness: Answer grounded in sources
- Relevance: Answer addresses the question
- Coherence: Logical flow and readability
- Citation Accuracy: Proper source attribution
```

### **Federated Performance**
```
- Communication Rounds: Number of server-client exchanges
- Latency: End-to-end query response time
- Throughput: Queries per second
- Client Contribution: Documents per client
```

---

## Implementation Phases

### **Phase 1: Single-Node Prototype (2-3 weeks)**
```
✓ Set up single Flower client with PubMedQA
✓ Implement basic RAG with LlamaIndex
✓ Test query-answer pipeline
✓ Evaluate baseline metrics
```

### **Phase 2: Federated Setup (3-4 weeks)**
```
✓ Deploy Flower server + 3 clients
✓ Implement FedRAG aggregation
✓ Test federated retrieval
✓ Optimize communication protocol
```

### **Phase 3: Production Features (4-6 weeks)**
```
✓ Add user authentication
✓ Implement caching layer
✓ Set up monitoring & logging
✓ Deploy on cloud infrastructure
```

### **Phase 4: Advanced Features (ongoing)**
```
✓ Multi-turn conversations
✓ Query expansion & reformulation
✓ Continuous learning from feedback
✓ Multi-modal support (images, tables)
```
