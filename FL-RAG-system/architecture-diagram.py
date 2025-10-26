import graphviz

# Create a new directed graph
dot = graphviz.Digraph(comment='Federated Medical RAG Architecture', format='png')
dot.attr(rankdir='TB', size='16,20', dpi='300')
dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='11')
dot.attr('edge', fontname='Arial', fontsize='10')

# Define color scheme
color_ui = '#E3F2FD'
color_orchestration = '#FFF3E0'
color_federated = '#F3E5F5'
color_client = '#E8F5E9'
color_storage = '#FFF9C4'

# User Interface Layer
with dot.subgraph(name='cluster_0') as c:
    c.attr(label='User Interface Layer', style='filled', color='lightgrey')
    c.node('ui', 'Web Dashboard\nAPI Endpoint\nMobile App', fillcolor=color_ui)

# Orchestration Layer
with dot.subgraph(name='cluster_1') as c:
    c.attr(label='Orchestration Layer (LangChain)', style='filled', color='lightgrey')
    c.node('orchestrator', 'Query Router\nPreprocessor\nResponse Synthesizer', fillcolor=color_orchestration)

# Federated Learning Layer
with dot.subgraph(name='cluster_2') as c:
    c.attr(label='Federated Learning Layer (Flower)', style='filled', color='lightgrey')
    c.node('flower_server', 'Flower Server\n\n• Query Distribution\n• RRF Aggregation\n• Result Merging', 
            fillcolor=color_federated, width='3')
    c.node('llm_engine', 'LLM Generation\n\n• Context Assembly\n• Answer Generation\n• Citation Management', 
            fillcolor=color_federated, width='3')

# Flower Clients
with dot.subgraph(name='cluster_3') as c:
    c.attr(label='Hospital A - Flower Client 1', style='filled', color='lightblue')
    c.node('client1_docs', 'Document Store\nPubMed Papers\nClinical Trials', fillcolor=color_client)
    c.node('client1_embed', 'Embedding\nBioBERT', fillcolor=color_storage)
    c.node('client1_vector', 'Vector Store\nQdrant', fillcolor=color_storage)
    c.node('client1_retrieval', 'Retrieval Engine\nLlamaIndex', fillcolor=color_client)

with dot.subgraph(name='cluster_4') as c:
    c.attr(label='Hospital B - Flower Client 2', style='filled', color='lightgreen')
    c.node('client2_docs', 'Document Store\nClinical Guidelines\nDrug Information', fillcolor=color_client)
    c.node('client2_embed', 'Embedding\nBioBERT', fillcolor=color_storage)
    c.node('client2_vector', 'Vector Store\nQdrant', fillcolor=color_storage)
    c.node('client2_retrieval', 'Retrieval Engine\nLlamaIndex', fillcolor=color_client)

with dot.subgraph(name='cluster_5') as c:
    c.attr(label='Research Center - Flower Client 3', style='filled', color='lightyellow')
    c.node('client3_docs', 'Document Store\nResearch Data\nCase Studies', fillcolor=color_client)
    c.node('client3_embed', 'Embedding\nBioBERT', fillcolor=color_storage)
    c.node('client3_vector', 'Vector Store\nQdrant', fillcolor=color_storage)
    c.node('client3_retrieval', 'Retrieval Engine\nLlamaIndex', fillcolor=color_client)

# Edges - Main flow
dot.edge('ui', 'orchestrator', label='User Query')
dot.edge('orchestrator', 'flower_server', label='Processed Query')
dot.edge('flower_server', 'client1_retrieval', label='Distribute Query')
dot.edge('flower_server', 'client2_retrieval', label='Distribute Query')
dot.edge('flower_server', 'client3_retrieval', label='Distribute Query')

# Client internal flows
dot.edge('client1_retrieval', 'client1_vector', label='Search')
dot.edge('client1_vector', 'client1_embed', style='dotted')
dot.edge('client1_embed', 'client1_docs', style='dotted')

dot.edge('client2_retrieval', 'client2_vector', label='Search')
dot.edge('client2_vector', 'client2_embed', style='dotted')
dot.edge('client2_embed', 'client2_docs', style='dotted')

dot.edge('client3_retrieval', 'client3_vector', label='Search')
dot.edge('client3_vector', 'client3_embed', style='dotted')
dot.edge('client3_embed', 'client3_docs', style='dotted')

# Return results
dot.edge('client1_retrieval', 'flower_server', label='Top-k Docs + Scores')
dot.edge('client2_retrieval', 'flower_server', label='Top-k Docs + Scores')
dot.edge('client3_retrieval', 'flower_server', label='Top-k Docs + Scores')

# LLM generation
dot.edge('flower_server', 'llm_engine', label='Aggregated Context')
dot.edge('llm_engine', 'orchestrator', label='Generated Answer')
dot.edge('orchestrator', 'ui', label='Final Response')

# Save the diagram
dot.render('federated_medical_rag_architecture', view=False, cleanup=True)
print("Architecture diagram created successfully at: federated_medical_rag_architecture.png")