from src.config import config
from src.data_loader import PubMedQALoader
from src.client import MedicalRAGClient
from src.server import FederatedRAGServer
from src.llm_generator import MedicalLLMGenerator
import argparse

def setup_clients():
    """Setup and prepare client data"""
    print("=== Setting up Federated Clients ===\n")
    
    # Load and split dataset
    loader = PubMedQALoader()
    loader.load_dataset()
    client_datasets = loader.split_by_specialty(num_clients=config.flower.num_clients)
    loader.save_client_data(client_datasets)
    
    print("\nClient data prepared successfully!")

def run_federated_query(query: str):
    """Run a federated query across all clients"""
    print(f"\n=== Federated Query: {query} ===\n")
    
    # Initialize clients
    clients = []
    client_paths = ['./data/hospital_a', './data/hospital_b', './data/research_center']
    
    for i, path in enumerate(client_paths):
        print(f"\nInitializing Client {i}...")
        client = MedicalRAGClient(client_id=i, data_path=path)
        clients.append(client)
    
    # Execute retrieval on each client
    print(f"\n=== Retrieving from {len(clients)} clients ===")
    client_results = []
    for i, client in enumerate(clients):
        print(f"\nClient {i} retrieving...")
        results = client.retrieve(query, top_k=config.retrieval.top_k)
        client_results.append(results)
        print(f"Client {i}: Found {len(results)} documents")
    
    # Aggregate results on server
    server = FederatedRAGServer(
        num_clients=config.flower.num_clients,
        k_rrf=config.flower.k_rrf
    )
    merged_results = server.aggregate_results(query, client_results)
    
    # Generate answer using LLM
    print("\n=== Generating Answer with LLM ===")
    generator = MedicalLLMGenerator(
        model_name=config.model.llm_model,
        temperature=config.model.llm_temperature
    )
    result = generator.generate_answer(query, merged_results, top_k=5)
    
    # Display result
    print("\n" + "="*80)
    print(f"QUERY: {result['query']}")
    print("="*80)
    print(f"\nANSWER:\n{result['answer']}")
    print("\n" + "="*80)
    print("SOURCES:")
    for source in result['sources']:
        print(f"\n[{source['id']}] Score: {source['score']:.4f}")
        print(f"    {source['text']}")
        print(f"    Metadata: {source['metadata']}")
    print("="*80)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Federated Medical RAG System')
    parser.add_argument('--setup', action='store_true', help='Setup client data')
    parser.add_argument('--query', type=str, help='Query to run')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_clients()
    elif args.query:
        run_federated_query(args.query)
    else:
        # Demo queries
        demo_queries = [
            "What are the side effects of Metformin?",
            "How effective is aspirin for cardiovascular disease prevention?",
            "What are the latest treatments for Type 2 Diabetes?"
        ]
        
        print("\n=== Running Demo Queries ===\n")
        for query in demo_queries:
            run_federated_query(query)
            print("\n\n")

if __name__ == "__main__":
    main()