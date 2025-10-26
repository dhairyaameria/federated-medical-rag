from src.config import config
from src.data_loader import PubMedQALoader
from src.client import MedicalRAGClient
from src.server import FederatedRAGServer
from src.llm_generator import MedicalLLMGenerator
import argparse

# ============================================
# GLOBAL CLIENTS (initialized once)
# ============================================
_clients = None
_server = None
_generator = None

def initialize_system():
    """Initialize clients, server, and generator once"""
    global _clients, _server, _generator
    
    if _clients is not None:
        print("System already initialized, reusing existing clients...")
        return _clients, _server, _generator
    
    print("=" * 80)
    print("INITIALIZING FEDERATED MEDICAL RAG SYSTEM")
    print("=" * 80)
    print("\nThis will take 2-3 minutes (but only happens once)...\n")
    
    # Initialize clients
    _clients = []
    client_paths = ['./data/hospital_a', './data/hospital_b', './data/research_center']
    
    for i, path in enumerate(client_paths):
        print(f"[{i+1}/3] Initializing Client {i}...")
        try:
            client = MedicalRAGClient(client_id=i, data_path=path)
            _clients.append(client)
            print(f"    ✓ Client {i} ready!\n")
        except Exception as e:
            print(f"    ✗ Error initializing Client {i}: {e}")
            raise
    
    # Initialize server
    print("Initializing Server...")
    _server = FederatedRAGServer(
        num_clients=config.flower.num_clients,
        k_rrf=config.flower.k_rrf
    )
    print("    ✓ Server ready!\n")
    
    # Initialize LLM generator
    print("Initializing LLM Generator...")
    _generator = MedicalLLMGenerator(
        provider=config.model.llm_provider,
        model_name=config.model.llm_model,
        temperature=config.model.llm_temperature
    )
    print("    ✓ Generator ready!\n")
    
    print("=" * 80)
    print("SYSTEM READY! Queries will now be fast (3-5 seconds each)")
    print("=" * 80 + "\n")
    
    return _clients, _server, _generator

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
    """Run a federated query across all clients (FAST version)"""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}\n")
    
    # Get or initialize system (only happens once)
    clients, server, generator = initialize_system()
    
    # Execute retrieval on each client (FAST - no reinitialization!)
    print(f"Retrieving from {len(clients)} clients...")
    client_results = []
    for i, client in enumerate(clients):
        print(f"  Client {i}: ", end="", flush=True)
        results = client.retrieve(query, top_k=config.retrieval.top_k)
        client_results.append(results)
        print(f"✓ Found {len(results)} documents")
    
    # Aggregate results on server
    print("\nAggregating results...")
    merged_results = server.aggregate_results(query, client_results)
    print(f"✓ Merged into {len(merged_results)} unique documents")
    
    # Generate answer using LLM
    print("\nGenerating answer with LLM...")
    result = generator.generate_answer(query, merged_results, top_k=5)
    print("✓ Answer generated!")
    
    # Display result
    print("\n" + "="*80)
    print("ANSWER:")
    print("="*80)
    print(f"\n{result['answer']}\n")
    print("="*80)
    print("SOURCES:")
    print("="*80)
    for source in result['sources']:
        print(f"\n[{source['id']}] Score: {source['score']:.4f}")
        print(f"Text: {source['text']}")
        if source['metadata']:
            print(f"Metadata: {source['metadata']}")
    print("="*80)
    
    return result

def run_interactive_mode():
    """Interactive query mode"""
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Type your medical questions (or 'quit' to exit)\n")
    
    # Initialize system once
    initialize_system()
    
    while True:
        try:
            query = input("\n🔍 Enter your question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋\n")
                break
            
            run_federated_query(query)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Federated Medical RAG System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --setup                                    # Setup client data
  python main.py --query "What are side effects of aspirin?"  # Single query
  python main.py --interactive                              # Interactive mode
  python main.py --demo                                     # Run demo queries
        """
    )
    parser.add_argument('--setup', action='store_true', 
                       help='Setup and split client data')
    parser.add_argument('--query', type=str, 
                       help='Run a single query')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive query mode')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo queries')
    parser.add_argument('--init-only', action='store_true',
                       help='Just initialize the system (for testing)')
    
    args = parser.parse_args()
    
    try:
        if args.setup:
            setup_clients()
            
        elif args.init_only:
            # Just initialize and exit (useful for warming up)
            initialize_system()
            print("✓ System initialized successfully!")
            
        elif args.query:
            run_federated_query(args.query)
            
        elif args.interactive:
            run_interactive_mode()
            
        elif args.demo:
            # Demo queries
            demo_queries = [
                "What are the side effects of Metformin?",
                "How effective is aspirin for cardiovascular disease prevention?",
                "What are the latest treatments for Type 2 Diabetes?"
            ]
            
            print("\n" + "="*80)
            print("RUNNING DEMO QUERIES")
            print("="*80)
            
            # Initialize once before all queries
            initialize_system()
            
            for i, query in enumerate(demo_queries, 1):
                print(f"\n\n{'#'*80}")
                print(f"DEMO QUERY {i}/{len(demo_queries)}")
                print(f"{'#'*80}")
                run_federated_query(query)
            
            print("\n\n" + "="*80)
            print("DEMO COMPLETED")
            print("="*80 + "\n")
        else:
            # Default: show help
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye! 👋\n")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()