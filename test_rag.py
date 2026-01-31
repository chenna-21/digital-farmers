from rag_engine import rag_engine

# Test queries
test_queries = [
    "medicine for potato crop",
    "Medicine for potato blight",
    "How to control blight in potato",
    "PM Kisan scheme",
    "fertilizer for wheat"
]

print("=" * 80)
print("Testing RAG Engine with different queries")
print("=" * 80)

for query in test_queries:
    print(f"\n\nQuery: '{query}'")
    print("-" * 80)
    
    # Get search results
    results = rag_engine.search(query, k=3)
    
    if results:
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i} (Score: {result['score']:.4f}):")
            print(f"    Q: {result['question']}")
            print(f"    A: {result['answer']}")
    else:
        print("  No results found!")
    
    # Get generated response
    print(f"\n  Generated Response:")
    response = rag_engine.generate_response(query)
    print(f"  {response}")
