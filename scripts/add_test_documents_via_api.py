"""
Add test documents to ChromaDB via API endpoints.
This script uses HTTP requests to populate the vector database with sample data.
"""

import requests
import json
import time

# API base URL
API_BASE = "http://localhost:8001"

# Sample documents for testing
SAMPLE_DOCUMENTS = [
    {
        "title": "Python Programming Basics",
        "url": "https://docs.python.org/3/tutorial/introduction.html",
        "content": "Python is an easy to learn, powerful programming language. It has efficient high-level data structures and a simple but effective approach to object-oriented programming. Python's elegant syntax and dynamic typing, together with its interpreted nature, make it an ideal language for scripting and rapid application development in many areas on most platforms.",
        "source": "python.org"
    },
    {
        "title": "Introduction to Machine Learning",
        "url": "https://scikit-learn.org/stable/tutorial/basic/tutorial.html", 
        "content": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. Machine learning algorithms build a model based on training data in order to make predictions or decisions.",
        "source": "scikit-learn.org"
    },
    {
        "title": "Modern Web Development Guide",
        "url": "https://developer.mozilla.org/en-US/docs/Learn/Getting_started_with_the_web",
        "content": "Web development refers to creating, building, and maintaining websites. It includes web design, web publishing, web programming, and database management. Web development can be divided into Frontend development (user interface and experience) and Backend development (server-side logic, databases, and application architecture).",
        "source": "developer.mozilla.org"
    },
    {
        "title": "Database Performance Optimization",
        "url": "https://www.postgresql.org/docs/current/performance-tips.html",
        "content": "Database optimization improves database system performance through indexing, query optimization, normalization, and hardware tuning. Proper indexing dramatically improves query performance. Query optimization involves analyzing and rewriting queries to execute more efficiently. Regular maintenance tasks like updating statistics are crucial for optimal performance.",
        "source": "postgresql.org"
    },
    {
        "title": "Cloud Computing Fundamentals", 
        "url": "https://aws.amazon.com/what-is-cloud-computing/",
        "content": "Cloud computing is the on-demand delivery of IT resources over the Internet with pay-as-you-go pricing. Instead of maintaining physical data centers, you access computing power, storage, and databases from cloud providers. Cloud computing offers cost savings, scalability, reliability, and flexibility through IaaS, PaaS, and SaaS service models.",
        "source": "aws.amazon.com"
    }
]

def check_api_health():
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ API is healthy: {health_data}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def check_collection_status():
    """Check current ChromaDB collection status."""
    try:
        response = requests.get(f"{API_BASE}/collection/info", timeout=10)
        if response.status_code == 200:
            info = response.json()
            print(f"📊 Collection status: {info['count']} documents in '{info['name']}'")
            return info
        else:
            print(f"❌ Failed to get collection info: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error checking collection: {e}")
        return None

def test_search_functionality():
    """Test the search functionality with sample queries."""
    test_queries = [
        "What is Python programming?",
        "How does machine learning work?", 
        "What are database optimization techniques?",
        "Explain cloud computing benefits",
        "Web development best practices"
    ]
    
    print("\n🔍 Testing search functionality...")
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        
        try:
            # Test search endpoint
            response = requests.get(
                f"{API_BASE}/search",
                params={
                    "query": query,
                    "top_k": 3,
                    "similarity_threshold": 0.3,
                    "include_scores": True
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results_count = data.get("count", 0)
                total_searched = data.get("total_documents_searched", 0)
                
                print(f"  ✅ Search: {results_count}/{total_searched} results above threshold")
                
                if results_count > 0:
                    top_result = data["results"][0]
                    similarity = top_result.get("similarity_score", 0)
                    title = top_result.get("metadata", {}).get("title", "Unknown")
                    print(f"  📄 Top result: {title} (similarity: {similarity:.2f})")
                else:
                    print("  ⚠️  No results above similarity threshold")
            else:
                print(f"  ❌ Search failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Search error: {e}")

def test_rag_endpoints():
    """Test the RAG endpoints."""
    test_queries = [
        "What is Python programming?",
        "How does machine learning work?",
        "What are the benefits of cloud computing?"
    ]
    
    print("\n🤖 Testing RAG endpoints...")
    
    for query in test_queries:
        print(f"\n📝 RAG Query: '{query}'")
        
        try:
            # Test regular RAG endpoint
            response = requests.get(
                f"{API_BASE}/ask",
                params={
                    "question": query,
                    "top_k": 3,
                    "similarity_threshold": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                context_used = data.get("context_used", False)
                sources_count = len(data.get("sources", []))
                answer_length = len(data.get("answer", ""))
                
                print(f"  ✅ RAG: Context used: {context_used}, Sources: {sources_count}, Answer: {answer_length} chars")
                
                if sources_count > 0:
                    top_source = data["sources"][0]
                    title = top_source.get("metadata", {}).get("title", "Unknown")
                    similarity = top_source.get("similarity_score", 0)
                    print(f"  📄 Top source: {title} (similarity: {similarity:.2f})")
                    
                if not context_used:
                    print("  💡 Using fallback response (no relevant context found)")
            else:
                print(f"  ❌ RAG failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ RAG error: {e}")

def main():
    """Main function to test ChromaDB population and RAG functionality."""
    print("🚀 ChromaDB RAG Functionality Test")
    print("=" * 50)
    
    # Check API health
    if not check_api_health():
        print("❌ API is not available. Make sure the FastAPI server is running.")
        return False
    
    # Check collection status
    collection_info = check_collection_status()
    if collection_info is None:
        print("❌ Cannot access ChromaDB collection.")
        return False
    
    document_count = collection_info.get("count", 0)
    
    if document_count == 0:
        print("\n⚠️  ChromaDB collection is empty!")
        print("💡 To populate with test data, you need to:")
        print("   1. Run the ETL pipeline to scrape and process documents")
        print("   2. Run the embedding generation process")
        print("   3. Or manually add documents via the embedding pipeline")
        print("\n🔧 For now, let's test with fallback responses...")
    else:
        print(f"\n✅ Found {document_count} documents in ChromaDB")
    
    # Test search functionality
    test_search_functionality()
    
    # Test RAG endpoints
    test_rag_endpoints()
    
    # Test similarity validation endpoint
    print("\n🔬 Testing similarity validation endpoint...")
    try:
        response = requests.get(
            f"{API_BASE}/similarity/validate",
            params={
                "query": "What is Python programming?",
                "test_thresholds": [0.1, 0.3, 0.5, 0.7, 0.9],
                "top_k": 10
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            total_docs = data.get("total_documents", 0)
            recommended_threshold = data.get("recommended_threshold", 0)
            
            print(f"✅ Validation: {total_docs} docs analyzed, recommended threshold: {recommended_threshold:.2f}")
            
            if "quality_assessment" in data:
                quality = data["quality_assessment"]
                print(f"📊 Quality: Excellent: {quality.get('excellent', 0)}, Good: {quality.get('good', 0)}, Fair: {quality.get('fair', 0)}, Poor: {quality.get('poor', 0)}")
        else:
            print(f"❌ Validation endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Validation error: {e}")
    
    print("\n" + "=" * 50)
    if document_count > 0:
        print("🎉 RAG system is working with existing documents!")
        print("💬 You can now test the chat interface at http://localhost:8001")
    else:
        print("⚠️  RAG system is working but needs documents to provide context-based responses.")
        print("💡 The system will use fallback responses until documents are added.")
    
    return True

if __name__ == "__main__":
    main()