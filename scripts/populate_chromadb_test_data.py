"""
Populate ChromaDB with test data to validate RAG functionality.
This script adds sample documents to test similarity search and context retrieval.
"""

import sys
import os
import json
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from embeddings.embedder import EmbeddingPipeline
    from common.config import config, setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

# Sample documents for testing RAG functionality
SAMPLE_DOCUMENTS = [
    {
        "id": "python_basics_1",
        "title": "Python Programming Basics",
        "url": "https://docs.python.org/3/tutorial/introduction.html",
        "content": """Python is an easy to learn, powerful programming language. It has efficient high-level data structures and a simple but effective approach to object-oriented programming. Python's elegant syntax and dynamic typing, together with its interpreted nature, make it an ideal language for scripting and rapid application development in many areas on most platforms. The Python interpreter and the extensive standard library are freely available in source or binary form for all major platforms from the Python Web site, https://www.python.org/, and may be freely distributed.""",
        "source": "python.org",
        "content_length": 645,
        "scraped_at": datetime.now().isoformat()
    },
    {
        "id": "machine_learning_intro",
        "title": "Introduction to Machine Learning",
        "url": "https://scikit-learn.org/stable/tutorial/basic/tutorial.html",
        "content": """Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. Machine learning algorithms build a model based on training data in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision.""",
        "source": "scikit-learn.org",
        "content_length": 612,
        "scraped_at": datetime.now().isoformat()
    },
    {
        "id": "web_development_guide",
        "title": "Modern Web Development Guide",
        "url": "https://developer.mozilla.org/en-US/docs/Learn/Getting_started_with_the_web",
        "content": "Web development refers to the creating, building, and maintaining of websites. It includes aspects such as web design, web publishing, web programming, and database management. It is the creation of an application that works over the internet i.e. websites. Web development can be divided into two parts: Frontend development and Backend development. Frontend development deals with the user interface and user experience, while backend development focuses on server-side logic, databases, and application architecture.",
        "source": "developer.mozilla.org",
        "content_length": 578,
        "scraped_at": datetime.now().isoformat()
    },
    {
        "id": "database_optimization",
        "title": "Database Performance Optimization",
        "url": "https://www.postgresql.org/docs/current/performance-tips.html",
        "content": """Database optimization is the process of improving the performance of a database system. This involves various techniques such as indexing, query optimization, normalization, and hardware tuning. Proper indexing can dramatically improve query performance by allowing the database to quickly locate specific rows. Query optimization involves analyzing and rewriting queries to execute more efficiently. Database normalization reduces data redundancy and improves data integrity. Regular maintenance tasks like updating statistics and rebuilding indexes are also crucial for optimal performance.""",
        "source": "postgresql.org",
        "content_length": 623,
        "scraped_at": datetime.now().isoformat()
    },
    {
        "id": "cloud_computing_basics",
        "title": "Cloud Computing Fundamentals",
        "url": "https://aws.amazon.com/what-is-cloud-computing/",
        "content": """Cloud computing is the on-demand delivery of IT resources over the Internet with pay-as-you-go pricing. Instead of buying, owning, and maintaining physical data centers and servers, you can access technology services, such as computing power, storage, and databases, on an as-needed basis from a cloud provider like Amazon Web Services (AWS). Cloud computing offers several benefits including cost savings, scalability, reliability, and flexibility. The main service models are Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS).""",
        "source": "aws.amazon.com",
        "content_length": 634,
        "scraped_at": datetime.now().isoformat()
    },
    {
        "id": "api_design_principles",
        "title": "RESTful API Design Best Practices",
        "url": "https://restfulapi.net/rest-api-design-tutorial-with-example/",
        "content": """REST (Representational State Transfer) is an architectural style for designing networked applications. RESTful APIs follow specific principles: they are stateless, use standard HTTP methods (GET, POST, PUT, DELETE), have a uniform interface, and use resource-based URLs. Good API design includes proper use of HTTP status codes, consistent naming conventions, versioning strategies, and comprehensive documentation. Security considerations include authentication, authorization, input validation, and rate limiting. Well-designed APIs are intuitive, consistent, and easy to use.""",
        "source": "restfulapi.net",
        "content_length": 598,
        "scraped_at": datetime.now().isoformat()
    }
]

def populate_chromadb():
    """Populate ChromaDB with sample test data."""
    setup_logging(config.log_level, config.log_format)
    
    print("🔄 Initializing embedding pipeline...")
    pipeline = EmbeddingPipeline()
    
    try:
        # Check current collection status
        collection_info = pipeline.vector_store.get_collection_info()
        print(f"📊 Current collection status: {collection_info['count']} documents")
        
        if collection_info['count'] > 0:
            response = input("⚠️  Collection already has documents. Reset and repopulate? (y/N): ")
            if response.lower() != 'y':
                print("❌ Aborted. Keeping existing documents.")
                return False
            
            print("🗑️  Resetting collection...")
            pipeline.vector_store.reset_collection()
        
        # Prepare data for embedding
        print(f"📝 Preparing {len(SAMPLE_DOCUMENTS)} documents for embedding...")
        
        ids = []
        documents = []
        metadatas = []
        
        for doc in SAMPLE_DOCUMENTS:
            ids.append(doc["id"])
            documents.append(doc["content"])
            
            metadata = {
                "url": doc["url"],
                "title": doc["title"],
                "source": doc["source"],
                "content_length": str(doc["content_length"]),
                "scraped_at": doc["scraped_at"],
                "id": doc["id"]
            }
            metadatas.append(metadata)
        
        # Generate embeddings
        print("🧠 Generating embeddings...")
        embeddings = pipeline.embedding_generator.embed_texts(documents)
        
        # Filter out any None embeddings
        valid_data = []
        for i, embedding in enumerate(embeddings):
            if embedding is not None:
                valid_data.append((ids[i], embedding, documents[i], metadatas[i]))
            else:
                print(f"⚠️  Failed to generate embedding for document: {ids[i]}")
        
        if not valid_data:
            print("❌ No valid embeddings generated. Check your embedding service.")
            return False
        
        # Unpack valid data
        valid_ids, valid_embeddings, valid_documents, valid_metadatas = zip(*valid_data)
        
        # Add to ChromaDB
        print(f"💾 Adding {len(valid_data)} documents to ChromaDB...")
        pipeline.vector_store.add_documents(
            ids=list(valid_ids),
            embeddings=list(valid_embeddings),
            documents=list(valid_documents),
            metadatas=list(valid_metadatas)
        )
        
        # Verify the data was added
        final_info = pipeline.vector_store.get_collection_info()
        print(f"✅ Successfully populated ChromaDB!")
        print(f"📊 Final collection status: {final_info['count']} documents")
        
        # Test a sample query
        print("\n🔍 Testing similarity search...")
        test_query = "How do I learn Python programming?"
        results = pipeline.query_embeddings(test_query, n_results=3)
        
        if results.get('documents') and results['documents'][0]:
            print(f"✅ Similarity search working! Found {len(results['documents'][0])} relevant documents")
            print(f"📄 Top result: {results['metadatas'][0][0]['title']}")
        else:
            print("⚠️  Similarity search returned no results")
        
        return True
        
    except Exception as e:
        print(f"❌ Error populating ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_endpoints():
    """Test the RAG endpoints with the populated data."""
    import requests
    
    print("\n🧪 Testing RAG endpoints...")
    
    test_queries = [
        "What is Python programming?",
        "How does machine learning work?",
        "What are database optimization techniques?",
        "Explain cloud computing benefits"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        
        try:
            # Test regular endpoint
            response = requests.get(
                "http://localhost:8001/ask",
                params={"question": query, "top_k": 3, "similarity_threshold": 0.3},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                context_used = data.get("context_used", False)
                sources_count = len(data.get("sources", []))
                
                print(f"  ✅ Regular endpoint: Context used: {context_used}, Sources: {sources_count}")
                
                if sources_count > 0:
                    print(f"  📄 Top source: {data['sources'][0].get('metadata', {}).get('title', 'Unknown')}")
            else:
                print(f"  ❌ Regular endpoint failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error testing regular endpoint: {e}")

if __name__ == "__main__":
    print("🚀 ChromaDB Test Data Population Script")
    print("=" * 50)
    
    success = populate_chromadb()
    
    if success:
        test_rag_endpoints()
        print("\n🎉 ChromaDB population and testing completed!")
        print("💡 You can now test the chat interface with real data.")
    else:
        print("\n❌ Failed to populate ChromaDB. Check the logs for details.")
        sys.exit(1)