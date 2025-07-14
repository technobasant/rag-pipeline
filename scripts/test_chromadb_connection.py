#!/usr/bin/env python3
"""
Test script to diagnose ChromaDB connection issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.config import Settings

def test_chromadb_connection():
    """Test ChromaDB connection with detailed diagnostics."""
    
    print("🔍 Testing ChromaDB Connection...")
    print("=" * 50)
    
    # Configuration
    host = "localhost"
    port = 8000
    
    print(f"📍 Connecting to ChromaDB at {host}:{port}")
    
    try:
        # Test 1: Basic connection
        print("\n1️⃣ Testing basic connection...")
        client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(allow_reset=True)
        )
        print("✅ ChromaDB client created successfully")
        
        # Test 2: Heartbeat
        print("\n2️⃣ Testing heartbeat...")
        try:
            heartbeat_result = client.heartbeat()
            print(f"✅ Heartbeat successful: {heartbeat_result}")
        except Exception as e:
            print(f"❌ Heartbeat failed: {e}")
            return False
        
        # Test 3: List collections
        print("\n3️⃣ Testing collection listing...")
        try:
            collections = client.list_collections()
            print(f"✅ Collections retrieved: {len(collections)} found")
            for i, col in enumerate(collections):
                print(f"   {i+1}. {col.name}")
            if not collections:
                print("   📝 No collections exist yet")
        except Exception as e:
            print(f"❌ Failed to list collections: {e}")
            return False
        
        # Test 4: Try to get or create target collection
        print("\n4️⃣ Testing target collection 'documents'...")
        try:
            collection = client.get_or_create_collection(name="documents")
            print(f"✅ Collection 'documents' accessed/created successfully")
            
            # Get collection info
            count = collection.count()
            print(f"📊 Collection has {count} documents")
            
        except Exception as e:
            print(f"❌ Failed to access/create collection 'documents': {e}")
            return False
        
        print("\n🎉 All ChromaDB tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to ChromaDB: {e}")
        print(f"🔧 Make sure ChromaDB is running on {host}:{port}")
        print("💡 Try: docker-compose up chromadb")
        return False

def test_with_config():
    """Test using the application configuration."""
    print("\n" + "=" * 50)
    print("🔍 Testing with Application Configuration...")
    print("=" * 50)
    
    try:
        from src.common.config import config
        print(f"📍 Config ChromaDB: {config.chroma.host}:{config.chroma.port}")
        print(f"📁 Target collection: {config.chroma.collection_name}")
        
        client = chromadb.HttpClient(
            host=config.chroma.host,
            port=config.chroma.port,
            settings=Settings(allow_reset=True)
        )
        
        heartbeat = client.heartbeat()
        print(f"✅ Config-based connection successful: {heartbeat}")
        
        collections = client.list_collections()
        collection_names = [col.name for col in collections]
        print(f"📋 Available collections: {collection_names}")
        
        if config.chroma.collection_name in collection_names:
            print(f"✅ Target collection '{config.chroma.collection_name}' exists")
        else:
            print(f"⚠️  Target collection '{config.chroma.collection_name}' not found")
            print("💡 You may need to run the embedding pipeline to create it")
        
        return True
        
    except Exception as e:
        print(f"❌ Config-based test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 ChromaDB Connection Diagnostic Tool")
    print("=" * 50)
    
    # Test basic connection
    basic_success = test_chromadb_connection()
    
    # Test with app config
    config_success = test_with_config()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    print(f"Basic Connection: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"Config Connection: {'✅ PASS' if config_success else '❌ FAIL'}")
    
    if basic_success and config_success:
        print("\n🎉 ChromaDB is working correctly!")
        print("💡 The 'Not Found' error in health check might be due to:")
        print("   - Collection doesn't exist yet (run embedding pipeline)")
        print("   - Server needs restart to pick up code changes")
    else:
        print("\n❌ ChromaDB connection issues detected")
        print("🔧 Troubleshooting steps:")
        print("   1. Check if ChromaDB container is running: docker ps")
        print("   2. Check ChromaDB logs: docker logs chromadb")
        print("   3. Restart ChromaDB: docker-compose restart chromadb")
        print("   4. Check port 8000 is not blocked: netstat -an | grep 8000")
    
    sys.exit(0 if (basic_success and config_success) else 1)