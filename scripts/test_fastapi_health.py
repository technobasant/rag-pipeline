#!/usr/bin/env python3
"""
Test script that mimics the exact FastAPI health check logic.
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_fastapi_health_logic():
    """Test the exact same logic used in FastAPI health check."""
    
    print("🔍 Testing FastAPI Health Check Logic...")
    print("=" * 50)
    
    try:
        # Import exactly like FastAPI does
        from src.api.main import get_embedding_pipeline, config
        
        print("✅ Imports successful")
        
        # Test the exact function used in health check
        def check_chroma():
            try:
                print("  📍 Getting embedding pipeline...")
                pipeline = get_embedding_pipeline()
                print("  ✅ Pipeline obtained")
                
                print("  📍 Connecting to vector store...")
                pipeline.vector_store.connect()
                print("  ✅ Connected to vector store")
                
                print("  📍 Getting collection info...")
                collection_info = pipeline.vector_store.get_collection_info()
                print(f"  ✅ Collection info: {collection_info}")
                
                return {"status": "healthy", "info": collection_info}
            except Exception as e:
                print(f"  ❌ Error in check_chroma: {e}")
                print(f"  🔍 Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                
                # Try fallback like in health check
                try:
                    print("  📍 Trying fallback connection...")
                    import chromadb
                    from chromadb.config import Settings
                    client = chromadb.HttpClient(
                        host=config.chroma.host,
                        port=config.chroma.port,
                        settings=Settings(allow_reset=True)
                    )
                    client.heartbeat()
                    return {"status": "connected_no_collection", "error": str(e)}
                except Exception as conn_e:
                    print(f"  ❌ Fallback also failed: {conn_e}")
                    return {"status": "connection_failed", "error": str(conn_e)}
        
        # Test synchronous execution
        print("\n1️⃣ Testing synchronous execution...")
        sync_result = check_chroma()
        print(f"✅ Sync result: {sync_result}")
        
        # Test async execution like FastAPI does
        print("\n2️⃣ Testing async execution (like FastAPI)...")
        async_result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, check_chroma),
            timeout=5.0
        )
        print(f"✅ Async result: {async_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_direct_health_endpoint():
    """Test calling the health endpoint directly."""
    print("\n" + "=" * 50)
    print("🔍 Testing Direct Health Endpoint Call...")
    print("=" * 50)
    
    try:
        import httpx
        
        print("📍 Making HTTP request to /health endpoint...")
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://localhost:8001/health")
            
        print(f"✅ Response status: {response.status_code}")
        print(f"📄 Response content: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Parsed data: {data}")
            return True
        else:
            print(f"❌ Non-200 status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ HTTP request failed: {e}")
        print("💡 Make sure FastAPI server is running on localhost:8001")
        return False

async def main():
    print("🧪 FastAPI Health Check Diagnostic Tool")
    print("=" * 50)
    
    # Test the health check logic
    logic_success = await test_fastapi_health_logic()
    
    # Test the actual endpoint
    endpoint_success = await test_direct_health_endpoint()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    print(f"Health Logic Test: {'✅ PASS' if logic_success else '❌ FAIL'}")
    print(f"Endpoint Test: {'✅ PASS' if endpoint_success else '❌ FAIL'}")
    
    if logic_success and not endpoint_success:
        print("\n🔍 DIAGNOSIS: Logic works but endpoint fails")
        print("💡 Possible causes:")
        print("   - FastAPI server not running")
        print("   - Server running old code (needs restart)")
        print("   - Port conflict or binding issue")
    elif not logic_success:
        print("\n🔍 DIAGNOSIS: Health check logic has issues")
        print("💡 Need to fix the health check implementation")
    else:
        print("\n🎉 Everything working correctly!")

if __name__ == "__main__":
    asyncio.run(main())