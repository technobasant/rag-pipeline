"""
Comprehensive similarity search validation tests for the RAG system.
Tests similarity search functionality, thresholds, and fallback responses.
"""

import pytest
import requests
import json
import time
from typing import Dict, List, Any

# Test configuration
API_BASE_URL = "http://localhost:8001"
TEST_QUERIES = [
    "What is Python?",
    "How to use machine learning?",
    "Database optimization techniques",
    "Web development best practices",
    "Cloud computing fundamentals",
    "This is a completely random query that should not match anything in the knowledge base"
]

class SimilaritySearchValidator:
    """Validates similarity search functionality across different scenarios."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_api_health(self) -> bool:
        """Test if the API is healthy and responsive."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def test_similarity_thresholds(self, query: str, thresholds: List[float] = None) -> Dict[str, Any]:
        """Test similarity search across different thresholds."""
        if thresholds is None:
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        results = {
            "query": query,
            "threshold_results": [],
            "validation_summary": {}
        }
        
        for threshold in thresholds:
            try:
                # Test regular endpoint
                response = self.session.get(
                    f"{self.base_url}/ask",
                    params={
                        "question": query,
                        "top_k": 5,
                        "similarity_threshold": threshold
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    threshold_result = {
                        "threshold": threshold,
                        "status": "success",
                        "context_used": data.get("context_used", False),
                        "relevant_documents": data.get("relevant_documents_count", 0),
                        "total_documents": data.get("total_documents_found", 0),
                        "sources_count": len(data.get("sources", [])),
                        "has_answer": bool(data.get("answer", "")),
                        "fallback_used": data.get("fallback_used", False),
                        "response_length": len(data.get("answer", ""))
                    }
                else:
                    threshold_result = {
                        "threshold": threshold,
                        "status": "error",
                        "error": f"HTTP {response.status_code}"
                    }
                    
            except Exception as e:
                threshold_result = {
                    "threshold": threshold,
                    "status": "exception",
                    "error": str(e)
                }
            
            results["threshold_results"].append(threshold_result)
        
        # Analyze results
        successful_tests = [r for r in results["threshold_results"] if r["status"] == "success"]
        
        if successful_tests:
            results["validation_summary"] = {
                "total_tests": len(thresholds),
                "successful_tests": len(successful_tests),
                "context_usage_rate": sum(1 for r in successful_tests if r.get("context_used", False)) / len(successful_tests),
                "fallback_usage_rate": sum(1 for r in successful_tests if r.get("fallback_used", False)) / len(successful_tests),
                "avg_relevant_docs": sum(r.get("relevant_documents", 0) for r in successful_tests) / len(successful_tests),
                "threshold_effectiveness": self._analyze_threshold_effectiveness(successful_tests)
            }
        
        return results
    
    def test_similarity_validation_endpoint(self, query: str) -> Dict[str, Any]:
        """Test the dedicated similarity validation endpoint."""
        try:
            response = self.session.get(
                f"{self.base_url}/similarity/validate",
                params={
                    "query": query,
                    "test_thresholds": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "top_k": 10
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "data": response.json()
                }
            else:
                return {
                    "status": "error",
                    "error": f"HTTP {response.status_code}",
                    "response": response.text
                }
                
        except Exception as e:
            return {
                "status": "exception",
                "error": str(e)
            }
    
    def test_streaming_with_thresholds(self, query: str, threshold: float = 0.3) -> Dict[str, Any]:
        """Test streaming endpoint with similarity thresholds."""
        try:
            response = self.session.get(
                f"{self.base_url}/ask/stream",
                params={
                    "question": query,
                    "top_k": 5,
                    "similarity_threshold": threshold
                },
                timeout=60,
                stream=True
            )
            
            if response.status_code == 200:
                chunks = []
                metadata = {}
                sources = []
                answer_chunks = []
                
                for line in response.iter_lines(decode_unicode=True):
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            chunks.append(data)
                            
                            if data.get("type") == "metadata":
                                metadata = data
                            elif data.get("type") == "sources":
                                sources = data.get("sources", [])
                            elif data.get("type") == "answer_chunk":
                                answer_chunks.append(data.get("chunk", ""))
                            elif data.get("type") == "answer_complete":
                                metadata.update(data)
                                
                        except json.JSONDecodeError:
                            continue
                
                return {
                    "status": "success",
                    "chunks_received": len(chunks),
                    "metadata": metadata,
                    "sources_count": len(sources),
                    "answer_length": len("".join(answer_chunks)),
                    "context_used": metadata.get("context_used", False),
                    "fallback_used": metadata.get("using_fallback", False)
                }
            else:
                return {
                    "status": "error",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "exception",
                "error": str(e)
            }
    
    def test_search_endpoint(self, query: str, threshold: float = 0.3) -> Dict[str, Any]:
        """Test the enhanced search endpoint."""
        try:
            response = self.session.get(
                f"{self.base_url}/search",
                params={
                    "query": query,
                    "top_k": 10,
                    "similarity_threshold": threshold,
                    "include_scores": True
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "success",
                    "results_count": data.get("count", 0),
                    "total_searched": data.get("total_documents_searched", 0),
                    "filtered_count": data.get("filtered_by_threshold", 0),
                    "threshold": data.get("similarity_threshold", threshold),
                    "has_results": data.get("count", 0) > 0
                }
            else:
                return {
                    "status": "error",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "status": "exception",
                "error": str(e)
            }
    
    def _analyze_threshold_effectiveness(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze the effectiveness of different thresholds."""
        if not results:
            return {}
        
        # Find optimal threshold (highest context usage with reasonable document count)
        context_results = [r for r in results if r.get("context_used", False)]
        
        if context_results:
            optimal = max(context_results, key=lambda x: (x.get("relevant_documents", 0), -x["threshold"]))
            return {
                "optimal_threshold": optimal["threshold"],
                "optimal_relevant_docs": optimal.get("relevant_documents", 0),
                "recommendation": self._get_threshold_recommendation(results)
            }
        else:
            return {
                "optimal_threshold": None,
                "recommendation": "No relevant documents found at any threshold. Consider lowering thresholds or improving document quality."
            }
    
    def _get_threshold_recommendation(self, results: List[Dict]) -> str:
        """Get threshold recommendation based on results."""
        context_usage = sum(1 for r in results if r.get("context_used", False))
        total_results = len(results)
        
        if context_usage == 0:
            return "Lower thresholds recommended - no relevant documents found"
        elif context_usage / total_results > 0.8:
            return "Current thresholds are effective"
        elif context_usage / total_results > 0.5:
            return "Consider slightly lowering thresholds for better coverage"
        else:
            return "Significantly lower thresholds recommended for better document retrieval"

def run_comprehensive_validation():
    """Run comprehensive similarity search validation."""
    validator = SimilaritySearchValidator()
    
    print("🔍 Starting Comprehensive Similarity Search Validation")
    print("=" * 60)
    
    # Test API health
    print("\n1. Testing API Health...")
    if not validator.test_api_health():
        print("❌ API health check failed. Please ensure the API is running.")
        return False
    print("✅ API is healthy and responsive")
    
    # Test each query across different scenarios
    all_results = []
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{i + 1}. Testing Query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        print("-" * 50)
        
        query_results = {
            "query": query,
            "threshold_test": None,
            "validation_endpoint": None,
            "streaming_test": None,
            "search_test": None
        }
        
        # Test similarity thresholds
        print("   📊 Testing similarity thresholds...")
        threshold_results = validator.test_similarity_thresholds(query)
        query_results["threshold_test"] = threshold_results
        
        if threshold_results["validation_summary"]:
            summary = threshold_results["validation_summary"]
            print(f"   ✅ Context usage: {summary['context_usage_rate']:.1%}")
            print(f"   📈 Avg relevant docs: {summary['avg_relevant_docs']:.1f}")
            
            if "optimal_threshold" in summary["threshold_effectiveness"]:
                opt_threshold = summary["threshold_effectiveness"]["optimal_threshold"]
                print(f"   🎯 Optimal threshold: {opt_threshold}")
        
        # Test validation endpoint
        print("   🔬 Testing validation endpoint...")
        validation_result = validator.test_similarity_validation_endpoint(query)
        query_results["validation_endpoint"] = validation_result
        
        if validation_result["status"] == "success":
            data = validation_result["data"]
            print(f"   ✅ Validation endpoint: {data.get('total_documents', 0)} docs analyzed")
            if "recommended_threshold" in data:
                print(f"   💡 Recommended threshold: {data['recommended_threshold']:.2f}")
        
        # Test streaming
        print("   📡 Testing streaming with thresholds...")
        streaming_result = validator.test_streaming_with_thresholds(query, 0.3)
        query_results["streaming_test"] = streaming_result
        
        if streaming_result["status"] == "success":
            print(f"   ✅ Streaming: {streaming_result['chunks_received']} chunks received")
            print(f"   📝 Answer length: {streaming_result['answer_length']} chars")
        
        # Test search endpoint
        print("   🔍 Testing search endpoint...")
        search_result = validator.test_search_endpoint(query, 0.3)
        query_results["search_test"] = search_result
        
        if search_result["status"] == "success":
            print(f"   ✅ Search: {search_result['results_count']}/{search_result['total_searched']} results")
        
        all_results.append(query_results)
        time.sleep(1)  # Brief pause between queries
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY REPORT")
    print("=" * 60)
    
    successful_queries = 0
    total_context_usage = 0
    total_fallback_usage = 0
    
    for result in all_results:
        if result["threshold_test"] and result["threshold_test"]["validation_summary"]:
            successful_queries += 1
            summary = result["threshold_test"]["validation_summary"]
            total_context_usage += summary["context_usage_rate"]
            total_fallback_usage += summary["fallback_usage_rate"]
    
    if successful_queries > 0:
        avg_context_usage = total_context_usage / successful_queries
        avg_fallback_usage = total_fallback_usage / successful_queries
        
        print(f"✅ Successful query tests: {successful_queries}/{len(TEST_QUERIES)}")
        print(f"📊 Average context usage: {avg_context_usage:.1%}")
        print(f"🔄 Average fallback usage: {avg_fallback_usage:.1%}")
        
        # Overall assessment
        if avg_context_usage > 0.7:
            print("🎉 EXCELLENT: High context usage indicates good similarity search performance")
        elif avg_context_usage > 0.4:
            print("👍 GOOD: Moderate context usage, consider threshold optimization")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Low context usage, review thresholds and document quality")
    
    print(f"\n📁 Detailed results available in validation output")
    return True

if __name__ == "__main__":
    success = run_comprehensive_validation()
    exit(0 if success else 1)