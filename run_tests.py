#!/usr/bin/env python3
"""
Automated test runner for RAG system evaluation
Runs 20 queries and measures real metrics
"""
import json
import requests
import time
from datetime import datetime
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000/api/v1/chat"
TEST_FILE = "test_queries.json"
RESULTS_DIR = Path("test_results")
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def load_test_queries():
    """Load test queries from JSON file"""
    with open(TEST_FILE, 'r') as f:
        return json.load(f)

def run_single_query(test_case):
    """Execute a single test query"""
    start_time = time.time()
    
    payload = {
        "session_id": f"test_{test_case['id']}",
        "message": test_case['question']
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'latency': latency,
                'answer': result.get('answer', ''),
                'sources': result.get('sources', []),
                'confidence': result.get('confidence', 0),
                'error': None
            }
        else:
            return {
                'success': False,
                'latency': latency,
                'answer': None,
                'sources': [],
                'confidence': 0,
                'error': f"HTTP {response.status_code}: {response.text[:200]}"
            }
    
    except Exception as e:
        latency = time.time() - start_time
        return {
            'success': False,
            'latency': latency,
            'answer': None,
            'sources': [],
            'confidence': 0,
            'error': str(e)
        }

def evaluate_result(test_case, result):
    """
    Evaluate the quality of the result
    Returns scores for different metrics
    """
    scores = {
        'retrieved_correct_document': False,
        'has_citation': False,
        'has_answer': False,
        'latency_acceptable': result['latency'] < 30.0,  # 30 seconds max
        'answer_not_hallucinated': True  # Assume true unless we detect issues
    }
    
    if not result['success']:
        return scores
    
    # Check if answer exists and is substantial
    scores['has_answer'] = bool(result['answer'] and len(result['answer']) > 50)
    
    # Check if citation exists
    scores['has_citation'] = len(result['sources']) > 0
    
    # Check if correct document was retrieved
    if test_case['expected_document'] != 'multiple' and result['sources']:
        for source in result['sources']:
            if test_case['expected_document'] in source.get('source_file', ''):
                scores['retrieved_correct_document'] = True
                break
    elif test_case['expected_document'] == 'multiple':
        # For multi-document queries, just check if we got sources from multiple docs
        unique_docs = set(s.get('source_file', '') for s in result['sources'])
        scores['retrieved_correct_document'] = len(unique_docs) >= 2
    
    # Check for hallucination indicators
    if "I don't have" in result['answer'] or "not available" in result['answer']:
        # System correctly saying it doesn't know
        scores['answer_not_hallucinated'] = True
    
    return scores

def calculate_metrics(results):
    """Calculate aggregate metrics"""
    total = len(results)
    if total == 0:
        return {}
    
    successful = sum(1 for r in results if r['result']['success'])
    
    # Calculate individual metrics
    correct_docs = sum(1 for r in results if r['scores']['retrieved_correct_document'])
    has_citation = sum(1 for r in results if r['scores']['has_citation'])
    has_answer = sum(1 for r in results if r['scores']['has_answer'])
    acceptable_latency = sum(1 for r in results if r['scores']['latency_acceptable'])
    
    # Latency metrics (only for successful queries)
    latencies = [r['result']['latency'] for r in results if r['result']['success']]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    
    # Confidence metrics
    confidences = [r['result']['confidence'] for r in results if r['result']['success']]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        'total_queries': total,
        'successful_queries': successful,
        'success_rate': successful / total,
        'document_retrieval_accuracy': correct_docs / total,
        'citation_rate': has_citation / total,
        'answer_generation_rate': has_answer / total,
        'acceptable_latency_rate': acceptable_latency / total,
        'avg_latency_seconds': avg_latency,
        'p95_latency_seconds': p95_latency,
        'avg_confidence': avg_confidence
    }

def print_progress_bar(current, total, bar_length=40):
    """Print a progress bar"""
    percent = float(current) * 100 / total
    arrow = '█' * int(percent/100 * bar_length - 1)
    spaces = ' ' * (bar_length - len(arrow))
    print(f'\rProgress: [{arrow}{spaces}] {current}/{total} ({percent:.1f}%)', end='', flush=True)

def main():
    """Run all tests and generate report"""
    print("=" * 70)
    print("RAG SYSTEM EVALUATION - PHASE 3")
    print("=" * 70)
    print()
    
    # Load test queries
    test_queries = load_test_queries()
    print(f"Loaded {len(test_queries)} test queries")
    print()
    
    # Run all tests
    results = []
    print("Running tests...")
    print()
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] {test_case['id']}: {test_case['question'][:60]}...")
        
        result = run_single_query(test_case)
        scores = evaluate_result(test_case, result)
        
        # Combine everything
        full_result = {
            'test_case': test_case,
            'result': result,
            'scores': scores
        }
        results.append(full_result)
        
        # Print result
        if result['success']:
            print(f"  ✓ Success (latency: {result['latency']:.2f}s, confidence: {result['confidence']:.2f})")
            print(f"    Retrieved from: {', '.join(set(s.get('source_file', 'unknown') for s in result['sources'][:3]))}")
        else:
            print(f"  ✗ Failed: {result['error'][:100]}")
        
        # Be nice to the API - small delay between queries
        if i < len(test_queries):
            time.sleep(1)
    
    # Calculate aggregate metrics
    print("\n")
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    metrics = calculate_metrics(results)
    
    print(f"\nTotal Queries:              {metrics['total_queries']}")
    print(f"Successful:                 {metrics['successful_queries']} ({metrics['success_rate']*100:.1f}%)")
    print()
    print(f"Document Retrieval Accuracy: {metrics['document_retrieval_accuracy']*100:.1f}%")
    print(f"Citation Rate:              {metrics['citation_rate']*100:.1f}%")
    print(f"Answer Generation Rate:     {metrics['answer_generation_rate']*100:.1f}%")
    print(f"Acceptable Latency Rate:    {metrics['acceptable_latency_rate']*100:.1f}%")
    print()
    print(f"Average Latency:            {metrics['avg_latency_seconds']:.2f}s")
    print(f"P95 Latency:                {metrics['p95_latency_seconds']:.2f}s")
    print(f"Average Confidence:         {metrics['avg_confidence']:.2f}")
    print()
    
    # Category breakdown
    print("BREAKDOWN BY CATEGORY:")
    categories = {}
    for r in results:
        cat = r['test_case']['category']
        if cat not in categories:
            categories[cat] = {'total': 0, 'success': 0, 'correct_doc': 0}
        categories[cat]['total'] += 1
        if r['result']['success']:
            categories[cat]['success'] += 1
        if r['scores']['retrieved_correct_document']:
            categories[cat]['correct_doc'] += 1
    
    for cat, stats in sorted(categories.items()):
        print(f"  {cat}:")
        print(f"    Success: {stats['success']}/{stats['total']} ({stats['success']/stats['total']*100:.1f}%)")
        print(f"    Correct retrieval: {stats['correct_doc']}/{stats['total']} ({stats['correct_doc']/stats['total']*100:.1f}%)")
    
    # Save results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': metrics,
        'category_breakdown': categories,
        'detailed_results': results
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print()
    print(f"Detailed results saved to: {RESULTS_FILE}")
    print()
    print("=" * 70)
    print("NEXT STEPS:")
    print("1. Review the detailed results JSON file")
    print("2. Manually assess 5 sample queries for answer quality")
    print("3. Document failure modes for your article")
    print("4. Calculate total costs from OpenAI usage dashboard")
    print("=" * 70)

if __name__ == '__main__':
    main()
