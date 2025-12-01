import asyncio
import aiohttp
import time
import argparse
from collections import Counter
from statistics import mean, median
import sys

# Configuration
DEFAULT_URL = "http://localhost:8000/"

async def fetch(session, url, request_id):
    """
    Sends a request and returns the backend ID that served it.
    """
    start_time = time.perf_counter()
    try:
        async with session.get(url) as response:
            content = await response.text()
            status = response.status
            latency = (time.perf_counter() - start_time) * 1000 # ms
            return {
                "status": status,
                "backend": content.strip(),
                "latency": latency,
                "error": None
            }
    except Exception as e:
        return {
            "status": 0,
            "backend": "Error",
            "latency": 0,
            "error": str(e)
        }

async def run_load_test(url, total_requests, concurrency):
    print(f"--- Starting Load Test ---")
    print(f"Target: {url}")
    print(f"Requests: {total_requests}")
    print(f"Concurrency: {concurrency}")
    print("-" * 30)

    # TCPConnector limits the number of open connections
    connector = aiohttp.TCPConnector(force_close=True)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        start_global = time.perf_counter()
        
        # specific batching logic allows us to respect concurrency limits
        # essentially acting as a semaphore
        semaphore = asyncio.Semaphore(concurrency)

        async def sem_fetch(req_id):
            async with semaphore:
                return await fetch(session, url, req_id)

        for i in range(total_requests):
            tasks.append(asyncio.create_task(sem_fetch(i)))

        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_global

    return results, total_time

def analyze_results(results, total_time):
    statuses = Counter(r['status'] for r in results)
    backends = Counter(r['backend'] for r in results)
    latencies = [r['latency'] for r in results if r['status'] == 200]
    
    print(f"\n--- Results ---")
    print(f"Total Time:     {total_time:.2f} seconds")
    print(f"RPS (Throughput): {len(results) / total_time:.2f} req/s")
    print(f"Success Rate:   {(statuses[200] / len(results)) * 100:.2f}%")
    
    print(f"\n--- Latency (ms) ---")
    if latencies:
        print(f"Mean:   {mean(latencies):.2f} ms")
        print(f"Median: {median(latencies):.2f} ms")
        print(f"Max:    {max(latencies):.2f} ms")
    
    print(f"\n--- Load Distribution (Traffic Balance) ---")
    # This proves if Nginx is actually balancing the load
    total_success = len(latencies)
    for backend, count in backends.items():
        percentage = (count / len(results)) * 100
        bar = "#" * int(percentage / 5)
        print(f"{backend:<15}: {count:>5} ({percentage:>5.1f}%) [{bar}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nginx Load Balancer Tester")
    parser.add_argument("--url", default=DEFAULT_URL, help="Target URL")
    parser.add_argument("-n", "--number", type=int, default=1000, help="Total requests")
    parser.add_argument("-c", "--concurrency", type=int, default=50, help="Concurrent requests")
    
    args = parser.parse_args()
    
    try:
        # Run Async Loop
        results, time_taken = asyncio.run(run_load_test(args.url, args.number, args.concurrency))
        analyze_results(results, time_taken)
    except KeyboardInterrupt:
        print("\nTest aborted.")