import os
import asyncio
import subprocess
import httpx
import time
import numpy as np
from typing import List, Dict

# --- Configuration ---
NGINX_CONF_PATH = "/etc/nginx/nginx.conf"
NGINX_TEMPLATE_PATH = "/etc/nginx/nginx.conf.template" # We'll use a template

# Adjust these thresholds based on the new metric (running + waiting requests per server)
SCALE_DOWN_THRESHOLD = 50   # e.g., scale down if avg load per server is below 2
SCALE_UP_THRESHOLD = 60  # e.g., scale up if avg load per server is over 10
MIN_ACTIVE_SERVERS = 1
SCALING_COOLDOWN_SECONDS = 30
MONITOR_INTERVAL_SECONDS = 2 # Check metrics every 10 seconds


# --- Server State Management ---
ALL_SERVERS = [
    {"host": "node1", "port": 8000, "status": "active"},
    {"host": "node1", "port": 8001, "status": "active"},
    {"host": "node1", "port": 8002, "status": "active"},
    {"host": "node1", "port": 8003, "status": "active"},
    # {"host": "10.10.3.2", "port": 8000, "status": "active"},
    # {"host": "10.10.3.2", "port": 8001, "status": "active"},
    # {"host": "10.10.3.2", "port": 8002, "status": "active"},
    # {"host": "10.10.3.2", "port": 8003, "status": "active"},
]

# --- NEW FUNCTION: Get metrics from a single server ---
async def get_server_metrics(server: Dict, client: httpx.AsyncClient) -> Dict:
    """Fetches and parses metrics from a vLLM server."""
    url = f"http://{server['host']}:{server['port']}/metrics"
    running = 0.0
    waiting = 0.0
    try:
        response = await client.get(url, timeout=5)
        response.raise_for_status()
        for line in response.text.split('\n'):
            if line.startswith("vllm:num_requests_running"):
                running = float(line.rsplit(' ', 1)[1])
            elif line.startswith("vllm:num_requests_waiting"):
                waiting = float(line.rsplit(' ', 1)[1])
    except httpx.RequestError as e:
        print(f"\nWARN: Could not connect to server {server['host']}:{server['port']} to get metrics: {e}")
    except Exception as e:
        print(f"\nERROR: Failed to parse metrics from {server['host']}:{server['port']}: {e}")
    return {"running": running, "waiting": waiting}

async def update_nginx_config(active_servers: List[Dict]):
    """Generates and writes a new nginx.conf from a template."""
    print("Updating Nginx configuration...")
    
    upstream_servers = ""
    for server in active_servers:
        upstream_servers += f"        server {server['host']}:{server['port']};\n"
        
    try:
        with open(NGINX_TEMPLATE_PATH, "r") as f:
            template = f.read()
            
        new_config = template.replace("{UPSTREAM_SERVERS}", upstream_servers)
        
        with open(NGINX_CONF_PATH, "w") as f:
            f.write(new_config)
            
        print(f"Nginx config updated with {len(active_servers)} active servers.")
        return True
    except FileNotFoundError:
        print(f"ERROR: Nginx template file not found at {NGINX_TEMPLATE_PATH}")
        return False
    except Exception as e:
        print(f"ERROR: Failed to write Nginx config: {e}")
        return False

def reload_nginx():
    """Executes the command to reload Nginx."""
    print("Reloading Nginx...")
    try:
        # This requires passwordless sudo permission for the user running the script
        subprocess.run(["sudo", "nginx", "-s", "reload"], check=True)
        print("Nginx reloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to reload Nginx: {e}")
    except FileNotFoundError:
        print("ERROR: 'sudo' or 'nginx' command not found. Is Nginx installed?")

async def set_server_sleep_state(server: Dict, sleep: bool):
    """Sends a POST request to put a server to sleep or wake it up."""
    level = 1 if sleep else 0
    url = f"http://{server['host']}:{server['port']}/sleep?level={level}"
    action = "Putting to sleep" if sleep else "Waking up"
    print(f"{action}: {server['host']}:{server['port']}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, timeout=5)
            response.raise_for_status()
        print(f"Successfully sent command to {server['host']}:{server['port']}")
    except httpx.RequestError as e:
        print(f"ERROR: Could not connect to server {server['host']}:{server['port']}: {e}")

async def set_server_sleep_state(server: Dict, sleep: bool):
    """Sends a POST request to put a server to sleep or wake it up."""
    if sleep:
        url = f"http://{server['host']}:{server['port']}/sleep?level=1"
        action = "Putting to sleep"
    else:
        # --- CHANGE: This is the new wake-up endpoint ---
        url = f"http://{server['host']}:{server['port']}/wake_up"
        action = "Waking up"

    print(f"{action}: {server['host']}:{server['port']}")
    try:
        async with httpx.AsyncClient() as client:
            # Use POST for both actions as requested
            response = await client.post(url, timeout=5)
            response.raise_for_status()
        print(f"Successfully sent command to {server['host']}:{server['port']}")
    except httpx.RequestError as e:
        print(f"ERROR: Could not connect to server {server['host']}:{server['port']}: {e}")

async def scale_down(count: int) -> bool:
    """Scales down by removing a specified number of active servers."""
    active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    
    # Clamp the count to what's possible to remove
    actual_count = min(count, len(active_servers) - MIN_ACTIVE_SERVERS)
    if actual_count <= 0:
        print("Scale-down skipped: Minimum number of active servers reached.")
        return False

    # Select servers to put to sleep
    servers_to_sleep = active_servers[-actual_count:]
    for server in servers_to_sleep:
        server['status'] = 'sleeping'
    
    new_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    
    if await update_nginx_config(new_active_servers):
        reload_nginx()
        # Concurrently send sleep commands
        sleep_tasks = [set_server_sleep_state(server, sleep=True) for server in servers_to_sleep]
        await asyncio.gather(*sleep_tasks)
        return True
    return False

async def scale_up(count: int) -> bool:
    """Scales up by adding a specified number of sleeping servers."""
    sleeping_servers = [s for s in ALL_SERVERS if s['status'] == 'sleeping']
    
    # Clamp the count to the number of available sleeping servers
    actual_count = min(count, len(sleeping_servers))
    if actual_count <= 0:
        print("Scale-up skipped: No sleeping servers available.")
        return False
        
    # Select servers to wake up
    servers_to_wake = sleeping_servers[:actual_count]
    for server in servers_to_wake:
        server['status'] = 'active'
    
    new_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']

    if await update_nginx_config(new_active_servers):
        reload_nginx()
        # Concurrently send wake-up commands
        wake_tasks = [set_server_sleep_state(server, sleep=False) for server in servers_to_wake]
        await asyncio.gather(*wake_tasks)
        return True
    return False

async def autoscaler_task():
    """The main autoscaler loop that polls server metrics and triggers proportional scaling."""
    print("🚀 Autoscaler started. Monitoring server metrics for proportional scaling...")
    last_scaling_time = 0

    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(MONITOR_INTERVAL_SECONDS)

            active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
            if not active_servers:
                print("\nNo active servers. Waiting...")
                continue

            # Concurrently fetch metrics from all active servers
            metric_tasks = [get_server_metrics(server, client) for server in active_servers]
            metric_results = await asyncio.gather(*metric_tasks)

            total_running = sum(r['running'] for r in metric_results)
            total_waiting = sum(r['waiting'] for r in metric_results)
            total_cluster_load = total_running + total_waiting
            avg_load_per_server = total_cluster_load / len(active_servers)

            print(
                f"\r[{time.strftime('%H:%M:%S')}] Active Servers: {len(active_servers)} | "
                f"Avg Load/Server: {avg_load_per_server:.2f}", end=""
            )

            cooldown_over = (time.time() - last_scaling_time) > SCALING_COOLDOWN_SECONDS
            if not cooldown_over:
                continue

            # --- NEW: Proportional Scaling Logic ---
            if avg_load_per_server < SCALE_DOWN_THRESHOLD:
                # Calculate how many servers to remove
                # The further below the threshold, the more we scale down
                load_ratio = avg_load_per_server / SCALE_DOWN_THRESHOLD
                num_to_remove = int(len(active_servers) * (1 - load_ratio))
                # Ensure we scale down at least one
                num_to_remove = max(1, num_to_remove)

                print(f"\nAverage load per server is low. Triggering scale-down of {num_to_remove} server(s)...")
                if await scale_down(count=num_to_remove):
                    last_scaling_time = time.time()

            elif avg_load_per_server > SCALE_UP_THRESHOLD:
                # Calculate how many servers to add
                # The further above the threshold, the more we scale up
                load_ratio = avg_load_per_server / SCALE_UP_THRESHOLD
                num_to_add = int(len(active_servers) * (load_ratio - 1))
                # Ensure we scale up at least one
                num_to_add = max(1, num_to_add)

                print(f"\nAverage load per server is high. Triggering scale-up of {num_to_add} server(s)...")
                if await scale_up(count=num_to_add):
                    last_scaling_time = time.time()

if __name__ == "__main__":
    if update_nginx_config(ALL_SERVERS):
        reload_nginx()

    loop = asyncio.get_event_loop()
    loop.create_task(autoscaler_task())
    loop.run_forever()