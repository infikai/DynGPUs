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
SCALE_DOWN_THRESHOLD = 2   # e.g., scale down if avg load per server is below 2
SCALE_UP_THRESHOLD = 10  # e.g., scale up if avg load per server is over 10
MIN_ACTIVE_SERVERS = 2
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

async def scale_down():
    """Scales down by removing one active server."""
    active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    if len(active_servers) <= MIN_ACTIVE_SERVERS:
        print("Scale-down skipped: Minimum number of active servers reached.")
        return

    # Select the last active server to put to sleep
    server_to_sleep = active_servers[-1]
    server_to_sleep['status'] = 'sleeping'
    
    new_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    
    if await update_nginx_config(new_active_servers):
        reload_nginx()
        await set_server_sleep_state(server_to_sleep, sleep=True)

async def scale_up():
    """Scales up by adding one sleeping server."""
    sleeping_servers = [s for s in ALL_SERVERS if s['status'] == 'sleeping']
    if not sleeping_servers:
        print("Scale-up skipped: No sleeping servers available.")
        return
        
    # Select the first sleeping server to wake up
    server_to_wake = sleeping_servers[0]
    server_to_wake['status'] = 'active'
    
    new_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']

    if await update_nginx_config(new_active_servers):
        reload_nginx()
        # --- CHANGE: Add the wake-up call after reloading Nginx ---
        await set_server_sleep_state(server_to_wake, sleep=False)
        # The server is "woken up" by being added back to the load balancer
        # You could add a POST request here if your server needs an explicit wake-up call

async def autoscaler_task():
    """The main autoscaler loop that polls server metrics and triggers scaling."""
    print("🚀 Autoscaler started. Monitoring server metrics...")
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

            # Calculate total load
            total_running = sum(r['running'] for r in metric_results)
            total_waiting = sum(r['waiting'] for r in metric_results)
            total_cluster_load = total_running + total_waiting
            
            # Calculate average load per server
            avg_load_per_server = total_cluster_load / len(active_servers)

            print(
                f"\r[{time.strftime('%H:%M:%S')}] Active Servers: {len(active_servers)} | "
                f"Total Running: {int(total_running)} | "
                f"Total Waiting: {int(total_waiting)} | "
                f"Avg Load/Server: {avg_load_per_server:.2f}", end=""
            )

            # Cooldown logic
            cooldown_over = (time.time() - last_scaling_time) > SCALING_COOLDOWN_SECONDS
            if not cooldown_over:
                continue

            # Scaling decision logic
            if avg_load_per_server < SCALE_DOWN_THRESHOLD:
                print("\nAverage load per server is low. Triggering scale-down...")
                await scale_down()
                last_scaling_time = time.time()
            elif avg_load_per_server > SCALE_UP_THRESHOLD:
                print("\nAverage load per server is high. Triggering scale-up...")
                await scale_up()
                last_scaling_time = time.time()

if __name__ == "__main__":
    if update_nginx_config(ALL_SERVERS):
        reload_nginx()

    loop = asyncio.get_event_loop()
    loop.create_task(autoscaler_task())
    loop.run_forever()