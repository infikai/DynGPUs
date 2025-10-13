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
SERVER_COUNT_LOG_FILE = "//mydata/Data/DynGPUs/sys/active_servers.log" # --- NEW: Log file path ---

# Adjust these thresholds based on the new metric (running + waiting requests per server)
SCALE_DOWN_THRESHOLD = 50   # e.g., scale down if avg load per server is below 2
SCALE_UP_THRESHOLD = 60  # e.g., scale up if avg load per server is over 10
MIN_ACTIVE_SERVERS = 1
SCALING_COOLDOWN_SECONDS = 30
MONITOR_INTERVAL_SECONDS = 2 # Check metrics every 10 seconds
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 120 # --- NEW: Max time to wait for a server to finish requests ---


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

async def log_active_servers():
    """Logs the number of active servers to a file every second."""
    print(f"📝 Logging active server count to {SERVER_COUNT_LOG_FILE} every second...")
    while True:
        try:
            active_server_count = sum(1 for s in ALL_SERVERS if s['status'] == 'active')
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"{timestamp}, {active_server_count}\n"
            
            with open(SERVER_COUNT_LOG_FILE, "a") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"\nERROR: Could not write to log file: {e}")
            
        await asyncio.sleep(1)

# --- Core Functions ---

async def get_server_metrics(server: Dict, client: httpx.AsyncClient) -> Dict:
    """Fetches and parses metrics from a vLLM server's /metrics endpoint."""
    url = f"http://{server['host']}:{server['port']}/metrics"
    running, waiting = 0.0, 0.0
    try:
        response = await client.get(url, timeout=5)
        response.raise_for_status()
        for line in response.text.split('\n'):
            if line.startswith("vllm:num_requests_running"):
                running = float(line.rsplit(' ', 1)[1])
            elif line.startswith("vllm:num_requests_waiting"):
                waiting = float(line.rsplit(' ', 1)[1])
    except httpx.RequestError as e:
        print(f"\nWARN: Could not connect to server {server['host']}:{server['port']} for metrics: {e}")
    except Exception as e:
        print(f"\nERROR: Failed to parse metrics from {server['host']}:{server['port']}: {e}")
    return {"running": running, "waiting": waiting}

async def update_nginx_config(active_servers: List[Dict]):
    """Generates and writes a new nginx.conf from a template."""
    print("\nUpdating Nginx configuration...")
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
    except Exception as e:
        print(f"ERROR: Failed to write Nginx config: {e}")
        return False

def reload_nginx():
    """Executes the command to reload Nginx gracefully."""
    print("Reloading Nginx...")
    try:
        subprocess.run(["sudo", "nginx", "-s", "reload"], check=True)
        print("Nginx reloaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to reload Nginx: {e}")

async def set_server_sleep_state(server: Dict, sleep: bool):
    """Sends a POST request to put a server to sleep or wake it up."""
    action, url = ("Putting to sleep", f"http://{server['host']}:{server['port']}/sleep?level=1") if sleep else \
                  ("Waking up", f"http://{server['host']}:{server['port']}/wake_up")
    
    print(f"{action}: {server['host']}:{server['port']}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, timeout=20)
            response.raise_for_status()
        print(f"Successfully sent command to {server['host']}:{server['port']}")
    except httpx.RequestError as e:
        print(f"ERROR: Could not send command to server {server['host']}:{server['port']}: {e}")

async def scale_down(count: int) -> bool:
    """
    Scales down gracefully and logs the total duration of the process.
    """
    start_time = time.time() # --- Start timing the full process ---
    
    active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    actual_count = min(count, len(active_servers) - MIN_ACTIVE_SERVERS)
    if actual_count <= 0:
        print("\nScale-down skipped: Minimum number of active servers would be breached.")
        return False

    servers_to_scale_down = active_servers[-actual_count:]
    
    for server in servers_to_scale_down:
        server['status'] = 'sleeping'
    
    new_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    if not await update_nginx_config(new_active_servers):
        for server in servers_to_scale_down:
            server['status'] = 'active'
        return False

    reload_nginx()
    
    async with httpx.AsyncClient() as client:
        shutdown_tasks = []
        for server in servers_to_scale_down:
            async def wait_and_sleep(s):
                print(f"\nGracefully shutting down {s['host']}:{s['port']}. Waiting for running requests to finish...")
                wait_start_time = time.time()
                while (time.time() - wait_start_time) < GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS:
                    metrics = await get_server_metrics(s, client)
                    if metrics.get("running", -1) == 0:
                        print(f"Server {s['host']}:{s['port']} is now idle.")
                        break
                    await asyncio.sleep(2)
                else:
                    print(f"\nWARN: Timeout reached waiting for {s['host']}:{s['port']} to become idle. Forcing sleep.")
                
                await set_server_sleep_state(s, sleep=True)

            shutdown_tasks.append(wait_and_sleep(server))
        
        await asyncio.gather(*shutdown_tasks)

    # --- CHANGE: Log the total process duration ---
    total_duration = time.time() - start_time
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SCALE_DOWN: Full scale-down of {actual_count} server(s) took {total_duration:.2f}s.\n"
    with open(SERVER_COUNT_LOG_FILE, "a") as f:
        f.write(log_entry)

    return True


async def scale_up(count: int) -> bool:
    """Wakes up servers, adds them to Nginx, and logs the total duration of the process."""
    start_time = time.time() # --- Start timing the full process ---

    sleeping_servers = [s for s in ALL_SERVERS if s['status'] == 'sleeping']
    actual_count = min(count, len(sleeping_servers))
    if actual_count <= 0:
        print("\nScale-up skipped: No sleeping servers available.")
        return False
        
    servers_to_wake = sleeping_servers[:actual_count]
    
    print(f"\nWaking up {len(servers_to_wake)} server(s)...")
    wake_tasks = [set_server_sleep_state(server, sleep=False) for server in servers_to_wake]
    await asyncio.gather(*wake_tasks)
    
    print("Server(s) reported ready. Updating Nginx...")
    for server in servers_to_wake:
        server['status'] = 'active'
    
    new_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    if await update_nginx_config(new_active_servers):
        reload_nginx()
        
        # --- CHANGE: Log the total process duration ---
        total_duration = time.time() - start_time
        log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SCALE_UP: Full scale-up of {actual_count} server(s) took {total_duration:.2f}s.\n"
        with open(SERVER_COUNT_LOG_FILE, "a") as f:
            f.write(log_entry)
            
        return True
    
    print("ERROR: Nginx update failed. Reverting server status.")
    for server in servers_to_wake:
        server['status'] = 'sleeping'
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

            metric_tasks = [get_server_metrics(server, client) for server in active_servers]
            metric_results = await asyncio.gather(*metric_tasks)

            total_running = sum(r['running'] for r in metric_results)
            total_waiting = sum(r['waiting'] for r in metric_results)
            total_cluster_load = total_running + total_waiting
            avg_load_per_server = total_cluster_load / len(active_servers)

            print(
                f"\r[{time.strftime('%H:%M:%S')}] Active Servers: {len(active_servers)} | "
                f"Total Running: {int(total_running)} | "
                f"Total Waiting: {int(total_waiting)} | "
                f"Avg Load/Server: {avg_load_per_server:.2f}", end=""
            )

            cooldown_over = (time.time() - last_scaling_time) > SCALING_COOLDOWN_SECONDS
            if not cooldown_over:
                continue

            if avg_load_per_server < SCALE_DOWN_THRESHOLD:
                load_ratio = avg_load_per_server / SCALE_DOWN_THRESHOLD if SCALE_DOWN_THRESHOLD > 0 else 0
                num_to_remove = int(len(active_servers) * (1 - load_ratio))
                num_to_remove = max(1, num_to_remove)
                if await scale_down(count=num_to_remove):
                    last_scaling_time = time.time()

            elif avg_load_per_server > SCALE_UP_THRESHOLD:
                load_ratio = avg_load_per_server / SCALE_UP_THRESHOLD if SCALE_UP_THRESHOLD > 0 else 1
                num_to_add = int(len(active_servers) * (load_ratio - 1))
                num_to_add = max(1, num_to_add)
                if await scale_up(count=num_to_add):
                    last_scaling_time = time.time()

if __name__ == "__main__":
    # Initialize the Nginx config with all servers active at the start
    if update_nginx_config(ALL_SERVERS):
        reload_nginx()

    # Start the main autoscaler loop
    try:
        asyncio.run(autoscaler_task())
    except KeyboardInterrupt:
        print("\nAutoscaler stopped by user.")