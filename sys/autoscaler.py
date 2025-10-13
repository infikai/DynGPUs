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

# Define a path for the shared file
CONCURRENCY_FILE_PATH = "/mydata/Data/DynGPUs/sys/vllm_concurrency.txt"

# Concurrency thresholds for scaling
SCALE_DOWN_THRESHOLD = 12  # e.g., scale down if avg concurrency is below 10
SCALE_UP_THRESHOLD = 17    # e.g., scale up if avg concurrency is over 50
MIN_ACTIVE_SERVERS = 2     # Always keep at least 2 servers active
AUTOSCALER_INTERVAL_SECONDS = 15 # Check concurrency every 15 seconds

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

# This list would be populated by your benchmark client in a real scenario.
# For this standalone script, we will simulate it.
mock_concurrency_list = []

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
    """The main autoscaler loop that reads concurrency from a file and triggers scaling."""
    print("🚀 Autoscaler started. Monitoring per-server concurrency...")
    # This list will store the calculated per-server concurrency readings
    concurrency_readings = []
    
    while True:
        await asyncio.sleep(AUTOSCALER_INTERVAL_SECONDS)
        
        try:
            # --- CHANGE: Calculate active servers first ---
            active_server_count = sum(1 for s in ALL_SERVERS if s['status'] == 'active')
            if active_server_count == 0:
                print("No active servers. Waiting...")
                continue

            # Read the latest TOTAL concurrency value from the shared file
            if not os.path.exists(CONCURRENCY_FILE_PATH):
                continue

            with open(CONCURRENCY_FILE_PATH, "r") as f:
                total_concurrency = int(f.read().strip())
            
            # --- CHANGE: Calculate concurrency PER SERVER ---
            # This is the key change to normalize the metric.
            concurrency_per_server = total_concurrency / active_server_count
            
            concurrency_readings.append(concurrency_per_server)
            
            # Use the average of the last 5 readings to smooth out spikes
            last_n_readings = concurrency_readings[-5:]
            avg_concurrency = np.mean(last_n_readings)
            
            print(
                f"[{time.strftime('%H:%M:%S')}] Active servers: {active_server_count}. "
                f"Total concurrent reqs: {total_concurrency}. "
                f"Avg reqs/server: {avg_concurrency:.2f}"
            )

            # --- CHANGE: Use per-server average for scaling logic ---
            if avg_concurrency < SCALE_DOWN_THRESHOLD:
                print("Average load per server is low. Triggering scale-down...")
                await scale_down()
            elif avg_concurrency > SCALE_UP_THRESHOLD:
                print("Average load per server is high. Triggering scale-up...")
                await scale_up()

        except (FileNotFoundError, ValueError):
            print("Waiting for benchmark client to write data...")
            continue
        except Exception as e:
            print(f"An error occurred in the autoscaler loop: {e}")

if __name__ == "__main__":
    if update_nginx_config(ALL_SERVERS):
        reload_nginx()

    loop = asyncio.get_event_loop()
    loop.create_task(autoscaler_task())
    loop.run_forever()