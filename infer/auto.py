import asyncio
import subprocess
import httpx
import time
from typing import List, Dict

# --- ⚙️ Configuration ---
# File Paths
NGINX_CONF_PATH = "/etc/nginx/nginx.conf"
NGINX_TEMPLATE_PATH = "/etc/nginx/nginx.conf.template"
AUTOSCALER_LOG_FILE = "/var/log/inference_autoscaler.log"

# Scaling Thresholds (based on average (running + waiting) requests per server)
SCALE_DOWN_THRESHOLD = 45
SCALE_UP_THRESHOLD = 60

# Scaling Rules
MIN_ACTIVE_SERVERS = 4
SCALING_COOLDOWN_SECONDS = 30
MONITOR_INTERVAL_SECONDS = 2
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 60

# --- 🖥️ Server State Management ---
# List of all potential inference servers.
# The initial 'status' determines the starting state.
ALL_SERVERS = [
    {"host": "10.10.3.1", "port": 8000, "status": "active"},
    {"host": "10.10.3.1", "port": 8001, "status": "active"},
    {"host": "10.10.3.1", "port": 8002, "status": "active"},
    {"host": "10.10.3.1", "port": 8003, "status": "active"},
    {"host": "10.10.3.2", "port": 8000, "status": "active"},
    {"host": "10.10.3.2", "port": 8001, "status": "active"},
    {"host": "10.10.3.2", "port": 8002, "status": "active"},
    {"host": "10.10.3.2", "port": 8003, "status": "active"},
]

# --- Helper Functions ---

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
    except httpx.RequestError:
        pass # Errors are expected if a server is sleeping
    return {"running": running, "waiting": waiting}

async def update_nginx_config(active_servers: List[Dict]) -> bool:
    """Generates and writes a new nginx.conf from a template."""
    print("\nUpdating Nginx configuration...")
    upstream_servers = "".join([f"        server {s['host']}:{s['port']};\n" for s in active_servers])
    try:
        with open(NGINX_TEMPLATE_PATH, "r") as f:
            template = f.read()
        with open(NGINX_CONF_PATH, "w") as f:
            f.write(template.replace("{UPSTREAM_SERVERS}", upstream_servers))
        print(f"Nginx config updated with {len(active_servers)} active servers.")
        return True
    except Exception as e:
        print(f"\nERROR: Failed to write Nginx config: {e}")
        return False

def reload_nginx():
    """Executes the command to reload Nginx gracefully."""
    print("Reloading Nginx...")
    try:
        subprocess.run(["sudo", "nginx", "-s", "reload"], check=True)
        print("Nginx reloaded successfully.")
    except Exception as e:
        print(f"\nERROR: Failed to reload Nginx: {e}")

async def set_server_sleep_state(server: Dict, sleep: bool):
    """Sends a POST request to put a server to sleep or wake it up."""
    if sleep:
        action, url = "Putting to sleep", f"http://{server['host']}:{server['port']}/sleep?level=1"
    else:
        action, url = "Waking up", f"http://{server['host']}:{server['port']}/wake_up"

    print(f"{action}: {server['host']}:{server['port']}")
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, timeout=20)
    except httpx.RequestError as e:
        print(f"\nERROR: Could not send command to server {server['host']}:{server['port']}: {e}")

def log_event(message: str):
    """Writes a timestamped event to the log file."""
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
    print(log_entry.strip())
    try:
        with open(AUTOSCALER_LOG_FILE, "a") as f:
            f.write(log_entry)
    except IOError as e:
        print(f"ERROR: Could not write to log file {AUTOSCALER_LOG_FILE}: {e}")

# --- Scaling Logic ---

async def scale_down(count: int) -> bool:
    """Gracefully scales down active inference servers."""
    start_time = time.time()
    active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']

    # Determine how many servers can actually be removed
    max_to_remove = len(active_servers) - MIN_ACTIVE_SERVERS
    actual_count = min(count, max_to_remove)

    if actual_count <= 0:
        print("\nScale-down skipped: Minimum server count would be breached.")
        return False

    servers_to_scale_down = active_servers[:actual_count]
    log_event(f"SCALE_DOWN: Attempting to scale down {len(servers_to_scale_down)} server(s).")

    # Step 1: Remove servers from Nginx load balancer
    for server in servers_to_scale_down:
        server['status'] = 'sleeping'

    new_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    if not await update_nginx_config(new_active_servers):
        # On failure, revert the status of the servers
        for server in servers_to_scale_down:
            server['status'] = 'active'
        return False
    reload_nginx()

    # Step 2: Concurrently wait for each server to be idle, then put it to sleep
    async def wait_and_sleep(s):
        print(f"Gracefully shutting down {s['host']}:{s['port']}...")
        wait_start_time = time.time()
        async with httpx.AsyncClient() as client:
            while (time.time() - wait_start_time) < GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS:
                metrics = await get_server_metrics(s, client)
                if metrics.get("running", -1) == 0:
                    print(f"Server {s['host']}:{s['port']} is now idle.")
                    break
                await asyncio.sleep(2)
            else:
                print(f"WARN: Timeout waiting for {s['host']}:{s['port']} to become idle. Forcing sleep.")
        await set_server_sleep_state(s, sleep=True)

    await asyncio.gather(*[wait_and_sleep(s) for s in servers_to_scale_down])

    total_duration = time.time() - start_time
    log_event(f"SCALE_DOWN: Completed scale-down of {actual_count} server(s) in {total_duration:.2f}s.")
    return True

async def scale_up(count: int) -> bool:
    """Wakes up sleeping inference servers and adds them to the load balancer."""
    start_time = time.time()
    sleeping_servers = [s for s in ALL_SERVERS if s['status'] == 'sleeping']

    actual_count = min(count, len(sleeping_servers))
    if actual_count <= 0:
        return False

    servers_to_wake = sleeping_servers[:actual_count]
    log_event(f"SCALE_UP: Attempting to scale up {len(servers_to_wake)} server(s).")

    # Step 1: Wake up the servers
    await asyncio.gather(*[set_server_sleep_state(s, sleep=False) for s in servers_to_wake])

    # Step 2: Update status and add them to Nginx
    for server in servers_to_wake:
        server['status'] = 'active'

    if await update_nginx_config([s for s in ALL_SERVERS if s['status'] == 'active']):
        reload_nginx()
        total_duration = time.time() - start_time
        log_event(f"SCALE_UP: Completed scale-up of {len(servers_to_wake)} server(s) in {total_duration:.2f}s.")
        return True

    # On failure, revert the status of the servers we tried to wake up
    print("ERROR: Nginx update failed during scale-up. Reverting status.")
    for server in servers_to_wake:
        server['status'] = 'sleeping'
    return False

# --- Main Task ---

async def autoscaler_task():
    """The main autoscaler loop that polls server metrics and triggers scaling."""
    print("🚀 Inference Autoscaler started...")
    last_scaling_time = 0
    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
            active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
            if not active_servers:
                print("No active servers found. Waiting...")
                continue

            metric_tasks = [get_server_metrics(server, client) for server in active_servers]
            metric_results = await asyncio.gather(*metric_tasks)

            total_load = sum(r['running'] + r['waiting'] for r in metric_results)
            avg_load_per_server = total_load / len(active_servers)

            # Use carriage return to print on the same line for clean logging
            print(f"\r[{time.strftime('%H:%M:%S')}] Active: {len(active_servers)} | Avg Load/Server: {avg_load_per_server:.2f} | Total Load: {total_load:.0f}", end="")

            if not ((time.time() - last_scaling_time) > SCALING_COOLDOWN_SECONDS):
                continue # Skip scaling if within the cooldown period

            if avg_load_per_server < SCALE_DOWN_THRESHOLD and len(active_servers) > MIN_ACTIVE_SERVERS:
                # Calculate how many servers to remove proportionally
                scale_factor = avg_load_per_server / SCALE_DOWN_THRESHOLD if SCALE_DOWN_THRESHOLD > 0 else 0
                num_to_scale = max(1, int(len(active_servers) * (1 - scale_factor)))
                if await scale_down(count=num_to_scale):
                    last_scaling_time = time.time()
            elif avg_load_per_server > SCALE_UP_THRESHOLD:
                # Calculate how many servers to add proportionally
                scale_factor = avg_load_per_server / SCALE_UP_THRESHOLD if SCALE_UP_THRESHOLD > 0 else 1
                num_to_scale = max(1, int(len(active_servers) * (scale_factor - 1)))
                if await scale_up(count=num_to_scale):
                    last_scaling_time = time.time()


# --- Main Execution ---

if __name__ == "__main__":
    # Ensure Nginx is configured with the initial set of active servers on startup
    initial_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    if not initial_active_servers:
        print("Warning: No servers are initially active. The load balancer will be empty.")
    
    if update_nginx_config(initial_active_servers):
        reload_nginx()
        
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(autoscaler_task())
    except KeyboardInterrupt:
        print("\nAutoscaler stopped by user.")
    finally:
        loop.close()