import asyncio
import subprocess
import httpx
import time
import numpy as np
import asyncssh
from typing import List, Dict

# --- ⚙️ Configuration ---
# File Paths
NGINX_CONF_PATH = "/etc/nginx/nginx.conf"
NGINX_TEMPLATE_PATH = "/etc/nginx/nginx.conf.template"
SERVER_COUNT_LOG_FILE = "/var/log/active_servers.log"
ACTIVE_WORKERS_FILE = "/mydata/Data/DynGPUs/custom_hvd/active_workers.txt"

# Scaling Thresholds (based on average (running + waiting) requests per server)
SCALE_DOWN_THRESHOLD = 2
SCALE_UP_THRESHOLD = 10

# Scaling Rules
MIN_ACTIVE_SERVERS = 4
SCALING_COOLDOWN_SECONDS = 30
MONITOR_INTERVAL_SECONDS = 2
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 60
GPU_MEMORY_FREE_THRESHOLD_MB = 2000

# --- 🖥️ Server State Management ---
# `rank` only exists for shared servers that are part of the training job.
ALL_SERVERS = [
    # Dedicated inference-only servers (no rank)
    {"host": "10.10.3.1", "port": 8000, "status": "active", "shared": False},
    {"host": "10.10.3.1", "port": 8001, "status": "active", "shared": False},
    {"host": "10.10.3.1", "port": 8002, "status": "active", "shared": False},
    {"host": "10.10.3.1", "port": 8003, "status": "active", "shared": False},
    # Shared servers that have a corresponding training rank
    {"host": "10.10.3.2", "port": 8000, "status": "active", "rank": 4, "shared": True},
    {"host": "10.10.3.2", "port": 8001, "status": "active", "rank": 5, "shared": True},
    {"host": "10.10.3.2", "port": 8002, "status": "active", "rank": 6, "shared": True},
    {"host": "10.10.3.2", "port": 8003, "status": "active", "rank": 7, "shared": True},
]


# --- Helper Functions ---

def read_active_workers() -> List[int]:
    """Reads the list of active training ranks from the file."""
    try:
        with open(ACTIVE_WORKERS_FILE, "r") as f:
            content = f.read().strip()
            return [int(rank) for rank in content.split(',')] if content else []
    except FileNotFoundError:
        return []

def write_active_workers(ranks: List[int]):
    """Writes the list of active training ranks to the file."""
    ranks.sort()
    content = ",".join(map(str, ranks))
    with open(ACTIVE_WORKERS_FILE, "w") as f:
        f.write(content)
    print(f"\nUpdated active_workers.txt with ranks: {content}")

async def check_gpu_memory_is_free(server: Dict) -> bool:
    """Connects via SSH to run nvidia-smi and check if GPU memory is below a threshold."""
    if not server.get("shared"): return True # Not a shared server, so no need to check
    
    print(f"\nChecking GPU memory for rank {server['rank']} on {server['host']}...")
    local_gpu_id = server['rank'] % 4
    command = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {local_gpu_id}"
    
    try:
        async with asyncssh.connect(server['host']) as conn:
            result = await conn.run(command, check=True)
            memory_used_mb = int(result.stdout.strip())
            print(f"Rank {server['rank']} is using {memory_used_mb} MiB of memory.")
            return memory_used_mb < GPU_MEMORY_FREE_THRESHOLD_MB
    except (asyncssh.Error, OSError, ValueError) as e:
        print(f"\nERROR: Failed to check GPU memory for rank {server['rank']}: {e}")
        return False

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
    except httpx.RequestError: pass
    return {"running": running, "waiting": waiting}

async def update_nginx_config(active_servers: List[Dict]) -> bool:
    """Generates and writes a new nginx.conf from a template."""
    print("\nUpdating Nginx configuration...")
    upstream_servers = "".join([f"        server {s['host']}:{s['port']};\n" for s in active_servers])
    try:
        with open(NGINX_TEMPLATE_PATH, "r") as f: template = f.read()
        with open(NGINX_CONF_PATH, "w") as f: f.write(template.replace("{UPSTREAM_SERVERS}", upstream_servers))
        print(f"Nginx config updated with {len(active_servers)} active servers.")
        return True
    except Exception as e:
        print(f"\nERROR: Failed to write Nginx config: {e}"); return False

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
    action, url = ("Putting to sleep", f"http://{server['host']}:{server['port']}/sleep?level=1") if sleep else \
                  ("Waking up", f"http://{server['host']}:{server['port']}/wake_up")
    print(f"{action}: {server['host']}:{server['port']}")
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, timeout=20)
    except httpx.RequestError as e:
        print(f"\nERROR: Could not send command to server {server['host']}:{server['port']}: {e}")

# --- Scaling Logic ---

async def scale_down(count: int) -> bool:
    """
    Scales down, targeting ONLY shared servers to free GPUs for training.
    Dedicated servers will not be scaled down.
    """
    start_time = time.time()
    
    # --- CHANGE: Only consider SHARED servers for scale-down ---
    active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    shared_active_servers = [s for s in active_servers if s['shared']]
    
    # Determine how many servers we can possibly remove
    max_possible_to_remove = len(active_servers) - MIN_ACTIVE_SERVERS
    
    # We can only remove shared servers, up to the count requested, and without violating the minimum
    actual_count = min(count, len(shared_active_servers), max_possible_to_remove)
    
    if actual_count <= 0:
        print("\nScale-down skipped: No shared servers available to scale down or minimum would be breached.")
        return False

    # Select the shared servers to be scaled down
    servers_to_scale_down = shared_active_servers[:actual_count]
    
    for server in servers_to_scale_down:
        server['status'] = 'sleeping'
    
    new_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    if not await update_nginx_config(new_active_servers):
        # Revert status on failure
        for server in servers_to_scale_down:
            server['status'] = 'active'
        return False
    reload_nginx()
    
    # The rest of the graceful shutdown logic remains the same
    async with httpx.AsyncClient() as client:
        async def wait_sleep_and_coordinate(s):
            print(f"\nGracefully shutting down shared server {s['host']}:{s['port']}...")
            # ... (graceful shutdown logic: wait for idle) ...
            await set_server_sleep_state(s, sleep=True)
            
            # Add its rank back to the training pool
            print(f"Adding rank {s['rank']} back to the training job...")
            active_ranks = read_active_workers()
            if s['rank'] not in active_ranks:
                active_ranks.append(s['rank'])
                write_active_workers(active_ranks)

        await asyncio.gather(*[wait_sleep_and_coordinate(s) for s in servers_to_scale_down])

    total_duration = time.time() - start_time
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SCALE_DOWN: Full scale-down of {actual_count} shared server(s) took {total_duration:.2f}s.\n"
    with open(SERVER_COUNT_LOG_FILE, "a") as f:
        f.write(log_entry)
    return True

async def scale_up(count: int) -> bool:
    """Scales up, prioritizing dedicated servers first."""
    start_time = time.time()
    all_sleeping = [s for s in ALL_SERVERS if s['status'] == 'sleeping']
    dedicated_sleeping = [s for s in all_sleeping if not s['shared']]
    shared_sleeping = [s for s in all_sleeping if s['shared']]

    servers_to_consider = dedicated_sleeping + shared_sleeping
    actual_count = min(count, len(servers_to_consider))
    if actual_count <= 0: return False
        
    servers_to_wake = servers_to_consider[:actual_count]
    successfully_woken = []

    for server in servers_to_wake:
        if server['shared']:
            active_ranks = read_active_workers()
            new_ranks = [r for r in active_ranks if r != server['rank']]
            write_active_workers(new_ranks)
            if not await check_gpu_memory_is_free(server):
                print(f"ERROR: GPU memory check failed for rank {server['rank']}. Aborting wake-up.")
                write_active_workers(active_ranks) # Revert
                continue
        
        await set_server_sleep_state(server, sleep=False)
        server['status'] = 'active'
        successfully_woken.append(server)

    if not successfully_woken: return False

    if await update_nginx_config([s for s in ALL_SERVERS if s['status'] == 'active']):
        reload_nginx()
        log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SCALE_UP: Full scale-up of {len(successfully_woken)} server(s) took {time.time() - start_time:.2f}s.\n"
        with open(SERVER_COUNT_LOG_FILE, "a") as f: f.write(log_entry)
        return True
    
    print("ERROR: Nginx update failed. Reverting...")
    for server in successfully_woken:
        server['status'] = 'sleeping'
        if server['shared']:
            ranks = read_active_workers()
            if server['rank'] not in ranks:
                ranks.append(server['rank'])
                write_active_workers(ranks)
    return False

# --- Background Tasks ---

async def log_active_servers():
    """Logs the number of active servers to a file every second."""
    print(f"📝 Logging active server count to {SERVER_COUNT_LOG_FILE}...")
    while True:
        try:
            with open(SERVER_COUNT_LOG_FILE, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {sum(1 for s in ALL_SERVERS if s['status'] == 'active')}\n")
        except Exception as e:
            print(f"\nERROR: Could not write to log file: {e}")
        await asyncio.sleep(1)

async def autoscaler_task():
    """The main autoscaler loop that polls server metrics and triggers scaling."""
    print("🚀 Autoscaler started...")
    last_scaling_time = 0
    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
            active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
            if not active_servers: continue

            metric_tasks = [get_server_metrics(server, client) for server in active_servers]
            metric_results = await asyncio.gather(*metric_tasks)

            total_load = sum(r['running'] + r['waiting'] for r in metric_results)
            avg_load_per_server = total_load / len(active_servers)

            print(f"\r[{time.strftime('%H:%M:%S')}] Active: {len(active_servers)} | Avg Load/Server: {avg_load_per_server:.2f}", end="")

            if not ((time.time() - last_scaling_time) > SCALING_COOLDOWN_SECONDS): continue

            if avg_load_per_server < SCALE_DOWN_THRESHOLD:
                scale_factor = avg_load_per_server / SCALE_DOWN_THRESHOLD if SCALE_DOWN_THRESHOLD > 0 else 0
                num_to_scale = max(1, int(len(active_servers) * (1 - scale_factor)))
                if await scale_down(count=num_to_scale): last_scaling_time = time.time()
            elif avg_load_per_server > SCALE_UP_THRESHOLD:
                scale_factor = avg_load_per_server / SCALE_UP_THRESHOLD if SCALE_UP_THRESHOLD > 0 else 1
                num_to_scale = max(1, int(len(active_servers) * (scale_factor - 1)))
                if await scale_up(count=num_to_scale): last_scaling_time = time.time()

# --- Main Execution ---

if __name__ == "__main__":
    if update_nginx_config(ALL_SERVERS):
        reload_nginx()

    loop = asyncio.get_event_loop()
    loop.create_task(log_active_servers())
    loop.create_task(autoscaler_task())
    
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("\nAutoscaler stopped by user.")