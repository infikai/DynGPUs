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
SERVER_COUNT_LOG_FILE = "./active_servers.log"
ACTIVE_WORKERS_FILE = "/mydata/Data/DynGPUs/exp/latency/custom_hvd/active_workers.txt"

# Scaling Thresholds (based on average (running + waiting) requests per server)
SCALE_DOWN_THRESHOLD = 25
SCALE_UP_THRESHOLD = 35

# Scaling Rules
MIN_ACTIVE_SERVERS = 1
SCALING_COOLDOWN_SECONDS = 10
MONITOR_INTERVAL_SECONDS = 2
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 60
GPU_MEMORY_FREE_THRESHOLD_MB = 5000
GPU_FREE_TIMEOUT_SECONDS = 15  # --- NEW: Max time to wait for GPU memory to become free ---
GPU_FREE_POLL_INTERVAL_SECONDS = 1 # --- NEW: How often to check the GPU memory ---


# --- 🖥️ Server State Management ---
# `rank` only exists for shared servers that are part of the training job.
ALL_SERVERS = [
    # Dedicated inference-only servers (no rank)
    {"host": "10.10.3.1", "port": 8000, "status": "sleeping", "rank": 8, "shared": True},
    {"host": "10.10.3.1", "port": 8001, "status": "sleeping", "rank": 9, "shared": True},
    {"host": "10.10.3.1", "port": 8002, "status": "sleeping", "rank": 10, "shared": True},
    {"host": "10.10.3.1", "port": 8003, "status": "active", "shared": False},
    # Shared servers that have a corresponding training rank
    {"host": "10.10.3.2", "port": 8000, "status": "sleeping", "rank": 4, "shared": True},
    {"host": "10.10.3.2", "port": 8001, "status": "sleeping", "rank": 5, "shared": True},
    {"host": "10.10.3.2", "port": 8002, "status": "sleeping", "rank": 6, "shared": True},
    {"host": "10.10.3.2", "port": 8003, "status": "sleeping", "rank": 7, "shared": True},

    {"host": "10.10.3.3", "port": 8002, "status": "sleeping", "rank": 2, "shared": True},
    {"host": "10.10.3.3", "port": 8003, "status": "sleeping", "rank": 3, "shared": True},
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
    """
    Connects to a server via SSH and polls nvidia-smi until the GPU's memory
    is below a threshold or a timeout is reached.
    """
    if not server.get("shared"):
        return True

    print(f"\nWaiting for GPU memory to be freed for rank {server['rank']} on {server['host']}...")
    local_gpu_id = server['rank'] % 4
    command = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {local_gpu_id}"
    
    start_time = time.time()
    # --- NEW: Polling loop with a timeout ---
    while (time.time() - start_time) < GPU_FREE_TIMEOUT_SECONDS:
        try:
            async with asyncssh.connect(server['host']) as conn:
                result = await conn.run(command, check=True)
                memory_used_mb = int(result.stdout.strip())
                
                print(f"\rRank {server['rank']} on {server['host']} is using {memory_used_mb} MiB of memory...", end="")
                
                # If memory is below the threshold, the GPU is free.
                if memory_used_mb < GPU_MEMORY_FREE_THRESHOLD_MB:
                    print(f"\nGPU for rank {server['rank']} is now free.")
                    return True
            
            # If not free, wait for the poll interval before checking again.
            await asyncio.sleep(GPU_FREE_POLL_INTERVAL_SECONDS)
            
        except (asyncssh.Error, OSError, ValueError) as e:
            print(f"\nERROR: Failed to check GPU memory for rank {server['rank']}: {e}. Retrying...")
            await asyncio.sleep(GPU_FREE_POLL_INTERVAL_SECONDS)

    # --- NEW: This code runs if the while loop finishes (timeout) ---
    print(f"\nERROR: Timeout reached. GPU memory for rank {server['rank']} on {server['host']} was not freed in time.")
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
    """
    Generates and writes a new nginx.conf from a template.
    
    --- MODIFIED ---
    Uses 'random two least_conn' as the balancing method.
    This is the most effective algorithm in open-source Nginx
    for this type of workload. It avoids the pitfalls of
    simple round-robin by picking two random servers and
    sending the request to the *better* of those two.
    """
    print("\nUpdating Nginx configuration...")
    
    # Create a list of server lines
    server_lines = "".join([f"        server {s['host']}:{s['port']};\n" for s in active_servers])
    
    # Prepend the balancing algorithm
    upstream_config = "        random two least_conn;\n" + server_lines
    
    try:
        with open(NGINX_TEMPLATE_PATH, "r") as f: template = f.read()
            
        with open(NGINX_CONF_PATH, "w") as f: 
            # Replace the placeholder with the full upstream config
            f.write(template.replace("{UPSTREAM_SERVERS}", upstream_config))
            
        print(f"Nginx config updated with {len(active_servers)} active servers (using 'random two least_conn').")
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
    Scales down gracefully, targeting ONLY shared servers. It removes them from Nginx,
    waits for them to be idle, puts them to sleep, and then returns their GPUs to the
    training pool in a single batch operation.
    """
    start_time = time.time()
    
    # --- Step 1: Select which servers to scale down ---
    active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    shared_active_servers = [s for s in active_servers if s['shared']]
    
    # Determine how many servers can actually be removed
    max_possible_to_remove = len(active_servers) - MIN_ACTIVE_SERVERS
    actual_count = min(count, len(shared_active_servers), max_possible_to_remove)
    
    if actual_count <= 0:
        print("\nScale-down skipped: No shared servers available to scale down or minimum would be breached.")
        return False

    # The list of servers that will be shut down
    servers_to_scale_down = shared_active_servers[:actual_count]
    
    # --- Step 2: Remove servers from Nginx load balancer ---
    for server in servers_to_scale_down:
        server['status'] = 'sleeping'
    
    new_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    if not await update_nginx_config(new_active_servers):
        # Revert status on failure
        for server in servers_to_scale_down:
            server['status'] = 'active'
        return False
    reload_nginx()
    
    # --- Step 3: Concurrently wait for each server to be idle, then put it to sleep ---
    async with httpx.AsyncClient() as client:
        async def wait_and_sleep(s):
            """Helper to handle graceful shutdown for one server."""
            print(f"\nGracefully shutting down shared server {s['host']}:{s['port']}...")
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

        await asyncio.gather(*[wait_and_sleep(s) for s in servers_to_scale_down])

    # --- Step 4: Update the training job file in a single batch operation ---
    ranks_to_add = [s['rank'] for s in servers_to_scale_down]
    print(f"\nAdding ranks {ranks_to_add} back to the training job...")
    active_ranks = read_active_workers()
    for rank in ranks_to_add:
        if rank not in active_ranks:
            active_ranks.append(rank)
    write_active_workers(active_ranks)
    
    # --- Step 5: Log the total duration of the event ---
    total_duration = time.time() - start_time
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SCALE_DOWN: Full scale-down of {actual_count} shared server(s) took {total_duration:.2f}s.\n"
    with open(SERVER_COUNT_LOG_FILE, "a") as f:
        f.write(log_entry)
        
    return True

async def scale_up(count: int) -> bool:
    """
    Scales up, prioritizing dedicated servers first. When scaling shared servers,
    it prioritizes the one with the highest rank number.
    """
    start_time = time.time()
    
    all_sleeping = [s for s in ALL_SERVERS if s['status'] == 'sleeping']
    dedicated_sleeping = [s for s in all_sleeping if not s['shared']]
    shared_sleeping = [s for s in all_sleeping if s['shared']]

    # --- NEW: Sort shared servers to prioritize the highest rank first ---
    shared_sleeping.sort(key=lambda s: s['rank'], reverse=True)
    
    # The final priority list: dedicated servers, then shared servers by highest rank
    servers_to_consider = dedicated_sleeping + shared_sleeping
    
    actual_count = min(count, len(servers_to_consider))
    if actual_count <= 0:
        print("\nScale-up skipped: No available servers to wake up.")
        return False
        
    servers_to_wake = servers_to_consider[:actual_count]
    
    successfully_woken = []
    
    shared_servers_to_wake = [s for s in servers_to_wake if s['shared']]
    if shared_servers_to_wake:
        original_active_ranks = read_active_workers()
        ranks_to_remove = [s['rank'] for s in shared_servers_to_wake]
        print(f"\nRequesting to remove ranks {ranks_to_remove} from the training job...")
        
        new_active_ranks = [r for r in original_active_ranks if r not in ranks_to_remove]
        write_active_workers(new_active_ranks)
        
        # Pre-check memory for all shared servers before proceeding
        memory_checks = await asyncio.gather(*[check_gpu_memory_is_free(s) for s in shared_servers_to_wake])
        if not all(memory_checks):
            print("ERROR: GPU memory check failed for one or more servers. Aborting scale-up and reverting training file.")
            write_active_workers(original_active_ranks) # Revert
            return False

    # --- Step 3: Proceed to wake up all selected servers individually ---
    for server in servers_to_wake:
        await set_server_sleep_state(server, sleep=False)
        server['status'] = 'active'
        successfully_woken.append(server)

    if not successfully_woken:
        # If we failed to wake any servers but modified the training file, revert it
        if shared_servers_to_wake:
             write_active_workers(original_active_ranks)
        return False

    if await update_nginx_config([s for s in ALL_SERVERS if s['status'] == 'active']):
        reload_nginx()
        log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SCALE_UP: Full scale-up of {len(successfully_woken)} server(s) took {time.time() - start_time:.2f}s.\n"
        with open(SERVER_COUNT_LOG_FILE, "a") as f: f.write(log_entry)
        return True
    
    # Revert logic if Nginx fails
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
    """Logs the number of active servers to a file every 5 seconds."""
    print(f"📝 Logging active server count to {SERVER_COUNT_LOG_FILE}...")
    while True:
        try:
            with open(SERVER_COUNT_LOG_FILE, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {sum(1 for s in ALL_SERVERS if s['status'] == 'active')}\n")
        except Exception as e:
            print(f"\nERROR: Could not write to log file: {e}")
        await asyncio.sleep(5)

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