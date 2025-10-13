import asyncio
import subprocess
import httpx
import time
import asyncssh
from typing import List, Dict

# --- ⚙️ Configuration (Production Ready) ---
NGINX_CONF_PATH = "/etc/nginx/nginx.conf"
NGINX_TEMPLATE_PATH = "/etc/nginx/nginx.conf.template"
SERVER_COUNT_LOG_FILE = "/mydata/Data/DynGPUs/sys_test1/active_servers.log"
ACTIVE_WORKERS_FILE = "/mydata/Data/DynGPUs/custom_hvd/active_workers.txt"
GPU_MEMORY_FREE_THRESHOLD_MB = 2000

# Scaling Thresholds
SCALE_DOWN_THRESHOLD = 2.0
SCALE_UP_THRESHOLD = 10.0

# Scaling Rules
MIN_ACTIVE_SERVERS = 1
SCALING_COOLDOWN_SECONDS = 15
MONITOR_INTERVAL_SECONDS = 2
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 60 # Timeout for waiting for server to be idle

# --- 🖥️ Server State Management (for 'node1' with 4 GPUs) ---
ALL_SERVERS = [
    {"host": "node1", "port": 8000, "status": "active", "rank": 0, "shared": True},
    {"host": "node1", "port": 8001, "status": "sleeping", "rank": 1, "shared": True},
    {"host": "node1", "port": 8002, "status": "sleeping", "rank": 2, "shared": True},
    {"host": "node1", "port": 8003, "status": "sleeping", "rank": 3, "shared": True},
]

# --- 🟢 ALL FUNCTIONS ARE LIVE 🟢 ---

def read_active_workers() -> List[int]:
    """Reads the list of active training ranks from the file."""
    try:
        with open(ACTIVE_WORKERS_FILE, "r") as f:
            content = f.read().strip()
            return [int(rank) for rank in content.split(',')] if content else []
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error reading {ACTIVE_WORKERS_FILE}: {e}.")
        return []

def write_active_workers(ranks: List[int]):
    """Writes the list of active training ranks to the file."""
    ranks.sort()
    content = ",".join(map(str, ranks))
    try:
        with open(ACTIVE_WORKERS_FILE, "w") as f:
            f.write(content)
        print(f"\n✅ SUCCESS: Updated {ACTIVE_WORKERS_FILE} with ranks: {content}")
    except (PermissionError, IOError) as e:
        print(f"\nERROR: Failed to write to {ACTIVE_WORKERS_FILE}: {e}.")

async def check_gpu_memory_is_free(server: Dict) -> bool:
    """[LIVE] Connects via SSH to run nvidia-smi and check if GPU memory is below a threshold."""
    if not server.get("shared"): return True
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
    """[LIVE] Fetches and parses metrics from a vLLM server's /metrics endpoint."""
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

async def set_server_sleep_state(server: Dict, sleep: bool):
    """[LIVE] Sends a POST request to put a server to sleep or wake it up."""
    action, url = ("Putting to sleep", f"http://{server['host']}:{server['port']}/sleep?level=1") if sleep else \
                  ("Waking up", f"http://{server['host']}:{server['port']}/wake_up")
    print(f"{action}: {server['host']}:{server['port']}")
    try:
        # Use a new client here as the main one is used in the calling loop
        async with httpx.AsyncClient() as client:
            await client.post(url, timeout=20)
    except httpx.RequestError as e:
        print(f"\nERROR: Could not send command to server {server['host']}:{server['port']}: {e}")

async def update_nginx_config(active_servers: List[Dict]) -> bool:
    """[LIVE] Generates and writes a new nginx.conf from a template."""
    print("\nUpdating Nginx configuration...")
    upstream_servers = "".join([f"        server {s['host']}:{s['port']};\n" for s in active_servers])
    try:
        with open(NGINX_TEMPLATE_PATH, "r") as f: template = f.read()
        with open(NGINX_CONF_PATH, "w") as f: f.write(template.replace("{UPSTREAM_SERVERS}", upstream_servers))
        print(f"Nginx config written to {NGINX_CONF_PATH} with {len(active_servers)} active servers.")
        return True
    except (IOError, PermissionError) as e:
        print(f"\nERROR: Failed to write Nginx config: {e}.")
        return False

def reload_nginx():
    """[LIVE] Executes the command to reload Nginx gracefully."""
    print("Reloading Nginx...")
    try:
        subprocess.run(["sudo", "nginx", "-s", "reload"], check=True, capture_output=True, text=True)
        print("Nginx reloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Failed to reload Nginx. Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"\nERROR: 'nginx' command not found.")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred while reloading Nginx: {e}")

async def log_active_servers():
    """Logs the number of active servers to a file every second."""
    while True:
        try:
            with open(SERVER_COUNT_LOG_FILE, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {sum(1 for s in ALL_SERVERS if s['status'] == 'active')}\n")
        except (IOError, PermissionError) as e:
            print(f"\nERROR: Could not write to log file {SERVER_COUNT_LOG_FILE}: {e}")
        await asyncio.sleep(1)

# --- Scaling Logic ---

async def scale_down(count: int, client: httpx.AsyncClient) -> bool: # MODIFIED: added client
    """Scales down shared servers using the new graceful shutdown logic."""
    active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    shared_active_servers = [s for s in active_servers if s['shared']]
    max_possible_to_remove = len(active_servers) - MIN_ACTIVE_SERVERS
    actual_count = min(count, len(shared_active_servers), max_possible_to_remove)
    if actual_count <= 0: return False
    
    servers_to_scale_down = shared_active_servers[:actual_count]
    print(f"\n🔽 Scaling DOWN by {actual_count} server(s).")
    
    for server in servers_to_scale_down:
        server['status'] = 'draining' # Use a new status to indicate it's shutting down
    
    # 1. Update Nginx to stop sending new requests
    new_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    if not await update_nginx_config(new_active_servers):
        for server in servers_to_scale_down: server['status'] = 'active' # Revert
        return False
    reload_nginx()
    
    # NEW HELPER LOGIC: Wait for server to be idle, then shut down
    async def coordinate_graceful_shutdown(server: Dict, client: httpx.AsyncClient):
        shutdown_start_time = time.time()
        print(f"\nWaiting for server {server['host']}:{server['port']} to finish running requests...")
        
        # 2. Wait until the server has no running requests or timeout is reached
        while (time.time() - shutdown_start_time) < GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS:
            metrics = await get_server_metrics(server, client)
            if metrics.get('running', 0) == 0:
                print(f"Server {server['host']}:{server['port']} is now idle.")
                break
            await asyncio.sleep(1) # Check every second
        else: # This 'else' belongs to the 'while' loop, runs if loop finishes without break
            print(f"WARNING: Timeout reached waiting for {server['host']}:{server['port']} to be idle. Proceeding with shutdown.")
            
        # 3. Put the now-idle server to sleep
        await set_server_sleep_state(server, sleep=True)
        
        # 4. Update the active worker file
        print(f"Adding rank {server['rank']} back to the training job...")
        active_ranks = read_active_workers()
        if server['rank'] not in active_ranks:
            active_ranks.append(server['rank'])
            write_active_workers(active_ranks)
        
        server['status'] = 'sleeping' # Final status change
    
    # MODIFIED: Pass client to the shutdown coordinator
    await asyncio.gather(*[coordinate_graceful_shutdown(s, client) for s in servers_to_scale_down])
    return True

async def scale_up(count: int) -> bool:
    all_sleeping = [s for s in ALL_SERVERS if s['status'] == 'sleeping']
    actual_count = min(count, len(all_sleeping))
    if actual_count <= 0: return False
    servers_to_wake = all_sleeping[:actual_count]
    successfully_woken = []
    print(f"\n🔼 Scaling UP by {actual_count} server(s).")
    for server in servers_to_wake:
        if server['shared']:
            print(f"Removing rank {server['rank']} from training to use for inference...")
            active_ranks = read_active_workers()
            new_ranks = [r for r in active_ranks if r != server['rank']]
            write_active_workers(new_ranks)
            if not await check_gpu_memory_is_free(server):
                print(f"ERROR: GPU memory check failed for rank {server['rank']}. Reverting.")
                write_active_workers(active_ranks)
                continue
        await set_server_sleep_state(server, sleep=False)
        server['status'] = 'active'
        successfully_woken.append(server)
    if not successfully_woken: return False
    if await update_nginx_config([s for s in ALL_SERVERS if s['status'] == 'active']):
        reload_nginx()
        return True
    else:
        for server in successfully_woken:
            server['status'] = 'sleeping'
            if server['shared']: write_active_workers(read_active_workers() + [server['rank']])
        return False

async def autoscaler_task():
    print("🚀 Fully Live Autoscaler Started...")
    last_scaling_time = 0
    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
            # MODIFIED: Only consider 'active' servers for metrics, not 'draining' ones
            active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
            if not active_servers: continue
            
            metric_tasks = [get_server_metrics(server, client) for server in active_servers]
            metric_results = await asyncio.gather(*metric_tasks)
            total_load = sum(r['running'] + r['waiting'] for r in metric_results)
            avg_load_per_server = total_load / len(active_servers) if active_servers else 0
            
            print(f"\r[{time.strftime('%H:%M:%S')}] Active: {len(active_servers)} | Avg Load/Server: {avg_load_per_server:.2f}", end="")
            
            if (time.time() - last_scaling_time) < SCALING_COOLDOWN_SECONDS: continue
            
            if avg_load_per_server < SCALE_DOWN_THRESHOLD and len(active_servers) > MIN_ACTIVE_SERVERS:
                scale_factor = avg_load_per_server / SCALE_DOWN_THRESHOLD if SCALE_DOWN_THRESHOLD > 0 else 0
                num_to_scale = max(1, int(len(active_servers) * (1 - scale_factor)))
                # MODIFIED: Pass client to scale_down
                if await scale_down(count=num_to_scale, client=client): 
                    last_scaling_time = time.time()
            elif avg_load_per_server > SCALE_UP_THRESHOLD:
                scale_factor = avg_load_per_server / SCALE_UP_THRESHOLD if SCALE_UP_THRESHOLD > 0 else 1
                num_to_scale = max(1, int(len(active_servers) * (scale_factor - 1)))
                if await scale_up(count=num_to_scale): 
                    last_scaling_time = time.time()

# --- Main Execution ---
if __name__ == "__main__":
    initial_active_ranks = []
    for i, server in enumerate(ALL_SERVERS):
        if i < MIN_ACTIVE_SERVERS:
            server['status'] = 'active'
            initial_active_ranks.append(server['rank'])
        else:
            server['status'] = 'sleeping'
    write_active_workers([r for r in [0,1,2,3] if r not in initial_active_ranks])
    
    print("\n--- Starting Live Autoscaler ---")
    print("WARNING: This script will modify live system files. Run with sudo.")
    
    loop = asyncio.get_event_loop()
    loop.create_task(log_active_servers())
    loop.create_task(autoscaler_task())
    
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("\nAutoscaler stopped by user.")