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
HOROVOD_HOSTFILE_PATH = "/mydata/Data/DynGPUs/horovod/hostfile.txt"

# Scaling Thresholds (based on average (running + waiting) requests per server)
SCALE_DOWN_THRESHOLD = 45
SCALE_UP_THRESHOLD = 60

# Scaling Rules
MIN_ACTIVE_SERVERS = 4
SCALING_COOLDOWN_SECONDS = 20
MONITOR_INTERVAL_SECONDS = 2
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 60
GPU_MEMORY_FREE_THRESHOLD_MB = 3500
GPU_FREE_TIMEOUT_SECONDS = 10
GPU_FREE_POLL_INTERVAL_SECONDS = 1


# --- 🖥️ Server State Management ---
ALL_SERVERS = [
    # Dedicated inference-only servers (no rank)
    {"host": "10.10.3.1", "port": 8000, "status": "active", "shared": False},
    {"host": "10.10.3.1", "port": 8001, "status": "active", "shared": False},
    {"host": "10.10.3.1", "port": 8002, "status": "active", "shared": False},
    {"host": "10.10.3.1", "port": 8003, "status": "active", "shared": False},
    # Shared servers that have a corresponding training rank
    {"host": "10.10.3.2", "port": 8000, "status": "sleeping", "rank": 4, "shared": True},
    {"host": "10.10.3.2", "port": 8001, "status": "sleeping", "rank": 5, "shared": True},
    {"host": "10.10.3.2", "port": 8002, "status": "sleeping", "rank": 6, "shared": True},
    {"host": "10.10.3.2", "port": 8003, "status": "sleeping", "rank": 7, "shared": True},
]
ALL_SHARED_RANKS = [4, 5, 6, 7]


# --- Helper Functions ---

def read_horovod_hostfile() -> List[int]:
    """Reads the horovod hostfile, parses the count for node2, and returns the corresponding ranks."""
    try:
        with open(HOROVOD_HOSTFILE_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("node2:"):
                    count = int(line.split(':')[1])
                    active_ranks = sorted(ALL_SHARED_RANKS)[-count:] if count > 0 else []
                    return active_ranks
    except (FileNotFoundError, IndexError, ValueError):
        pass
    return []

def write_horovod_hostfile(ranks: List[int]):
    """
    Counts active shared ranks and writes the result to the horovod hostfile,
    ensuring node3:4 is always the first line.
    """
    shared_rank_count = len([r for r in ranks if r in ALL_SHARED_RANKS])
    
    content_lines = ["node3:4"]
    
    if shared_rank_count > 0:
        content_lines.append(f"node2:{shared_rank_count}")
    
    content = "\n".join(content_lines)
    
    with open(HOROVOD_HOSTFILE_PATH, "w") as f:
        f.write(content)
    print(f"\nUpdated {HOROVOD_HOSTFILE_PATH} with content:\n---\n{content}\n---")


async def check_gpu_memory_is_free(server: Dict) -> bool:
    if not server.get("shared"):
        return True

    print(f"\nWaiting for GPU memory to be freed for rank {server['rank']} on {server['host']}...")
    local_gpu_id = server['rank'] % 4
    command = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {local_gpu_id}"
    
    start_time = time.time()
    while (time.time() - start_time) < GPU_FREE_TIMEOUT_SECONDS:
        try:
            async with asyncssh.connect(server['host']) as conn:
                result = await conn.run(command, check=True)
                memory_used_mb = int(result.stdout.strip())
                
                print(f"\rRank {server['rank']} on {server['host']} is using {memory_used_mb} MiB of memory...", end="")
                
                if memory_used_mb < GPU_MEMORY_FREE_THRESHOLD_MB:
                    print(f"\nGPU for rank {server['rank']} is now free.")
                    return True
            
            await asyncio.sleep(GPU_FREE_POLL_INTERVAL_SECONDS)
            
        except (asyncssh.Error, OSError, ValueError) as e:
            print(f"\nERROR: Failed to check GPU memory for rank {server['rank']}: {e}. Retrying...")
            await asyncio.sleep(GPU_FREE_POLL_INTERVAL_SECONDS)

    print(f"\nERROR: Timeout reached. GPU memory for rank {server['rank']} on {server['host']} was not freed in time.")
    return False

async def get_server_metrics(server: Dict, client: httpx.AsyncClient) -> Dict:
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
    print("Reloading Nginx...")
    try:
        subprocess.run(["sudo", "nginx", "-s", "reload"], check=True)
        print("Nginx reloaded successfully.")
    except Exception as e:
        print(f"\nERROR: Failed to reload Nginx: {e}")

async def set_server_sleep_state(server: Dict, sleep: bool):
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
    Scales down gracefully by shutting down the N highest-ranked shared servers.
    """
    start_time = time.time()
    
    active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    shared_active_servers = [s for s in active_servers if s['shared']]
    
    # POLICY: Sort servers to always scale down the HIGHEST rank first
    shared_active_servers.sort(key=lambda s: s.get('rank', -1), reverse=True)
    
    max_possible_to_remove = len(active_servers) - MIN_ACTIVE_SERVERS
    actual_count = min(count, len(shared_active_servers), max_possible_to_remove)
    
    if actual_count <= 0:
        print("\nScale-down skipped: No shared servers to scale down or minimum would be breached.")
        return False

    servers_to_scale_down = shared_active_servers[:actual_count]
    
    for server in servers_to_scale_down:
        server['status'] = 'sleeping'
    
    new_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    if not await update_nginx_config(new_active_servers):
        for server in servers_to_scale_down:
            server['status'] = 'active'
        return False
    reload_nginx()
    
    async with httpx.AsyncClient() as client:
        async def wait_and_sleep(s):
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

    num_slots_to_add = len(servers_to_scale_down)
    print(f"\nAdding {num_slots_to_add} slots back to the training job...")
    
    active_ranks_before = read_horovod_hostfile()
    current_count = len(active_ranks_before)
    
    new_count = current_count + num_slots_to_add
    
    new_active_ranks = sorted(ALL_SHARED_RANKS)[-new_count:] if new_count > 0 else []
    
    write_horovod_hostfile(new_active_ranks)
    
    total_duration = time.time() - start_time
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SCALE_DOWN: Took {total_duration:.2f}s for {actual_count} server(s).\n"
    with open(SERVER_COUNT_LOG_FILE, "a") as f:
        f.write(log_entry)
        
    return True

async def scale_up(count: int) -> bool:
    """
    Scales up by waking servers, prioritizing dedicated servers, then shared
    servers with the LOWEST rank.
    """
    all_sleeping = [s for s in ALL_SERVERS if s['status'] == 'sleeping']
    dedicated_sleeping = [s for s in all_sleeping if not s['shared']]
    shared_sleeping = [s for s in all_sleeping if s['shared']]
    
    # --- POLICY CHANGE: Sort servers to always scale up the LOWEST rank first ---
    shared_sleeping.sort(key=lambda s: s['rank']) # Removed reverse=True
    
    servers_to_consider = dedicated_sleeping + shared_sleeping
    
    actual_count = min(count, len(servers_to_consider))
    if actual_count <= 0:
        print("\nScale-up skipped: No available servers to wake up.")
        return False
        
    servers_to_wake = servers_to_consider[:actual_count]
    
    shared_servers_to_wake = [s for s in servers_to_wake if s['shared']]
    active_ranks_before = read_horovod_hostfile()
    
    if shared_servers_to_wake:
        ranks_to_remove = [s['rank'] for s in shared_servers_to_wake]
        print(f"\nAttempting to reclaim ranks {ranks_to_remove} from training job...")
        
        new_ranks = [r for r in active_ranks_before if r not in ranks_to_remove]
        write_horovod_hostfile(new_ranks)
        await asyncio.sleep(5)

    successfully_woken = []
    failed_ranks = []
    
    for server in servers_to_wake:
        wake_server = True
        if server['shared']:
            if not await check_gpu_memory_is_free(server):
                print(f"ERROR: GPU for rank {server['rank']} not free. It will not be woken up.")
                failed_ranks.append(server['rank'])
                wake_server = False
        
        if wake_server:
            await set_server_sleep_state(server, sleep=False)
            server['status'] = 'active'
            successfully_woken.append(server)

    if failed_ranks:
        print(f"Adding failed ranks {failed_ranks} back to the training job...")
        current_ranks = read_horovod_hostfile()
        all_ranks = list(set(current_ranks + failed_ranks))
        write_horovod_hostfile(all_ranks)

    if not successfully_woken:
        print("No servers were successfully woken up.")
        return False

    if await update_nginx_config([s for s in ALL_SERVERS if s['status'] == 'active']):
        reload_nginx()
        return True
    
    print("ERROR: Nginx update failed. Reverting all woken servers...")
    ranks_to_re_add = []
    for server in successfully_woken:
        server['status'] = 'sleeping'
        if server['shared']:
            ranks_to_re_add.append(server['rank'])

    if ranks_to_re_add:
        final_ranks = read_horovod_hostfile()
        all_ranks = list(set(final_ranks + ranks_to_re_add))
        write_horovod_hostfile(all_ranks)
        
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