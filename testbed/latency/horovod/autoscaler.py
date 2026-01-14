import asyncio
import subprocess
import httpx
import time
import numpy as np
import asyncssh
from typing import List, Dict

# ---  Configuration ---
# File Paths
HAPROXY_CONF_PATH = "/etc/haproxy/haproxy.cfg"  # The file to be overwritten
HAPROROXY_TEMPLATE_PATH = "/etc/haproxy/haproxy.cfg.template" # The template file
SERVER_COUNT_LOG_FILE = "./active_servers.log"
HOROVOD_HOSTFILE_PATH = "./hostfile.txt"

# Scaling Thresholds (based on average (running + waiting) requests per server)
SCALE_DOWN_THRESHOLD = 20
SCALE_UP_THRESHOLD = 34

# Scaling Rules
MIN_ACTIVE_SERVERS = 3
SCALING_COOLDOWN_SECONDS = 30
MONITOR_INTERVAL_SECONDS = 2
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 60
GPU_MEMORY_FREE_THRESHOLD_MB = 3500
GPU_FREE_TIMEOUT_SECONDS = 15
GPU_FREE_POLL_INTERVAL_SECONDS = 1

LOAD_HISTORY_SIZE = 8

# ---  Server State Management ---
ALL_SERVERS = [
    # node1 (10.10.3.1)
    {"host": "10.10.3.1", "port": 8000, "status": "sleeping", "rank": 8, "shared": True},
    {"host": "10.10.3.1", "port": 8001, "status": "sleeping", "rank": 9, "shared": True},
    {"host": "10.10.3.1", "port": 8002, "status": "sleeping", "rank": 10, "shared": True},
    {"host": "10.10.3.1", "port": 8003, "status": "active", "shared": False}, # Dedicated server
    
    # node2 (10.10.3.2)
    {"host": "10.10.3.2", "port": 8000, "status": "sleeping", "rank": 4, "shared": True},
    {"host": "10.10.3.2", "port": 8001, "status": "sleeping", "rank": 5, "shared": True},
    {"host": "10.10.3.2", "port": 8002, "status": "sleeping", "rank": 6, "shared": True},
    {"host": "10.10.3.2", "port": 8003, "status": "sleeping", "rank": 7, "shared": True}, # On Node 2
    
    # node3 (10.10.3.3)
    {"host": "10.10.3.3", "port": 8000, "status": "sleeping", "rank": 0, "shared": True},
    {"host": "10.10.3.3", "port": 8001, "status": "sleeping", "rank": 1, "shared": True},
    {"host": "10.10.3.3", "port": 8002, "status": "sleeping", "rank": 2, "shared": True},
    {"host": "10.10.3.3", "port": 8003, "status": "sleeping", "rank": 3, "shared": True},
]

# --- ⚠️ Rank Management ---
# Master list of all ranks this script can share/reclaim
ALL_SHARED_RANKS = sorted([s['rank'] for s in ALL_SERVERS if s['shared']])

# Ranks on node3 (10.10.3.3) that are *always* reserved for training
# These servers can NEVER be woken up for inference.
RESERVED_TRAINING_RANKS = [0, 1]

# In-memory state of ranks currently assigned to training
# This is the source of truth
CURRENT_TRAINING_RANKS: List[int] = []


# --- Helper Functions ---

def write_horovod_hostfile(ranks: List[int]):
    """
    Counts active shared ranks *per node* and writes the result to the horovod hostfile.
    """
    
    # 1. Define mappings
    rank_to_host = {s['rank']: s['host'] for s in ALL_SERVERS if s['shared']}
    host_to_nodename = {
        "10.10.3.1": "node1",
        "10.10.3.2": "node2",
        "10.10.3.3": "node3"
    }
    
    # 2. Count active training ranks per nodename
    nodename_counts = {"node3": 0, "node2": 0, "node1": 0}
    
    # Force reserved ranks to be in the list, just in case.
    active_shared_ranks = list(set(ranks) | set(RESERVED_TRAINING_RANKS))
    
    for rank in active_shared_ranks:
        host = rank_to_host.get(rank)
        if not host:
            print(f"WARN: Rank {rank} has no host mapping.")
            continue
            
        nodename = host_to_nodename.get(host)
        if nodename:
            nodename_counts[nodename] += 1
        else:
            print(f"WARN: Host {host} (for rank {rank}) has no nodename mapping.")

    # 3. Generate file content
    content_lines = []
    for nodename, count in nodename_counts.items():
        if count > 0:
            content_lines.append(f"{nodename}:{count}")
    
    # Ensure stable order
    # content_lines.sort()
    content = "\n".join(content_lines)
    
    if not content:
        print("\nNo active training ranks. Writing empty hostfile.")
    
    try:
        with open(HOROVOD_HOSTFILE_PATH, "w") as f:
            f.write(content)
        print(f"\nUpdated {HOROVOD_HOSTFILE_PATH} with content:\n---\n{content}\n---")
    except Exception as e:
        print(f"\nERROR: Failed to write hostfile: {e}")


async def check_gpu_memory_is_free(server: Dict) -> bool:
    if not server.get("shared"):
        return True

    print(f"\nWaiting for GPU memory to be freed for rank {server['rank']} on {server['host']}...")
    
    # NOTE: This logic assumes ranks map to local GPU IDs 0-3
    local_gpu_id = server['rank'] % 4
    command = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {local_gpu_id}"
    
    start_time = time.time()
    while (time.time() - start_time) < GPU_FREE_TIMEOUT_SECONDS:
        try:
            async with asyncssh.connect(server['host']) as conn:
                result = await conn.run(command, check=True)
                memory_used_mb = int(result.stdout.strip())
                
                print(f"\rRank {server['rank']} (Local GPU {local_gpu_id}) on {server['host']} is using {memory_used_mb} MiB of memory...", end="")
                
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

async def update_haproxy_config(active_servers: List[Dict]) -> bool:
    """
    Generates and writes a new haproxy.cfg from a template.
    
    The template must contain the placeholder: {UPSTREAM_SERVERS}
    """
    print("\nUpdating HAProxy configuration...")
    
    # Create a list of HAProxy server lines
    # The 'check' directive ensures HAProxy performs health checks on the backend.
    server_lines = [
        f"    server web{i:02d} {s['host']}:{s['port']}\n"
        for i, s in enumerate(active_servers, start=1)
    ]
    
    upstream_config = "".join(server_lines)
    
    try:
        with open(HAPROROXY_TEMPLATE_PATH, "r") as f: 
            template = f.read()
            
        with open(HAPROXY_CONF_PATH, "w") as f: 
            # Replace the placeholder with the full upstream server list
            f.write(template.replace("{UPSTREAM_SERVERS}", upstream_config))
            
        print(f"HAProxy config updated with {len(active_servers)} active servers.")
        return True
    except Exception as e:
        print(f"\nERROR: Failed to write HAProxy config: {e}")
        return False

def reload_haproxy():
    """Executes the command to reload HAProxy gracefully."""
    print("Reloading HAProxy...")
    try:
        # The 'soft-stop' reload command is often used for HAProxy
        subprocess.run(["sudo", "systemctl", "reload", "haproxy"], check=True)
        print("HAProxy reloaded successfully.")
    except Exception as e:
        print(f"\nERROR: Failed to reload HAProxy: {e}. Check if the service is running or configuration is valid.")

# --- Renaming the functions called by the scaling logic ---
update_nginx_config = update_haproxy_config # Alias the old name to the new function
reload_nginx = reload_haproxy # Alias the old name to the new function

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
    Scales down gracefully by shutting down the N lowest-ranked shared servers.
    """
    global CURRENT_TRAINING_RANKS
    start_time = time.time()
    
    active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    shared_active_servers = [s for s in active_servers if s['shared']]
    
    # POLICY: Sort servers to always scale down the LOWEST rank first
    # (Note: Ranks 0 and 1 are reserved, so they will never be in this list)
    shared_active_servers.sort(key=lambda s: s.get('rank', -1))
    
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
    
    active_ranks_before = CURRENT_TRAINING_RANKS
    current_count = len(active_ranks_before)
    
    new_count = current_count + num_slots_to_add
    
    # The training job always gets the *lowest* N available ranks
    new_active_ranks = sorted(ALL_SHARED_RANKS)[:new_count] if new_count > 0 else []
    
    write_horovod_hostfile(new_active_ranks)
    CURRENT_TRAINING_RANKS = new_active_ranks # Update state
    
    total_duration = time.time() - start_time
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SCALE_DOWN: Took {total_duration:.2f}s for {actual_count} server(s).\n"
    with open(SERVER_COUNT_LOG_FILE, "a") as f:
        f.write(log_entry)
        
    return True

async def scale_up(count: int) -> bool:
    """
    Scales up by waking servers, prioritizing dedicated servers, then shared
    servers with the HIGHEST rank.
    
    *** MODIFIED: Will NOT scale up servers with ranks in RESERVED_TRAINING_RANKS ***
    """
    global CURRENT_TRAINING_RANKS
    start_time = time.time()

    # --- POLICY CHANGE ---
    # Get all sleeping servers *except* those reserved for training
    all_sleeping = [
        s for s in ALL_SERVERS 
        if s['status'] == 'sleeping' 
        and s.get('rank') not in RESERVED_TRAINING_RANKS
    ]
    # --- End Policy Change ---

    dedicated_sleeping = [s for s in all_sleeping if not s['shared']]
    shared_sleeping = [s for s in all_sleeping if s['shared']]
    
    # POLICY: Sort servers to always scale up the HIGHEST rank first
    shared_sleeping.sort(key=lambda s: s['rank'], reverse=True)
    
    servers_to_consider = dedicated_sleeping + shared_sleeping
    
    actual_count = min(count, len(servers_to_consider))
    if actual_count <= 0:
        print("\nScale-up skipped: No available (non-reserved) servers to wake up.")
        return False
        
    servers_to_wake = servers_to_consider[:actual_count]
    
    shared_servers_to_wake = [s for s in servers_to_wake if s['shared']]
    active_ranks_before = CURRENT_TRAINING_RANKS
    
    if shared_servers_to_wake:
        ranks_to_remove = [s['rank'] for s in shared_servers_to_wake]
        print(f"\nAttempting to reclaim ranks {ranks_to_remove} from training job...")
        
        new_ranks = [r for r in active_ranks_before if r not in ranks_to_remove]
        write_horovod_hostfile(new_ranks)
        CURRENT_TRAINING_RANKS = new_ranks # Update state
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
        current_ranks = CURRENT_TRAINING_RANKS
        all_ranks = list(set(current_ranks + failed_ranks))
        write_horovod_hostfile(all_ranks)
        CURRENT_TRAINING_RANKS = all_ranks # Update state

    if not successfully_woken:
        print("No servers were successfully woken up.")
        return False

    if await update_nginx_config([s for s in ALL_SERVERS if s['status'] == 'active']):
        reload_nginx()
        log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SCALE_UP: Full scale-up of {len(successfully_woken)} server(s) took {time.time() - start_time:.2f}s.\n"
        with open(SERVER_COUNT_LOG_FILE, "a") as f: f.write(log_entry)
        return True
    
    print("ERROR: Nginx update failed. Reverting all woken servers...")
    ranks_to_re_add = []
    for server in successfully_woken:
        server['status'] = 'sleeping'
        if server['shared']:
            ranks_to_re_add.append(server['rank'])

    if ranks_to_re_add:
        final_ranks = CURRENT_TRAINING_RANKS
        all_ranks = list(set(final_ranks + ranks_to_re_add))
        write_horovod_hostfile(all_ranks)
        CURRENT_TRAINING_RANKS = all_ranks # Update state
        
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
    load_history = []

    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
            active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
            if not active_servers: continue

            metric_tasks = [get_server_metrics(server, client) for server in active_servers]
            metric_results = await asyncio.gather(*metric_tasks)

            total_load = sum(r['running'] + r['waiting'] for r in metric_results)
            instantaneous_avg_load = total_load / len(active_servers)
            load_history.append(instantaneous_avg_load)
            if len(load_history) > LOAD_HISTORY_SIZE: load_history.pop(0)
            smoothed_avg_load = np.mean(load_history)

            server_details = []
            for server, metrics in zip(active_servers_for_metrics, metric_results):
                r, w = metrics.get('running', 0), metrics.get('waiting', 0)
                server_details.append(f"[{server['host']}:{server['port']}] R:{r:.0f} W:{w:.0f}")

            print(f"\n[{time.strftime('%H:%M:%S')}] --- MONITORING REPORT ---")
            print(f"LOAD : Inst: {instantaneous_avg_load:.2f} | Smooth: {smoothed_avg_load:.2f} | Saturation: {saturation_ratio:.2f}")
            print(f"DTLS : {' | '.join(server_details)}")

            if not ((time.time() - last_scaling_time) > SCALING_COOLDOWN_SECONDS): continue

            if (time.time() - last_scaling_time) > SCALING_COOLDOWN_SECONDS:
                if (smoothed_avg_load < current_down_threshold and 
                    instantaneous_avg_load < current_down_threshold and 
                    (total_load / (len(active_servers)-1) if len(active_servers) > 1 else 1)+2 < current_up_threshold):
                    
                    deviation = (current_down_threshold - smoothed_avg_load) / current_down_threshold
                    num_to_scale = max(1, int(len(active_servers) * deviation))
                    num_to_scale = min(num_to_scale, 2)
                    print(f" (Scaling Down by {num_to_scale})")
                    if await scale_down(count=num_to_scale): last_scaling_time = time.time()
                
                elif (smoothed_avg_load > current_up_threshold and 
                      instantaneous_avg_load > current_up_threshold and 
                      (total_load / (len(active_servers)+1)) > current_down_threshold):
                    
                    deviation = (smoothed_avg_load - current_up_threshold) / current_up_threshold
                    num_to_scale = max(1, int(len(active_servers) * deviation))
                    num_to_scale = min(num_to_scale, 2)
                    print(f" (Scaling Up by {num_to_scale})")
                    if await scale_up(count=num_to_scale): last_scaling_time = time.time()

# --- Main Execution ---

if __name__ == "__main__":
    # Set global rank state
    # --- FIX: Removed 'global' keyword ---
    CURRENT_TRAINING_RANKS = sorted(ALL_SHARED_RANKS)
    
    # On startup, set Nginx to only the initially 'active' servers
    initial_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    print(f"Starting... Initial active servers: {len(initial_active_servers)}")
    if update_nginx_config(initial_active_servers):
        reload_nginx()
    
    # Also, ensure the hostfile reflects the initial state.
    # All shared servers are 'sleeping' (from inference perspective), 
    # so all shared ranks are available for training.
    print(f"Setting initial hostfile to all {len(CURRENT_TRAINING_RANKS)} shared ranks...")
    write_horovod_hostfile(CURRENT_TRAINING_RANKS)

    loop = asyncio.get_event_loop()
    loop.create_task(log_active_servers())
    loop.create_task(autoscaler_task())
    
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("\nAutoscaler stopped by user.")