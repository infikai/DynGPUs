import asyncio
import subprocess
import httpx
import time
import numpy as np
import asyncssh
from typing import List, Dict

# --- ⚙️ Configuration ---
# File Paths
# --- MODIFIED FOR HAPROXY ---
HAPROXY_CONF_PATH = "/etc/haproxy/haproxy.cfg"  # The file to be overwritten
HAPROROXY_TEMPLATE_PATH = "/etc/haproxy/haproxy.cfg.template" # The template file
SERVER_COUNT_LOG_FILE = "./active_servers.log"
ACTIVE_WORKERS_FILE = "./active_workers.txt"

# Scaling Thresholds (based on average (running + waiting) requests per server)
SCALE_DOWN_THRESHOLD = 10.5
SCALE_UP_THRESHOLD = 21

# Scaling Rules
MIN_ACTIVE_SERVERS = 1
SCALING_COOLDOWN_SECONDS = 20
MONITOR_INTERVAL_SECONDS = 3
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 180
GPU_MEMORY_FREE_THRESHOLD_MB = 3000
GPU_FREE_TIMEOUT_SECONDS = 15
GPU_FREE_POLL_INTERVAL_SECONDS = 1

# --- Unaggressive/Anticipatory Scaling Parameters ---
LOAD_HISTORY_SIZE = 10
DELTA_HISTORY_SIZE = 5
MEDIAN_DELTA_TRIGGER = 0.25
# -------------------------------------


# --- 🖥️ Server State Management (Retained) ---
ALL_SERVERS = [
    # Dedicated inference-only servers (no rank)
    # {"host": "localhost", "port": 8000, "status": "sleeping", "rank": 0, "shared": True},
    {"host": "localhost", "port": 8001, "status": "sleeping", "rank": 1, "shared": True},
    {"host": "localhost", "port": 8002, "status": "active", "rank": 2, "shared": True},
]


# --- Helper Functions (Retained) ---

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
    # ... (content remains the same) ...
    """
    Connects to a server via SSH and polls nvidia-smi until the GPU's memory
    is below a threshold or a timeout is reached.
    """
    if not server.get("shared"):
        return True

    print(f"\nWaiting for GPU memory to be freed for rank {server['rank']} on {server['host']}...")
    local_gpu_id = server['rank'] % 4 + 1
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
    # ... (content remains the same) ...
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


# --- MODIFIED: HAProxy Configuration Update ---
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

# --- MODIFIED: HAProxy Reload ---
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
    """Sends a POST request to put a server to sleep or wake it up."""
    action, url = ("Putting to sleep", f"http://{server['host']}:{server['port']}/sleep?level=1") if sleep else \
                  ("Waking up", f"http://{server['host']}:{server['port']}/wake_up")
    print(f"{action}: {server['host']}:{server['port']}")
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, timeout=20)
    except httpx.RequestError as e:
        print(f"\nERROR: Could not send command to server {server['host']}:{server['port']}: {e}")


# --- Scaling Logic (Retained, now using HAProxy aliases) ---

async def scale_down(count: int) -> bool:
    """
    Scales down gracefully, targeting ONLY shared servers.
    """
    start_time = time.time()
    
    active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    shared_active_servers = [s for s in active_servers if s['shared']]
    
    max_possible_to_remove = len(active_servers) - MIN_ACTIVE_SERVERS
    actual_count = min(count, len(shared_active_servers), max_possible_to_remove)
    
    if actual_count <= 0:
        print("\nScale-down skipped: No shared servers available to scale down or minimum would be breached.")
        return False

    servers_to_scale_down = shared_active_servers[:actual_count]
    
    # Temporarily mark the servers as 'sleeping' before HAProxy update
    for server in servers_to_scale_down:
        server['status'] = 'sleeping'
    
    # HAProxy update must use servers that are currently 'active'
    new_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    
    # Use the aliased function
    if not await update_haproxy_config(new_active_servers):
        # Revert status on failure
        for server in servers_to_scale_down:
            server['status'] = 'active'
        return False
    # Use the aliased function
    reload_haproxy()
    
    async with httpx.AsyncClient() as client:
        # ... (wait_and_sleep logic remains the same) ...
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

    # ... (ranks update logic remains the same) ...
    ranks_to_add = [s['rank'] for s in servers_to_scale_down]
    print(f"\nAdding ranks {ranks_to_add} back to the training job...")
    active_ranks = read_active_workers()
    for rank in ranks_to_add:
        if rank not in active_ranks:
            active_ranks.append(rank)
    write_active_workers(active_ranks)
    
    total_duration = time.time() - start_time
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SCALE_DOWN: Full scale-down of {actual_count} shared server(s) took {total_duration:.2f}s.\n"
    with open(SERVER_COUNT_LOG_FILE, "a") as f:
        f.write(log_entry)
        
    return True

async def scale_up(count: int) -> bool:
    """
    Scales up, prioritizing dedicated servers first.
    """
    start_time = time.time()
    
    all_sleeping = [s for s in ALL_SERVERS if s['status'] == 'sleeping']
    dedicated_sleeping = [s for s in all_sleeping if not s['shared']]
    shared_sleeping = [s for s in all_sleeping if s['shared']]

    shared_sleeping.sort(key=lambda s: s['rank'], reverse=True)
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
        
        memory_checks = await asyncio.gather(*[check_gpu_memory_is_free(s) for s in shared_servers_to_wake])
        if not all(memory_checks):
            print("ERROR: GPU memory check failed for one or more servers. Aborting scale-up and reverting training file.")
            write_active_workers(original_active_ranks)
            return False

    for server in servers_to_wake:
        await set_server_sleep_state(server, sleep=False)
        server['status'] = 'active'
        successfully_woken.append(server)

    if not successfully_woken:
        if shared_servers_to_wake:
             write_active_workers(original_active_ranks)
        return False

    # Use the aliased function
    if await update_haproxy_config([s for s in ALL_SERVERS if s['status'] == 'active']):
        # Use the aliased function
        reload_haproxy()
        log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SCALE_UP: Full scale-up of {len(successfully_woken)} server(s) took {time.time() - start_time:.2f}s.\n"
        with open(SERVER_COUNT_LOG_FILE, "a") as f: f.write(log_entry)
        return True
    
    print("ERROR: HAProxy update failed. Reverting...")
    for server in successfully_woken:
        server['status'] = 'sleeping'
        if server['shared']:
            ranks = read_active_workers()
            if server['rank'] not in ranks:
                ranks.append(server['rank'])
                write_active_workers(ranks)
    return False

# --- Background Tasks (Retained) ---

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
    # ... (content remains the same, but uses the aliased functions) ...
    """
    The main autoscaler loop that polls server metrics, triggers scaling, and 
    isolates servers that are severely bottlenecked (high waiting queue disparity).
    """
    print("🚀 Autoscaler started...")
    last_scaling_time = 0
    load_history = []
    delta_history = []
    last_total_load = 0
    
    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
            
            # Flag to track if HAProxy needs a reload due to isolation status change
            haproxy_reload_needed = False
            
            # --- 1. ISOLATION CLEANUP: Re-activate servers whose isolation time has expired ---
            for server in ALL_SERVERS:
                if server.get('status') == 'bottlenecked':
                    if server.get('isolation_end_time', 0.0) < time.time():
                        print(f"\n✅ Server {server['host']}:{server['port']} isolation expired. Reverting status to active.")
                        server['status'] = 'active'
                        server['isolation_end_time'] = 0.0
                        haproxy_reload_needed = True 
            
            # Identify servers that are currently active (serving traffic)
            active_servers_for_metrics = [s for s in ALL_SERVERS if s['status'] == 'active']
            if not active_servers_for_metrics: 
                continue 

            # --- 2. METRIC GATHERING ---
            metric_tasks = [get_server_metrics(server, client) for server in active_servers_for_metrics]
            metric_results = await asyncio.gather(*metric_tasks)

            # Map metrics back to servers for easy look up
            server_metrics_map = {
                (s['host'], s['port']): r for s, r in zip(active_servers_for_metrics, metric_results)
            }

            # --- 3. ISOLATION CHECK: Find and isolate bottlenecked servers ---
            
            # waiting_requests = [r['waiting'] for r in metric_results if r['waiting'] >= 0]
            
            # if len(waiting_requests) >= 2: 
            #     median_waiting = np.median(waiting_requests)
            #     isolation_threshold = median_waiting * ISOLATION_THRESHOLD_FACTOR
                
            #     if isolation_threshold > 0:
            #         for server in active_servers_for_metrics:
            #             metrics = server_metrics_map.get((server['host'], server['port']))
                        
            #             if metrics and metrics['waiting'] > isolation_threshold:
            #                 print(f"\n⚠️ Isolating server {server['host']}:{server['port']}! Waiting queue ({metrics['waiting']:.0f}) > {ISOLATION_THRESHOLD_FACTOR}x median ({median_waiting:.2f}).")
                            
            #                 server['status'] = 'bottlenecked'
            #                 server['isolation_end_time'] = time.time() + ISOLATION_DURATION_SECONDS
            #                 haproxy_reload_needed = True 
                            
            # # --- 4. HAPROXY RELOAD ---
            # if haproxy_reload_needed:
            #     final_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
            #     if await update_haproxy_config(final_active_servers):
            #         reload_haproxy()
            #     continue 

            # --- 5. SCALING DECISION ---
            
            total_load = sum(r['running'] + r['waiting'] for r in metric_results)
            instantaneous_avg_load = total_load / len(active_servers_for_metrics)

            # --- Load Averaging (Absolute Load) ---
            load_history.append(instantaneous_avg_load)
            if len(load_history) > LOAD_HISTORY_SIZE:
                load_history.pop(0)
            smoothed_avg_load = np.mean(load_history)
            
            # --- DELTA Calculation and Smoothing ---
            load_delta = total_load - last_total_load
            percent_change = load_delta / last_total_load if last_total_load > 0 else 0
            
            delta_history.append(percent_change if percent_change > 0 else 0)
            if len(delta_history) > DELTA_HISTORY_SIZE:
                delta_history.pop(0)
            
            median_delta = np.median(delta_history) if delta_history else 0
            
            
            # --- MONITORING OUTPUT ---
            server_details = []
            for server, metrics in zip(active_servers_for_metrics, metric_results):
                r = metrics.get('running', 0)
                w = metrics.get('waiting', 0)
                server_details.append(f"[{server['host']}:{server['port']}] R:{r:.0f} W:{w:.0f}")

            print(f"\n[{time.strftime('%H:%M:%S')}] --- MONITORING REPORT ---")
            print(f"STATUS: Active Servers: {len(active_servers_for_metrics)} | Smoothed Avg Load: {smoothed_avg_load:.2f} | Median Delta: {median_delta:.0%}")
            print(f"DETAILS: {' | '.join(server_details)}")
            
            # --- DECISION LOGIC ---

            # 1. Median Delta Trigger (Anticipatory Scaling - overrides cooldown)
            if (time.time() - last_scaling_time) > SCALING_COOLDOWN_SECONDS:
                
                if smoothed_avg_load < SCALE_DOWN_THRESHOLD and instantaneous_avg_load < SCALE_DOWN_THRESHOLD and (total_load / (len(active_servers_for_metrics)-1) if len(active_servers_for_metrics) > 1 else 1)+2 < SCALE_UP_THRESHOLD:
                    deviation = (SCALE_DOWN_THRESHOLD - smoothed_avg_load) / SCALE_DOWN_THRESHOLD
                    num_to_scale = max(1, int(len(active_servers_for_metrics) * deviation))
                    
                    print(f" (Scaling Down by {num_to_scale} servers, Smoothed Load: {smoothed_avg_load:.2f})")
                    if await scale_down(count=num_to_scale): 
                        last_scaling_time = time.time()
                        
                elif smoothed_avg_load > SCALE_UP_THRESHOLD and instantaneous_avg_load > SCALE_UP_THRESHOLD and (total_load / (len(active_servers_for_metrics)+1)) > SCALE_DOWN_THRESHOLD:
                    deviation = (smoothed_avg_load - SCALE_UP_THRESHOLD) / SCALE_UP_THRESHOLD
                    num_to_scale = max(1, int(len(active_servers_for_metrics) * deviation))
                    
                    print(f" (Scaling Up by {num_to_scale} servers, Smoothed Load: {smoothed_avg_load:.2f})")
                    if await scale_up(count=num_to_scale): 
                        last_scaling_time = time.time()
                        
            # --- END OF CYCLE ---
            last_total_load = total_load


# --- Main Execution ---

async def startup_tasks():
    """Performs asynchronous setup and starts the recurring tasks."""
    
    # AWAIT the configuration update
    # The initial HAProxy update must only include servers marked 'active'
    initial_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    if await update_haproxy_config(initial_active_servers):
        reload_haproxy()
    
    # Get the loop once the async environment is active
    loop = asyncio.get_event_loop()
    
    # Start the background tasks
    loop.create_task(log_active_servers())
    loop.create_task(autoscaler_task())
    
    # Create a never-ending future to keep the loop alive indefinitely.
    await asyncio.Future() 


if __name__ == "__main__":
    
    try:
        asyncio.run(startup_tasks())
    except KeyboardInterrupt:
        print("\nAutoscaler stopped by user.")
    except RuntimeError:
        pass