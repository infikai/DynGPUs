import asyncio
import subprocess
import httpx
import time
import numpy as np
import asyncssh
from typing import List, Dict
from collections import deque

# --- ⚙️ Configuration ---
# File Paths
# --- MODIFIED FOR HAPROXY ---
HAPROXY_CONF_PATH = "/etc/haproxy/haproxy.cfg"  # The file to be overwritten
HAPROROXY_TEMPLATE_PATH = "/etc/haproxy/haproxy.cfg.template" # The template file
SERVER_COUNT_LOG_FILE = "./active_servers.log"
TTFT_LOG_FILE = "./ttft_controller.log"
ACTIVE_WORKERS_FILE = "/home/pacs/Kevin/DynGPUs/sys_v100/active_workers.txt"

# Scaling Thresholds (based on average (running + waiting) requests per server)
SCALE_DOWN_THRESHOLD = 11
SCALE_UP_THRESHOLD = 21
LOAD_HISTORY_SIZE = 12

# Scaling Rules
MIN_ACTIVE_SERVERS = 1
SCALING_COOLDOWN_SECONDS = 15
MONITOR_INTERVAL_SECONDS = 3
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 180
GPU_MEMORY_FREE_THRESHOLD_MB = 3000
GPU_FREE_TIMEOUT_SECONDS = 15
GPU_FREE_POLL_INTERVAL_SECONDS = 1

# --- P-Controller Configuration (TTFT) ---
TTFT_TARGET_SECONDS = 1.5
TTFT_KP = 10.0 

# STABILIZATION SETTINGS
# Number of monitor intervals to average TTFT over (e.g., 5 * 3s = 15s window)
TTFT_HISTORY_SIZE = 5 

# Deadband: If TTFT is within ±10% of target, do not change thresholds.
# This prevents constant micro-adjustments.
TTFT_DEADBAND = 0.1 * TTFT_TARGET_SECONDS  

# Max change per step: Prevent thresholds from jumping too fast (e.g. max +/- 2 per cycle)
MAX_THRESHOLD_CHANGE_PER_STEP = 2.0

# Limits
MIN_DYNAMIC_UP_THRESHOLD = 10
MAX_DYNAMIC_UP_THRESHOLD = 40
MIN_DYNAMIC_DOWN_THRESHOLD = 5
MAX_DYNAMIC_DOWN_THRESHOLD = 30


# --- 🖥️ Server State Management (Retained) ---
ALL_SERVERS = [
    # Dedicated inference-only servers (no rank)
    # {"host": "localhost", "port": 8000, "status": "active", "rank": 0, "shared": True},
    {"host": "localhost", "port": 8001, "status": "sleeping", "rank": 1, "shared": True},
    {"host": "localhost", "port": 8002, "status": "active", "rank": 2, "shared": True},
    # {"host": "localhost", "port": 8003, "status": "active", "rank": 3, "shared": True},
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
    """Fetches metrics including TTFT counters."""
    url = f"http://{server['host']}:{server['port']}/metrics"
    running, waiting = 0.0, 0.0
    ttft_sum, ttft_count = 0.0, 0.0
    
    try:
        response = await client.get(url, timeout=5)
        response.raise_for_status()
        for line in response.text.split('\n'):
            if line.startswith("vllm:num_requests_running"):
                running = float(line.rsplit(' ', 1)[1])
            elif line.startswith("vllm:num_requests_waiting"):
                waiting = float(line.rsplit(' ', 1)[1])
            # --- TTFT Parsing ---
            elif line.startswith("vllm:time_to_first_token_seconds_sum"):
                ttft_sum = float(line.rsplit(' ', 1)[1])
            elif line.startswith("vllm:time_to_first_token_seconds_count"):
                ttft_count = float(line.rsplit(' ', 1)[1])
                
    except httpx.RequestError: 
        pass
    
    return {
        "running": running, 
        "waiting": waiting, 
        "ttft_sum": ttft_sum, 
        "ttft_count": ttft_count
    }


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
    print("🚀 Autoscaler started with Stabilized P-Controller (TTFT)...")
    last_scaling_time = 0
    load_history = []
    last_total_load = 0
    
    # State for TTFT Calculation
    last_ttft_sum = 0.0
    last_ttft_count = 0.0
    
    # --- STABILIZATION STATE ---
    # We use a deque to keep a rolling window of recent TTFT averages
    ttft_history = deque(maxlen=TTFT_HISTORY_SIZE)
    
    # Initialize Dynamic Thresholds
    current_up_threshold = SCALE_UP_THRESHOLD
    current_down_threshold = SCALE_DOWN_THRESHOLD

    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
            
            # ... (Keep Isolation Cleanup Logic) ...

            active_servers_for_metrics = [s for s in ALL_SERVERS if s['status'] == 'active']
            if not active_servers_for_metrics: 
                continue 

            # --- METRIC GATHERING ---
            metric_tasks = [get_server_metrics(server, client) for server in active_servers_for_metrics]
            metric_results = await asyncio.gather(*metric_tasks)

            # ... (Keep Isolation Check Logic) ...

            # --- TTFT DATA COLLECTION ---
            curr_ttft_sum = sum(r['ttft_sum'] for r in metric_results)
            curr_ttft_count = sum(r['ttft_count'] for r in metric_results)
            
            delta_ttft_sum = curr_ttft_sum - last_ttft_sum
            delta_ttft_count = curr_ttft_count - last_ttft_count
            
            # 1. Calculate Instantaneous TTFT
            if delta_ttft_count > 0:
                instant_ttft = delta_ttft_sum / delta_ttft_count
                ttft_history.append(instant_ttft) # Add to history
            else:
                # If no requests, do we append 0? 
                # Better to append the LAST known value or nothing to avoid dragging average down artificially.
                # Here we skip appending to keep the average "honest" to active traffic.
                pass 
            
            # Update cumulative counters
            if curr_ttft_count >= last_ttft_count:
                last_ttft_sum = curr_ttft_sum
                last_ttft_count = curr_ttft_count
            else:
                last_ttft_sum, last_ttft_count = 0.0, 0.0

            # --- STABILIZED P-CONTROLLER LOGIC ---
            
            # 2. Calculate Smoothed TTFT (Mean of history)
            if len(ttft_history) > 0:
                smoothed_ttft = np.mean(ttft_history)
            else:
                smoothed_ttft = 0.0

            adjustment = 0.0
            
            if smoothed_ttft > 0:
                # 3. Check Deadband
                # Only act if we are OUTSIDE the safe zone (Target +/- Deadband)
                if abs(smoothed_ttft - TTFT_TARGET_SECONDS) > TTFT_DEADBAND:
                    
                    ttft_error = smoothed_ttft - TTFT_TARGET_SECONDS
                    raw_adjustment = TTFT_KP * ttft_error
                    
                    # 4. Clamp the Rate of Change (Slew Rate Limiting)
                    # This ensures we don't change the threshold by more than X in a single step
                    adjustment = max(-MAX_THRESHOLD_CHANGE_PER_STEP, min(MAX_THRESHOLD_CHANGE_PER_STEP, raw_adjustment))

            # Apply Adjustment (Subtracting because High TTFT -> Lower Threshold)
            # We apply it to the CURRENT threshold, not the BASE threshold, to make it integral-like behavior?
            # Actually, P-Controllers usually apply to a base. 
            # To make it "Dynamic" but stable, we calculate off the BASE config every time.
            
            new_up = SCALE_UP_THRESHOLD - adjustment
            new_down = SCALE_DOWN_THRESHOLD - adjustment
            
            # Clamp final values
            current_up_threshold = max(MIN_DYNAMIC_UP_THRESHOLD, min(MAX_DYNAMIC_UP_THRESHOLD, new_up))
            current_down_threshold = max(MIN_DYNAMIC_DOWN_THRESHOLD, min(MAX_DYNAMIC_DOWN_THRESHOLD, new_down))
            
            try:
                log_line = (
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"{smoothed_ttft*1000:.2f}, "
                    f"{TTFT_TARGET_SECONDS*1000:.0f}, "
                    f"{ttft_error*1000:.2f}, "
                    f"{adjustment:.4f}, "
                    f"{current_up_threshold:.2f}, "
                    f"{current_down_threshold:.2f}, "
                    f"{len(active_servers_for_metrics)}\n"
                )
                with open(TTFT_LOG_FILE, "a") as f:
                    f.write(log_line)
            except Exception as e:
                print(f"Logging Error: {e}")

            # --- SCALING DECISION ---
            
            total_load = sum(r['running'] + r['waiting'] for r in metric_results)
            instantaneous_avg_load = total_load / len(active_servers_for_metrics)

            load_history.append(instantaneous_avg_load)
            if len(load_history) > LOAD_HISTORY_SIZE: load_history.pop(0)
            smoothed_avg_load = np.mean(load_history)
            
            print(f"\n[{time.strftime('%H:%M:%S')}] --- MONITORING REPORT ---")
            print(f"TTFT: Smooth: {smoothed_ttft*1000:.0f}ms | Target: {TTFT_TARGET_SECONDS*1000:.0f}ms | Error: {(smoothed_ttft - TTFT_TARGET_SECONDS)*1000:.0f}ms")
            print(f"CTRL: Adj: {adjustment:.2f} | Thresholds: UP {current_up_threshold:.1f} / DOWN {current_down_threshold:.1f}")
            print(f"LOAD: Active: {len(active_servers_for_metrics)} | Load: {smoothed_avg_load:.2f}")

            # DECISION LOGIC (Using Dynamic Thresholds)
            if (time.time() - last_scaling_time) > SCALING_COOLDOWN_SECONDS:
                
                if (smoothed_avg_load < current_down_threshold and 
                    instantaneous_avg_load < current_down_threshold and 
                    (total_load / (len(active_servers_for_metrics)-1) if len(active_servers_for_metrics) > 1 else 1)+2 < current_up_threshold):
                    
                    deviation = (current_down_threshold - smoothed_avg_load) / current_down_threshold
                    num_to_scale = max(1, int(len(active_servers_for_metrics) * deviation))
                    
                    print(f" (Scaling Down by {num_to_scale})")
                    if await scale_down(count=num_to_scale): 
                        last_scaling_time = time.time()
                        
                elif (smoothed_avg_load > current_up_threshold and 
                      instantaneous_avg_load > current_up_threshold and 
                      (total_load / (len(active_servers_for_metrics)+1)) > current_down_threshold):
                    
                    deviation = (smoothed_avg_load - current_up_threshold) / current_up_threshold
                    num_to_scale = max(1, int(len(active_servers_for_metrics) * deviation))
                    
                    print(f" (Scaling Up by {num_to_scale})")
                    if await scale_up(count=num_to_scale): 
                        last_scaling_time = time.time()

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