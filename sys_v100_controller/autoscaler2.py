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
HAPROXY_CONF_PATH = "/etc/haproxy/haproxy.cfg" 
HAPROROXY_TEMPLATE_PATH = "/etc/haproxy/haproxy.cfg.template"
SERVER_COUNT_LOG_FILE = "./active_servers.log"
ACTIVE_WORKERS_FILE = "/home/pacs/Kevin/DynGPUs/sys_v100/active_workers.txt"
TTFT_LOG_FILE = "./ttft_controller.log"

# Scaling Thresholds (Base)
SCALE_DOWN_THRESHOLD = 11
SCALE_UP_THRESHOLD = 21

# Scaling Rules
MIN_ACTIVE_SERVERS = 1
SCALING_COOLDOWN_SECONDS = 15
MONITOR_INTERVAL_SECONDS = 3
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 180
GPU_MEMORY_FREE_THRESHOLD_MB = 3000
GPU_FREE_TIMEOUT_SECONDS = 15
GPU_FREE_POLL_INTERVAL_SECONDS = 1

# --- Unaggressive/Anticipatory Scaling Parameters ---
LOAD_HISTORY_SIZE = 12
DELTA_HISTORY_SIZE = 5
MEDIAN_DELTA_TRIGGER = 0.25

# --- P-Controller Configuration (Physically Tuned) ---
TTFT_TARGET_SECONDS = 0.2 
QUEUE_COST_MS_PER_REQUEST = 521.62 

# Physics-based Gain
THEORETICAL_KP = 1.0 / (QUEUE_COST_MS_PER_REQUEST / 1000.0)
GAIN_FACTOR = 1.2
TTFT_KP = THEORETICAL_KP * GAIN_FACTOR

# --- Stabilization Settings ---
TTFT_HISTORY_SIZE = 5
TTFT_DEADBAND = 0.1 * TTFT_TARGET_SECONDS  
MAX_THRESHOLD_CHANGE_PER_STEP = 2.0 

# --- ACTIVATION TRIGGERS ---
# 1. Saturation: Trigger if >50% of servers have waiting requests.
QUEUE_SATURATION_RATIO = 0.5

# 2. Proximity: Trigger if System Load > 80% of the Base Up Threshold.
# If Threshold is 21, controller wakes up at load ~16.8 to prevent latency spikes.
LOAD_PROXIMITY_RATIO = 0.9

# Limits
MIN_DYNAMIC_UP_THRESHOLD = 10
MAX_DYNAMIC_UP_THRESHOLD = 40
MIN_DYNAMIC_DOWN_THRESHOLD = 5
MAX_DYNAMIC_DOWN_THRESHOLD = 30
# -------------------------------------


# --- 🖥️ Server State Management ---
ALL_SERVERS = [
    # {"host": "localhost", "port": 8000, "status": "sleeping", "rank": 0, "shared": True},
    {"host": "localhost", "port": 8001, "status": "sleeping", "rank": 1, "shared": True},
    {"host": "localhost", "port": 8002, "status": "active", "rank": 2, "shared": True},
]


# --- Helper Functions (Retained) ---

def read_active_workers() -> List[int]:
    try:
        with open(ACTIVE_WORKERS_FILE, "r") as f:
            content = f.read().strip()
            return [int(rank) for rank in content.split(',')] if content else []
    except FileNotFoundError:
        return []

def write_active_workers(ranks: List[int]):
    ranks.sort()
    content = ",".join(map(str, ranks))
    with open(ACTIVE_WORKERS_FILE, "w") as f:
        f.write(content)
    print(f"\nUpdated active_workers.txt with ranks: {content}")

async def check_gpu_memory_is_free(server: Dict) -> bool:
    if not server.get("shared"): return True
    print(f"\nWaiting for GPU memory to be freed for rank {server['rank']} on {server['host']}...")
    local_gpu_id = server['rank'] % 4 + 1
    command = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {local_gpu_id}"
    start_time = time.time()
    while (time.time() - start_time) < GPU_FREE_TIMEOUT_SECONDS:
        try:
            async with asyncssh.connect(server['host']) as conn:
                result = await conn.run(command, check=True)
                memory_used_mb = int(result.stdout.strip())
                print(f"\rRank {server['rank']} using {memory_used_mb} MiB...", end="")
                if memory_used_mb < GPU_MEMORY_FREE_THRESHOLD_MB:
                    print(f"\nGPU for rank {server['rank']} is now free.")
                    return True
            await asyncio.sleep(GPU_FREE_POLL_INTERVAL_SECONDS)
        except Exception as e:
            print(f"\nERROR checking GPU: {e}. Retrying...")
            await asyncio.sleep(GPU_FREE_POLL_INTERVAL_SECONDS)
    print(f"\nERROR: Timeout waiting for GPU free.")
    return False

async def get_server_metrics(server: Dict, client: httpx.AsyncClient) -> Dict:
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
            elif line.startswith("vllm:time_to_first_token_seconds_sum"):
                ttft_sum = float(line.rsplit(' ', 1)[1])
            elif line.startswith("vllm:time_to_first_token_seconds_count"):
                ttft_count = float(line.rsplit(' ', 1)[1])
    except httpx.RequestError: pass
    return {"running": running, "waiting": waiting, "ttft_sum": ttft_sum, "ttft_count": ttft_count}


# --- HAProxy Logic ---
async def update_haproxy_config(active_servers: List[Dict]) -> bool:
    print("\nUpdating HAProxy configuration...")
    server_lines = [f"    server web{i:02d} {s['host']}:{s['port']}\n" for i, s in enumerate(active_servers, start=1)]
    upstream_config = "".join(server_lines)
    try:
        with open(HAPROROXY_TEMPLATE_PATH, "r") as f: template = f.read()
        with open(HAPROXY_CONF_PATH, "w") as f: f.write(template.replace("{UPSTREAM_SERVERS}", upstream_config))
        print(f"HAProxy config updated with {len(active_servers)} active servers.")
        return True
    except Exception as e:
        print(f"\nERROR: Failed to write HAProxy config: {e}")
        return False

def reload_haproxy():
    print("Reloading HAProxy...")
    try:
        subprocess.run(["sudo", "systemctl", "reload", "haproxy"], check=True)
        print("HAProxy reloaded successfully.")
    except Exception as e:
        print(f"\nERROR: Failed to reload HAProxy: {e}")

async def set_server_sleep_state(server: Dict, sleep: bool):
    action, url = ("Putting to sleep", f"http://{server['host']}:{server['port']}/sleep?level=1") if sleep else \
                  ("Waking up", f"http://{server['host']}:{server['port']}/wake_up")
    print(f"{action}: {server['host']}:{server['port']}")
    try:
        async with httpx.AsyncClient() as client: await client.post(url, timeout=20)
    except Exception as e:
        print(f"\nERROR: Could not send command: {e}")


# --- Scaling Logic ---

async def scale_down(count: int) -> bool:
    start_time = time.time()
    active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    shared_active_servers = [s for s in active_servers if s['shared']]
    max_possible = len(active_servers) - MIN_ACTIVE_SERVERS
    actual_count = min(count, len(shared_active_servers), max_possible)
    
    if actual_count <= 0: return False

    servers_to_scale_down = shared_active_servers[:actual_count]
    for server in servers_to_scale_down: server['status'] = 'sleeping'
    
    if not await update_haproxy_config([s for s in ALL_SERVERS if s['status'] == 'active']):
        for server in servers_to_scale_down: server['status'] = 'active'
        return False
    reload_haproxy()
    
    async with httpx.AsyncClient() as client:
        async def wait_and_sleep(s):
            print(f"\nGracefully shutting down {s['host']}:{s['port']}...")
            start = time.time()
            while (time.time() - start) < GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS:
                metrics = await get_server_metrics(s, client)
                if metrics.get("running", -1) == 0: break
                await asyncio.sleep(2)
            await set_server_sleep_state(s, sleep=True)
        await asyncio.gather(*[wait_and_sleep(s) for s in servers_to_scale_down])

    ranks_to_add = [s['rank'] for s in servers_to_scale_down]
    active_ranks = read_active_workers()
    for rank in ranks_to_add:
        if rank not in active_ranks: active_ranks.append(rank)
    write_active_workers(active_ranks)
    
    log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SCALE_DOWN: {actual_count} servers took {time.time() - start_time:.2f}s.\n"
    with open(SERVER_COUNT_LOG_FILE, "a") as f: f.write(log_entry)
    return True

async def scale_up(count: int) -> bool:
    start_time = time.time()
    all_sleeping = [s for s in ALL_SERVERS if s['status'] == 'sleeping']
    dedicated = [s for s in all_sleeping if not s['shared']]
    shared = [s for s in all_sleeping if s['shared']]
    shared.sort(key=lambda s: s['rank'], reverse=True)
    
    candidates = dedicated + shared
    actual_count = min(count, len(candidates))
    if actual_count <= 0: return False
        
    servers_to_wake = candidates[:actual_count]
    shared_to_wake = [s for s in servers_to_wake if s['shared']]
    
    if shared_to_wake:
        orig_ranks = read_active_workers()
        ranks_to_rm = [s['rank'] for s in shared_to_wake]
        print(f"\nRequesting to remove ranks {ranks_to_rm}...")
        write_active_workers([r for r in orig_ranks if r not in ranks_to_rm])
        
        if not all(await asyncio.gather(*[check_gpu_memory_is_free(s) for s in shared_to_wake])):
            print("ERROR: GPU memory check failed. Reverting.")
            write_active_workers(orig_ranks)
            return False

    for server in servers_to_wake:
        await set_server_sleep_state(server, sleep=False)
        server['status'] = 'active'

    if await update_haproxy_config([s for s in ALL_SERVERS if s['status'] == 'active']):
        reload_haproxy()
        log_entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SCALE_UP: {len(servers_to_wake)} servers took {time.time() - start_time:.2f}s.\n"
        with open(SERVER_COUNT_LOG_FILE, "a") as f: f.write(log_entry)
        return True
    
    return False

async def log_active_servers():
    print(f"📝 Logging active server count to {SERVER_COUNT_LOG_FILE}...")
    while True:
        try:
            with open(SERVER_COUNT_LOG_FILE, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {sum(1 for s in ALL_SERVERS if s['status'] == 'active')}\n")
        except: pass
        await asyncio.sleep(5)


# --- MAIN AUTOSCALER TASK ---

async def autoscaler_task():
    print(f"🚀 Autoscaler started with Hybrid P-Controller (Saturation + Proximity)...")
    print(f"ℹ️  Queue Cost: {QUEUE_COST_MS_PER_REQUEST}ms/req -> Kp: {TTFT_KP:.3f}")
    print(f"ℹ️  Triggers: Saturation > {QUEUE_SATURATION_RATIO*100:.0f}%  OR  Load > {LOAD_PROXIMITY_RATIO*100:.0f}% of Threshold")

    last_scaling_time = 0
    load_history = []
    last_total_load = 0
    
    # State for TTFT Calculation
    last_ttft_sum = 0.0
    last_ttft_count = 0.0
    ttft_history = deque(maxlen=TTFT_HISTORY_SIZE)
    
    current_up_threshold = SCALE_UP_THRESHOLD
    current_down_threshold = SCALE_DOWN_THRESHOLD

    # Ensure Log Header
    try:
        with open(TTFT_LOG_FILE, "x") as f:
            f.write("Timestamp, Instant_TTFT_ms, Smoothed_TTFT_ms, Target_ms, Error_ms, Adjustment, Up_Threshold, Down_Threshold, Active_Servers, Instant_Load, Smoothed_Load, Saturation_Ratio, Proximity_Triggered\n")
    except FileExistsError: pass

    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(MONITOR_INTERVAL_SECONDS)

            active_servers_for_metrics = [s for s in ALL_SERVERS if s['status'] == 'active']
            if not active_servers_for_metrics: continue 

            # --- METRIC GATHERING ---
            metric_tasks = [get_server_metrics(server, client) for server in active_servers_for_metrics]
            metric_results = await asyncio.gather(*metric_tasks)

            # --- CALCULATE LOAD & TRIGGERS ---
            waiting_counts = [r['waiting'] for r in metric_results]
            servers_with_waiting = sum(1 for w in waiting_counts if w > 0)
            saturation_ratio = servers_with_waiting / len(active_servers_for_metrics)
            
            total_load = sum(r['running'] + r['waiting'] for r in metric_results)
            instantaneous_avg_load = total_load / len(active_servers_for_metrics)

            load_history.append(instantaneous_avg_load)
            if len(load_history) > LOAD_HISTORY_SIZE: load_history.pop(0)
            smoothed_avg_load = np.mean(load_history)
            
            # --- TTFT DATA COLLECTION ---
            curr_ttft_sum = sum(r['ttft_sum'] for r in metric_results)
            curr_ttft_count = sum(r['ttft_count'] for r in metric_results)
            
            delta_ttft_sum = curr_ttft_sum - last_ttft_sum
            delta_ttft_count = curr_ttft_count - last_ttft_count
            
            instant_ttft = 0.0
            if delta_ttft_count > 0:
                instant_ttft = delta_ttft_sum / delta_ttft_count
                ttft_history.append(instant_ttft)
            
            if curr_ttft_count >= last_ttft_count:
                last_ttft_sum = curr_ttft_sum
                last_ttft_count = curr_ttft_count
            else:
                last_ttft_sum, last_ttft_count = 0.0, 0.0

            # --- STABILIZED P-CONTROLLER LOGIC ---
            if len(ttft_history) > 0:
                smoothed_ttft = np.mean(ttft_history)
            else:
                smoothed_ttft = 0.0

            adjustment = 0.0
            ttft_error = 0.0
            
            # --- DUAL TRIGGER CHECK ---
            is_saturated = saturation_ratio >= QUEUE_SATURATION_RATIO
            # Check against BASE threshold to see if we are "full"
            is_near_threshold = smoothed_avg_load >= (SCALE_UP_THRESHOLD * LOAD_PROXIMITY_RATIO)
            
            should_activate = is_saturated or is_near_threshold

            if should_activate and smoothed_ttft > 0:
                ttft_error = smoothed_ttft - TTFT_TARGET_SECONDS
                if abs(ttft_error) > TTFT_DEADBAND:
                    raw_adjustment = TTFT_KP * ttft_error
                    adjustment = max(-MAX_THRESHOLD_CHANGE_PER_STEP, min(MAX_THRESHOLD_CHANGE_PER_STEP, raw_adjustment))
            else:
                adjustment = 0.0

            # Apply Adjustment
            new_up = SCALE_UP_THRESHOLD - adjustment
            new_down = SCALE_DOWN_THRESHOLD - adjustment
            
            current_up_threshold = max(MIN_DYNAMIC_UP_THRESHOLD, min(MAX_DYNAMIC_UP_THRESHOLD, new_up))
            current_down_threshold = max(MIN_DYNAMIC_DOWN_THRESHOLD, min(MAX_DYNAMIC_DOWN_THRESHOLD, new_down))
            
            # --- LOGGING ---
            try:
                log_line = (
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"{instant_ttft*1000:.2f}, {smoothed_ttft*1000:.2f}, {TTFT_TARGET_SECONDS*1000:.0f}, "
                    f"{ttft_error*1000:.2f}, {adjustment:.4f}, "
                    f"{current_up_threshold:.2f}, {current_down_threshold:.2f}, "
                    f"{len(active_servers_for_metrics)}, {instantaneous_avg_load:.2f}, {smoothed_avg_load:.2f}, "
                    f"{saturation_ratio:.2f}, {1 if is_near_threshold else 0}\n"
                )
                with open(TTFT_LOG_FILE, "a") as f: f.write(log_line)
            except Exception as e: print(f"Logging Error: {e}")

            # --- CONSOLE REPORT ---
            server_details = []
            for server, metrics in zip(active_servers_for_metrics, metric_results):
                r = metrics.get('running', 0)
                w = metrics.get('waiting', 0)
                server_details.append(f"[{server['host']}:{server['port']}] R:{r:.0f} W:{w:.0f}")
            
            # Determine Trigger Status String
            triggers = []
            if is_saturated: triggers.append("SAT")
            if is_near_threshold: triggers.append("PROX")
            status_tag = f"ACTIVE ({'+'.join(triggers)})" if triggers else "IDLE"
            
            print(f"\n[{time.strftime('%H:%M:%S')}] --- MONITORING REPORT ---")
            print(f"TTFT : Smooth: {smoothed_ttft*1000:.0f}ms | Err: {ttft_error*1000:.0f}ms")
            print(f"CTRL : [{status_tag}] Sat: {saturation_ratio*100:.0f}% | Prox: {is_near_threshold}")
            print(f"ADJ  : {adjustment:.2f} -> UP {current_up_threshold:.1f} / DOWN {current_down_threshold:.1f}")
            print(f"LOAD : Inst: {instantaneous_avg_load:.2f} | Smooth: {smoothed_avg_load:.2f} | Servers: {len(active_servers_for_metrics)}")
            print(f"DTLS : {' | '.join(server_details)}")

            # --- DECISION LOGIC ---
            if (time.time() - last_scaling_time) > SCALING_COOLDOWN_SECONDS:
                if (smoothed_avg_load < current_down_threshold and 
                    instantaneous_avg_load < current_down_threshold and 
                    (total_load / (len(active_servers_for_metrics)-1) if len(active_servers_for_metrics) > 1 else 1)+2 < current_up_threshold):
                    
                    deviation = (current_down_threshold - smoothed_avg_load) / current_down_threshold
                    num_to_scale = max(1, int(len(active_servers_for_metrics) * deviation))
                    print(f" (Scaling Down by {num_to_scale})")
                    if await scale_down(count=num_to_scale): last_scaling_time = time.time()
                        
                elif (smoothed_avg_load > current_up_threshold and 
                      instantaneous_avg_load > current_up_threshold and 
                      (total_load / (len(active_servers_for_metrics)+1)) > current_down_threshold):
                    
                    deviation = (smoothed_avg_load - current_up_threshold) / current_up_threshold
                    num_to_scale = max(1, int(len(active_servers_for_metrics) * deviation))
                    print(f" (Scaling Up by {num_to_scale})")
                    if await scale_up(count=num_to_scale): last_scaling_time = time.time()

            last_total_load = total_load

async def startup_tasks():
    initial_active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    if await update_haproxy_config(initial_active_servers): reload_haproxy()
    loop = asyncio.get_event_loop()
    loop.create_task(log_active_servers())
    loop.create_task(autoscaler_task())
    await asyncio.Future() 

if __name__ == "__main__":
    try: asyncio.run(startup_tasks())
    except KeyboardInterrupt: print("\nAutoscaler stopped by user.")
    except RuntimeError: pass