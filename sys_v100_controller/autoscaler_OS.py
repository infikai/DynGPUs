import asyncio
import subprocess
import httpx
import time
import numpy as np
import asyncssh
from typing import List, Dict, Tuple
from collections import deque

# --- ⚙️ Configuration ---
HAPROXY_CONF_PATH = "/etc/haproxy/haproxy.cfg" 
HAPROROXY_TEMPLATE_PATH = "/etc/haproxy/haproxy.cfg.template"
SERVER_COUNT_LOG_FILE = "./active_servers.log"
ACTIVE_WORKERS_FILE = "./active_workers.txt"
TTFT_LOG_FILE = "./ttft_adaptive_window.log"

# --- 🎯 SLO & Adaptive Logic Configuration ---
TTFT_SLO_TARGET_SECONDS = 6.9

# --- 🔒 ONE-SHOT TUNING CONFIGURATION ---
TUNING_MODE_DURATION_SECONDS = 100
ADAPTIVE_WINDOW_SECONDS = 100      

# Minimum data required to allow a pre-emptive adjustment
MIN_WINDOW_DURATION_FOR_DECISION = 15 

# Percentile Targets
MAX_SLO_VIOLATION_RATIO = 0.12    
SAFE_SLO_VIOLATION_RATIO = 0.12

# Threshold tuning
INITIAL_UP_THRESHOLD = 16.0
MIN_UP_THRESHOLD = 3.0
MAX_UP_THRESHOLD = 50.0
THRESHOLD_STEP_SIZE = 1       
DOWN_THRESHOLD_RATIO = 0.5

# Standard Autoscaling Rules
MIN_ACTIVE_SERVERS = 1
SCALING_COOLDOWN_SECONDS = 20

# --- ⏱️ INTERVAL CONFIGURATION ---
TTFT_MONITOR_INTERVAL_SECONDS = 0.5
LOAD_MONITOR_INTERVAL_SECONDS = 3.0
LOAD_HISTORY_SIZE = 10

GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 180
GPU_MEMORY_FREE_THRESHOLD_MB = 3000
GPU_FREE_TIMEOUT_SECONDS = 15
GPU_FREE_POLL_INTERVAL_SECONDS = 1
# -------------------------------------


# --- 🖥️ Server State Management ---
ALL_SERVERS = [
    {"host": "localhost", "port": 8000, "status": "sleeping", "rank": 0, "shared": True},
    {"host": "localhost", "port": 8001, "status": "sleeping", "rank": 1, "shared": True},
    {"host": "localhost", "port": 8002, "status": "active", "rank": 2, "shared": True}, 
]

# --- Helper Functions ---

def read_active_workers() -> List[int]:
    try:
        with open(ACTIVE_WORKERS_FILE, "r") as f:
            content = f.read().strip()
            return [int(rank) for rank in content.split(',')] if content else []
    except FileNotFoundError: return []

def write_active_workers(ranks: List[int]):
    ranks.sort()
    content = ",".join(map(str, ranks))
    with open(ACTIVE_WORKERS_FILE, "w") as f: f.write(content)
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
    metrics = {
        "running": 0.0, "waiting": 0.0, 
        "ttft_sum": 0.0, "ttft_count": 0.0,
        "prompt_tokens": 0.0 
    }
    try:
        response = await client.get(url, timeout=5)
        response.raise_for_status()
        for line in response.text.split('\n'):
            if line.startswith("vllm:num_requests_running"):
                metrics["running"] = float(line.rsplit(' ', 1)[1])
            elif line.startswith("vllm:num_requests_waiting"):
                metrics["waiting"] = float(line.rsplit(' ', 1)[1])
            elif line.startswith("vllm:time_to_first_token_seconds_sum"):
                metrics["ttft_sum"] = float(line.rsplit(' ', 1)[1])
            elif line.startswith("vllm:time_to_first_token_seconds_count"):
                metrics["ttft_count"] = float(line.rsplit(' ', 1)[1])
            elif line.startswith("vllm:prompt_tokens_total"):
                metrics["prompt_tokens"] = float(line.rsplit(' ', 1)[1])
    except httpx.RequestError: pass
    return metrics


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
    for server in servers_to_scale_down: 
        server['status'] = 'sleeping'
    
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
    print(f"🚀 Autoscaler started (Pre-Emptive Tuning, Skipping First Adjustment)...")
    print(f"ℹ️  Tuning: {TUNING_MODE_DURATION_SECONDS}s. Pre-emptive check enabled.")
    print(f"ℹ️  Sampling: TTFT={TTFT_MONITOR_INTERVAL_SECONDS}s | Load={LOAD_MONITOR_INTERVAL_SECONDS}s")

    last_scaling_time = time.time()
    last_load_check_time = 0
    
    # State flags for the One-Shot logic
    tuning_event_completed = False 
    
    # --- NEW FLAG: Skip First Adjustment ---
    skip_initial_adjustment = True

    load_history = []
    smoothed_avg_load = 0.0 
    
    current_up_threshold = INITIAL_UP_THRESHOLD
    current_down_threshold = current_up_threshold * DOWN_THRESHOLD_RATIO
    
    ttft_window: deque[Tuple[float, float]] = deque()
    previous_server_metrics = {}

    try:
        with open(TTFT_LOG_FILE, "x") as f:
            f.write("Timestamp, Event, Value, Threshold, Window_Size_Sec, Violation_Ratio\n")
    except FileExistsError: pass

    async with httpx.AsyncClient() as client:
        while True:
            # --- 1. FAST LOOP (TTFT Data Collection) ---
            await asyncio.sleep(TTFT_MONITOR_INTERVAL_SECONDS)
            current_time = time.time()

            active_servers_for_metrics = [s for s in ALL_SERVERS if s['status'] == 'active']
            if not active_servers_for_metrics: continue 

            metric_tasks = [get_server_metrics(server, client) for server in active_servers_for_metrics]
            metric_results = await asyncio.gather(*metric_tasks)

            for server, current_m in zip(active_servers_for_metrics, metric_results):
                srv_key = f"{server['host']}:{server['port']}"
                prev_m = previous_server_metrics.get(srv_key, {'ttft_sum': 0, 'ttft_count': 0})
                
                delta_sum = current_m['ttft_sum'] - prev_m['ttft_sum']
                delta_count = current_m['ttft_count'] - prev_m['ttft_count']
                
                previous_server_metrics[srv_key] = {
                    'ttft_sum': current_m['ttft_sum'], 
                    'ttft_count': current_m['ttft_count']
                }

                if delta_count > 0:
                    instant_ttft = delta_sum / delta_count
                    ttft_window.append((current_time, instant_ttft))
            
            # Prune Window
            while ttft_window and (current_time - ttft_window[0][0]) > ADAPTIVE_WINDOW_SECONDS:
                ttft_window.popleft()

            
            # --- 2. SLOW LOOP (One-Shot Adjustment & Load Logic) ---
            if (current_time - last_load_check_time) >= LOAD_MONITOR_INTERVAL_SECONDS:
                last_load_check_time = current_time
                time_since_scale = current_time - last_scaling_time

                # --- A. STANDARD TIMER-BASED TUNING ---
                if time_since_scale >= TUNING_MODE_DURATION_SECONDS and not tuning_event_completed:
                    if ttft_window:
                        window_duration = ttft_window[-1][0] - ttft_window[0][0]
                        if window_duration >= MIN_WINDOW_DURATION_FOR_DECISION:
                            all_values = [v for t, v in ttft_window]
                            violation_count = sum(1 for v in all_values if v > TTFT_SLO_TARGET_SECONDS)
                            violation_ratio = violation_count / len(all_values)

                            adjustment = 0.0
                            if violation_ratio > MAX_SLO_VIOLATION_RATIO: adjustment = -THRESHOLD_STEP_SIZE 
                            elif violation_ratio < SAFE_SLO_VIOLATION_RATIO: adjustment = THRESHOLD_STEP_SIZE
                            
                            # SKIP LOGIC
                            if skip_initial_adjustment:
                                print(f"\n✋ Skipping first tuning adjustment (Startup Stabilization).")
                                print(f"   (Calculated Adj would be: {adjustment})")
                                skip_initial_adjustment = False
                            elif adjustment != 0:
                                current_up_threshold = max(MIN_UP_THRESHOLD, min(MAX_UP_THRESHOLD, current_up_threshold + adjustment))
                                current_down_threshold = current_up_threshold * DOWN_THRESHOLD_RATIO
                                print(f"\n🎯 ONE-SHOT TUNING COMPLETED (Timer)!")
                                print(f"   Violation: {violation_ratio*100:.1f}% -> New Thresh: {current_up_threshold}")
                                try:
                                    with open(TTFT_LOG_FILE, "a") as f: 
                                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, TIMER_ADJUST, {adjustment}, {current_up_threshold}, {window_duration:.1f}, {violation_ratio:.2f}\n")
                                except: pass
                    tuning_event_completed = True 


                # --- B. LOAD CALCULATION ---
                total_load = sum(r['running'] + r['waiting'] for r in metric_results)
                instantaneous_avg_load = total_load / len(active_servers_for_metrics)
                load_history.append(instantaneous_avg_load)
                if len(load_history) > LOAD_HISTORY_SIZE: load_history.pop(0)
                smoothed_avg_load = np.mean(load_history)


                # --- C. PRE-EMPTIVE INTERRUPT TUNING ---
                draft_scale_decision = False
                
                # Draft Scale Down Check
                if (smoothed_avg_load < current_down_threshold and 
                    instantaneous_avg_load < current_down_threshold and 
                    (total_load / (len(active_servers_for_metrics)-1) if len(active_servers_for_metrics) > 1 else 1)+2 < current_up_threshold):
                    draft_scale_decision = True
                
                # Draft Scale Up Check
                elif (smoothed_avg_load > current_up_threshold and 
                      instantaneous_avg_load > current_up_threshold and 
                      (total_load / (len(active_servers_for_metrics)+1)) > current_down_threshold):
                    draft_scale_decision = True
                
                if draft_scale_decision and not tuning_event_completed and ttft_window:
                     if (time.time() - last_scaling_time) > SCALING_COOLDOWN_SECONDS:
                        window_duration = ttft_window[-1][0] - ttft_window[0][0]
                        if window_duration >= MIN_WINDOW_DURATION_FOR_DECISION:
                            all_values = [v for t, v in ttft_window]
                            violation_count = sum(1 for v in all_values if v > TTFT_SLO_TARGET_SECONDS)
                            violation_ratio = violation_count / len(all_values)

                            pre_adj = 0.0
                            if violation_ratio > MAX_SLO_VIOLATION_RATIO: pre_adj = -THRESHOLD_STEP_SIZE 
                            elif violation_ratio < SAFE_SLO_VIOLATION_RATIO: pre_adj = THRESHOLD_STEP_SIZE
                            
                            # SKIP LOGIC
                            if skip_initial_adjustment:
                                print(f"\n✋ Skipping pre-emptive adjustment (Startup Stabilization).")
                                skip_initial_adjustment = False
                            elif pre_adj != 0:
                                old_thresh = current_up_threshold
                                current_up_threshold = max(MIN_UP_THRESHOLD, min(MAX_UP_THRESHOLD, current_up_threshold + pre_adj))
                                current_down_threshold = current_up_threshold * DOWN_THRESHOLD_RATIO
                                tuning_event_completed = True 
                                print(f"\n⚡ PRE-EMPTIVE TUNING INTERRUPT! (Load triggered)")
                                print(f"   Load: {smoothed_avg_load:.1f} vs Old Thresh: {old_thresh:.1f}")
                                print(f"   Violation: {violation_ratio*100:.1f}% -> New Thresh: {current_up_threshold:.1f}")
                                try:
                                    with open(TTFT_LOG_FILE, "a") as f: 
                                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, PRE_EMPT_ADJUST, {pre_adj}, {current_up_threshold}, {window_duration:.1f}, {violation_ratio:.2f}\n")
                                except: pass


                # --- D. FINAL SCALING DECISION ---
                if (time.time() - last_scaling_time) > SCALING_COOLDOWN_SECONDS:
                    did_scale_up = False
                    did_scale_down = False
                    
                    if (smoothed_avg_load < current_down_threshold and 
                        instantaneous_avg_load < current_down_threshold and 
                        (total_load / (len(active_servers_for_metrics)-1) if len(active_servers_for_metrics) > 1 else 1)+2 < current_up_threshold):
                        
                        deviation = (current_down_threshold - smoothed_avg_load) / current_down_threshold
                        num_to_scale = max(1, int(len(active_servers_for_metrics) * deviation))
                        print(f" (Scaling Down by {num_to_scale})")
                        if await scale_down(count=num_to_scale): did_scale_down = True
                    
                    elif (smoothed_avg_load > current_up_threshold and 
                          instantaneous_avg_load > current_up_threshold and 
                          (total_load / (len(active_servers_for_metrics)+1)) > current_down_threshold):
                        
                        deviation = (smoothed_avg_load - current_up_threshold) / current_up_threshold
                        num_to_scale = max(1, int(len(active_servers_for_metrics) * deviation))
                        print(f" (Scaling Up by {num_to_scale})")
                        if await scale_up(count=num_to_scale): did_scale_up = True

                    # --- RESET LOGIC ---
                    if did_scale_up:
                        print("🔄 Scaling UP occurred! Resetting One-Shot Tuning logic.")
                        ttft_window.clear()
                        last_scaling_time = time.time()
                        tuning_event_completed = False 
                        for s in active_servers_for_metrics: pass 
                    
                    elif did_scale_down:
                        print("📉 Scaling DOWN occurred! Skipping re-tuning.")
                        last_scaling_time = time.time()
                        tuning_event_completed = True 

                # Report Status
                server_details = [f"[{s['port']}]:{int(m['running']+m['waiting'])}" for s, m in zip(active_servers_for_metrics, metric_results)]
                
                if not tuning_event_completed:
                    remaining = int(TUNING_MODE_DURATION_SECONDS - time_since_scale)
                    print(f"[{time.strftime('%H:%M:%S')}] LOAD:{smoothed_avg_load:.1f} | ⏳ COLLECTING: {remaining}s")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] LOAD:{smoothed_avg_load:.1f} | 🔒 LOCKED: {current_up_threshold:.1f}")
                print(f"   ↳ {' '.join(server_details)}")

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