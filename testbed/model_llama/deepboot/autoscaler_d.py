import asyncio
import subprocess
import httpx
import time
import numpy as np
import asyncssh
from typing import List, Dict, Optional

# --- Configuration ---
HAPROXY_CONF_PATH = "/etc/haproxy/haproxy.cfg"
HAPROROXY_TEMPLATE_PATH = "/etc/haproxy/haproxy.cfg.template"
SERVER_COUNT_LOG_FILE = "./active_servers.log"
ACTIVE_WORKERS_FILE = "./active_workers.txt"

# Scaling Thresholds
SCALE_DOWN_THRESHOLD = 22
SCALE_UP_THRESHOLD = 38

# Scaling Rules
MIN_ACTIVE_SERVERS = 1
SCALING_COOLDOWN_SECONDS = 45
MONITOR_INTERVAL_SECONDS = 3
GPU_MEMORY_FREE_THRESHOLD_MB = 6000
GPU_FREE_TIMEOUT_SECONDS = 15
GPU_FREE_POLL_INTERVAL_SECONDS = 1

# vLLM Startup settings
VLLM_STARTUP_DELAY_SECONDS = 25  # Fixed warm-up time

# --- DeepBoot / ATS-I Parameters ---
DEEPBOOT_PROTECT_MIN_SECONDS = 30
DEEPBOOT_PROTECT_MAX_SECONDS = 120
DEEPBOOT_PROTECT_INCREMENT = 15

# --- Server State Management ---
ALL_SERVERS = [
    # Dedicated inference-only servers (no rank)
    # {"host": "10.10.3.3", "port": 8001, "status": "sleeping", "rank": 1, "shared": True},
    {"host": "10.10.3.3", "port": 8002, "status": "training", "rank": 2, "shared": True, "protect_expiry": 0.0, "usage_counter": 0},
    {"host": "10.10.3.3", "port": 8003, "status": "training", "rank": 3, "shared": True, "protect_expiry": 0.0, "usage_counter": 0},

    {"host": "10.10.3.2", "port": 8000, "status": "training", "rank": 4, "shared": True, "protect_expiry": 0.0, "usage_counter": 0},
    {"host": "10.10.3.2", "port": 8001, "status": "training", "rank": 5, "shared": True, "protect_expiry": 0.0, "usage_counter": 0},
    {"host": "10.10.3.2", "port": 8002, "status": "training", "rank": 6, "shared": True, "protect_expiry": 0.0, "usage_counter": 0},
    {"host": "10.10.3.2", "port": 8003, "status": "training", "rank": 7, "shared": True, "protect_expiry": 0.0, "usage_counter": 0},

    {"host": "10.10.3.1", "port": 8000, "status": "training", "rank": 8, "shared": True, "protect_expiry": 0.0, "usage_counter": 0},
    {"host": "10.10.3.1", "port": 8001, "status": "training", "rank": 9, "shared": True, "protect_expiry": 0.0, "usage_counter": 0},
    {"host": "10.10.3.1", "port": 8002, "status": "training", "rank": 10, "shared": True, "protect_expiry": 0.0, "usage_counter": 0},
    {"host": "10.10.3.1", "port": 8003, "status": "active", "shared": False},
    # Shared servers that have a corresponding training rank
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
    print(f"\n[DeepBoot] Updated active_workers.txt with ranks: {content}")

async def check_gpu_memory_is_free(server: Dict) -> bool:
    if not server.get("shared"): return True
    print(f"\n[DeepBoot] Waiting for GPU memory to be freed for rank {server['rank']}...")
    local_gpu_id = server['rank'] % 4
    command = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {local_gpu_id}"
    
    start_time = time.time()
    while (time.time() - start_time) < GPU_FREE_TIMEOUT_SECONDS:
        try:
            async with asyncssh.connect(server['host']) as conn:
                result = await conn.run(command, check=True)
                if int(result.stdout.strip()) < GPU_MEMORY_FREE_THRESHOLD_MB:
                    return True
            await asyncio.sleep(GPU_FREE_POLL_INTERVAL_SECONDS)
        except Exception:
            await asyncio.sleep(GPU_FREE_POLL_INTERVAL_SECONDS)
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

async def update_haproxy_config() -> bool:
    """Updates HAProxy with ONLY servers that are fully 'active'."""
    # Note: 'starting_up' servers are explicitly EXCLUDED until they finish warming up
    active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    
    server_lines = [
        f"    server web{i:02d} {s['host']}:{s['port']}\n"
        for i, s in enumerate(active_servers, start=1)
    ]
    try:
        with open(HAPROROXY_TEMPLATE_PATH, "r") as f: template = f.read()
        with open(HAPROXY_CONF_PATH, "w") as f: 
            f.write(template.replace("{UPSTREAM_SERVERS}", "".join(server_lines)))
        return True
    except Exception as e:
        print(f"ERROR: HAProxy config failed: {e}")
        return False

def reload_haproxy():
    try:
        subprocess.run(["sudo", "systemctl", "reload", "haproxy"], check=True)
        print("HAProxy reloaded.")
    except Exception: pass

async def set_server_sleep_state(server: Dict, sleep: bool):
    action, url = ("Releasing GPU", f"http://{server['host']}:{server['port']}/sleep?level=1") if sleep else \
                  ("Acquiring GPU", f"http://{server['host']}:{server['port']}/wake_up")
    print(f"[DeepBoot] {action}: {server['host']}:{server['port']}")
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, timeout=20)
    except Exception as e:
        print(f"ERROR: Failed to toggle sleep state: {e}")

# --- 🧠 DeepBoot Lifecycle Logic ---

async def transition_protected_to_training(server: Dict):
    print(f"\n[DeepBoot] 🛡️->🏋️ Server {server['host']}:{server['port']} protection expired. Giving empty GPU to training.")
    server['usage_counter'] = 0 
    server['status'] = 'training'
    
    current_ranks = read_active_workers()
    if server['rank'] not in current_ranks:
        current_ranks.append(server['rank'])
        write_active_workers(current_ranks)

async def deepboot_scale_down(count: int) -> bool:
    """ACTIVE -> PROTECTED. Releases GPU immediately."""
    active_shared = [s for s in ALL_SERVERS if s['status'] == 'active' and s['shared']]
    servers_to_protect = active_shared[:count]
    if not servers_to_protect: return False

    print(f"\n[DeepBoot] Scaling down {len(servers_to_protect)} servers to IDLE-PROTECTED state.")

    for server in servers_to_protect:
        server['status'] = 'protected'
        
        # [cite_start]Calculate protection time [cite: 586-588]
        protection_duration = min(
            DEEPBOOT_PROTECT_MAX_SECONDS, 
            DEEPBOOT_PROTECT_MIN_SECONDS + (server['usage_counter'] * DEEPBOOT_PROTECT_INCREMENT)
        )
        server['protect_expiry'] = time.time() + protection_duration
        
        # Release GPU immediately (Non-blocking HTTP call)
        await set_server_sleep_state(server, sleep=True)
        print(f"   -> {server['host']}:{server['port']} Idle-Protected for {protection_duration}s")

    # Update HAProxy immediately to stop traffic
    if await update_haproxy_config():
        reload_haproxy()
        return True
    
    return False

# --- NEW: Background Warmup Task ---
async def background_warmup_server(server: Dict):
    """Waits for fixed delay then promotes server to active."""
    print(f"   ⏳ [Background] Warming up {server['host']}:{server['port']} for {VLLM_STARTUP_DELAY_SECONDS}s...")
    
    # This sleep happens in the background, main loop continues!
    await asyncio.sleep(VLLM_STARTUP_DELAY_SECONDS)
    
    # Promotion
    print(f"   ✅ [Background] {server['host']}:{server['port']} warm-up complete. Adding to Load Balancer.")
    server['status'] = 'active'
    
    if await update_haproxy_config():
        reload_haproxy()

async def deepboot_scale_up(count: int) -> bool:
    """
    Initiates scale up but returns IMMEDIATELY so monitoring continues.
    """
    needed = count
    servers_to_start = []
    
    # --- Phase 1: Select from PROTECTED ---
    protected_servers = [s for s in ALL_SERVERS if s['status'] == 'protected']
    protected_servers.sort(key=lambda s: s['usage_counter'], reverse=True)
    
    while needed > 0 and protected_servers:
        srv = protected_servers.pop(0)
        servers_to_start.append(srv)
        needed -= 1

    # --- Phase 2: Reclaim from TRAINING ---
    if needed > 0:
        training_servers = [s for s in ALL_SERVERS if s['status'] == 'training']
        training_servers.sort(key=lambda s: s['rank']) 
        to_reclaim = training_servers[:needed]
        
        if to_reclaim:
            # Synchronous reclaim steps (Blocking briefly is OK for file I/O and checks)
            current_ranks = read_active_workers()
            reclaim_ranks = [s['rank'] for s in to_reclaim]
            new_ranks = [r for r in current_ranks if r not in reclaim_ranks]
            write_active_workers(new_ranks)
            
            print(f"[DeepBoot] Reclaiming {len(to_reclaim)} GPUs from training...")
            checks = await asyncio.gather(*[check_gpu_memory_is_free(s) for s in to_reclaim])
            
            if all(checks):
                for srv in to_reclaim:
                    srv['usage_counter'] = 0 # Reset for cold start
                    servers_to_start.append(srv)
            else:
                print("ERROR: Failed to reclaim GPUs. Rolling back.")
                write_active_workers(current_ranks)

    # --- LAUNCH BACKGROUND STARTUP ---
    if servers_to_start:
        for srv in servers_to_start:
            # Set intermediate status so autoscaler counts it as "pending"
            srv['status'] = 'starting_up'
            srv['usage_counter'] += 1
            
            # 1. Wake up vLLM (Start loading weights)
            print(f"[DeepBoot] 🛡️->🚀 Triggering wake-up for {srv['host']}:{srv['port']}")
            await set_server_sleep_state(srv, sleep=False)
            
            # 2. Fire and Forget the wait
            asyncio.create_task(background_warmup_server(srv))
            
        return True # Returns immediately!

    return False

# --- Background Tasks ---
async def lifecycle_manager_task():
    print("🚀 DeepBoot Lifecycle Manager started...")
    while True:
        now = time.time()
        expired = [s for s in ALL_SERVERS if s['status'] == 'protected' and s['protect_expiry'] < now]
        for server in expired:
            await transition_protected_to_training(server)
        await asyncio.sleep(1)

async def log_active_servers():
    while True:
        active = sum(1 for s in ALL_SERVERS if s['status'] == 'active')
        protected = sum(1 for s in ALL_SERVERS if s['status'] == 'protected')
        training = sum(1 for s in ALL_SERVERS if s['status'] == 'training')
        # We can also log 'starting_up' if we want visibility
        starting = sum(1 for s in ALL_SERVERS if s['status'] == 'starting_up')
        
        try:
            with open(SERVER_COUNT_LOG_FILE, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, Active:{active}, Starting:{starting}, Protected:{protected}, Training:{training}\n")
        except Exception: pass
        await asyncio.sleep(5)

async def autoscaler_task():
    print("🚀 Autoscaler started...")
    last_scaling_time = 0
    
    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
            
            # Metrics: Only check ACTIVE servers (starting_up can't answer yet)
            active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
            pending_servers = [s for s in ALL_SERVERS if s['status'] == 'starting_up']
            
            # Capacity Calculation: Include Pending servers to avoid over-scaling
            # If we have 2 active and 2 starting, effective count is 4 for decision making
            total_capacity_count = len(active_servers) + len(pending_servers)

            if not active_servers: 
                # Edge case: No active servers. 
                # If we have pending servers, just wait. If neither, we might be in trouble (or initial boot).
                if pending_servers:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Waiting for {len(pending_servers)} servers to start...")
                continue

            metric_tasks = [get_server_metrics(server, client) for server in active_servers]
            results = await asyncio.gather(*metric_tasks)
            
            total_load = sum(r['running'] + r['waiting'] for r in results)
            
            # Load Calculation: 
            # We divide load by (active + pending) to simulate "projected load" once pending join.
            # This prevents triggering another scale-up while 2 servers are warming up.
            projected_avg_load = total_load / max(1, total_capacity_count)

            print(f"\n[{time.strftime('%H:%M:%S')}] Active: {len(active_servers)} (Pending: {len(pending_servers)}) | Projected Load: {projected_avg_load:.2f}")
            
            for s in ALL_SERVERS:
                if s['status'] == 'protected':
                    ttl = int(s['protect_expiry'] - time.time())
                    print(f"   [Idle-Protected] {s['host']} (TTL: {ttl}s)")
                elif s['status'] == 'starting_up':
                    print(f"   [Starting Up] {s['host']}")

            # --- Scaling Logic ---
            if (time.time() - last_scaling_time) > SCALING_COOLDOWN_SECONDS:
                
                # Scale Down
                if projected_avg_load < SCALE_DOWN_THRESHOLD:
                    excess = int(total_capacity_count * 0.2) or 1
                    actual_remove = min(excess, len([s for s in active_servers if s['shared']]))
                    
                    if actual_remove > 0 and (total_capacity_count - actual_remove) >= MIN_ACTIVE_SERVERS:
                        if await deepboot_scale_down(actual_remove):
                            last_scaling_time = time.time()

                # Scale Up
                elif projected_avg_load > SCALE_UP_THRESHOLD:
                    # We scale up by 1 at a time, but since it's non-blocking,
                    # the COOLDOWN ensures we don't spam 10 requests in 1 second.
                    if await deepboot_scale_up(1):
                        last_scaling_time = time.time()

# --- Main ---
async def startup_tasks():
    # Initial Sync
    await update_haproxy_config()
    reload_haproxy()
    
    loop = asyncio.get_event_loop()
    loop.create_task(log_active_servers())
    loop.create_task(lifecycle_manager_task())
    loop.create_task(autoscaler_task())
    await asyncio.Future() 

if __name__ == "__main__":
    try: asyncio.run(startup_tasks())
    except KeyboardInterrupt: pass