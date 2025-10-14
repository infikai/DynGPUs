import asyncio
import subprocess
import httpx
import time
import numpy as np
import asyncssh
from typing import List, Dict
from collections import Counter

# --- ⚙️ Configuration ---
# File Paths
NGINX_CONF_PATH = "/etc/nginx/nginx.conf"
NGINX_TEMPLATE_PATH = "/etc/nginx/nginx.conf.template"
SERVER_COUNT_LOG_FILE = "./active_servers.log"
HOROVOD_HOSTFILE_PATH = "/mydata/Data/DynGPUs/horovod/hostfile.txt"

# Node Mapping & GPU Info
NODE_IP_MAPPING = {"node1": "10.10.3.1", "node2": "10.10.3.2"}
GPUS_PER_NODE = 4

# Scaling Thresholds (based on average (running + waiting) requests per server)
SCALE_DOWN_THRESHOLD = 45
SCALE_UP_THRESHOLD = 60

# Scaling Rules
MIN_ACTIVE_SERVERS = 4
SCALING_COOLDOWN_SECONDS = 40
MONITOR_INTERVAL_SECONDS = 2
GPU_MEMORY_FREE_THRESHOLD_MB = 2000

# --- 🖥️ Server State Management ---
# `rank` is used again to map a server to a specific GPU slot.
ALL_SERVERS = [
    # Dedicated inference-only servers (on node1)
    {"host": "10.10.3.1", "port": 8000, "status": "active", "shared": False},
    {"host": "10.10.3.1", "port": 8001, "status": "active", "shared": False},
    {"host": "10.10.3.1", "port": 8002, "status": "active", "shared": False},
    {"host": "10.10.3.1", "port": 8003, "status": "active", "shared": False},
    # Shared servers that can be used for training (on node2)
    {"host": "10.10.3.2", "port": 8000, "status": "sleeping", "rank": 4, "shared": True},
    {"host": "10.10.3.2", "port": 8001, "status": "sleeping", "rank": 5, "shared": True},
    {"host": "10.10.3.2", "port": 8002, "status": "sleeping", "rank": 6, "shared": True},
    {"host": "10.10.3.2", "port": 8003, "status": "sleeping", "rank": 7, "shared": True},
]


# --- Helper Functions ---

def read_active_hosts() -> Dict[str, int]:
    """Reads the horovod hostfile and returns a dict of {ip: gpu_slot_count}."""
    active_hosts = {}
    try:
        with open(HOROVOD_HOSTFILE_PATH, "r") as f:
            for line in f:
                if not line.strip(): continue
                name, count = line.strip().split(':')
                ip = NODE_IP_MAPPING.get(name)
                if ip:
                    active_hosts[ip] = int(count)
    except FileNotFoundError:
        pass
    return active_hosts

def write_active_hosts(hosts: Dict[str, int]):
    """Writes the dict of {ip: gpu_slot_count} to the horovod hostfile."""
    ip_to_node_mapping = {v: k for k, v in NODE_IP_MAPPING.items()}
    lines = []
    for ip, count in sorted(hosts.items()):
        node_name = ip_to_node_mapping.get(ip)
        if node_name and count > 0: # Only write hosts with slots > 0
            lines.append(f"{node_name}:{count}")

    content = "\n".join(lines)
    with open(HOROVOD_HOSTFILE_PATH, "w") as f:
        f.write(content)
    print(f"\nUpdated {HOROVOD_HOSTFILE_PATH} with content:\n---\n{content}\n---")

async def check_gpu_memory_is_free(server: Dict) -> bool:
    """Connects via SSH to run nvidia-smi and check if a specific GPU's memory is free."""
    if not server.get("shared"): return True
    
    local_gpu_id = server['rank'] % GPUS_PER_NODE
    print(f"\nChecking GPU memory for rank {server['rank']} (GPU {local_gpu_id}) on {server['host']}...")
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
    url = f"http://{server['host']}:{server['port']}/metrics"
    running, waiting = 0.0, 0.0
    try:
        response = await client.get(url, timeout=5)
        response.raise_for_status()
        for line in response.text.split('\n'):
            if line.startswith("vllm:num_requests_running"): running = float(line.rsplit(' ', 1)[1])
            elif line.startswith("vllm:num_requests_waiting"): waiting = float(line.rsplit(' ', 1)[1])
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
    """Scales down by shutting down `count` shared servers and adding their slots to Horovod."""
    active_servers = [s for s in ALL_SERVERS if s['status'] == 'active']
    shared_active_servers = [s for s in active_servers if s['shared']]
    
    max_possible_to_remove = len(active_servers) - MIN_ACTIVE_SERVERS
    actual_count = min(count, len(shared_active_servers), max_possible_to_remove)
    
    if actual_count <= 0:
        print("\nScale-down skipped: No shared servers available or minimum would be breached.")
        return False

    servers_to_scale_down = shared_active_servers[:actual_count]
    print(f"\nScaling down by {len(servers_to_scale_down)} servers...")
    
    for server in servers_to_scale_down:
        server['status'] = 'sleeping'
    
    if not await update_nginx_config([s for s in ALL_SERVERS if s['status'] == 'active']):
        for server in servers_to_scale_down: server['status'] = 'active' # Revert
        return False
    reload_nginx()
    
    # Gracefully shut down servers, then update hostfile once.
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
    
    hosts_to_update = Counter(s['host'] for s in servers_to_scale_down)
    active_hosts = read_active_hosts()
    for host, num_slots in hosts_to_update.items():
        print(f"Adding {num_slots} slots to host {host} for training.")
        active_hosts[host] = active_hosts.get(host, 0) + num_slots
    write_active_hosts(active_hosts)
    
    return True

async def scale_up(count: int) -> bool:
    """Scales up by reclaiming `count` slots from Horovod and waking up their servers."""
    sleeping_shared_servers = [s for s in ALL_SERVERS if s['status'] == 'sleeping' and s['shared']]
    actual_count = min(count, len(sleeping_shared_servers))
    
    if actual_count <= 0:
        print("\nScale-up skipped: No sleeping shared servers available.")
        return False
        
    servers_to_wake = sleeping_shared_servers[:actual_count]
    print(f"\nScaling up by {len(servers_to_wake)} servers...")
    successfully_woken = []
    
    # --- Step 1: Update Horovod hostfile first ---
    hosts_to_update = Counter(s['host'] for s in servers_to_wake)
    active_hosts_before = read_active_hosts()
    active_hosts_after = active_hosts_before.copy()
    
    for host, num_slots in hosts_to_update.items():
        if active_hosts_after.get(host, 0) < num_slots:
            print(f"ERROR: Cannot reclaim {num_slots} from host {host}, only {active_hosts_after.get(host, 0)} are in use. Aborting scale up.")
            return False
        active_hosts_after[host] -= num_slots
    write_active_hosts(active_hosts_after)
    await asyncio.sleep(3) # Give Horovod time to adjust

    # --- Step 2: Wake up servers one by one ---
    for server in servers_to_wake:
        if not await check_gpu_memory_is_free(server):
            print(f"ERROR: GPU memory check failed for rank {server['rank']}. Skipping.")
            continue
        
        await set_server_sleep_state(server, sleep=False)
        server['status'] = 'active'
        successfully_woken.append(server)

    if not successfully_woken:
        print("No servers were woken up. Reverting hostfile change.")
        write_active_hosts(active_hosts_before)
        return False

    # --- Step 3: Update Nginx ---
    if await update_nginx_config([s for s in ALL_SERVERS if s['status'] == 'active']):
        reload_nginx()
        return True
    
    print("ERROR: Nginx update failed. Reverting state and hostfile...")
    # Revert status for woken servers and adjust hostfile back
    for server in successfully_woken:
        server['status'] = 'sleeping'
    write_active_hosts(active_hosts_before)
    return False

# --- Background Tasks ---

async def log_active_servers():
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

            # Proportional scaling logic
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
    if update_nginx_config([s for s in ALL_SERVERS if s['status'] == 'active']):
        reload_nginx()

    loop = asyncio.get_event_loop()
    loop.create_task(log_active_servers())
    loop.create_task(autoscaler_task())
    
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("\nAutoscaler stopped by user.")