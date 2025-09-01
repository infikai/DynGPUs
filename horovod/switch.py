import subprocess
import time
import random

# --- Configuration ---
# 1. Define all servers in the cluster.
ALL_SERVERS = ["node0", "node1", "node2", "node3", "node4"]
HOSTS_FILE = "hosts.txt"

# 2. Define the two thresholds for resource adjustment.
HIGH_LOAD_THRESHOLD = 75  # If load goes ABOVE this, add an inference server.
LOW_LOAD_THRESHOLD = 25   # If load goes BELOW this, remove an inference server.

# 3. Define the minimum number of servers required for training.
MIN_TRAINING_SERVERS = 2

# --- Commands (Templates) ---
TRAINING_COMMAND = (
    f"horovodrun -np {len(ALL_SERVERS)} --min-np {MIN_TRAINING_SERVERS} --max-np {len(ALL_SERVERS)} "
    "--network-interface eth0 --host-discovery-script ./discover_hosts.sh "
    "python train_elastic_measurement.py --batch-size 64 "
    "--train-dir /mydata/Data/imagenet/train --val-dir /mydata/Data/imagenet/val"
)

INFERENCE_COMMAND_TEMPLATE = (
    "docker run --name triton_server_{hostname} --gpus=1 --rm --net=host "
    "-v /mydata/Data/server/docs/examples/model_repository:/models "
    "nvcr.io/nvidia/tritonserver:25.02-py3 tritonserver --model-repository=/models "
    "--model-control-mode explicit --load-model densenet_onnx"
)

STOP_INFERENCE_COMMAND_TEMPLATE = "docker stop triton_server_{hostname}"

# Command to check for active compute processes on a GPU
GPU_CHECK_COMMAND = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"


def update_hosts_file(servers):
    """Writes the list of active training servers to the hosts.txt file."""
    print(f"INFO: Updating '{HOSTS_FILE}' with active training hosts: {sorted(servers)}")
    with open(HOSTS_FILE, "w") as f:
        for server in sorted(servers):
            f.write(f"{server}\n")

def run_remote_command_async(hostname, command):
    """Executes a command on a remote host via SSH in a non-blocking way."""
    ssh_command = f"ssh {hostname} \"{command}\""
    print(f"INFO: Executing async on {hostname}: {ssh_command}")
    return subprocess.Popen(ssh_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def wait_for_gpu_to_be_free(hostname, poll_interval=3, timeout=60):
    """
    Polls a remote node using nvidia-smi until the GPU is free or a timeout occurs.
    Returns True if the GPU becomes free, False otherwise.
    """
    print(f"INFO: Waiting for GPU on '{hostname}' to become free... (Timeout: {timeout}s)")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Use subprocess.run for a blocking call that captures output
            ssh_command = f"ssh {hostname} \"{GPU_CHECK_COMMAND}\""
            result = subprocess.run(
                ssh_command,
                shell=True,
                capture_output=True,
                text=True,
                check=True # Raise exception on non-zero exit code
            )
            # If stdout is empty (just whitespace), no processes are running.
            if not result.stdout.strip():
                print(f"SUCCESS: GPU on '{hostname}' is now free.")
                return True
            else:
                print(f"INFO: GPU on '{hostname}' is still busy. Processes found: {result.stdout.strip().replace(chr(10),', ')}. Retrying in {poll_interval}s...")

        except subprocess.CalledProcessError as e:
            print(f"WARNING: Error checking GPU status on {hostname}: {e.stderr}")
            # Potentially handle specific errors, e.g., if SSH fails
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while checking GPU on {hostname}: {e}")
            return False # Exit on unexpected errors

        time.sleep(poll_interval)

    print(f"ERROR: Timeout! GPU on '{hostname}' did not become free within {timeout} seconds.")
    return False


def main():
    """Main control loop for managing training and inference servers."""
    active_training_servers = list(ALL_SERVERS)
    inference_servers = []

    # --- Initial State ---
    update_hosts_file(active_training_servers)
    print("="*60 + f"\nINITIAL SETUP (Time: {time.ctime()})\n" + "="*60)
    print("Please ensure the Horovod master process is running in a separate terminal:")
    print(f"\n{TRAINING_COMMAND}\n")
    input("Press Enter to begin the dynamic control loop...")

    load_variable = 50.0  # Start in the "dead zone"

    try:
        while True:
            print("\n" + "="*60)
            print(f"TIMESTAMP: {time.ctime()}")
            print(f"CURRENT LOAD METRIC: {load_variable:.2f} (Thresholds: Low < {LOW_LOAD_THRESHOLD} | High > {HIGH_LOAD_THRESHOLD})")
            print(f"STATE: {len(active_training_servers)} Training Servers | {len(inference_servers)} Inference Servers")
            print("="*60)

            # --- SCALE DOWN LOGIC (Training -> Inference) ---
            if load_variable > HIGH_LOAD_THRESHOLD and len(active_training_servers) > MIN_TRAINING_SERVERS:
                server_to_reclaim = active_training_servers.pop(0)
                print(f"\nACTION: High load! Attempting to reclaim '{server_to_reclaim}' for inference.")
                update_hosts_file(active_training_servers)

                # Actively wait for the Horovod process to terminate on the remote node
                if wait_for_gpu_to_be_free(server_to_reclaim, timeout=45):
                    inference_servers.append(server_to_reclaim)
                    inference_command = INFERENCE_COMMAND_TEMPLATE.format(hostname=server_to_reclaim)
                    run_remote_command_async(server_to_reclaim, inference_command)
                    print(f"INFO: Inference server launch command sent to '{server_to_reclaim}'.")
                else:
                    print(f"ERROR: Failed to reclaim '{server_to_reclaim}'. Adding it back to the training pool.")
                    active_training_servers.append(server_to_reclaim) # Revert change
                    update_hosts_file(active_training_servers) # Update file back to original state

            # --- SCALE UP LOGIC (Inference -> Training) ---
            elif load_variable < LOW_LOAD_THRESHOLD and len(inference_servers) > 0:
                server_to_return = inference_servers.pop(0)
                print(f"\nACTION: Low load! Attempting to return '{server_to_return}' to training pool.")
                stop_command = STOP_INFERENCE_COMMAND_TEMPLATE.format(hostname=server_to_return)
                run_remote_command_async(server_to_return, stop_command)

                # Actively wait for the Docker container to stop and release the GPU
                if wait_for_gpu_to_be_free(server_to_return, timeout=30):
                    active_training_servers.append(server_to_return)
                    update_hosts_file(active_training_servers)
                    print("INFO: Horovod will now discover the new node and scale up automatically.")
                else:
                    print(f"ERROR: Failed to stop inference on '{server_to_return}'. It remains in the inference pool.")
                    inference_servers.append(server_to_return) # Revert change

            # --- SIMULATE LOAD CHANGES ---
            time.sleep(20)
            if len(inference_servers) > 0:
                load_variable -= random.uniform(5, 15)
            else:
                load_variable += random.uniform(5, 15)
            load_variable = max(0, min(100, load_variable))

    except KeyboardInterrupt:
        print("\n\n--- Control script interrupted. ---")

if __name__ == "__main__":
    main()