import time
import os
import signal
import multiprocessing
import threading
import subprocess # For running Docker command
import shutil # For removing checkpoint directory if needed (optional)

# Attempt to import PyTorch, provide guidance if not found
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder # For loading ImageNet
    from torch.utils.data import DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("WARNING: PyTorch is not installed. The 'train' function will be a placeholder or will fail if real data loading is expected.")
    print("Please install PyTorch and torchvision with: pip install torch torchvision")
    # Define dummy classes if PyTorch is not available to prevent NameErrors later
    class Dataset: pass
    class ImageFolder(Dataset): pass


# --- User-defined functions ---

CHECKPOINT_DIR = "./checkpoints"
IMAGENET_DATA_PATH = "/mydata/Data/imagenet" # User-specified path
DOCKER_CONTAINER_NAME = "triton_server_instance_pm" # Unique name for the container

def save_checkpoint(epoch, model, optimizer, loss, filename_prefix="checkpoint"):
    """Saves model checkpoint and measures time taken."""
    if not PYTORCH_AVAILABLE:
        return

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"[TRAIN PID {os.getpid()}] Created checkpoint directory: {CHECKPOINT_DIR}")

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    epoch_filename = os.path.join(CHECKPOINT_DIR, f"{filename_prefix}_epoch_{epoch}.pth")
    start_time_epoch_save = time.time()
    torch.save(state, epoch_filename)
    end_time_epoch_save = time.time()
    print(f"[TRAIN PID {os.getpid()}] Saved epoch checkpoint to {epoch_filename} (took {end_time_epoch_save - start_time_epoch_save:.2f}s)")

    latest_filename = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
    start_time_latest_save = time.time()
    torch.save(state, latest_filename)
    end_time_latest_save = time.time()
    print(f"[TRAIN PID {os.getpid()}] Updated latest checkpoint to {latest_filename} (took {end_time_latest_save - start_time_latest_save:.2f}s)")


def load_checkpoint(model, optimizer, filename="latest_checkpoint.pth"):
    """Loads model checkpoint."""
    if not PYTORCH_AVAILABLE:
        return 0, None

    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.isfile(filepath):
        print(f"[TRAIN PID {os.getpid()}] Loading checkpoint '{filepath}'")
        try:
            checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            last_loss = checkpoint.get('loss', None)
            print(f"[TRAIN PID {os.getpid()}] Loaded checkpoint '{filepath}' (epoch {checkpoint['epoch']}, loss: {last_loss})")
            print(f"[TRAIN PID {os.getpid()}] Resuming training from epoch {start_epoch}")
            return start_epoch, last_loss
        except Exception as e:
            print(f"[TRAIN PID {os.getpid()}] Error loading checkpoint {filepath}: {e}. Starting from scratch.")
            return 0, None
    else:
        print(f"[TRAIN PID {os.getpid()}] No checkpoint found at '{filepath}'. Starting from scratch.")
        return 0, None


def train(st):
    """
    PyTorch training function for ResNet50 with ImageNet data.
    Includes checkpointing.
    """
    if not PYTORCH_AVAILABLE:
        print(f"[TRAIN PID {os.getpid()}] PyTorch not available. Running placeholder training.")
        # Placeholder loop if PyTorch is not available
        try:
            while True:
                print(f"[TRAIN PID {os.getpid()}] Placeholder training batch processing...")
                time.sleep(5)
        except KeyboardInterrupt:
            print(f"[TRAIN PID {os.getpid()}] Placeholder training received KeyboardInterrupt. Exiting.")
        finally:
            print(f"[TRAIN PID {os.getpid()}] Placeholder training finished.")
        return

    start_time_training_init = time.time()

    print(f"[TRAIN PID {os.getpid()}] PyTorch Training started with ResNet50 on ImageNet data.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN PID {os.getpid()}] Using device: {device}")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) # Removed PYTORCH_AVAILABLE check as it's guarded above
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    start_epoch, _ = load_checkpoint(model, optimizer)

    model.train()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = os.path.join(IMAGENET_DATA_PATH, 'train')
    print(f"[TRAIN PID {os.getpid()}] Attempting to load ImageNet training data from: {train_dir}")

    try:
        train_dataset = ImageFolder(train_dir, train_transform)
    except FileNotFoundError:
        print(f"[TRAIN PID {os.getpid()}] ERROR: ImageNet training data not found at {train_dir}. Please check the path.")
        print(f"[TRAIN PID {os.getpid()}] Exiting training function due to missing data.")
        return
    except Exception as e:
        print(f"[TRAIN PID {os.getpid()}] ERROR: Could not load ImageNet dataset: {e}")
        print(f"[TRAIN PID {os.getpid()}] Exiting training function.")
        return


    if len(train_dataset) == 0:
        print(f"[TRAIN PID {os.getpid()}] ERROR: ImageNet training dataset at {train_dir} is empty or could not find any images.")
        print(f"[TRAIN PID {os.getpid()}] Please ensure the directory structure is correct (e.g., {train_dir}/class_name/image.JPEG).")
        print(f"[TRAIN PID {os.getpid()}] Exiting training function.")
        return

    print(f"[TRAIN PID {os.getpid()}] Successfully loaded {len(train_dataset)} images from ImageNet training set.")

    num_dataloader_workers = 32 if device.type == 'cuda' else 0
    batch_size = 256 if device.type == 'cuda' else 32
    print(f"[TRAIN PID {os.getpid()}] DataLoader using num_workers={num_dataloader_workers}, batch_size={batch_size}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_dataloader_workers, pin_memory=True if device.type == 'cuda' else False)

    max_epochs = 100
    running = True
    current_epoch_loss = 0.0
    current_epoch_batches = 0
    epoch_for_interrupt_save = start_epoch

    end_time_training_init = time.time()
    print(f"fully starting training job took {end_time_training_init - start_time_training_init:.2f}s")

    try:
        for epoch in range(start_epoch, max_epochs):
            epoch_for_interrupt_save = epoch
            if not running: break
            print(f"[TRAIN PID {os.getpid()}] Epoch {epoch+1}/{max_epochs}")
            epoch_loss_aggregator = 0.0
            num_batches_in_epoch = 0

            start_time_moving = time.time()

            for i, (inputs, labels) in enumerate(train_loader):
                if not running: break

                end_time_moving = time.time()
                if i == 0:
                    print(f"Init batches loop took {end_time_moving - start_time_moving:.2f}s")
                if start_epoch == epoch and i == 0:
                    END_TIME_I2T = time.time()
                    print(f"Fully I2T took {END_TIME_I2T - st:.2f}s")

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss_aggregator += loss.item()
                num_batches_in_epoch += 1
                current_epoch_loss = epoch_loss_aggregator
                current_epoch_batches = num_batches_in_epoch

                end_time_batch = time.time()

                if (i + 1) % 100 == 0:
                    print(f"The {i+1}th batch took {end_time_batch - end_time_moving:.2f}s")
                    print(f"[TRAIN PID {os.getpid()}] Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

            if not running: break

            avg_epoch_loss = epoch_loss_aggregator / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
            print(f"[TRAIN PID {os.getpid()}] Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")

            save_checkpoint(epoch, model, optimizer, avg_epoch_loss)

        if running:
             print(f"[TRAIN PID {os.getpid()}] Training completed {max_epochs} epochs (or ran out of specified range).")

    except KeyboardInterrupt:
        print(f"[TRAIN PID {os.getpid()}] Training function received KeyboardInterrupt. Saving final checkpoint...")
        running = False
        final_loss_to_save = current_epoch_loss / current_epoch_batches if current_epoch_batches > 0 else None

        print(f"[TRAIN PID {os.getpid()}] Attempting to save interrupt checkpoint for epoch {epoch_for_interrupt_save}...")
        save_checkpoint(epoch_for_interrupt_save, model, optimizer, final_loss_to_save, filename_prefix="interrupt_checkpoint")
        print(f"[TRAIN PID {os.getpid()}] Training function cleanup complete after interrupt. Exiting.")
    except Exception as e:
        print(f"[TRAIN PID {os.getpid()}] Error in PyTorch training: {e}")
        try:
            print(f"[TRAIN PID {os.getpid()}] Attempting to save crash checkpoint for epoch {epoch_for_interrupt_save}...")
            save_checkpoint(epoch_for_interrupt_save, model, optimizer, None, filename_prefix="crash_checkpoint")
        except Exception as ce:
            print(f"[TRAIN PID {os.getpid()}] Could not save crash checkpoint: {ce}")
    finally:
        print(f"[TRAIN PID {os.getpid()}] PyTorch Training function finished.")


def serving(st):
    """
    Starts the NVIDIA Triton Inference Server using Docker.
    Manages the Docker subprocess.
    """
    print(f"[SERVING PID {os.getpid()}] Serving function started. Attempting to launch Triton Docker container...")

    docker_command_str = (
        f"docker run --name {DOCKER_CONTAINER_NAME} --gpus=1 --rm --net=host "
        f"-v /mydata/Data/server/docs/examples/model_repository:/models "
        f"nvcr.io/nvidia/tritonserver:25.02-py3 "
        f"tritonserver --model-repository=/models --model-control-mode explicit --load-model densenet_onnx"
    )
    docker_command_list = docker_command_str.split()
    docker_process = None
    process_group_id = None

    try:
        print(f"[SERVING PID {os.getpid()}] Executing Docker command: {' '.join(docker_command_list)}")
        docker_process = subprocess.Popen(
            docker_command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        if docker_process.pid is not None:
            process_group_id = os.getpgid(docker_process.pid)
            print(f"[SERVING PID {os.getpid()}] Triton Docker container process started (PID: {docker_process.pid}, Process Group ID: {process_group_id}). Monitoring...")
            END_TIME_T2I = time.time()
            print(f"Fully T2I took {END_TIME_T2I - st:.2f}s")
        else:
            print(f"[SERVING PID {os.getpid()}] Failed to get PID from Popen for Docker command. Cannot get PGID.")
            raise Exception("Failed to start Docker process and get PID.")

        while docker_process.poll() is None:
            time.sleep(1)
        print(f"[SERVING PID {os.getpid()}] Docker Popen process (PID: {docker_process.pid}) has exited with code {docker_process.returncode}.")

    except FileNotFoundError:
        print(f"[SERVING PID {os.getpid()}] Error: 'docker' command not found. Is Docker installed and in PATH?")
    except Exception as e:
        print(f"[SERVING PID {os.getpid()}] Error in serving function (launching/managing Docker): {e}")
    finally:
        print(f"[SERVING PID {os.getpid()}] ##### SERVING FUNCTION FINALLY BLOCK ENTERED ({DOCKER_CONTAINER_NAME}) #####")

        print(f"[SERVING PID {os.getpid()}] Attempting 'docker stop {DOCKER_CONTAINER_NAME}' (if not already stopped)...")
        try:
            stop_result = subprocess.run(
                ["docker", "stop", DOCKER_CONTAINER_NAME],
                timeout=20,
                capture_output=True, text=True, check=False
            )
            if stop_result.returncode == 0:
                print(f"[SERVING PID {os.getpid()}] 'docker stop {DOCKER_CONTAINER_NAME}' succeeded via finally block.")
            else:
                print(f"[SERVING PID {os.getpid()}] 'docker stop {DOCKER_CONTAINER_NAME}' (in finally) finished with RC {stop_result.returncode}. Stderr: {stop_result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"[SERVING PID {os.getpid()}] 'docker stop {DOCKER_CONTAINER_NAME}' (in finally) timed out.")
        except FileNotFoundError:
            print(f"[SERVING PID {os.getpid()}] 'docker' command not found (in finally) while trying to stop container.")
        except Exception as e_docker_stop:
            print(f"[SERVING PID {os.getpid()}] Error during 'docker stop {DOCKER_CONTAINER_NAME}' (in finally): {e_docker_stop}")

        if docker_process and docker_process.pid is not None:
            if docker_process.poll() is None:
                print(f"[SERVING PID {os.getpid()}] Popen object for 'docker run' (PID: {docker_process.pid}) still active (in finally). Attempting to terminate its process group.")
                if process_group_id:
                    try:
                        os.killpg(process_group_id, signal.SIGTERM)
                        docker_process.wait(timeout=5)
                        print(f"[SERVING PID {os.getpid()}] Process group (PGID: {process_group_id}) for 'docker run' terminated after SIGTERM (in finally).")
                    except ProcessLookupError:
                        print(f"[SERVING PID {os.getpid()}] Process group (PGID: {process_group_id}) not found during SIGTERM (in finally).")
                    except subprocess.TimeoutExpired:
                        print(f"[SERVING PID {os.getpid()}] Process group (PGID: {process_group_id}) did not terminate after SIGTERM (in finally). Killing group...")
                        try:
                            os.killpg(process_group_id, signal.SIGKILL)
                            docker_process.wait(timeout=2)
                            print(f"[SERVING PID {os.getpid()}] Process group (PGID: {process_group_id}) killed (in finally).")
                        except Exception as e_pg_kill:
                            print(f"[SERVING PID {os.getpid()}] Error killing process group (PGID: {process_group_id}) (in finally): {e_pg_kill}")
                    except Exception as e_pg_term:
                         print(f"[SERVING PID {os.getpid()}] Error terminating process group (PGID: {process_group_id}) (in finally): {e_pg_term}")
                else:
                    print(f"[SERVING PID {os.getpid()}] No PGID, attempting to terminate Popen process (PID: {docker_process.pid}) directly (in finally).")
                    docker_process.terminate()
                    try:
                        docker_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        docker_process.kill()
                        docker_process.wait(timeout=2) # Added wait after kill

            if docker_process.poll() is not None:
                 print(f"[SERVING PID {os.getpid()}] Popen object for 'docker run' (PID: {docker_process.pid}) has exited (in finally).")
            else:
                 print(f"[SERVING PID {os.getpid()}] Popen object for 'docker run' (PID: {docker_process.pid}) still alive after all attempts (in finally). Forcibly killing.")
                 docker_process.kill()
                 try:
                    docker_process.wait(timeout=2)
                 except subprocess.TimeoutExpired:
                    print(f"[SERVING PID {os.getpid()}] Popen object kill timed out (in finally).")
        print(f"[SERVING PID {os.getpid()}] Serving function finished.")


# --- Global state and configuration ---
Load = 0
Max_Load = 50
SYS_STATE = 1

train_process = None
serving_process = None
process_management_lock = threading.Lock()
stop_monitor_event = threading.Event()

# --- Worker entry points for multiprocessing ---
def train_worker_entry(st):
    print(f"MONITOR: Train worker process (PID: {os.getpid()}) starting train().")
    s = st.value
    train(s)
    print(f"MONITOR: Train worker process (PID: {os.getpid()}) exiting.")

def serving_worker_entry(st):
    print(f"MONITOR: Serving worker process (PID: {os.getpid()}) starting serving().")
    s = st.value
    serving(s)
    print(f"MONITOR: Serving worker process (PID: {os.getpid()}) exiting.")

# --- Core logic: Process Management ---
def manage_processes():
    global train_process, serving_process, Load, Max_Load, SYS_STATE
    START_TIME_I2T = multiprocessing.Value('d', 1)

    with process_management_lock:
        train_is_effectively_running = train_process is not None and train_process.is_alive()
        serving_is_effectively_running = serving_process is not None and serving_process.is_alive()

        if Load > Max_Load:
            if SYS_STATE == 0:
                strat_time_switching = time.time()
                START_TIME_T2I = multiprocessing.Value('d', time.time())
            if train_is_effectively_running:
                print(f"MONITOR: Load ({Load}) > Max_Load ({Max_Load}). Attempting to shut down training (PID: {train_process.pid}).")
                try:
                    os.kill(train_process.pid, signal.SIGINT)
                    train_process.join(timeout=30)
                    if train_process.is_alive():
                        print(f"MONITOR: Training (PID: {train_process.pid}) did not stop after SIGINT (30s). Forcing kill.")
                        train_process.kill()
                        train_process.join(timeout=5)
                except ProcessLookupError:
                    print(f"MONITOR: Training process (PID: {train_process.pid}) not found.")
                except Exception as e:
                    print(f"MONITOR: Error stopping training process (PID: {train_process.pid if train_process else 'N/A'}): {e}")
                finally:
                    if train_process and not train_process.is_alive():
                        train_process.close()
                        train_process = None
                        print("MONITOR: Training process confirmed shut down.")
                    elif train_process:
                         print(f"MONITOR: Training process (PID: {train_process.pid}) still alive after shutdown attempts.")

            serving_is_effectively_running = serving_process is not None and serving_process.is_alive()
            if not serving_is_effectively_running:
                if serving_process and not serving_process.is_alive():
                    serving_process.close()
                    serving_process = None
                print(f"MONITOR: Load ({Load}) > Max_Load ({Max_Load}). Starting serving...")
                serving_process = multiprocessing.Process(target=serving_worker_entry, args=(START_TIME_T2I,), daemon=False)
                serving_process.start()
                print(f"MONITOR: Serving process started (PID: {serving_process.pid}).")
            if SYS_STATE == 0:
                end_time_switching = time.time()
                print(f"Shuting train and call infer up took {end_time_switching - strat_time_switching:.2f}s")
            SYS_STATE = 1


        elif Load <= Max_Load:
            if SYS_STATE == 1:
                strat_time_switching = time.time()
                START_TIME_I2T = multiprocessing.Value('d', time.time())
            if serving_is_effectively_running:
                print(f"MONITOR: Load ({Load}) <= Max_Load ({Max_Load}). Shutting down serving (Docker container: {DOCKER_CONTAINER_NAME}, Python Process PID: {serving_process.pid}).")

                # Step 1: Directly try to stop the Docker container by name
                print(f"MONITOR: Attempting 'docker stop {DOCKER_CONTAINER_NAME}' directly...")
                try:
                    stop_result = subprocess.run(
                        ["docker", "stop", DOCKER_CONTAINER_NAME],
                        timeout=20, # Give Docker time to stop Triton
                        capture_output=True, text=True, check=False
                    )
                    if stop_result.returncode == 0:
                        print(f"MONITOR: 'docker stop {DOCKER_CONTAINER_NAME}' succeeded from manage_processes.")
                    else:
                        print(f"MONITOR: 'docker stop {DOCKER_CONTAINER_NAME}' (from manage_processes) finished with RC {stop_result.returncode}. Stderr: {stop_result.stderr.strip()}")
                except subprocess.TimeoutExpired:
                    print(f"MONITOR: 'docker stop {DOCKER_CONTAINER_NAME}' (from manage_processes) timed out.")
                except FileNotFoundError:
                    print(f"MONITOR: 'docker' command not found (from manage_processes) while trying to stop container.")
                except Exception as e_docker_stop_direct:
                    print(f"MONITOR: Error during direct 'docker stop {DOCKER_CONTAINER_NAME}' (from manage_processes): {e_docker_stop_direct}")

                # Step 2: Terminate the Python process hosting the serving() function
                print(f"MONITOR: Terminating Python serving process (PID: {serving_process.pid}).")
                try:
                    serving_process.terminate()
                    serving_process.join(timeout=15) # Reduced timeout as docker stop was attempted first
                                                      # This join is for the Python process itself.
                    if serving_process.is_alive():
                        print(f"MONITOR: Serving Python process (PID: {serving_process.pid}) did not stop after terminate (15s). Forcing kill.")
                        serving_process.kill()
                        serving_process.join(timeout=10)
                except ProcessLookupError:
                     print(f"MONITOR: Serving Python process (PID: {serving_process.pid}) not found during terminate/kill.")
                except Exception as e:
                    print(f"MONITOR: Error stopping serving Python process (PID: {serving_process.pid if serving_process else 'N/A'}): {e}")
                finally:
                    if serving_process and not serving_process.is_alive():
                        serving_process.close()
                        serving_process = None
                        print("MONITOR: Serving Python process confirmed shut down.")
                    elif serving_process:
                        print(f"MONITOR: Serving Python process (PID: {serving_process.pid}) still alive after all attempts.")

            train_is_effectively_running = train_process is not None and train_process.is_alive()
            if not train_is_effectively_running:
                if train_process and not train_process.is_alive():
                    train_process.close()
                    train_process = None
                print(f"MONITOR: Load ({Load}) <= Max_Load ({Max_Load}). Starting training...")
                train_process = multiprocessing.Process(target=train_worker_entry, args=(START_TIME_I2T,), daemon=False)
                train_process.start()
                print(f"MONITOR: Training process started (PID: {train_process.pid}).")
            if SYS_STATE == 1:
                end_time_switching = time.time()
                print(f"Shuting infer and call train up took {end_time_switching - strat_time_switching:.2f}s")
            SYS_STATE = 0

def monitor_thread_worker():
    print("MONITOR: Monitor thread started. Will check load every 2 seconds.")
    while not stop_monitor_event.is_set():
        manage_processes()
        time.sleep(5) # Check frequency
    print("MONITOR: Monitor thread stopped.")

def cleanup_all_processes():
    print("MAIN: Initiating cleanup of all child processes...")
    global train_process, serving_process

    with process_management_lock:
        if train_process and train_process.is_alive():
            print(f"MAIN: Cleaning up training process (PID: {train_process.pid})...")
            try:
                os.kill(train_process.pid, signal.SIGINT)
                train_process.join(timeout=30)
                if train_process.is_alive():
                    print(f"MAIN: Training (PID: {train_process.pid}) still alive after SIGINT (30s). Killing.")
                    train_process.kill()
                    train_process.join(timeout=5)
            except Exception as e:
                print(f"MAIN: Error cleaning up training process: {e}")
            finally:
                if train_process: train_process.close()
                train_process = None

        if serving_process and serving_process.is_alive():
            print(f"MAIN: Cleaning up serving Python process (PID: {serving_process.pid}). Docker container: {DOCKER_CONTAINER_NAME}")
            # Attempt direct docker stop first during final cleanup as well
            print(f"MAIN: Attempting 'docker stop {DOCKER_CONTAINER_NAME}' during final cleanup...")
            try:
                stop_result = subprocess.run(
                    ["docker", "stop", DOCKER_CONTAINER_NAME],
                    timeout=20, capture_output=True, text=True, check=False
                )
                if stop_result.returncode == 0:
                    print(f"MAIN: 'docker stop {DOCKER_CONTAINER_NAME}' (final cleanup) succeeded.")
                else:
                    print(f"MAIN: 'docker stop {DOCKER_CONTAINER_NAME}' (final cleanup) finished with RC {stop_result.returncode}. Stderr: {stop_result.stderr.strip()}")
            except Exception as e_docker_stop_final:
                print(f"MAIN: Error during final 'docker stop {DOCKER_CONTAINER_NAME}': {e_docker_stop_final}")

            # Then proceed to terminate the Python process
            try:
                serving_process.terminate()
                serving_process.join(timeout=15)
                if serving_process.is_alive():
                    print(f"MAIN: Serving Python process (PID: {serving_process.pid}) still alive after terminate (15s). Killing.")
                    serving_process.kill()
                    serving_process.join(timeout=10)
            except Exception as e:
                print(f"MAIN: Error cleaning up serving Python process: {e}")
            finally:
                if serving_process: serving_process.close()
                serving_process = None
    print("MAIN: Cleanup finished.")

# --- Main execution ---
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.freeze_support()

    print(f"MAIN: Program started. Default Max_Load is {Max_Load}. Initial Load is {Load}.")
    if not PYTORCH_AVAILABLE:
        print("MAIN: Note - PyTorch is not installed, training will use a placeholder.")
    else:
        print(f"MAIN: Attempting to use ImageNet data from: {IMAGENET_DATA_PATH}")
        print(f"MAIN: Please ensure '{os.path.join(IMAGENET_DATA_PATH, 'train')}' exists and is structured for ImageFolder.")

    print("MAIN: Ensure Docker is running and you have permissions if serving is activated.")
    print(f"MAIN: Docker container will be named: {DOCKER_CONTAINER_NAME}")
    print("MAIN: Ensure model repository path for Triton is correct (e.g., /mydata/Data/server/docs/examples/model_repository)")
    print(f"MAIN: Checkpoints will be saved in: {os.path.abspath(CHECKPOINT_DIR)}")
    print("MAIN: Enter an integer value for 'Load' to change system behavior.")
    print("MAIN: Type 'q' or 'quit' to exit.")

    monitor = threading.Thread(target=monitor_thread_worker, daemon=True)
    monitor.start()

    try:
        while True:
            try:
                user_input = input(f"Enter new Load (current: {Load}, max: {Max_Load}) (or 'q' to quit): ")
                if user_input.lower() in ['q', 'quit']:
                    print("MAIN: Quit command received. Shutting down...")
                    break

                new_load = int(user_input)
                Load = new_load
                print(f"MAIN: Load updated to {Load}. Monitor will adjust processes shortly.")

            except ValueError:
                print("MAIN: Invalid input. Please enter an integer for Load or 'q' to quit.")
            except EOFError:
                print("MAIN: EOF detected, exiting...")
                break

    except KeyboardInterrupt:
        print("MAIN: KeyboardInterrupt received in main loop. Shutting down...")
    finally:
        print("MAIN: Starting final shutdown sequence...")
        stop_monitor_event.set()

        if monitor.is_alive():
            print("MAIN: Waiting for monitor thread to finish...")
            monitor.join(timeout=10)
            if monitor.is_alive():
                print("MAIN: Monitor thread did not stop in time (this is okay as it's daemonic).")

        cleanup_all_processes()

        print("MAIN: Program terminated.")