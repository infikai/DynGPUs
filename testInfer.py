import subprocess
import os
import time
import signal
import threading
import sys

# Event to signal termination to all threads/loops
stop_event = threading.Event()
triton_process_global = None # Global reference for signal handler and input thread

def run_triton_server():
    """
    Runs the Triton Inference Server Docker container and returns the process object.
    """
    global triton_process_global
    current_working_directory = '/mydata/Data/server/docs/examples'
    model_repository_path = os.path.join(current_working_directory, "model_repository")

    if not os.path.exists(model_repository_path):
        print(f"Warning: Host model repository directory '{model_repository_path}' does not exist.")
        # os.makedirs(model_repository_path, exist_ok=True) # Optionally create
        # return None # Or exit

    command = [
        'docker', 'run',
        '--gpus=1',
        '--rm',
        '--net=host',
        '-v', f'{model_repository_path}:/models',
        'nvcr.io/nvidia/tritonserver:25.02-py3', # Make sure this tag is current if needed
        'tritonserver',
        '--model-repository=/models',
        '--model-control-mode', 'explicit',
        '--load-model', 'densenet_onnx'
    ]

    print(f"Executing command: {' '.join(command)}")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True # Ensures text mode and handles line endings
        )
        triton_process_global = process # Store for global access
        print(f"Triton server started with PID (of docker run command): {process.pid}")
        return process
    except FileNotFoundError:
        print("Error: The 'docker' command was not found. Please ensure Docker is installed and in your PATH.")
        stop_event.set()
        return None
    except Exception as e:
        print(f"An error occurred while trying to start the Docker container: {e}")
        stop_event.set()
        return None

def terminate_server(process_to_terminate):
    """
    Terminates the given process. Sets the stop_event.
    """
    if not process_to_terminate:
        print("No process to terminate.")
        stop_event.set() # Ensure event is set even if no process
        return

    stop_event.set() # Signal all parts of the script to stop

    if process_to_terminate.poll() is None: # Check if the process is still running
        print(f"Terminating process {process_to_terminate.pid} (Docker container)...")
        # Sending SIGTERM to `process` (the `docker run` command) should signal Docker
        # to stop the container. The `--rm` flag will then ensure it's removed.
        process_to_terminate.terminate() # Sends SIGTERM
        try:
            process_to_terminate.wait(timeout=15) # Increased timeout
            print("Docker container terminated gracefully.")
        except subprocess.TimeoutExpired:
            print("Docker container did not terminate gracefully after 15s. Forcing kill...")
            process_to_terminate.kill() # Sends SIGKILL
            try:
                process_to_terminate.wait(timeout=5) # Wait for kill
                print("Docker container killed.")
            except subprocess.TimeoutExpired:
                print("Failed to confirm kill within 5s. The process might be orphaned.")
            except Exception as e_kill_wait:
                print(f"Exception during kill wait: {e_kill_wait}")
        except Exception as e_term_wait:
            print(f"Exception during termination wait: {e_term_wait}")
        finally:
            # Ensure streams are closed after termination attempt
            if process_to_terminate.stdout:
                process_to_terminate.stdout.close()
            if process_to_terminate.stderr:
                process_to_terminate.stderr.close()
    elif process_to_terminate.returncode is not None:
        print(f"Process {process_to_terminate.pid} already terminated with code {process_to_terminate.returncode}.")
    else:
        # This case should ideally not be hit if poll() is None means running.
        # But as a fallback.
        print(f"Process {process_to_terminate.pid} is in an indeterminate state.")


def input_listener_thread(process_to_watch):
    """
    Runs in a separate thread, listens for 'quit' command.
    """
    print("\nType 'quit' or 'exit' and press Enter to stop the server.")
    while not stop_event.is_set():
        try:
            command = input() # This will block the input thread
            if command.lower() in ['quit', 'exit']:
                print("User requested termination via input command.")
                terminate_server(process_to_watch)
                break
            else:
                if not stop_event.is_set(): # Only show if not already stopping
                    print(f"Unknown command: '{command}'. Type 'quit' or 'exit' to stop.")
        except EOFError: # Happens if stdin is closed unexpectedly
            print("EOF received on input thread, exiting listener.")
            if not stop_event.is_set():
                terminate_server(process_to_watch) # Attempt to stop if not already
            break
        except Exception as e:
            if not stop_event.is_set():
                print(f"Error in input listener: {e}")
            break # Exit thread on other errors
    print("Input listener thread finished.")


def stream_output(process_to_watch, stream_name, stream):
    """
    Reads and prints lines from a given stream until the stream is closed or stop_event is set.
    """
    try:
        for line in iter(stream.readline, ''):
            if stop_event.is_set() and not line: # If stopping and no more output, exit
                break
            if line:
                print(f"Server {stream_name}: {line.strip()}", flush=True)
            elif stop_event.is_set(): # If no line but stop_event is set, break
                break
    except IOError: # Handle cases where the pipe might be closed abruptly
        if not stop_event.is_set():
             print(f"IOError reading {stream_name}. Stream may have closed.", flush=True)
    except ValueError: # Can happen if trying to operate on a closed stream
        if not stop_event.is_set():
            print(f"ValueError reading {stream_name}. Stream likely closed.", flush=True)
    finally:
        stream.close()
        # print(f"{stream_name} streaming finished.") # Optional debug


def monitor_server(process_to_monitor):
    """
    Monitors the server process output and waits for it to complete or be terminated.
    """
    if not process_to_monitor:
        return

    # Start threads for streaming stdout and stderr
    stdout_thread = threading.Thread(target=stream_output, args=(process_to_monitor, "STDOUT", process_to_monitor.stdout))
    stderr_thread = threading.Thread(target=stream_output, args=(process_to_monitor, "STDERR", process_to_monitor.stderr))

    stdout_thread.daemon = True # Daemon threads exit when the main program exits
    stderr_thread.daemon = True

    stdout_thread.start()
    stderr_thread.start()

    # Wait for the process to complete or for a stop signal
    while not stop_event.is_set() and process_to_monitor.poll() is None:
        try:
            time.sleep(0.2) # Check periodically
        except KeyboardInterrupt: # Should be caught by the main SIGINT handler
            print("KeyboardInterrupt in monitoring loop (should be handled by SIGINT).")
            # The SIGINT handler will call terminate_server and set stop_event
            break # Exit loop, SIGINT handler will do the work

    # If the loop exited because the process terminated on its own
    if process_to_monitor.poll() is not None and not stop_event.is_set():
        print("Server process appears to have terminated on its own.")
        stop_event.set() # Ensure other threads know to stop

    # Wait for output streaming threads to finish
    # print("Waiting for output streamers to finish...") # Optional debug
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    # Final check on process status
    if process_to_monitor.returncode is not None:
        print(f"Server exited. Final return code: {process_to_monitor.returncode}")
    else:
        # This might happen if termination was forced and we couldn't get a return code quickly
        print("Server monitoring finished. Process state might be post-kill.")


def main_signal_handler(sig, frame):
    """
    Handles Ctrl+C (SIGINT).
    """
    global triton_process_global
    print(f"\nCtrl+C (Signal {sig}) detected. Initiating server termination...")
    if not stop_event.is_set(): # Prevent multiple calls if already handling
        terminate_server(triton_process_global)
    # To ensure the script exits after handling Ctrl+C if it's stuck somewhere else:
    # sys.exit(1) # Consider if this is too abrupt or if cleanup should fully complete.
                 # For now, let the main loop and finally blocks handle exit.

if __name__ == "__main__":
    # --- Create a dummy model_repository and model for testing ---
    current_dir = os.getcwd()
    model_repo = os.path.join(current_dir, "model_repository")
    densenet_model_dir = os.path.join(model_repo, "densenet_onnx")
    densenet_config_path = os.path.join(densenet_model_dir, "config.pbtxt")
    densenet_versions_dir = os.path.join(densenet_model_dir, "1")
    dummy_model_path = os.path.join(densenet_versions_dir, "model.onnx")

    if not os.path.exists(densenet_versions_dir):
        os.makedirs(densenet_versions_dir)
    if not os.path.exists(densenet_config_path):
        config_content = 'name: "densenet_onnx"\nplatform: "onnxruntime_onnx"\nmax_batch_size: 128\ninput [{name: "data_0",data_type: TYPE_FP32,dims: [3, 224, 224]}]\noutput [{name: "fc6_1",data_type: TYPE_FP32,dims: [1000]}]'
        with open(densenet_config_path, "w") as f: f.write(config_content)
    if not os.path.exists(dummy_model_path):
        with open(dummy_model_path, "w") as f: f.write("dummy onnx")
    # --- End of dummy setup ---

    # Set up the SIGINT (Ctrl+C) handler
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, main_signal_handler)

    triton_process = None
    input_thread = None

    try:
        triton_process = run_triton_server()

        if triton_process and triton_process.poll() is None:
            # Start the input listener thread
            input_thread = threading.Thread(target=input_listener_thread, args=(triton_process,))
            input_thread.daemon = True # Allow main program to exit even if this thread is stuck on input()
            input_thread.start()

            monitor_server(triton_process)
        elif triton_process and triton_process.poll() is not None:
            print(f"Triton server process exited very quickly with code: {triton_process.returncode}")
            # Read any remaining output
            if triton_process.stdout:
                for line in triton_process.stdout: print(f"Early STDOUT: {line.strip()}")
            if triton_process.stderr:
                for line in triton_process.stderr: print(f"Early STDERR: {line.strip()}")
        else:
            print("Triton server failed to start or was immediately unavailable.")

    except Exception as e: # Catch any other unexpected exceptions in the main flow
        print(f"An unexpected error occurred in the main execution: {e}")
        if triton_process and not stop_event.is_set():
            terminate_server(triton_process) # Attempt cleanup
    finally:
        print("Main script block: Initiating final cleanup...")
        if not stop_event.is_set(): # If not already triggered by handler or input
            # This will mainly be for cases where the server stops on its own
            # and we need to signal the input thread to stop.
            stop_event.set()

        if triton_process and triton_process.poll() is None:
             print("Main finally: Server still seems to be running, ensuring termination.")
             terminate_server(triton_process) # Ensure it's terminated

        if input_thread and input_thread.is_alive():
            print("Main finally: Waiting for input listener thread to join...")
            # The input thread should exit once stop_event is set or its input() unblocks.
            # If input() is blocking, we can't easily force it to stop cleanly without platform-specific tricks
            # or closing sys.stdin, which can be problematic. Daemon=True helps.
            input_thread.join(timeout=2) # Wait for a short period
            if input_thread.is_alive():
                print("Main finally: Input listener thread did not exit cleanly.")

        # Restore original SIGINT handler
        signal.signal(signal.SIGINT, original_sigint_handler)
        print("Script finished.")
