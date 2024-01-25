import subprocess
import os
import re

def clear_existing_logs():
    """Delete existing logs in the log_fedsper directory."""
    for log_file in os.listdir(log_dir):
        log_file_path = os.path.join(log_dir, log_file)
        try:
            os.remove(log_file_path)
        except Exception as e:
            print(f"Error deleting file {log_file_path}: {e}")

def extract_accuracy_from_log(log_file_path):
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "200:" in line:
                    match = re.search(r"accuracy': '([\d\.]+)%", line)
                    if match:
                        return match.group(1)
    except (UnicodeDecodeError, FileNotFoundError) as e:
        print(f"An error occurred with the file {log_file_path}: {e}")
        return None

dataset_names = ['medmnistC', 'medmnistA', 'svhn','cinic10','fmnist', 'mnist', 'cifar10', 'cifar100', 'tiny_imagenet', 'emnist']

max_parallel_runs = 1
processes = []

log_dir = "FedAMD"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Clear existing logs
clear_existing_logs()

python_path = "C:\\Users\\Admin\\.conda\\envs\\FL\\python.exe"
script_name = "fedamd.py"

for dataset_name in dataset_names:
    cmd = f"{python_path} {script_name} --dataset {dataset_name} --visible 1 --global_epoch 200 --join_ratio 0.1"
    process = subprocess.Popen(cmd, shell=True)
    processes.append(process)

    # If we've started 'max_parallel_runs' processes, wait for them to finish.
    if len(processes) == max_parallel_runs:
        for p in processes:
            p.wait()
        processes = []

# Wait for any remaining processes to finish.
for p in processes:
    p.wait()

# Extract accuracies and write to all.log
with open(os.path.join(log_dir, 'all.log'), 'w') as all_log:
    for dataset_name in dataset_names:
        log_file_path = os.path.join(log_dir, f"{dataset_name}.log")
        accuracy = extract_accuracy_from_log(log_file_path)
        if accuracy:
            all_log.write(f"{dataset_name}: {accuracy}\n")
