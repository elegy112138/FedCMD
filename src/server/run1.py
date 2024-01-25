import subprocess
import time
import os
import re
from math import ceil


def clear_existing_logs(dir_name):
    if os.path.exists(dir_name):
        for file_name in os.listdir(dir_name):
            file_path = os.path.join(dir_name, file_name)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")


def run_scripts(scripts, args, max_parallel_runs, dir_name):
    processes = []
    for i, script in enumerate(scripts):
        if i >= max_parallel_runs:
            script_to_wait, process_to_wait = processes[i % max_parallel_runs]
            ret = process_to_wait.wait()
            print(f"{script_to_wait} has finished with return code {ret}.")

        cmd = f"C:\\Users\\Admin\\.conda\\envs\\FL\\python.exe {script} {args}"
        log_file = os.path.join(dir_name, f"{os.path.splitext(script)[0]}.log")
        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
            if i < max_parallel_runs:
                processes.append((script, process))
            else:
                processes[i % max_parallel_runs] = (script, process)
        print(f"Started {script}...")
    return processes


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


def main():
    scripts = [
        "fedbabu.py", "fedrep.py",  "fedfomo.py", "fedprox.py",
        "pfedsim.py", "feddyn.py", "fedbn.py", "fedap.py", "fedper.py", 'fedavg.py'
    ]
    dataset_names = ['cifar10', 'cifar100', 'medmnistA', 'medmnistC', 'tiny_imagenet', 'svhn', 'cinic10', 'emnist',
                     'fmnist', 'mnist']

    max_parallel_runs = ceil(len(scripts) / 4)

    for dataset_name in dataset_names:
        args = f"--dataset {dataset_name} --visible 1 --global_epoch 200 --join_ratio 0.1"
        dir_name = f'logs_{dataset_name}'

        clear_existing_logs(dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        processes = run_scripts(scripts, args, max_parallel_runs, dir_name)

        for script, process in processes:
            ret = process.wait()
            print(f"{script} has finished with return code {ret}.")

        with open(os.path.join(dir_name, 'all.log'), 'w') as all_log:
            for script in scripts:
                log_file = os.path.join(dir_name, f"{os.path.splitext(script)[0]}.log")
                accuracy = extract_accuracy_from_log(log_file)
                if accuracy:
                    all_log.write(f"{script}: {accuracy}%\n")


if __name__ == '__main__':
    main()
