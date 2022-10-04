import json
from itertools import product
import os
import sys
import time

import submitit
import torch

from main import submitit_main


log_folder = os.path.join("logs", "%j")

# generate the generic executor
executor = submitit.AutoExecutor(folder=log_folder)

# specify the sbatch options

executor.update_parameters(
    slurm_partition="gpu",
    timeout_min=60,
    nodes=1,
    # ntasks=1,
    tasks_per_node=1,
    cpus_per_task=4,
    # gres="gpu:1",
    gpus_per_node=1,
    mem_gb=16,
    wckey="",
)


jobs = []
labels = []
config = "config_" + sys.argv[1]
last_arg = sys.argv[2] if len(sys.argv) > 2 else "-a"
if len(sys.argv) > 3:
    labels = sys.argv[3:]
num_batch = 4
for seed_batch, label in product(range(0, num_batch), labels):
    with executor.batch():
        for seed in range(seed_batch * 10, (seed_batch + 1) * 10):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            exp_dir = os.path.join(current_dir, config)
            config_path = os.path.join(exp_dir, "config.json")
            output_path = os.path.join(exp_dir, f"{str(seed).zfill(4)}_{label}.pt")
            if os.path.exists(output_path):
                input_dict = torch.load(output_path)
                existing_iterations = input_dict["existing_iterations"]
                with open(config_path, "r") as f:
                    kwargs = json.load(f)
                if existing_iterations == kwargs["iterations"]:
                    continue
            job = executor.submit(submitit_main, config, label, seed, last_arg)
        time.sleep(10)
