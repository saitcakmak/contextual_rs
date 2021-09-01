from itertools import product
import os
import sys

import submitit
from main import submitit_main


log_folder = os.path.join("logs", "%j")

# generate the generic executor
executor = submitit.AutoExecutor(folder=log_folder)

# specify the sbatch options

executor.update_parameters(
    slurm_partition="default_gpu",
    timeout_min=240,  # 4 hrs
    nodes=1,
    # ntasks=1,
    tasks_per_node=1,
    cpus_per_task=4,
    # gres="gpu:1",
    gpus_per_node=1,
    mem_gb=16,
)

# # to submit batches of jobs
# # this line sets the maximum to run in parallel
# executor.update_parameters(slurm_array_parallelism=2)
# jobs = []
# with executor.batch():
#     for arg in whatever:
#         job = executor.submit(myfunc, arg)  # This is where the actual job goes
#         jobs.append(job)
#
# outputs = [job.result() for job in jobs]
# print(outputs)
#
# # for submitting bash commands
# function = submitit.helpers.CommandFunction(["which", "python"])

# let's try to put it together.
executor.update_parameters(slurm_array_parallelism=20)
jobs = []
labels = [
    "ML_IKG",
    "ML_Gao_infer_p",
]
config = "config_" + sys.argv[1]
last_arg = sys.argv[2] if len(sys.argv) > 2 else "---"
if len(sys.argv) > 3:
    labels = sys.argv[3:]
with executor.batch():
    for seed, label in product(range(0, 20), labels):
        # function = submitit.helpers.CommandFunction(
        #     ["python", "main.py", config, label, str(seed)]
        # )
        # job = executor.submit(function)
        job = executor.submit(submitit_main, config, label, seed, last_arg)
        # jobs.append(job)

# for job in jobs:
#     print(job.result())
