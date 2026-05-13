#!/usr/bin/env python3
import os
import subprocess
import uuid
from pathlib import Path
import submitit
import shutil

# ---------- USER CONFIG ----------
REPO_ROOT = "/scratch/abircs/sudipta/atlas"   # where main.py lives
CONFIG = "/scratch/abircs/sudipta/atlas/configs/test.yaml"
EXPERIMENT_ROOT = "/scratch/abircs/sudipta/atlas/experiments"   # shared folder for logs & inits
NGPUS = 2           # GPUs per node
NNODES = 2          # number of nodes
TIME_MIN = 24*60    # timeout in minutes
PARTITION = "gpu"
CPUS_PER_TASK = 12
PYTHON = "/scratch/abircs/condaenvs/atlas-env-new/bin/python"  # exact python in your conda env
MASTER_PORT = 29500
# ---------------------------------

Path(EXPERIMENT_ROOT).mkdir(parents=True, exist_ok=True)

def shutil_which(name):
    """Check if an executable exists in PATH."""
    return shutil.which(name)

def get_master_ip():
    """Return the IP of the first node in the SLURM node list."""
    node_list = os.environ.get("SLURM_NODELIST")
    if not node_list:
        raise RuntimeError("SLURM_NODELIST is not set — cannot determine master IP")

    # Get first hostname in the allocation
    master_host = subprocess.check_output(
        ["scontrol", "show", "hostnames", node_list]
    ).decode().splitlines()[0]

    # Get IP from hostname
    if shutil_which("getent"):
        master_addr = subprocess.check_output(
            ["getent", "hosts", master_host]
        ).decode().split()[0]
    else:
        # If getent not available, trust hostname
        master_addr = master_host

    return master_addr

class Trainer:
    def __init__(self, config_path, run_id):
        self.config = config_path
        self.run_id = run_id

    def __call__(self):
        # inside SLURM job; create job-specific dir inside EXPERIMENT_ROOT
        job_id = os.environ.get("SLURM_JOB_ID", str(self.run_id))
        job_dir = Path(EXPERIMENT_ROOT) / f"job_{job_id}"
        job_dir.mkdir(parents=True, exist_ok=True)

        master_addr = get_master_ip()

        # torchrun command
        cmd = [
            PYTHON, "-m", "torch.distributed.run",
            f"--nnodes={NNODES}",
            f"--nproc_per_node={NGPUS}",
            f"--node_rank={os.environ.get('SLURM_NODEID', '0')}",
            f"--master_addr={master_addr}",
            f"--master_port={MASTER_PORT}",
            "main.py",
            self.config
        ]

        out_path = job_dir / "out.log"
        err_path = job_dir / "err.log"
        print(f"[Job {job_id}] Running command:\n", " ".join(cmd))
        with open(out_path, "ab") as out_f, open(err_path, "ab") as err_f:
            process = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=out_f, stderr=err_f)
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"torchrun failed with exit code {return_code}. "
                    f"See {out_path} and {err_path} for logs."
                )

def submit():
    executor = submitit.AutoExecutor(folder=str(Path(EXPERIMENT_ROOT) / "submitit_logs"))
    executor.update_parameters(
        gpus_per_node=NGPUS,
        tasks_per_node=1,
        cpus_per_task=CPUS_PER_TASK,
        nodes=NNODES,
        timeout_min=TIME_MIN,
        slurm_partition=PARTITION,
        name="atlas_train",
    )
    run_id = uuid.uuid4().hex[:8]
    trainer = Trainer(CONFIG, run_id)
    job = executor.submit(trainer)
    print("Submitted job:", job.job_id)
    return job

if __name__ == "__main__":
    submit()
