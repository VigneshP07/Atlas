# main_with_logging.py
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def setup_logging():
    """Set up logging dir per rank and redirect stdout/stderr."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Create root experiments dir
    exp_root = Path("/scratch/abircs/sudipta/atlas/experiment")
    exp_root.mkdir(parents=True, exist_ok=True)

    # One folder per run (created by rank 0)
    timestamp = os.environ.get("RUN_TIMESTAMP")
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.environ["RUN_TIMESTAMP"] = timestamp
    exp_dir = exp_root / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save args (only on rank 0)
    if rank == 0:
        with open(exp_dir / "args.txt", "w") as f:
            f.write(" ".join(sys.argv) + "\n")

    # Separate log per rank
    sys.stdout = open(exp_dir / f"out_rank{rank}_gpu{local_rank}.log", "w")
    sys.stderr = open(exp_dir / f"err_rank{rank}_gpu{local_rank}.log", "w")

    return exp_dir

if __name__ == "__main__":
    log_dir = setup_logging()
    print(f"[INFO][Rank {os.environ.get('RANK')}] Logging to {log_dir}")

    from main import main as original_main
    original_main()
