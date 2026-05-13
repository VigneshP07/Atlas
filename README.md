# Atlas

## Directory Structure

```
Atlas/
├── sandstone/                  # Source code
│   ├── models/
│   │   ├── atlas.py            # Atlas model (modified)
│   │   └── msa.py              # MSA model (modified)
│   └── ...
├── configs/                    # Configuration files for experiments
├── jobs/                       # Scripts used to run experiments
├── checkpoints/
├── Output_Logs/                # Output and error logs for all experiments
├── main.py                     # Main entry point (modified)
└── environment.yaml            # Conda environment specification
```

> For environment setup instructions, refer to [ENV_BUILD.md](./ENV_BUILD.md).

---

## Source Code

The source code resides in the `sandstone/` directory. The following files have been modified from the original codebase:

- `main.py` — Main training and evaluation entry point
- `sandstone/models/atlas.py` — Atlas model definition
- `sandstone/models/msa.py` — MSA model definition

---

## Experiments

### Configuration Files

All configuration files are located in the `configs/` directory. Each config file corresponds to a specific experiment and is named accordingly.

The following parameters are set inside the config file before running an experiment:

- **Model resolution and hyperparameters** — architecture settings, learning rate, batch size, etc.
- **Dataset path** — path to the dataset to be used for training/evaluation
- **Checkpoint directory path** — directory where model checkpoints will be saved

When **resuming** a previously interrupted experiment, set the path to the latest checkpoint under the following key in the config file:

```yaml
engine:
  kwargs:
    resume: <path_to_latest_checkpoint>
```

#### Finetuning

To finetune from a pretrained checkpoint, set the following parameters under `model:kwargs:` in the config file:

```yaml
model:
  kwargs:
    pretrained: true
    pretrained_path: <path_to_pretrained_checkpoint>
    freeze_mode: <type_of_freeze>
```

- **`pretrained`** — set to `true` to load weights from a pretrained checkpoint before training
- **`pretrained_path`** — path to the pretrained checkpoint file to load from
- **`freeze_mode`** — controls which parts of the model are frozen during finetuning (e.g., backbone, encoder, etc.)

---

### Job Scripts

The `jobs/` directory contains shell scripts used to launch experiments. Refer to these scripts for experiment-specific launch configurations and to understand how experiments are structured and submitted.

---

### Output and Error Logs

All output and error logs are stored in the `Output_Logs/` directory. Logs are named after their corresponding experiment.

For experiments that were resumed, the logs from subsequent runs are suffixed with `_1`, `_2`, and so on, appended to the original log name. For example:

```
Output_Logs/
├── experiment_name.out         # Initial run — output log
├── experiment_name.err         # Initial run — error log
├── experiment_name_1.out       # First resume — output log
├── experiment_name_1.err       # First resume — error log
├── experiment_name_2.out       # Second resume — output log
└── experiment_name_2.err       # Second resume — error log
```

---

## Running an Experiment

### Step 1 — Activate the Environment

```bash
source <path_to_miniconda3>/miniconda3/etc/profile.d/conda.sh
conda activate <path_to_env>
```

### Step 2 — Navigate to the Working Directory

```bash
cd <path_to_Atlas>
```

### Step 3 — Launch the Experiment

Use PyTorch's distributed launcher to run the experiment with the desired config file:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=28428 \
    main.py configs/<config_file_name>
```

> **Note:** `--nproc_per_node=4` specifies the number of GPUs to use. Adjust this value based on your available hardware.

Refer to the scripts inside the `jobs/` directory for complete, experiment-specific run commands.

---