# Environment Setup Guide

This guide walks you through setting up the conda environment for the Atlas codebase. Because `flash-attn` does not install cleanly via the YAML file, it is removed from the pip dependencies in `environment.yaml` and installed separately after the environment is created.

---

## Prerequisites

### Step 1 — Install Miniconda (if conda is not already present)

Download and run the Miniconda installer:

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
```

After installation, reload your shell configuration to activate conda:

```bash
source ~/.bashrc
```

Verify that conda is working correctly:

```bash
conda list
```

---

## Environment Setup

### Step 2 — Remove `flash-attn` from `environment.yaml`

Before creating the environment, open `environment.yaml` and remove `flash-attn` from the `pip` dependencies section. This prevents build errors during environment creation.

For example, locate and delete any line resembling the following inside the `pip:` block:

```yaml
# Remove this line from environment.yaml before proceeding
- flash-attn
```

Save the file after making this change.

---

### Step 3 — Create the Conda Environment

Create the conda environment from the modified `environment.yaml`, using a local prefix directory named `atlas_env`:

```bash
conda env create -f environment.yaml --prefix ./atlas_env
```

> **Note:** Using `--prefix ./atlas_env` creates the environment inside the current project directory rather than the default conda environments folder.

---

### Step 4 — Activate the Environment

```bash
conda activate ./atlas_env
```

---

### Step 5 — Install `flash-attn` Separately

Install `flash-attn` using the prebuilt release binaries to avoid compilation errors:

```bash
pip install flash-attn --no-build-isolation --find-links https://github.com/Dao-AILab/flash-attention/releases/tag/v2.6.3
```

---

### Step 6 — Verify the Installation

Confirm that `flash-attn` has been installed successfully:

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```

If the version number is printed without errors, the installation is complete.

---

### Step 7 — Deactivate the Environment

Once setup is complete, deactivate the environment:

```bash
conda deactivate
```

---

## Troubleshooting

### Environment build stopped with errors

If the environment creation in Step 3 was interrupted or exited with errors, you can resume and update the environment after `flash-attn` has been installed separately. Run the following command with the full path to your environment prefix:

```bash
conda env update -f environment.yaml --prefix /home/others/abir_proj/Env/atlas_env
```

> **Note:** Replace `/home/others/abir_proj/Env/atlas_env` with the actual path to your environment if it differs.

---

## Summary

| Step | Command |
|------|---------|
| Install Miniconda | `bash ~/Miniconda3-latest-Linux-x86_64.sh` |
| Reload shell | `source ~/.bashrc` |
| Verify conda | `conda list` |
| Create environment | `conda env create -f environment.yaml --prefix ./atlas_env` |
| Activate environment | `conda activate ./atlas_env` |
| Install flash-attn | `pip install flash-attn --no-build-isolation --find-links ...` |
| Verify flash-attn | `python -c "import flash_attn; print(flash_attn.__version__)"` |
| Deactivate | `conda deactivate` |
| Update (if needed) | `conda env update -f environment.yaml --prefix <path>` |