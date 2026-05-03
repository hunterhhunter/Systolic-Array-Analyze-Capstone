# Systolic Array Analyze Capstone

This project provides a Docker-based development and experiment environment for running MLIR tiling, SCALE-Sim simulation, and result aggregation workflows.

---

## 1. Requirements

Before running this project, install the following tools.

### Common requirements

- Docker
- Docker Compose

You can check whether Docker is installed correctly with:

```bash
docker --version
docker compose version
```

---

## 2. Docker-based Development Environment

This project uses Docker to provide a Linux-compatible build and experiment environment.

The container includes:

- Ubuntu-based Linux environment
- Python 3.12
- uv
- make
- build tools
- project dependencies installed through `make env`

The Python virtual environment is stored inside a Docker volume, so it can be reused between container runs.

---

## 3. OS-specific Setup

### 3.1 Windows

#### Step 1. Install Docker Desktop

Install Docker Desktop for Windows.

Recommended settings:

```text
Settings -> General -> Use the WSL 2 based engine
```

Make sure Docker Desktop is running before executing commands.

#### Step 2. Open PowerShell

Move to the project directory:

```powershell
cd C:\path\to\repository
```

#### Step 3. Build the Docker image

```powershell
docker compose build
```

#### Step 4. Create the Python environment

```powershell
docker compose run --rm dev make env
```

#### Step 5. Run tests

```powershell
docker compose run --rm dev make test
```

#### Step 6. Run an experiment

```powershell
docker compose run --rm dev make experiment
```

---

### 3.2 macOS

#### Step 1. Install Docker Desktop

Install Docker Desktop for macOS and make sure it is running.

#### Step 2. Open Terminal

Move to the project directory:

```bash
cd /path/to/repository
```

#### Step 3. Build the Docker image

```bash
docker compose build
```

#### Step 4. Create the Python environment

```bash
docker compose run --rm dev make env
```

#### Step 5. Run tests

```bash
docker compose run --rm dev make test
```

#### Step 6. Run an experiment

```bash
docker compose run --rm dev make experiment
```

---

### 3.3 Linux

#### Step 1. Install Docker and Docker Compose

On Ubuntu, Docker can be installed using:

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
```

Start Docker:

```bash
sudo systemctl enable --now docker
```

Optional: allow the current user to run Docker without `sudo`.

```bash
sudo usermod -aG docker $USER
```

After running this command, log out and log back in.

#### Step 2. Move to the project directory

```bash
cd /path/to/repository
```

#### Step 3. Build the Docker image

```bash
docker compose build
```

If your Docker installation requires root permission, use:

```bash
sudo docker compose build
```

#### Step 4. Create the Python environment

```bash
docker compose run --rm dev make env
```

or:

```bash
sudo docker compose run --rm dev make env
```

#### Step 5. Run tests

```bash
docker compose run --rm dev make test
```

#### Step 6. Run an experiment

```bash
docker compose run --rm dev make experiment
```

---

## 4. Common Commands

### Enter the development container

```bash
docker compose run --rm dev
```

Inside the container, you can run:

```bash
make env
make test
make experiment
```

---

### Build the Docker image

```bash
docker compose build
```

---

### Install project dependencies

```bash
docker compose run --rm dev make env
```

This command creates the Python virtual environment and installs the required packages.

---

### Install optional model conversion dependencies

```bash
docker compose run --rm dev make env-model
```

Use this when running Hugging Face, ONNX, or model conversion workflows.

---

### Run tests

```bash
docker compose run --rm dev make test
```

---

### Run MLIR tiling

```bash
docker compose run --rm dev make tile
```

Example with custom options:

```bash
docker compose run --rm dev make tile \
  MODEL_MLIR=mlir/inputs/matmul_small.mlir \
  KIND=matmul \
  TILE_ARGS="--tile-m 8 --tile-n 8 --tile-k 8"
```

---

### Emit SCALE-Sim topology

```bash
docker compose run --rm dev make emit-topology
```

---

### Run SCALE-Sim experiment

```bash
docker compose run --rm dev make run-exp
```

---

### Run the full experiment pipeline

```bash
docker compose run --rm dev make experiment
```

This runs:

```text
tile -> emit-topology -> run-exp
```

---

## 5. Example Workflows

### 5.1 Basic matmul experiment

```bash
docker compose run --rm dev make experiment \
  MODEL_MLIR=mlir/inputs/matmul_small.mlir \
  KIND=matmul \
  TILE_ARGS="--tile-m 8 --tile-n 8 --tile-k 8"
```

---

### 5.2 Run with custom architecture config

```bash
docker compose run --rm dev make experiment \
  ARCH_CFG=SCALE-Sim/configs/tpuv2.cfg
```

---

### 5.3 Run sweep

```bash
docker compose run --rm dev make sweep
```

Example with custom sweep values:

```bash
docker compose run --rm dev make sweep \
  SWEEP_MNKS="32x32x64 64x64x128" \
  SWEEP_TILES="8x8x8 16x16x16" \
  SWEEP_ARCH_CFGS="SCALE-Sim/configs/walking_8x8_ws.cfg"
```

---

### 5.4 Aggregate results

```bash
docker compose run --rm dev make aggregate
```

---

### 5.5 Reuse analysis

```bash
docker compose run --rm dev make reuse
```

---

## 6. Output Files

Generated experiment files are usually stored in:

```text
outputs/
results/
```

Typical outputs include:

```text
outputs/experiments/<run-name>/
results/<run-name>.parquet
results/results.csv
results/aggregate.csv
```

---

## 7. Cleaning the Environment

### Remove stopped containers

```bash
docker compose down
```

### Remove containers and Docker volumes

```bash
docker compose down -v
```

Warning: this removes the Docker volume that stores the Python virtual environment.
After running this command, you need to run:

```bash
docker compose run --rm dev make env
```

again.

---

## 8. Troubleshooting

### Problem: `.venv/bin/python` not found

This can happen if a Windows virtual environment was created in the project directory.

Recommended solution:

```bash
docker compose run --rm dev make env
```

This project uses a Docker-managed virtual environment, so users should not manually create or use a Windows `.venv` for Docker execution.

---

### Problem: `/opt/venv/bin/python: No such file or directory`

Run the environment setup first:

```bash
docker compose run --rm dev make env
```

If the problem continues, reset the Docker volume:

```bash
docker compose down -v
docker compose build
docker compose run --rm dev make env
```

Then run:

```bash
docker compose run --rm dev make test
```

---

### Problem: Docker permission denied on Linux

Add the current user to the Docker group:

```bash
sudo usermod -aG docker $USER
```

Then log out and log back in.

Alternatively, run Docker with `sudo`:

```bash
sudo docker compose run --rm dev make test
```

---

### Problem: Docker Desktop is not running on Windows or macOS

Start Docker Desktop first, then run:

```bash
docker compose build
```

---

## 9. Notes for Developers

The Makefile is written for a Linux environment.
Instead of modifying the Makefile for each OS, this project runs all build and experiment commands inside a Linux Docker container.

Therefore, the recommended way to run this project is:

```bash
docker compose run --rm dev make <target>
```

For example:

```bash
docker compose run --rm dev make test
docker compose run --rm dev make experiment
```

---

## 10. Available Make Targets

```bash
docker compose run --rm dev make help
```

Main targets:

```text
env            Create Python virtual environment and install dependencies
env-model      Install optional model conversion dependencies
test           Run pytest
hf-onnx-mlir   Export Hugging Face model to ONNX and import to IREE MLIR
tile           Tile MLIR input
emit-topology  Emit SCALE-Sim topology CSV
run-exp        Run SCALE-Sim experiment
experiment     Run tile -> emit-topology -> run-exp
sweep          Run parameter sweep
aggregate      Aggregate SCALE-Sim report
reuse          Run reuse analysis
resnet         Run ResNet workflow
resnet-smoke   Run ResNet smoke test
```

---

## 11. Expected Docker Compose Configuration

The examples above assume that `docker-compose.yml` is configured similarly to the following:

```yaml
services:
  dev:
    build: .
    working_dir: /workspace
    volumes:
      - .:/workspace
      - venv:/opt/venv
    environment:
      - PYTHONUNBUFFERED=1
      - VENV=/opt/venv
    stdin_open: true
    tty: true

volumes:
  venv:
```

The Makefile should also use the `VENV` variable:

```makefile
VENV ?= .venv
PY   := $(VENV)/bin/python
UV   := uv
IREE_OPT ?= $(VENV)/bin/iree-opt
export IREE_OPT
```

The `env` target should check the actual Python executable instead of only checking whether the virtual environment directory exists:

```makefile
env:
	test -x $(PY) || $(UV) venv $(VENV) --python 3.12
	$(UV) pip install --python $(PY) -r requirements-dev.txt
	$(UV) pip install --python $(PY) -e ./SCALE-Sim
```
