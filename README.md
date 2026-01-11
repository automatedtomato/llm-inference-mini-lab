# LLM Inference Optimization "Mini Lab"

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/automatedtomato/llm-inference-mini-lab/actions/workflows/ci.yml/badge.svg)

A personal project to quantify the trade-offs between Speed, VRAM usage, and Accuracy in LLM inference on consumer hardware (RTX 3060 Ti / 8GB).

This project focuses on separating Prefill and Decode metrics to understand the true bottlenecks of LLM inference, moving beyond simple "tokens/sec" averages.

## Key Features

* **Granular Benchmarking:** Separates *Time-To-First-Token (Prefill)* and *Generation Speed (Decode)*.
* **Configurable Experiments:** Manage experiments via YAML configs.
* **Production-Ready Structure:** Built with Docker, `uv`, Ruff, and CI/CD pipelines.

## Preliminary Results (Baseline FP16)

Running **Llama 3.2 3B Instruct** on NVIDIA RTX 3060 Ti (8GB):

| Metric | Value | Note |
| :--- | :--- | :--- |
| **Prefill Speed** | **~3,190 tokens/s** | Compute-bound (TFLOPS) |
| **Decode Speed** | **~38 tokens/s** | Memory-bound (Bandwidth) |
| **VRAM Usage** | **6.02 GB** | Near limit for 8GB cards |

*(More quantization results is coming later)*

## Quick Start

### Prerequisites
* NVIDIA GPU (Driver 550+)
* Docker & Docker Compose (with NVIDIA Container Toolkit)

### Running the Benchmark

1.  **Clone & Start Container**
    ```bash
    git clone [https://github.com/automatedtomato/llm-inference-mini-lab.git](https://github.com/automatedtomato/llm-inference-mini-lab.git)
    cd llm-inference-mini-lab

    # Open in VSCode DevContainers or run manually:
    docker compose up -d
    docker compose exec llm-lab bash
    ```

2.  **Run Benchmark**
    ```bash
    # Run with default config
    python src/benchmark.py -c configs/benchmark_config.yaml
    ```

3.  **Check Results**
    Results are saved to `src/results/benchmark_log.csv`.

## Roadmap

* [x] **Phase 1:** Build Benchmarking Infrastructure (Prefill/Decode separation)
* [ ] **Phase 2:** Quantization Analysis (INT8/INT4 with bitsandbytes)
* [ ] **Phase 3:** Layer-wise Sensitivity Analysis & Mixed Precision
* [ ] **Phase 4:** Custom CUDA Kernel Implementation