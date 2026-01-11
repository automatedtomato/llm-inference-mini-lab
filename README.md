# LLM Inference Optimization Mini Lab

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/automatedtomato/llm-inference-mini-lab/actions/workflows/ci.yml/badge.svg)

A personal project to quantify the trade-offs between Speed, VRAM usage, and Accuracy in LLM inference on consumer hardware (RTX 3060 Ti / 8GB).

This project focuses on separating Prefill and Decode metrics to understand the true bottlenecks of LLM inference, moving beyond simple "tokens/sec" averages.

## Key Features

* Granular Benchmarking: Separates *Time-To-First-Token (Prefill)* and *Generation Speed (Decode)*.
* Quantization Support: Easy switching between FP16, INT8, and NF4 (4-bit) via YAML configs.
* Production-Ready Structure: Built with Docker, `uv`, Ruff, and CI/CD pipelines.

## Benchmark Results

Running `Llama 3.2 3B Instruct` on NVIDIA RTX 3060 Ti (8GB):

| Configuration | VRAM Usage | Prefill Speed | Decode Speed |
| :--- | :--- | :--- | :--- |
| FP16 (Baseline) | 6.02 GB | ~3,194 t/s | ~37.8 t/s |
| 8-bit (Int8) | 3.40 GB | ~875 t/s | ~9.6 t/s |
| 4-bit (NF4) | 2.21 GB | ~1,292 t/s | ~14.9 t/s |

### Observations

1.  **Memory Efficiency:** 4-bit quantization (NF4) reduces VRAM usage by 63% compared to FP16, allowing larger models or larger batch sizes on consumer GPUs.
2.  **The "4-bit is Faster than 8-bit" Paradox:** Surprisingly, 4-bit inference is faster than 8-bit.
    * **Reason:** LLM inference is memory-bound. 4-bit reduces memory transfer time (VRAM → Compute Units) significantly enough to offset the dequantization overhead.
    * Additionally, the NF4 kernels (introduced in QLoRA) are more optimized than the older `Linear8bitLt` kernels.

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
    # Run with default config (FP16)
    python src/benchmark.py -c configs/benchmark_config.yaml
    
    # Run with 4-bit quantization
    # (Edit config yaml to set load_in_4bit: true)
    python src/benchmark.py -c configs/benchmark_config.yaml
    ```

3.  **Check Results**
    Results are saved to `src/results/benchmark_log.csv`.

## 📅 Roadmap

* [x] **Phase 1:** Build Benchmarking Infrastructure (Prefill/Decode separation)
* [x] **Phase 2:** Quantization Analysis (INT8 vs NF4 trade-offs)
* [ ] **Phase 3:** Layer-wise Sensitivity Analysis (Mixed Precision)
    * *Goal: Recover accuracy by keeping specific sensitive layers in FP16.*
* [ ] **Phase 4:** Custom CUDA Kernel Implementation