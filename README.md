# LLM Inference Optimization Mini Lab

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A personal project to quantify the trade-offs between Speed, VRAM usage, and Accuracy in LLM inference on consumer hardware (RTX 3060 Ti / 8GB).

This project focuses on identifying the specific "brain damage" caused by quantization, moving beyond simple PPL averages to layer-by-layer sensitivity analysis.

## Key Features

* **Granular Benchmarking**: Separates *Time-To-First-Token (Prefill)* and *Generation Speed (Decode)*.
* **Layer-wise Sensitivity Analysis**: Visualizes how quantization noise (MSE) and semantic drift (Cosine Similarity) propagate through each layer of the model.
* **Quantization Support**: Easy switching between FP16, INT8, and NF4 (4-bit) via YAML configs.
* **Production-Ready Structure**: Built with Docker, `uv`, Ruff, and CI/CD pipelines.

## Benchmark Results

Running `Llama 3.2 3B Instruct` on NVIDIA RTX 3060 Ti (8GB):

| Configuration | VRAM Usage | Prefill Speed | Decode Speed | Perplexity (PPL) |
| :--- | :--- | :--- | :--- | :--- |
| FP16 (Baseline) | 6.03 GB | ~3,995 t/s | ~32.1 t/s | 9.1797 |
| 4-bit (NF4) | 2.21 GB | ~4,167 t/s | ~32.6 t/s | 9.5845 |

## Quick Start

### Prerequisites
* NVIDIA GPU (Driver 550+)
* Docker & Docker Compose (with NVIDIA Container Toolkit)

### Configuration (Environment Variables)

* **`SAVE_EVAL_RESULTS`**: Set this environment variable (to any value, e.g., `1`) to append benchmark metrics to `src/results/metrics.csv`. If undefined, results will only be printed to the console.

### Running the Lab

1.  **Clone & Start Container**
    ```bash
    git clone [https://github.com/automatedtomato/llm-inference-mini-lab.git](https://github.com/automatedtomato/llm-inference-mini-lab.git)
    cd llm-inference-mini-lab

    # Open in VSCode DevContainers or run manually:
    docker compose up -d
    docker compose exec llm-lab bash
    ```

2.  **Option A: Standard Benchmark (Speed/VRAM/PPL)**
    Evaluate both FP16 and Quantized models to compare basic metrics.

    ```bash
    python -m src.llm_lab \
        --float-eval \
        --quant-eval \
        --qconfig experiment.yaml
    ```

3.  **Option B: Layer-by-Layer Sensitivity Analysis (Deep Dive)**
    Perform a two-pass analysis to compare FP16 vs 4-bit activations for every single operation. This generates a detailed report and scatter plots.

    ```bash
    python -m src.llm_lab \
        --qconfig experiment.yaml \
        --lbl-dump-dir src/results/analysis_report
    ```

    **Output:**
    * `src/results/analysis_report/lbl_analysis.md`: Detailed markdown report.
    * `src/results/analysis_report/images/summary.png`: Global trend of MSE/Cosine Similarity.

## Roadmap

* [x] **Phase 1:** Build Benchmarking Infrastructure (Prefill/Decode separation)
* [x] **Phase 2:** Quantization Analysis (INT8 vs NF4 trade-offs)
* [x] **Phase 3:** Layer-wise Sensitivity Analysis (The "Stethoscope")
    * *Implemented a tool to capture and compare activations per layer (MSE/Cosine Similarity).*
    * *Visualize quantization noise to identify "victim" layers.*
* [ ] **Phase 4:** The Surgery (Mixed Precision Loading)
    * *Implement a custom loader to keep sensitive layers in FP16 while quantizing others.*
    * *Goal: Restore accuracy to FP16 levels with <1GB VRAM overhead.*
* [ ] **Phase 5:** Custom CUDA Kernel Implementation