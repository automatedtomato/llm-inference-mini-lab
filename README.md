# LLM Inference Optimization Mini Lab (Gen 1)

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A personal research project quantifying the trade-offs between Speed, VRAM usage, and Accuracy in LLM inference on consumer hardware (RTX 3060 Ti / 8GB).

This project moves beyond simple "load_in_4bit" by implementing **Layer-wise Sensitivity Analysis** to visualize quantization damage and a **Surgical Mixed Precision Patch** to selectively restore sensitive layers to FP16.

## Key Features

* **Granular Benchmarking**: Precise measurement of *Time-To-First-Token (Prefill)* and *Generation Speed (Decode)* separation.
* **Sensitivity Analysis**: Tools to capture activations layer-by-layer, calculating MSE (Noise) and Cosine Similarity (Semantic Drift) between FP16 and 4-bit models.
* **Mixed Precision**: A custom loading mechanism that injects specific FP16 weights from `safetensors` into a 4-bit quantized model, enabling sub-1GB fine-tuning of accuracy vs VRAM.
* **Reproducibility**: Fully containerized with Docker, NVIDIA Container Toolkit, and `uv` package management.

## Benchmark Results (Snapshot)

Target: `Llama-3.2-3B-Instruct` on RTX 3060 Ti (8GB)

| Configuration | VRAM Usage | Prefill Speed | Decode Speed | Perplexity (PPL) |
| :--- | :--- | :--- | :--- | :--- |
| **FP16 (Baseline)** | 6.08 GB | ~4,430 t/s | ~33.0 t/s | **9.1871** |
| **4-bit (NF4)** | 2.27 GB | ~2,640 t/s | ~16.8 t/s | 10.2583 |
| **Mixed Precision** | 2.40 GB | ~2,686 t/s | ~17.6 t/s | 10.2566 |

*Insight: The 4-bit NF4 quantization is incredibly robust. surgically restoring 4 sensitive layers (down_proj/o_proj) yielded minimal PPL gain, proving the base quantization quality is high for general text.*

## Quick Start

### Prerequisites
* NVIDIA GPU (Driver 550+)
* Docker & Docker Compose

### Configuration (Environment Variables)

* **`SAVE_EVAL_RESULTS`**: Set this environment variable (to any value, e.g., `1`) to append benchmark metrics to `src/results/metrics.csv`. If undefined, results will only be printed to the console.

### Running the Lab

1.  **Start Container**
    ```bash
    docker compose up -d
    docker compose exec llm-lab bash
    ```

2.  **Run Benchmark (FP16 vs 4-bit)**
    Evaluate both FP16 and Quantized models to compare basic metrics.
    ```bash
    python -m src.llm_lab --float-eval --quant-eval --qconfig configs/experiment.yaml
    ```

3.  **Run Sensitivity Analysis**
    Perform a two-pass analysis to compare FP16 vs 4-bit activations for every single operation. This generates a detailed report and scatter plots.
    ```bash
    python -m src.llm_lab --qconfig configs/experiment.yaml --lbl-dump-dir specify/save/directory
    ```
    **Output:**
    * `src/results/analysis_report/lbl_analysis.md`: Detailed markdown report.
    * `src/results/analysis_report/images/summary.png`: Global trend of MSE/Cosine Similarity.

4.  **Run Mixed Precision Evaluation**
    Edit `configs/experiment.yaml` to specify target layers in `fp16_layers`.
    ```bash
    python -m src.llm_lab --quant-eval --qconfig configs/experiment.yaml --mixed-prec
    ```
