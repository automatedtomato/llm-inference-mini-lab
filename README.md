# LLM Inference Optimization Lab

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A personal research project quantifying the trade-offs between Speed, VRAM usage, and Accuracy in LLM inference on consumer hardware (RTX 3060 Ti / 8GB).

This project consists of two generations:

* **Gen 1 (Analysis):** "Surgical" mixed-precision optimization based on layer-wise sensitivity analysis.
* **Gen 2 (Build):** Deconstructing the black box by building a custom quantization engine (INT8/INT4) from scratch to understand the "Bitwise Nightmare" of LLM inference.

---

## Key Features

### Gen 2: The Quantization Engine (Current)

* **Custom Quantization Logic:** Implemented `nn.Linear` replacements for **INT8 (AbsMax)** and **INT4 (Group-wise)** from scratch using pure PyTorch.
* **Bit Packing:** Manually packing two 4-bit integers into `uint8` containers to achieve real VRAM reduction in Python.
* **Group-wise (Block) Quantization:** Solved the "Accuracy Collapse" problem (PPL > 6000) by implementing block-wise scaling (block_size=128), recovering FP16-level accuracy.
* **The "Python Bottleneck":** Intentionally exposed the massive performance cost of performing bitwise unpacking in Python (preparation for CUDA kernel implementation).

### Gen 1: Analysis & Surgery (Completed)

* **Granular Benchmarking:** Precise separation of *Time-To-First-Token (Prefill)* and *Generation Speed (Decode)*.
* **Sensitivity Analysis:** Tools to visualize quantization damage (MSE/Cosine Similarity) layer-by-layer.
* **Surgical Mixed Precision:** A mechanism to selectively load sensitive layers in FP16 while keeping the rest in 4-bit, optimizing the Accuracy/VRAM curve.

---

## Benchmark Results

### 1. Group-wise Quantization (Gen 2 Result)

Comparison of **Custom Block-wise INT4** vs **FP16** vs **BitsAndBytes (NF4)** on `Qwen/Qwen2.5-3B-Instruct`.

| Configuration | VRAM Usage | Decode Speed | Perplexity (PPL) | Status |
| --- | --- | --- | --- | --- |
| **FP16 (Baseline)** | 5.81 GB | ~28.2 t/s | **9.00** | Reference |
| **BitsAndBytes (NF4)** | **2.01 GB** | ~13.0 t/s | 9.98 | Optimized Library |
| **Custom Naive INT4** | 2.06 GB | ~6.5 t/s | **6645.0** (Collapse) | *Phase 7 (Failed)* |
| **Custom Block-INT4** | **2.58 GB** | ~6.6 t/s | **9.20** (Recovered) | *Phase 7 (Success)* |

* **Insight (Accuracy):** Group-wise quantization (Block=128) successfully handled outliers in Qwen 2.5, recovering PPL from unusable (>6000) to near-FP16 levels (9.20).
* **Insight (Speed):** The current implementation is slow (6.6 t/s)** because unpacking bits in Python is computationally expensive. This sets the stage for **Phase 8: Custom CUDA Kernels**.

### 2. Sensitivity Analysis (Gen 1 Result)

Target: `Llama-3.2-3B-Instruct`

| Configuration | VRAM Usage | PPL | Note |
| --- | --- | --- | --- |
| **FP16** | 6.08 GB | 9.18 |  |
| **4-bit (NF4)** | 2.27 GB | 10.25 | Robust baseline |
| **Mixed Precision** | 2.40 GB | 10.25 | Restored sensitive layers |

---

## Quick Start

### Prerequisites

* NVIDIA GPU (Driver 550+)
* Docker & Docker Compose

### Configuration (Environment Variables)
* **`SAVE_EVAL_RESULTS`**: Set this environment variable (to any value, e.g., `1`) to append benchmark metrics to `src/results/metrics.csv`. If undefined, results will only be printed to the console.

### Running the Lab

1. **Start Container**
```bash
docker compose up -d
docker compose exec llm-lab bash
```


2. **Run Gen 2 Custom Quantization (INT4)**
Evaluate your custom quantization engine. This runs the Block-wise INT4 implementation.
```bash
# Expect low VRAM, good PPL, but slow speed (Python overhead)
python -m src.llm_lab --custom-quant-eval --arch Qwen/Qwen2.5-3B-Instruct
```


3. **Run Gen 1 Benchmarks (FP16 vs BNB)**
```bash
python -m src.llm_lab --float-eval --bnb-quant-eval --arch meta-llama/Llama-3.2-3B-Instruct
```


4. **Run Layer-wise Analysis**
Generate a report on which layers are most sensitive to quantization noise.
```bash
python -m src.llm_lab --lbl-dump-dir src/results/analysis_report
```
**Output:**
    * `src/results/analysis_report/lbl_analysis.md`: Detailed markdown report.



---

## Project Roadmap

* [x] **Phase 1-4 (Gen 1):** Build infrastructure, Sensitivity Analysis, Mixed Precision Loader.
* [x] **Phase 5:** Architecture Agnostic Model Loader.
* [x] **Phase 6:** Naive INT8 Quantization (Per-Channel).
* [x] **Phase 7:** Naive INT4 & Bit Packing (Group-wise / Block-wise).
* [ ] **Phase 8:** **Writing Custom CUDA Kernels** (Next Step: Accelerating INT4 Dequantization).