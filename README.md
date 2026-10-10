# Estimated LLM Throughput & Memory Benchmark

A zero-checkpoint, hardware-level CLI benchmarking utility to measure **Time-To-First-Token (TTFT)**, **Prefill Throughput**, **Time-Per-Output-Token (TPOT)**, **Decode Throughput**, and **CUDA Allocator Memory Fragmentation** directly on GPU VRAM without downloading model weights.

By instantiating model graphs directly on PyTorch's `meta` device and materializing pseudo-random weights straight into target precision (BF16, FP16, INT8, or NF4 4-bit), this tool allows you to profile hardware feasibility, memory ceilings, and baseline inference latency for models up to 14B parameters on consumer GPUs (e.g., RTX 3060 12GB) in seconds.

---

## Key Features

* **Zero Checkpoint Downloads:** Uses Hugging Face `AutoConfig` to build the model skeleton on the PyTorch `meta` device (0 MB VRAM) and materializes directly into GPU memory with Philox pseudo-random initialization.
* **On-the-Fly Quantization:** Materializes weights directly into `bitsandbytes` 4-bit (NF4) or 8-bit linear layers on CUDA. A 14B model never touches the 28 GB BF16 host memory ceiling.
* **Tied-Weight Safety & LM Head Control:** Optional quantization for the vocabulary projection (`lm_head`) with safe weight-untying so input embeddings remain intact.
* **Hardware-Accurate CUDA Telemetry:** Measures exact kernel execution boundaries via asynchronous `torch.cuda.Event` timing—strictly isolating compute-bound prefill from memory-bandwidth-bound decode.
* **Allocator Fragmentation Tax:** Tracks the gap between driver-reserved memory (`torch.cuda.memory_reserved`) and active tensor allocations (`torch.cuda.memory_allocated`).
* **Automated Parameter Audit:** Inspects the active model graph post-materialization and prints a breakdown of 4-bit vs. 16-bit parameter distribution, flagging memory-heavy layers.

---

## Architectural Scope: Dense Models Only

This benchmark utility is explicitly designed for **dense autoregressive transformer architectures** (e.g., LLaMA 2/3/3.1/3.2, Mistral, Qwen 2/2.5 Dense, Phi-2/3/3.5, Starcoder).

Support for **Mixture-of-Experts (MoE)** and **Multimodal** architectures is intentionally excluded. If you attempt to benchmark models like `LiquidAI/LFM2-MoE`, `Mixtral-8x7B`, or `google/gemma-4-E4B`, you will encounter high memory usage and reduced throughput (~7–8 TPS) due to foundational constraints within vanilla PyTorch and `bitsandbytes`:

### Why MoE & Massive Multimodal Models Fail in Pure PyTorch + BNB

1. **The 3D Fused Expert Barrier:**
In modern MoE architectures (such as `LFM2` and `Qwen-MoE`), expert projections are not individual `nn.Linear` layers. To enable batched computation, expert weights are packed into fused 3-dimensional parameter tensors (e.g., shape `[num_experts, out_dim, in_dim]`). `bitsandbytes` **only** provides CUDA kernels for 2D `nn.Linear` layers; it has no kernels for 3D tensors or batched matrix multiplication (`torch.bmm`).
2. **Missing Fused Grouped-GEMM in Core PyTorch:**
Standard PyTorch has no native 4-bit or 8-bit grouped matrix multiplication. Dedicated inference runtimes (vLLM, TensorRT-LLM) achieve high MoE throughput via custom Triton or CUTLASS `fused_moe` kernels that dynamically dequantize packed weights inside GPU SRAM during routing. In standard PyTorch, executing against 3D weights leaves them in full 16-bit BF16 (~12–14 GB for an 8B model).
3. **The Active vs. Resident VRAM Trap:**
Even if an MoE model only activates 1.5B parameters per token, **all 8B+ parameters must reside in VRAM simultaneously**. On a 12 GB GPU, unquantized expert weights exceed physical memory. The Windows Display Driver Model (WDDM) silently pages the overflow across the PCIe bus into host system RAM (16–32 GB/s vs. 360 GB/s VRAM bandwidth), stalling decode throughput at 7–8 TPS.
4. **Massive Vocabulary & Multimodal Non-Linearities:**
Models with massive token vocabularies (e.g., Gemma's 262k vocabulary = ~5.4 GB in BF16) and vision/audio towers rely on operations like `F.embedding` and depthwise convolutions. These operators enforce strict PyTorch C++ tensor constraints and reject arbitrary quantized wrapper subclasses.

> **Takeaway:** Benchmarking quantized MoE and multimodal architectures requires a dedicated C++/Triton inference engine with custom fused kernels (such as vLLM or llama.cpp), rather than an in-place PyTorch prototyping script.

---

## Baseline PyTorch vs. Production Inference Engines

The throughput figures reported by this utility represent an **un-fused, direct PyTorch execution baseline**.

When deploying models into production runtimes (such as **vLLM**, **llama.cpp / GGUF**, **ExLlamaV2**, or **TensorRT-LLM**), real-world generation speeds are typically **1.5× to 3× faster**:

| Model Size | Precision | This Benchmark (PyTorch + BNB Baseline) | Dedicated Engines (ExLlamaV2 / llama.cpp / vLLM) |
| --- | --- | --- | --- |
| **1B** | BF16 / 4-bit | ~40 TPS | **120 – 160 TPS** |
| **3B** | BF16 / 4-bit | ~22 TPS | **60 – 80 TPS** |
| **14B** | 4-bit (NF4) | ~10 TPS | **20 – 28 TPS** |

### Why Dedicated Engines Run Faster:

* **Fused Dequantization Kernels (Marlin, AWQ, EXL2):** Production engines run fused GEMM kernels that unpack 4-bit weights directly into GPU registers and compute matrix math in a single pass, avoiding temporary memory round-trips.
* **CUDA Graphs:** Dedicated runtimes capture the autoregressive decode step into static execution graphs, eliminating Python dispatch overhead and CPU-GPU synchronization bubbles.
* **PagedAttention:** Physical memory allocation is managed at the page level, completely eliminating memory reallocation overhead during context expansion.

---

## Installation & Requirements

Ensure you have a working Python environment with a CUDA-enabled PyTorch build:

```bash
# Clone the repository
git clone https://github.com/your-username/llm-tps-benchmark.git
cd llm-tps-benchmark

# Install PyTorch with CUDA support (example for CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Hugging Face Transformers, Accelerate, and BitsAndBytes
pip install transformers accelerate bitsandbytes

# Optional: Install FlashAttention-2 (Requires Ampere GPU or newer, CC >= 8.0)
pip install flash-attn --no-build-isolation

```

---

## CLI Usage & Examples

### 1. Standard Dense Baseline (PyTorch SDPA + Static Cache)

Runs a baseline evaluation on a 3B model in native BF16 precision with prefill sequence length 512 and 128 decode steps:

```bash
python llm_tps_benchmark_2.py --model meta-llama/Llama-3.2-3B-Instruct --cache static --prefill 512 --decode 128

```

### 2. Testing Large Dense Models in 4-Bit (e.g., 14B on a 12GB GPU)

Instantiates a 14B parameter model directly on the `meta` device and materializes all 2D linear layers directly as 4-bit NF4 tensors on CUDA. The model fits within ~8.5 GB of active VRAM without triggering out-of-memory errors:

```bash
python llm_tps_benchmark_2.py --model Qwen/Qwen2.5-14B-Instruct --quant 4bit --cache static --prefill 512 --decode 128

```

### 3. Quantizing the LM Head

By default, `lm_head` is kept in full floating-point precision to maintain generation stability. To quantize the LM head alongside internal linear projections:

```bash
python llm_tps_benchmark_2.py --model meta-llama/Llama-3.1-8B-Instruct --quant 4bit --quant-lm-head --lm-head-quant 4bit

```

---

## Understanding the Metrics

### 1. Time-To-First-Token (TTFT) & Prefill Throughput

* **Prefill Stage:** The model processes prompt tokens in parallel. This phase is **compute-bound (GEMM)** and saturates GPU Tensor Cores.
* **TTFT ($ms$):** The wall-clock time from prompt arrival until the first new logit is generated.
* **Prefill Throughput ($tokens/sec$):** $\frac{\text{Batch Size} \times \text{Prefill Tokens}}{\text{TTFT (seconds)}}$.

### 2. Time-Per-Output-Token (TPOT) & Decode Throughput

* **Decode Stage:** The model generates tokens autoregressively one by one. This phase is **memory-bandwidth-bound (GEMV)**; at every token, the entire active model weight payload must be transferred from global VRAM into on-chip cache.
* **TPOT ($ms/token$):** The average latency across all generated decode steps.
* **Decode Throughput ($tokens/sec$):** $\frac{\text{Batch Size} \times 1000}{\text{Average TPOT (ms)}}$.

### 3. Allocator Fragmentation Tax

* **Peak Active Memory:** The maximum VRAM actively storing model weights, activations, and KV cache tensors (`torch.cuda.max_memory_allocated`).
* **Peak Reserved Memory:** The total VRAM reserved by PyTorch's caching allocator from the NVIDIA driver (`torch.cuda.max_memory_reserved`).
* **Fragmentation Tax:** $\text{Peak Reserved Memory} - \text{Peak Active Memory}$. Represents driver memory locked by the caching allocator due to intermediate tensor deallocations that cannot be reclaimed by other applications.

---

## Sample Benchmark Output

```text
==============================================================================
      UNIVERSAL DIRECT-TO-VRAM THROUGHPUT & FRAGMENTATION BENCHMARK
==============================================================================
Model ID        : meta-llama/Llama-3.2-3B-Instruct
Quantization    : 4BIT (LM Head: BF16)
Device          : cuda (NVIDIA GeForce RTX 3060)
------------------------------------------------------------------------------
[*] Fetching configuration for 'meta-llama/Llama-3.2-3B-Instruct'...
[*] Constructing model skeleton on meta device (0 MB VRAM)...
[*] Quantizing Linear & Expert layers to 4BIT on cuda...
[*] Materializing standard parameters (Embeddings, Norms, Router Biases) on cuda...
[+] Model ready in 1420.12 ms (3.21B parameters).

==============================================================================
                      PARAMETER & VRAM AUDIT BREAKDOWN
==============================================================================
  * Quantized Parameters   : 2.42B (1.21 GB in VRAM)
  * Unquantized Parameters : 0.79B (1.58 GB in VRAM)

  Top Unquantized Memory Consumers (> 5M params):
    - model.embed_tokens.weight                        | Shape: [128256, 3072]     | 751.5 MB
    - lm_head.weight                                    | Shape: [128256, 3072]     | 751.5 MB
==============================================================================

[*] Running 2 warmup passes...
[*] Benchmarking Prefill (512 tokens, Batch=1)...
[*] Benchmarking Decode (128 autoregressive steps)...

==============================================================================
                      BENCHMARK PERFORMANCE REPORT
==============================================================================
| Metric                           | Measured Value | Unit        |
|----------------------------------|----------------|-------------|
| Model Parameter Count            | 3.21           | Billion (B) |
| Attention Backend                | sdpa           | string      |
| Cache Strategy                   | static         | string      |
| Model Quantization               | 4BIT           | string      |
| LM Head Status                   | BF16           | string      |
| Batch Size                       | 1              | sequences   |
| Prefill Tokens                   | 512            | tokens      |
| Decode Steps                     | 128            | tokens      |
| Time-To-First-Token (TTFT)       | 194.20         | ms          |
| Prefill Throughput               | 2636.45        | tokens/sec  |
| Avg Time-Per-Output-Token (TPOT) | 44.18          | ms/token    |
| Decode Throughput                | 22.63          | tokens/sec  |
| Peak Active Memory               | 3.142          | GB          |
| Peak Reserved Driver Memory      | 3.488          | GB          |
| Allocator Fragmentation Tax      | 0.346          | GB          |
|----------------------------------|----------------|-------------|

Step Latency: Min: 33.12 ms | Median: 35.80 ms | P95: 84.10 ms | Max: 147.20 ms

```

---

## CLI Reference Guide

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--model` | `str` | `meta-llama/Llama-3.2-3B-Instruct` | Hugging Face model repository ID. |
| `--attn` | `str` | `sdpa` | Attention backend: `sdpa` or `eager`. |
| `--cache` | `str` | `static` | KV Cache buffer strategy: `static` or `dynamic`. |
| `--quant` | `str` | `none` | Core model quantization: `none`, `8bit`, or `4bit` (NF4). |
| `--quant-lm-head` | `flag` | `False` | Enable quantization on the `lm_head` projection. |
| `--lm-head-quant` | `str` | `4bit` | Target precision for the LM Head (`4bit` or `8bit`). |
| `--prefill` | `int` | `512` | Prompt sequence context length ($n$ tokens). |
| `--decode` | `int` | `128` | Number of autoregressive generation steps ($m$ tokens). |
| `--batch-size` | `int` | `1` | Batch size ($B$ sequences). |
| `--dtype` | `str` | `bfloat16` | Compute precision: `bfloat16`, `float16`, or `float32`. |
| `--device` | `str` | `cuda` | Target execution device. |
| `--warmup-steps` | `int` | `2` | Number of dummy warmup passes executed before measuring. |

---

## License

This project is released under the [MIT License](https://www.google.com/search?q=LICENSE).
