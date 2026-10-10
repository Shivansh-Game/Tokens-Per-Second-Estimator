"""
Universal Zero-Checkpoint LLM Throughput & Memory Fragmentation Benchmark

Direct-to-VRAM benchmarking using standard PyTorch and bitsandbytes modules only.
Features an automatic parameter audit to inspect 4-bit vs 16-bit memory allocation.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache, StaticCache

warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    bnb = None
    HAS_BNB = False


# ==============================================================================
# In-Place Quantization & Meta-to-CUDA Materialization
# ==============================================================================

def replace_linear_layers(
    module: torch.nn.Module,
    quant: str,
    compute_dtype: torch.dtype,
    device: torch.device,
    quant_lm_head: bool,
    lm_head_quant: str,
) -> None:
    """Recursively replaces standard nn.Linear layers with official bitsandbytes layers."""
    if quant not in ["4bit", "8bit"] and not quant_lm_head:
        return

    if not HAS_BNB:
        raise ImportError("bitsandbytes is required: pip install bitsandbytes")

    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            is_lm_head = (name == "lm_head")
            active_quant = lm_head_quant if is_lm_head else quant

            if is_lm_head and not quant_lm_head:
                continue
            if not is_lm_head and quant not in ["4bit", "8bit"]:
                continue

            has_bias = child.bias is not None
            in_features = child.in_features
            out_features = child.out_features

            if active_quant == "4bit":
                new_layer = bnb.nn.Linear4bit(
                    in_features,
                    out_features,
                    bias=has_bias,
                    compute_dtype=compute_dtype,
                    quant_type="nf4",
                    device=device,
                )
                raw_data = torch.randn(out_features, in_features, dtype=compute_dtype, device=device)
                new_layer.weight = bnb.nn.Params4bit(raw_data, requires_grad=False, quant_type="nf4").to(device)
                del raw_data

            elif active_quant == "8bit":
                new_layer = bnb.nn.Linear8bitLt(
                    in_features,
                    out_features,
                    bias=has_bias,
                    has_fp16_weights=False,
                    device=device,
                )
                raw_data = torch.randn(out_features, in_features, dtype=compute_dtype, device=device)
                new_layer.weight = bnb.nn.Int8Params(raw_data, requires_grad=False, has_fp16_weights=False).to(device)
                del raw_data

            if has_bias:
                new_layer.bias.data = torch.zeros(out_features, dtype=compute_dtype, device=device)

            setattr(module, name, new_layer)
        else:
            replace_linear_layers(child, quant, compute_dtype, device, quant_lm_head, lm_head_quant)


def materialize_meta_tensors(model: torch.nn.Module, device: torch.device) -> None:
    """Materializes remaining meta tensors directly as real torch.Tensors on CUDA."""
    for module in model.modules():
        for p_name, param in list(module._parameters.items()):
            if param is not None and param.is_meta:
                new_param = torch.empty_like(param, device=device)
                with torch.no_grad():
                    if new_param.is_floating_point():
                        new_param.normal_(mean=0.0, std=0.02)
                    elif new_param.dtype in [torch.int32, torch.int64]:
                        new_param.zero_()
                module._parameters[p_name] = torch.nn.Parameter(
                    new_param, requires_grad=param.requires_grad
                )

        for b_name, buf in list(module._buffers.items()):
            if buf is not None and buf.is_meta:
                new_buf = torch.empty_like(buf, device=device)
                with torch.no_grad():
                    if new_buf.is_floating_point():
                        new_buf.normal_(mean=0.0, std=0.02)
                    else:
                        new_buf.zero_()
                module._buffers[b_name] = new_buf


def print_parameter_audit(model: torch.nn.Module) -> None:
    """Audits the model parameters and reports the exact 4-bit vs BF16 memory distribution."""
    quant_params = 0
    quant_bytes = 0
    unquant_params = 0
    unquant_bytes = 0
    unquant_layers = []

    for name, p in model.named_parameters():
        numel = p.numel()
        if hasattr(p, "quant_state") or type(p).__name__ in ["Params4bit", "Int8Params"]:
            bytes_per_elem = 0.5 if type(p).__name__ == "Params4bit" else 1.0
            quant_params += numel * (2 if type(p).__name__ == "Params4bit" else 1)
            quant_bytes += int(numel * bytes_per_elem)
        else:
            bytes_per_elem = p.element_size()
            unquant_params += numel
            unquant_bytes += numel * bytes_per_elem
            if numel > 5_000_000:
                unquant_layers.append((name, list(p.shape), (numel * bytes_per_elem) / (1024**2)))

    unquant_layers.sort(key=lambda x: x[2], reverse=True)

    print("\n" + "=" * 78)
    print("                      PARAMETER & VRAM AUDIT BREAKDOWN")
    print("=" * 78)
    print(f"  * Quantized Parameters   : {quant_params / 1e9:.2f}B ({quant_bytes / (1024**3):.2f} GB in VRAM)")
    print(f"  * Unquantized Parameters : {unquant_params / 1e9:.2f}B ({unquant_bytes / (1024**3):.2f} GB in VRAM)")
    
    if unquant_layers:
        print("\n  Top Unquantized Memory Consumers (> 5M params):")
        for layer_name, shape, mb in unquant_layers[:5]:
            print(f"    - {layer_name:<48} | Shape: {str(shape):<18} | {mb:.1f} MB")
    print("=" * 78 + "\n")


def create_model_on_vram(
    model_id: str,
    attn_impl: str = "sdpa",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    quant: str = "none",
    quant_lm_head: bool = False,
    lm_head_quant: str = "4bit",
) -> Tuple[Any, Any]:
    print(f"[*] Fetching configuration for '{model_id}'...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    target_device = torch.device(device)

    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = attn_impl
    setattr(config, "attn_implementation", attn_impl)

    t0 = time.perf_counter()

    print(f"[*] Constructing model skeleton on meta device (0 MB VRAM)...")
    with torch.device("meta"):
        try:
            model = AutoModelForCausalLM.from_config(
                config,
                attn_implementation=attn_impl,
                dtype=dtype,
                trust_remote_code=True,
            )
        except (TypeError, ValueError):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    model.total_params_b = sum(p.numel() for p in model.parameters()) / 1e9

    if quant_lm_head and hasattr(model, "get_output_embeddings") and hasattr(model, "get_input_embeddings"):
        out_emb = model.get_output_embeddings()
        in_emb = model.get_input_embeddings()
        if out_emb is not None and in_emb is not None and getattr(out_emb, "weight", None) is getattr(in_emb, "weight", None):
            out_emb.weight = torch.nn.Parameter(in_emb.weight.clone())

    if quant in ["4bit", "8bit"] or quant_lm_head:
        print(f"[*] Quantizing Linear & Expert layers to {quant.upper()} on {device}...")
        replace_linear_layers(
            model,
            quant=quant,
            compute_dtype=dtype,
            device=target_device,
            quant_lm_head=quant_lm_head,
            lm_head_quant=lm_head_quant,
        )

    print(f"[*] Materializing standard parameters (Embeddings, Norms, Router Biases) on {device}...")
    materialize_meta_tensors(model, device=target_device)

    model.eval()
    init_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[+] Model ready in {init_time_ms:.2f} ms ({model.total_params_b:.2f}B parameters).")

    print_parameter_audit(model)
    return model, config


# ==============================================================================
# Benchmark Harness
# ==============================================================================

@dataclass
class BenchmarkResults:
    param_count_b: float
    ttft_ms: float
    prefill_tps: float
    tpot_ms: float
    decode_tps: float
    step_latencies: List[float]
    peak_active_gb: float
    peak_reserved_gb: float
    frag_tax_gb: float


def run_benchmark(
    model: Any,
    config: Any,
    batch_size: int = 1,
    prefill_len: int = 512,
    decode_len: int = 128,
    cache_type: str = "static",
    device: str = "cuda",
    warmup_steps: int = 2,
) -> BenchmarkResults:
    target_device = torch.device(device)
    vocab_size = getattr(config, "vocab_size", 32000)
    dtype = next(model.parameters()).dtype

    print(f"[*] Running {warmup_steps} warmup passes...")
    warmup_ids = torch.randint(0, vocab_size, (batch_size, 16), device=target_device)
    with torch.inference_mode():
        for _ in range(warmup_steps):
            _ = model(warmup_ids, use_cache=True)
            torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(target_device)

    total_seq_len = prefill_len + decode_len + 16
    cache_obj = None
    if cache_type == "static":
        try:
            cache_obj = StaticCache(
                config=config,
                max_batch_size=batch_size,
                max_cache_len=total_seq_len,
                device=target_device,
                dtype=dtype,
            )
        except Exception:
            cache_obj = DynamicCache()
    else:
        cache_obj = DynamicCache()

    # Prefill Phase
    print(f"[*] Benchmarking Prefill ({prefill_len} tokens, Batch={batch_size})...")
    prompt_ids = torch.randint(0, vocab_size, (batch_size, prefill_len), device=target_device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.inference_mode():
        try:
            output = model(prompt_ids, past_key_values=cache_obj, use_cache=True)
            past_key_values = cache_obj
        except Exception:
            output = model(prompt_ids, use_cache=True)
            past_key_values = getattr(output, "past_key_values", None)

    end_event.record()
    torch.cuda.synchronize()

    if hasattr(output, "past_key_values"):
        past_key_values = output.past_key_values
    elif isinstance(output, tuple) and len(output) > 1:
        past_key_values = output[1]

    ttft_ms = start_event.elapsed_time(end_event)
    prefill_tps = (batch_size * prefill_len) / (ttft_ms / 1000.0)

    # Decode Phase
    print(f"[*] Benchmarking Decode ({decode_len} autoregressive steps)...")
    if hasattr(output, "logits"):
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    elif isinstance(output, tuple):
        next_token = output[0][:, -1, :].argmax(dim=-1, keepdim=True)
    else:
        next_token = torch.randint(0, vocab_size, (batch_size, 1), device=target_device)

    step_latencies: List[float] = []

    with torch.inference_mode():
        for _ in range(decode_len):
            step_start = torch.cuda.Event(enable_timing=True)
            step_end = torch.cuda.Event(enable_timing=True)

            step_start.record()
            step_output = model(next_token, past_key_values=past_key_values, use_cache=True)
            step_end.record()
            torch.cuda.synchronize()

            step_latencies.append(step_start.elapsed_time(step_end))

            if hasattr(step_output, "past_key_values"):
                past_key_values = step_output.past_key_values
            elif isinstance(step_output, tuple) and len(step_output) > 1:
                past_key_values = step_output[1]

            if hasattr(step_output, "logits"):
                next_token = step_output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            elif isinstance(step_output, tuple):
                next_token = step_output[0][:, -1, :].argmax(dim=-1, keepdim=True)
            else:
                next_token = torch.randint(0, vocab_size, (batch_size, 1), device=target_device)

    tpot_ms = sum(step_latencies) / len(step_latencies)
    decode_tps = (batch_size * 1000.0) / tpot_ms

    peak_active_gb = torch.cuda.max_memory_allocated(target_device) / (1024**3)
    peak_reserved_gb = torch.cuda.max_memory_reserved(target_device) / (1024**3)
    frag_tax_gb = max(0.0, peak_reserved_gb - peak_active_gb)

    return BenchmarkResults(
        param_count_b=model.total_params_b,
        ttft_ms=ttft_ms,
        prefill_tps=prefill_tps,
        tpot_ms=tpot_ms,
        decode_tps=decode_tps,
        step_latencies=step_latencies,
        peak_active_gb=peak_active_gb,
        peak_reserved_gb=peak_reserved_gb,
        frag_tax_gb=frag_tax_gb,
    )


# ==============================================================================
# CLI Entrypoint & Reporting
# ==============================================================================

def print_table(args: argparse.Namespace, res: BenchmarkResults) -> None:
    headers = ["Metric", "Measured Value", "Unit"]
    rows = [
        ["Model Parameter Count", f"{res.param_count_b:.2f}", "Billion (B)"],
        ["Attention Backend", f"{args.attn}", "string"],
        ["Cache Strategy", f"{args.cache}", "string"],
        ["Model Quantization", f"{args.quant.upper()}", "string"],
        ["LM Head Status", f"{args.lm_head_quant.upper() if args.quant_lm_head else 'Full Precision (BF16)'}", "string"],
        ["Batch Size", f"{args.batch_size}", "sequences"],
        ["Prefill Tokens", f"{args.prefill}", "tokens"],
        ["Decode Steps", f"{args.decode}", "tokens"],
        ["Time-To-First-Token (TTFT)", f"{res.ttft_ms:.2f}", "ms"],
        ["Prefill Throughput", f"{res.prefill_tps:.2f}", "tokens/sec"],
        ["Avg Time-Per-Output-Token (TPOT)", f"{res.tpot_ms:.2f}", "ms/token"],
        ["Decode Throughput", f"{res.decode_tps:.2f}", "tokens/sec"],
        ["Peak Active Memory", f"{res.peak_active_gb:.3f}", "GB"],
        ["Peak Reserved Driver Memory", f"{res.peak_reserved_gb:.3f}", "GB"],
        ["Allocator Fragmentation Tax", f"{res.frag_tax_gb:.3f}", "GB"],
    ]

    col_widths = [max(len(r[i]) for r in [headers] + rows) for i in range(len(headers))]

    def render_row(items: List[str]) -> str:
        return "| " + " | ".join(item.ljust(col_widths[i]) for i, item in enumerate(items)) + " |"

    sep = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"

    print("\n" + "=" * 78)
    print("                      BENCHMARK PERFORMANCE REPORT")
    print("=" * 78)
    print(render_row(headers))
    print(sep)
    for row in rows:
        print(render_row(row))
    print(sep)

    sorted_lats = sorted(res.step_latencies)
    print(f"\nStep Latency: Min: {sorted_lats[0]:.2f} ms | Median: {sorted_lats[len(sorted_lats)//2]:.2f} ms | P95: {sorted_lats[int(len(sorted_lats)*0.95)]:.2f} ms | Max: {sorted_lats[-1]:.2f} ms\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal Zero-Checkpoint LLM Throughput Benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--attn", type=str, choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")
    parser.add_argument("--cache", type=str, choices=["static", "dynamic"], default="static")
    parser.add_argument("--quant", type=str, choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--quant-lm-head", action="store_true", default=False, help="Quantize the LM head alongside weights")
    parser.add_argument("--lm-head-quant", type=str, choices=["4bit", "8bit"], default="4bit")
    parser.add_argument("--prefill", type=int, default=512)
    parser.add_argument("--decode", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("\n[!] Error: CUDA is not available.\n")
        sys.exit(1)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    selected_dtype = dtype_map[args.dtype]

    print("=" * 78)
    print("      UNIVERSAL DIRECT-TO-VRAM THROUGHPUT & FRAGMENTATION BENCHMARK")
    print("=" * 78)
    print(f"Model ID        : {args.model}")
    print(f"Quantization    : {args.quant.upper()} (LM Head: {args.lm_head_quant.upper() if args.quant_lm_head else 'BF16'})")
    print(f"Device          : {args.device} ({torch.cuda.get_device_name(0)})")
    print("-" * 78)

    try:
        model, config = create_model_on_vram(
            model_id=args.model,
            attn_impl=args.attn,
            dtype=selected_dtype,
            device=args.device,
            quant=args.quant,
            quant_lm_head=args.quant_lm_head,
            lm_head_quant=args.lm_head_quant,
        )

        results = run_benchmark(
            model=model,
            config=config,
            batch_size=args.batch_size,
            prefill_len=args.prefill,
            decode_len=args.decode,
            cache_type=args.cache,
            device=args.device,
            warmup_steps=args.warmup_steps,
        )

        print_table(args, results)

    except Exception as e:
        print("\n[!] Execution Error:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
