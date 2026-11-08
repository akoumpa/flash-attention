"""Compare compact, physical, and caller-masked FA4 through the public autograd API.

Run in an existing FA4 GPU environment; see repro_seqused.md. No Automodel dependency.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="FA4 checkout to import"
    )
    parser.add_argument("--documents", type=int, default=256)
    parser.add_argument("--span", type=int, default=128)
    parser.add_argument("--used", type=int, default=96)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument(
        "--window", type=int, default=-1, help="Left distance; -1 means full causal"
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--checkpoint", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-zero",
        action="store_true",
        help="Fail unless physical padding is zero",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not (
        args.documents > 0
        and 1 <= args.used < args.span
        and args.iterations > 0
        and args.repeats > 0
    ):
        parser.error("Require documents/iterations/repeats > 0 and 1 <= used < span")
    if args.kv_heads < 1 or args.heads < 1 or args.heads % args.kv_heads:
        parser.error("heads must be a positive multiple of kv-heads")

    import torch
    from torch.utils.checkpoint import checkpoint

    import flash_attn

    # Keep the installed package root: the repository root imports optional FA2
    # binaries. Select only the checkout's cute package, not a different FA2 ABI.
    flash_attn.__path__ = [
        str(args.source.resolve() / "flash_attn"),
        *flash_attn.__path__,
    ]
    from flash_attn.cute import interface

    source = Path(interface.__file__).resolve()
    if not source.is_relative_to(args.source.resolve()):
        raise RuntimeError(f"Wrong imported source: {source}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Requires a working CUDA environment with FA4 and its dependencies"
        )
    print(
        json.dumps(
            {
                "source": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "arguments": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
            }
        )
    )
    torch.manual_seed(17)
    tokens = args.documents * args.span
    physical_cu = (
        torch.arange(args.documents + 1, device="cuda", dtype=torch.int32) * args.span
    )
    # Alternating document lengths, with padding at the END of each physical span.
    used = (
        args.used - torch.arange(args.documents, device="cuda", dtype=torch.int32) % 2
    )
    compact_cu = torch.cat((used.new_zeros(1), used.cumsum(0, dtype=torch.int32)))
    valid = (torch.arange(tokens, device="cuda") % args.span) < used.repeat_interleave(
        args.span
    )
    indices = valid.nonzero().flatten()
    mask = valid[:, None, None]
    inputs = tuple(
        torch.randn(
            tokens, h, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for h in (args.heads, args.kv_heads, args.kv_heads)
    )
    sink = torch.zeros(
        args.heads, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    upstream = torch.randn_like(inputs[0]).masked_fill(~mask, 0)

    def forward(mode: str, q, k, v, sinks):
        if mode == "compact":
            q, k, v = (x.index_select(0, indices) for x in (q, k, v))
        elif mode == "masked":
            # Where backward also masks dQ/dK/dV; output-only masking is insufficient.
            q, k, v = (torch.where(mask, x, 0) for x in (q, k, v))
        offsets = compact_cu if mode == "compact" else physical_cu
        lengths = {} if mode == "compact" else {"seqused_q": used, "seqused_k": used}
        output, _ = interface.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=offsets,
            cu_seqlens_k=offsets,
            max_seqlen_q=args.used,
            max_seqlen_k=args.used,
            causal=True,
            window_size=(None if args.window < 0 else args.window, None),
            learnable_sink=sinks,
            **lengths,
        )
        if mode == "compact":
            return output.new_zeros(tokens, args.heads, 64).index_copy(
                0, indices, output
            )
        return torch.where(mask, output, 0) if mode == "masked" else output

    def cycle(mode: str):
        values = (*inputs, sink)
        fn = lambda *xs: forward(mode, *xs)
        output = (
            checkpoint(fn, *values, use_reentrant=False)
            if args.checkpoint
            else fn(*values)
        )
        return output, torch.autograd.grad(output, values, upstream)

    # Poison empty floating allocations to expose unwritten rows deterministically.
    # Restore both controls before timing; no patching of FA4 or allocator functions.
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    safe = []
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        reference_out, reference_grads = cycle("compact")
        for mode in ("compact", "physical", "masked"):
            output, grads = cycle(mode)
            # Compare all logical outputs/gradients, and separately inspect physical padding.
            for actual, expected in zip(
                (output, *grads[:3]), (reference_out, *reference_grads[:3]), strict=True
            ):
                torch.testing.assert_close(
                    actual[valid], expected[valid], atol=0.02, rtol=0.02
                )
            torch.testing.assert_close(
                grads[3], reference_grads[3], atol=0.02, rtol=0.02
            )
            zeros = [bool(torch.all(x[~valid] == 0)) for x in (output, *grads[:3])]
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "valid_values_close": True,
                        "zero_padding_O_dQ_dK_dV": zeros,
                    }
                )
            )
            if all(zeros):
                safe.append(mode)
        del reference_out, reference_grads, output, grads
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill
    if "compact" not in safe or "masked" not in safe:
        raise AssertionError("A correctness baseline failed zero-padding checks")
    if args.require_zero and "physical" not in safe:
        raise AssertionError(
            "Physical buffers have unwritten padding; do not use this path unmasked"
        )
    if args.check_only:
        return

    samples = {mode: [] for mode in safe}
    for mode in safe:
        for _ in range(3):
            cycle(mode)
    # Alternate order; time complete checkpointed forward/backward, including copies.
    for repeat in range(args.repeats):
        for mode in safe if repeat % 2 == 0 else list(reversed(safe)):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.iterations):
                cycle(mode)
            torch.cuda.synchronize()
            samples[mode].append(1000 * (time.perf_counter() - start) / args.iterations)
    medians = {mode: statistics.median(values) for mode, values in samples.items()}
    print(
        json.dumps(
            {
                "median_cycle_ms": medians,
                "samples_ms": samples,
                "speedup_vs_compact": {
                    mode: medians["compact"] / ms for mode, ms in medians.items()
                },
                "scope": "synthetic packed attention cycle, not a full-model throughput claim",
            }
        )
    )


if __name__ == "__main__":
    main()
