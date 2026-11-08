# Physical padded attention: correctness and cost

`repro_seqused.py` uses only PyTorch and FA4's public autograd API. It compares
three complete forward/backward cycles, with checkpoint recomputation by default:

| Path | Work included |
| --- | --- |
| `compact` | Gather valid Q/K/V rows, run attention, scatter into a zero output. Autograd restores zero input gradients in padding. |
| `physical` | Pass physical offsets and logical `seqused_q/k` directly. Requires zero unused O/dQ/dK/dV from the dependency. |
| `masked` | Pass physical buffers, with differentiable `torch.where` masks around Q/K/V and O. This caller-side alternative also masks input gradients. |

Use a working Blackwell GPU environment with this checkout's FA4 dependencies
already installed. The original evidence used GB200, BF16 and
`flash-attn-4==4.0.0b31.dev7+g3c2c6b3`. The script does not install or modify
dependencies. `--source` explicitly selects and verifies the imported checkout.

```bash
# Run from this checkout. Select your existing GPU Python environment.
uv run --no-project --python /path/to/gpu-env/bin/python \
  benchmarks/repro_seqused.py --source "$PWD" --require-zero

# Nontrivial sliding window: some valid documents exceed the left distance.
uv run --no-project --python /path/to/gpu-env/bin/python \
  benchmarks/repro_seqused.py --source "$PWD" --require-zero --window 128 \
  --span 256 --used 138 --documents 128

# Compare the unpatched parent without altering this checkout:
git worktree add --detach /tmp/fa4-seqused-parent 3c2c6b31e730fd13e7d6f0d384af22684b24c532
uv run --no-project --python /path/to/gpu-env/bin/python \
  benchmarks/repro_seqused.py --source /tmp/fa4-seqused-parent
```

Defaults are 32,768 physical tokens: 256 spans of 128 tokens, alternating 96/95
valid tokens, 64 query heads, 8 KV heads, head dimension 64, learned sinks, causal
attention and BF16. This is a synthetic reviewer fixture, not captured private
training data. Try `--used 64`, `--used 127`, `--no-checkpoint`, or a smaller smoke
test with `--documents 4 --heads 8 --kv-heads 2 --check-only`.

The correctness phase enables PyTorch's fill-uninitialized-memory diagnostic to
poison empty floating allocations with NaNs. It compares valid O/dQ/dK/dV/sink
gradients with compact attention (`atol=rtol=0.02`) and separately requires exact
zeros in padding. These are BF16 smoke checks, not an FP64 accuracy or convergence
proof. The diagnostic controls are restored before timing. Unsafe physical paths
are reported but excluded from timing; `--require-zero` makes them an error.
No allocator or FA4 function is monkeypatched.

Timing excludes compilation, metadata construction and correctness checks;
it includes gather/scatter or masks, output allocation, and backward/recomputation.
Both Q/K/V and output masks are included for the caller-side alternative. Modes
alternate order across repetitions. JSON prints source hash, GPU, parameters,
padding checks, timing samples and speedup relative to compact.

## Interpretation and upstream design choice

The allocation patch itself is a correctness guarantee, not a faster attention
kernel. It adds zero fills when used lengths are supplied. The potential benefit
is avoiding gather/scatter in an application that otherwise needs defined
physical outputs and gradients. Prior captured GB200 fixtures showed roughly
17.5–18.0% lower checkpointed attention-cycle latency for the physical path;
that is not a promised result for these synthetic inputs or other hardware.

FA4 tests already mask unused gradients and explicitly preserve caller-provided
output-buffer sentinels. Therefore unused rows are not universally promised to be
zero today. This patch preserves supplied buffers and strengthens only internally
allocated standard Q/K/V buffers. It makes no guarantee for unused LSE values or
sparse-MLA paths. Upstream review should decide whether zeroing belongs behind an
opt-in API option or in the default public autograd contract. The `masked` mode
provides a complete caller-side alternative for that discussion; output-only
masking is insufficient, and multiplication by zero does not sanitize NaNs.
