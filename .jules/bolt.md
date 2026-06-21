## 2025-05-24 - [Vectorized Multi-Source Sync & Flow Caching]
**Learning:** Python-level nested loops in tactical planning (Sources x Targets x Turns) are the primary bottleneck in Orbit Wars agents. Moving synchronization logic to 3D/4D tensor operations reduces per-turn planning time by ~40-50%.
**Action:** Always prefer `torch.topk` on masked cost/score tensors over iterating through target-source pairs.

## 2025-05-24 - [Baseline Flow Caching]
**Learning:** Re-simulating the "no-action" state for every candidate set is redundant. Caching the baseline garrison flow for the turn horizon saves significant O(P*H*A) compute.
**Action:** Implement `precompute_flow_baseline` and pass it to candidate scoring kernels.
