# Bolt's Journal

## 2025-05-15 - [Vectorized Owner Strength]
**Learning:** Python loops over player counts (even small ones like 4) in high-frequency functions like `_owner_strength` trigger significant overhead and host-device syncs when using boolean masks inside the loop.
**Action:** Use `scatter_add_` for vectorized summation across categories to keep operations on the device and avoid Python-level iteration.

## 2025-05-15 - [Distance Masking Efficiency]
**Learning:** Calling `.clone()` followed by multiple `masked_fill_` calls on large tensors (like the [K, P, P] distance cache) is significantly slower than a single `torch.where` with a broadcasted combined mask.
**Action:** Use `torch.where` with pre-combined boolean masks to avoid redundant memory copies and sequential kernel launches.

## 2025-05-15 - [Orbital Centrality Vectorization]
**Learning:** The previous centrality calculation used a loop-equivalent `torch.where` on a [P, P] distance matrix.
**Action:** Replaced with `torch.mv(d0, alive)`, which is a standard linear algebra operation optimized in BLAS/cuBLAS, reducing complexity and kernel launches.

## 2025-05-15 - [Defense Logic Resource Reuse]
**Learning:** `_build_defense_entries` was performing its own garrison simulation.
**Action:** Refactored to accept the already-computed `status` tensor from `run_turn`, eliminating redundant O(H*P*A) work each turn.

## 2025-05-15 - [2-Player Optimized Combat]
**Learning:** `topk(2)` is relatively expensive for just 2 players.
**Action:** Implemented a dedicated branch for `A=2` in `_per_step_survivor` using simple subtraction and `abs()`, which is significantly faster for the most common game mode.
