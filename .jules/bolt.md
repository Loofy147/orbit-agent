# Bolt's Journal

## 2025-05-15 - [Vectorized Owner Strength]
**Learning:** Python loops over player counts (even small ones like 4) in high-frequency functions like `_owner_strength` trigger significant overhead and host-device syncs when using boolean masks inside the loop.
**Action:** Use `scatter_add_` for vectorized summation across categories to keep operations on the device and avoid Python-level iteration.

## 2025-05-15 - [Distance Masking Efficiency]
**Learning:** Calling `.clone()` followed by multiple `masked_fill_` calls on large tensors (like the [K, P, P] distance cache) is significantly slower than a single `torch.where` with a broadcasted combined mask.
**Action:** Use `torch.where` with pre-combined boolean masks to avoid redundant memory copies and sequential kernel launches.
