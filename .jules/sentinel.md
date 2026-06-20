
## 2025-05-15 - [Player Count Inferred DoS Protection]
**Vulnerability:** The agent inferred player count from the observation without bounds. A malformed observation with many "unique" owners could trigger OOM or extreme CPU/GPU usage during planning (DoS).
**Learning:** Trusting game-state metadata (like owner IDs) for tensor sizing without validation is a security risk in competitive environments.
**Prevention:** Always cap inferred counts to the maximum supported by the environment (4 players in Orbit Wars).

## 2025-05-15 - [Empty Tensor Guard in Survivor Logic]
**Vulnerability:** `_per_step_survivor` assumed `arrivals` would always be populated. Malformed observations or unexpected game states could pass empty tensors, leading to index errors or OOM in `topk`.
**Learning:** High-performance kernels must have non-blocking validation for input shapes.
**Prevention:** Added explicit `None` and `numel() == 0` guards returning safe defaults.

## 2025-05-15 - [Candidate Count Bounding]
**Vulnerability:** `plan_lite_waves` could theoretically attempt to generate an unbounded number of candidates if target/source shortlists were very large, risking resource exhaustion (DoS).
**Learning:** Planning complexity should be bounded by static limits rather than raw observation size.
**Prevention:** Capped source and target counts to 64 each in the core planner loop.
