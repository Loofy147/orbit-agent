## 2025-05-15 - [Player Count Inferred DoS Protection]
**Vulnerability:** The agent inferred player count from the observation without bounds. A malformed observation with many "unique" owners could trigger OOM or extreme CPU/GPU usage during planning (DoS).
**Learning:** Trusting game-state metadata (like owner IDs) for tensor sizing without validation is a security risk in competitive environments.
**Prevention:** Always cap inferred counts to the maximum supported by the environment (4 players in Orbit Wars).

## 2025-05-15 - [Baseline Shape Validation]
**Vulnerability:** Reusing a cached baseline could lead to invalid results or crashes if the planet count or player count changed unexpectedly.
**Learning:** Caches must be verified against current observation dimensions before use.
**Prevention:** Added a shape-check guard in `sparse_launch_flow_delta` that falls back to recomputation if the baseline dimensions don't match the current `ParsedObs`.
