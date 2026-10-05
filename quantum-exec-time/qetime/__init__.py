"""qetime -- Understanding and estimating the execution time of quantum circuits.

Implementation of Ma & Li, "Understanding and Estimating the Execution Time of
Quantum Circuits" (ACM TOSEM 2025, arXiv:2411.15631).

Modules
-------
circuits         loading / generating benchmark circuits (MQT Bench)
backends         simulator and IBM backends, qubit coherence (T1/T2) lookup
features         41-d global features and 178-d per-node graph features
measure          simulator execution-time measurement (Eq. 1 repeats, timeouts)
hardware         IBM Quantum job submission and usage (estimate vs. actual) retrieval
dataset          dataset assembly, scaling and PyG conversion
model            graph-transformer execution-time model (Fig. 6)
train            training, split / k-fold / algorithm-family evaluation, fine-tuning
active_learning  greedy sampling on the input domain (GSx, Algorithm 1)
analysis         RQ1 statistics, IBM-estimate evaluation, SHAP importance
"""

__version__ = "1.0.0"
