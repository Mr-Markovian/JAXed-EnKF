# JAXed-EnKF

A jaxed version of Ensemble Kalman Filter (EnKF). It is extremely fast due to JIT-compilation + vectorization over the ensemble integration in the forecast step. Same code can be run on GPU/TPU without any modification. 

__Referrence__
1. (Ensemble Kalman Filters) [https://en.wikipedia.org/wiki/Ensemble_Kalman_filter]
2. (Jax)[https://jax.readthedocs.io/en/latest/notebooks/quickstart.html]
