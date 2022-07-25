# JAXed-EnKF

A jaxed version of Ensemble Kalman Filter (EnKF). It is extremely fast due to JIT-compilation + vectorization over the ensemble integration in the forecast step. This is the fastest that it can be in plain Python. Same code can be run on GPU/TPU without any mddification.

* Referrence *
(Ensemble Kalman Filters) [https://en.wikipedia.org/wiki/Ensemble_Kalman_filter]
(Jax)[https://jax.readthedocs.io/en/latest/notebooks/quickstart.html]
