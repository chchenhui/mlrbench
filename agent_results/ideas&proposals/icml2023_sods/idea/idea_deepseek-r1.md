**Title:** Gradient-Guided Discrete Sampling via Differentiable Latent Autoencoders  

**Motivation:** Current discrete sampling methods struggle with black-box objectives and high-dimensional, correlated variables (e.g., text, molecular sequences), where gradients of the objective are unavailable or noisy. Existing embedding approaches often produce invalid samples when mapped back to discrete space, limiting their practicality.  

**Main Idea:** Train a variational autoencoder (VAE) to embed discrete structures (e.g., token sequences) into a continuous latent space while ensuring invertible, validity-preserving decoding. For sampling/optimization:  
1. Apply gradient-based MCMC (e.g., Langevin dynamics) in the *latent space*, using a surrogate gradient derived from a learned, lightweight proxy model that approximates the black-box objective.  
2. Decode latent samples to valid discrete configurations via the VAE decoder, which is regularized during training to prioritize semantically meaningful outputs.  

The proxy model is iteratively refined using queried black-box evaluations to reduce approximation error. Experiments on text generation with language models and protein design tasks would validate efficiency and sample quality. This approach bypasses direct discrete space gradients, handles complex dependencies via the VAE’s structure, and maintains validity, enabling scalable sampling for high-dimensional, correlated problems.