Title: Permutation-Equivariant Contrastive Embeddings for Model Zoo Retrieval

Motivation:  
As model repositories swell past a million entries, practitioners struggle to discover pre-trained networks suited to new tasks. Current metadata-based search fails to capture latent functional similarities hidden in raw weights, leading to redundant training and wasted compute.

Main Idea:  
We introduce a permutation-equivariant encoder that maps a network’s weight tensors into a compact embedding space respecting layer symmetries (e.g., neuron permutations and scaling). Each weight matrix is treated as a graph structure—nodes for neurons, edges for connections—and processed by a shared GNN module with equivariant message passing. We train this encoder via contrastive learning: positive pairs derive from symmetry-preserving augmentations (permuted neurons, scaled filters), while negatives are weights from functionally distinct models. Optionally, downstream performance metrics serve as weak supervisors. The learned embeddings cluster models by task and capability, enabling k-NN retrieval of networks that best transfer to unseen datasets. We will evaluate retrieval precision, clustering coherence, and downstream fine-tuning efficiency.  
Expected Impact:  
This framework streamlines model selection in massive zoos, reduces redundant training, and paves the way for automated, weight-space–driven architecture search and transfer learning.