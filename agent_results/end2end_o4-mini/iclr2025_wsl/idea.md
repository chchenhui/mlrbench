Title: Permutation-Equivariant Graph Embeddings of Neural Weights for Model Retrieval and Synthesis

Motivation:
Neural network weights carry rich information about architecture, training dynamics, and performance, but direct comparison is hindered by weight‐space symmetries (e.g., neuron permutations and scaling). Learning symmetry-aware embeddings of weights can unlock fast model retrieval, transfer initialization, and principled model merging without expensive fine-tuning.

Main Idea:
We represent each layer of a trained network as a fully connected graph (nodes = neurons, edges = weights) augmented with bias attributes. A permutation-equivariant graph neural network (GNN) processes these layer graphs hierarchically, producing a global weight embedding invariant to neuron reorderings and rescalings. Training uses contrastive learning: positive pairs are weight graphs under random permutations/scalings, negatives are graphs from different architectures or training runs. Once trained, the embedding space clusters models by architecture, task similarity, and performance. We demonstrate applications:  
1) Fast model retrieval—given a target task embedding, retrieve pre-trained weights for warm-start training.  
2) Model merging—interpolate embeddings of two models to generate hybrid weight initializations.  
3) Zero-shot performance prediction—regress accuracy directly from embeddings.  
This approach paves the way for efficient, symmetry-aware weight-space search and synthesis.