# DiffuNA – Diffusion-Powered Generative Design of RNA Therapeutics

## Introduction

### Background

RNA molecules play a critical role in various biological processes, including gene expression, regulation, and catalytic functions. The design of functional RNA molecules, such as aptamers, ribozymes, and siRNAs, is crucial for therapeutic applications. However, traditional methods for designing RNA therapeutics are labor-intensive, time-consuming, and often rely on trial-and-error approaches. The advent of artificial intelligence (AI) offers promising solutions to accelerate the discovery and design of functional RNA molecules.

### Research Objectives

The primary objective of this research is to develop a novel generative framework, named DiffuNA, that leverages diffusion models to design functional RNA molecules. Specifically, the research aims to:

1. Develop a 3D graph-based diffusion model that jointly learns RNA sequence, secondary, and tertiary structure distributions from public databases.
2. Implement a reverse diffusion process to sample candidate sequences and structures based on user-specified targets.
3. Refine the candidate sequences and structures using an embedded reinforcement-learning loop to optimize folding stability and binding affinity.
4. Validate the performance of DiffuNA on standard benchmarks and compare it to existing generative baselines.

### Significance

The successful development of DiffuNA has the potential to significantly accelerate the discovery and design of RNA therapeutics. By automating the design process, DiffuNA can reduce costs, expand the therapeutic repertoire, and enable the rapid generation of high-novelty, high-affinity RNA molecules. This research also contributes to the broader AI community by showcasing the application of diffusion models in the field of nucleic acids research.

## Methodology

### Research Design

#### Data Collection

The data for training DiffuNA will be sourced from public databases, including the Protein Data Bank (PDB) for RNA 3D structures and SHAPE reactivity data for RNA secondary structures. The dataset will include a diverse range of RNA sequences and structures to ensure the model's generalizability.

#### Model Architecture

DiffuNA is based on a 3D graph-based diffusion model that consists of two main components: a graph neural network (GNN) for structure representation and a Transformer-based sequence module for sequence representation. The model's architecture is inspired by RiboDiffusion (Huang et al., 2024) and other recent generative diffusion models for molecular design (Schneuing et al., 2022; Luo et al., 2022).

The GNN module takes the 3D coordinates of RNA atoms as input and encodes them into a graph representation. The Transformer module takes the RNA sequence as input and encodes it into a sequence representation. The two representations are then concatenated and passed through a series of transformer layers to learn the joint distribution of RNA sequence and structure.

#### Training Procedure

During training, DiffuNA iteratively corrupts the RNA graphs and sequences and denoises them to reconstruct the native conformations and motifs. The corruption process involves adding noise to the graph and sequence representations, while the denoising process involves learning to remove the noise and reconstruct the original representations. The model is trained using a variational autoencoder (VAE) loss, which encourages the learned representations to be close to the ground truth data.

#### Inference Procedure

At inference, users specify a target binding pocket or structural scaffold. DiffuNA samples candidate sequences and structures via reverse diffusion, starting from random noise and iteratively denoising the representations to generate candidates that match the target. An embedded reinforcement-learning loop then refines the candidates by optimizing predicted folding stability and binding affinity.

The folding stability is predicted using a pretrained folding predictor, such as trRosettaRNA (trRosettaRNA, 2023), which is trained on a large dataset of RNA sequences and structures. The binding affinity is predicted using a docking surrogate, such as AutoDock (Morris et al., 1998), which estimates the affinity of the candidate RNA molecules to the target binding pocket.

#### Evaluation Metrics

The performance of DiffuNA will be evaluated using the following metrics:

1. **Sequence Recovery**: The accuracy of the model in recovering the original RNA sequences from the corrupted representations.
2. **Structure Recovery**: The accuracy of the model in reconstructing the original RNA structures from the corrupted representations.
3. **Binding Affinity**: The predicted binding affinity of the generated RNA molecules to the target binding pocket.
4. **Folding Stability**: The predicted folding stability of the generated RNA molecules.
5. **Novelty**: The diversity and novelty of the generated RNA molecules, as measured by the similarity to existing RNA sequences and structures.

### Experimental Design

The performance of DiffuNA will be validated on standard benchmarks, including thrombin-binding aptamers and hammerhead ribozymes. The model will be compared to existing generative baselines, such as RiboDiffusion (Huang et al., 2024) and DiffSBDD (Schneuing et al., 2022).

## Expected Outcomes & Impact

### Expected Outcomes

1. **High-Novelty, High-Affinity RNA Therapeutics**: DiffuNA is expected to generate RNA molecules with high binding affinity and favorable folding stability, significantly speeding up lead generation and broadening the scope of RNA-based drug design.
2. **Improved Efficiency**: By automating the design process, DiffuNA can reduce the time and cost associated with traditional RNA design methods.
3. **Generalization to Novel Targets**: DiffuNA is expected to generalize to novel RNA targets and binding pockets without extensive retraining, enabling rapid adaptation to new therapeutic applications.
4. **Contribution to the AI Community**: By showcasing the application of diffusion models in the field of nucleic acids research, DiffuNA will contribute to the broader AI community and inspire further research in this area.

### Impact

The successful development of DiffuNA has the potential to revolutionize the field of RNA-based drug design. By automating the design process, DiffuNA can accelerate the discovery and development of new therapeutic agents, reducing the time and cost associated with traditional methods. This research also has the potential to expand the therapeutic repertoire of RNA-based drugs, enabling the treatment of a wider range of diseases.

In addition to its impact on RNA-based drug design, DiffuNA also contributes to the broader AI community by showcasing the application of diffusion models in the field of nucleic acids research. By demonstrating the effectiveness of diffusion models in generating high-novelty, high-affinity RNA molecules, DiffuNA will inspire further research in this area and encourage the development of new generative models for molecular design.

## Conclusion

In conclusion, the development of DiffuNA represents a significant advance in the field of RNA-based drug design. By leveraging diffusion models to generate high-novelty, high-affinity RNA molecules, DiffuNA has the potential to accelerate the discovery and development of new therapeutic agents. This research also contributes to the broader AI community by showcasing the application of diffusion models in the field of nucleic acids research. With further development and refinement, DiffuNA has the potential to revolutionize the field of RNA-based drug design and enable the rapid generation of new therapeutic agents.