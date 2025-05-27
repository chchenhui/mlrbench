**Title:** Unified Interpretable Latent Representations for Joint Perception and Prediction in Autonomous Driving  

**Motivation:** Current autonomous systems often process perception (e.g., detecting objects) and prediction (e.g., forecasting trajectories) separately, leading to redundant computations, error propagation, and opaque decision-making. A unified, interpretable representation that bridges these tasks could improve efficiency, accuracy, and trust in self-driving systems.  

**Main Idea:** Develop a transformer-based architecture that learns a shared latent space for perception and prediction. The model ingests multi-modal sensor data (LiDAR, cameras) and outputs both detected objects and their future trajectories. Key innovations include:  
1. **Multi-task attention mechanisms** to align spatial-temporal features for joint reasoning.  
2. **Explainability layers** that highlight critical regions in input data (e.g., attention maps) and trajectory uncertainty estimates.  
3. **Contrastive learning** to ensure latent embeddings encode actionable driving semantics (e.g., "yielding" vs "accelerating").  

The framework will be trained on datasets like nuScenes, with metrics evaluating prediction accuracy (ADE/FDE) and interpretability (via user studies or saliency consistency). Expected outcomes include reduced computational overhead compared to modular pipelines, improved trajectory forecasting, and human-understandable explanations for model decisions. This could enable safer planning and regulatory compliance by demystifying complex scene reasoning.