**Research Idea**  

**Title**  
**Prompt-Driven Iterative Adaptation for Efficient Biological Discovery**  

**Motivation**  
Biological labs face significant barriers in adopting large foundation models due to computational costs and the need to iteratively refine models with new experimental data. Current fine-tuning methods are resource-intensive, limiting rapid adaptation. To bridge this gap, efficient, user-friendly techniques for updating pre-trained models with minimal data and computational overhead are critical for democratizing access to ML-driven discovery.  

**Main Idea**  
This work proposes *prompt-based iterative adaptation*, where lightweight, modular prompts are optimized to recalibrate pre-trained foundation models (e.g., for protein engineering) using sparse experimental feedback from the lab. Instead of retraining the entire model, small, learnable parameter vectors (prompts) are trained on newly collected data, enabling rapid hypothesis validation. A Bayesian framework integrates uncertainty estimation into the prompt, guiding experiment prioritization (e.g., selecting which protein variants to test next). This approach reduces computational load (only training >0.1% of parameters), preserves model performance, and facilitates non-experts to adapt models via intuitive interfaces. The method will be evaluated on real-world datasets (e.g., AAV capsid design) by comparing adaptation speed, cost, and discovery yield against full fine-tuning and LoRA baselines. Success in accelerating discovery cycles could lower adoption barriers for smaller labs and enable dynamic collaboration between ML and biomedical researchers.