**Title:** Socially-Aligned Intrinsic Reward Learning via Multimodal Human Feedback  

**Motivation:** Current interactive learning systems rely on hand-crafted rewards, failing to leverage implicit, socially rich feedback (e.g., gestures, tone) that humans naturally provide. This limits their ability to adapt to individual users and dynamic environments. Bridging this gap is critical for deploying AI assistants in real-world, socially complex settings like healthcare or education.  

**Main Idea:** Propose a framework where agents learn *intrinsic reward functions* by interpreting multimodal implicit feedback (speech, facial expressions, gaze) during interaction. A transformer-based model encodes these signals into a joint latent space, predicting human intent as a contextual reward. Using inverse reinforcement learning, the agent infers rewards from feedback without predefined semantics, while meta-learning adapts to non-stationary human preferences and environments. For example, a robot tutor could associate a student’s confused expression with a negative reward, adjusting its teaching strategy accordingly.  

**Methodology:**  
1. Collect multimodal interaction data (e.g., dialogue paired with gaze/gestures).  
2. Train a contrastive model to map feedback modalities to latent reward proxies.  
3. Integrate rewards into a meta-reinforcement learning loop, enabling rapid adaptation to user-specific cues.  

**Outcomes & Impact:** Agents that dynamically align with human intent, improving collaboration in assistive robotics or personalized education. Reduces reliance on explicit rewards, enabling scalable, socially aware AI systems.