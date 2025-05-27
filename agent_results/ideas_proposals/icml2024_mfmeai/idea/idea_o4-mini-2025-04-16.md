Title: Hierarchical Multi-Modal Controller for Embodied Agents

Motivation:  
While foundation models like CLIP or GPT-4V provide rich semantic understanding, embodied agents struggle to translate high-level insights into precise low-level actions. Bridging this gap is crucial for robust, adaptable robots in open-ended environments—e.g., home assistants that must reason about objects and then manipulate them safely.

Main Idea:  
We propose a two-tiered architecture. The top tier is a frozen multi-modal foundation model (MFM) that takes raw sensor streams (RGB, depth, audio) and outputs semantic affordance maps and goal representations. The bottom tier is a hierarchical reinforcement learning (HRL) controller:  
• High-level policy ingests MFM outputs to select subgoals (e.g., “pick up the red cup”).  
• Low-level controllers are specialized motion primitives (grasping, navigation) trained via imitation and on-policy RL.  

Training leverages self-supervised exploration in a photorealistic simulator, where MFM generates pseudo-instructions and affordances to bootstrap HRL. We expect improved sample efficiency, generalization to novel tasks, and real-world transfer. This framework unifies semantics and control, empowering agents with both nuanced perception and agile manipulation.