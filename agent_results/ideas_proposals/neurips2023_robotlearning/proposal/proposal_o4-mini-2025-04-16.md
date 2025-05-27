1. Title  
Safe Adapter-Based Fine-Tuning for Vision–Language Robotics

2. Introduction  
2.1 Background  
Recent advances in large-scale vision–language‐pretrained models (VLPMs), such as CLIP, Flamingo and LLaVA, have demonstrated remarkable capabilities in semantic understanding, open‐world object recognition, and high‐level planning. However, directly fine‐tuning these massive backbones on robotic platforms faces two fundamental challenges: (1) prohibitive computational and data requirements, and (2) safety risks when executing untested policy adaptations on real hardware. At the same time, robotics applications often operate under tight latency, compute and data constraints, and demand formal safety guarantees to prevent damage or disruption during learning and deployment.  

2.2 Research Objectives  
This proposal seeks to develop a parameter‐efficient, safety‐aware adaptation framework—Safe Adapters—for customizing a frozen VLPM backbone to new robotic tasks. We aim to answer the following research questions:  
• How can we pre‐train small “safety adapters” to align semantic features with robot state–action pairs using offline multi‐modal logs?  
• How can we safely fine‐tune these adapters on hardware‐in‐the‐loop with minimal risk and sample complexity?  
• What theoretical and empirical safety guarantees can we provide during both learning and deployment?  

2.3 Significance  
By decoupling semantic reasoning (frozen backbone) from control adaptation (learnable adapters), our approach (i) reduces the tunable parameters to under 5% of the full model, enabling rapid (<1 hr) fine‐tuning on a single GPU; (ii) enforces safety via a learned critic and shield mechanism, preventing catastrophic failures; and (iii) demonstrates strong generalization to novel objects and tasks. This work will democratize deployment of large VLPMs in robotics and push forward safe, sample‐efficient learning paradigms.

3. Methodology  
3.1 Overview  
Our framework has two phases (Fig. 1):  
A. Pre‐training: Learn safety adapters $A_\phi$ that map VLPM embeddings to robot‐control embeddings via contrastive learning on offline data.  
B. Fine‐tuning: Freeze the backbone, update only $\phi$ via a safety‐constrained reinforcement learning (RL) loop using a shielded policy and conservative Q‐learning.  

3.2 Adapter Architecture  
• Backbone: A frozen VLPM $f_\theta: (\text{RGB},\text{Depth},\text{Text})\!\to\!\mathbb{R}^d$.  
• Safety Adapters: At each transformer block $l$, insert two lightweight MLP layers:  
  $$h_l = \mathrm{Adapter}_\phi^{(l)}\bigl(f_\theta^{(l)}(x)\bigr)\quad\in\mathbb{R}^k,\quad k\ll d.$$  
• Action Head: A small policy network $\pi_\phi(a\mid s)$ and safety critic $Q_c^\phi(s,a)$ that take the final adapter embedding $h_L$ and robot proprioception $s_p$ as input.  

3.3 Pre‐Training Phase  
3.3.1 Data Collection  
Collect an offline multi‐modal dataset $\mathcal D_{\mathrm{pre}}=\{(I_i,D_i,s_i,a_i)\}$ from teleoperated logs, self‐play in simulation, and human demonstrations, where $I_i$ are RGB frames, $D_i$ depth maps, $s_i$ robot states, and $a_i$ recorded actions.  

3.3.2 Contrastive Loss  
We align image–action pairs in embedding space via InfoNCE: For a minibatch of $N$ samples, define  
$$z_i = h_L(I_i,D_i),\quad w_i = \psi(s_i,a_i)\in\mathbb{R}^k$$  
where $\psi$ is a small MLP mapping state–action to embedding. The loss for positive pair $(i,i)$ is  
$$\mathcal L_{\mathrm{NCE}} = -\frac{1}{N}\sum_{i=1}^N \log\frac{\exp\!\bigl(z_i^\top w_i/\tau\bigr)}{\sum_{j=1}^N\exp\!\bigl(z_i^\top w_j/\tau\bigr)}\,. $$  
We optimize $\phi$ (adapters) and $\psi$ jointly, keeping backbone $\theta$ fixed.

3.4 Fine‐Tuning Phase  
3.4.1 Safe RL Formulation  
After pre‐training, we fine‐tune adapters in the target environment using a Constrained Markov Decision Process (CMDP):  
$$(\mathcal S,\mathcal A,P,R,C,\gamma)$$  
with cost function $c(s,a)$ measuring risk (e.g., collision or torque limits). We seek policy $\pi_\phi$ that  
maximize $\mathbb E\bigl[\sum_t\gamma^tR(s_t,a_t)\bigr]$  
subject to $\mathbb E\bigl[\sum_t \gamma^t C(s_t,a_t)\bigr]\le d$.  

3.4.2 Safety Critic and Shield  
Train a safety critic $Q_c^\phi(s,a)$ via Conservative Q‐Learning (CQL) to estimate worst‐case cumulative cost. We impose a shield: at each decision step  
if $Q_c^\phi(s_t,a^\star) > \epsilon$, veto $a^\star$ and replace with backup policy $\pi_b(s_t)$. Formally:  
$$a_t = 
\begin{cases}
a^\star\!\sim\pi_\phi(\cdot\mid s_t) & \text{if }Q_c^\phi(s_t,a^\star)\le\epsilon,\\
a_b\!\sim\pi_b(\cdot\mid s_t)   & \text{otherwise.}
\end{cases}$$  

3.4.3 Policy Optimization  
We employ a Lagrangian‐based actor‐critic with CQL‐style penalties. The joint loss for adapters $\phi$ is:  
\[
\begin{aligned}
\mathcal L(\phi,\lambda) =\;& -\mathbb E_{(s,a)\sim\mathcal D_t}\bigl[Q_r^\phi(s,a)\bigr] \;+\;\lambda\bigl(\mathbb E[C(s,a)]-d\bigr)\\
&+\;\alpha\,\mathcal L_{\mathrm{CQL}}(Q_r^\phi)\;+\;\beta\,\mathcal L_{\mathrm{CQL}}(Q_c^\phi)\,,
\end{aligned}
\]  
where $Q_r^\phi$ is the reward critic, $\lambda$ a Lagrange multiplier, and $\mathcal L_{\mathrm{CQL}}$ enforces conservative Q‐value lower bounds. We update $\phi$ by stochastic gradient descent and adapt $\lambda$ via dual ascent.

Algorithm 1 (outline):  
1. Initialize $\phi$, $\psi$, $\lambda\leftarrow 0$, replay buffer $\mathcal B\leftarrow\emptyset$.  
2. Pre‐train $\phi,\psi$ on $\mathcal D_{\mathrm{pre}}$ with $\mathcal L_{\mathrm{NCE}}$.  
3. For each environment step:  
   a. Observe $s_t=(I_t,D_t,s_{p,t})$.  
   b. Sample $a^\star\sim\pi_\phi(\cdot\mid s_t)$; apply shield to get $a_t$.  
   c. Execute $a_t$, receive $(r_t,c_t,s_{t+1})$, store in $\mathcal B$.  
   d. Periodically sample minibatch from $\mathcal B$ and update  
      – $Q_r^\phi$ and $Q_c^\phi$ via CQL‐style TD learning,  
      – $\pi_\phi$ (adapters) and dual variable $\lambda$ via $\nabla_\phi\mathcal L,\ \nabla_\lambda\mathcal L$.  

3.5 Experimental Design  
3.5.1 Simulated Benchmarks  
We evaluate on RoboSuite pick‐and‐place, Meta‐World manipulation, and Habitat navigation tasks with unseen objects. Metrics: task success rate, constraint violations per episode, adaptation time, and parameter‐update ratio.  

3.5.2 Real‐World Evaluation  
Deploy on a Franka Panda arm for object rearrangement and a TurtleBot for semantic navigation. Measure collisions, hardware faults, and end‐task success in 50 randomized trials.  

3.5.3 Evaluation Metrics  
• Adaptation Efficiency: GPU hours and number of episodes to reach 90% of final performance.  
• Safety: average cost per episode and frequency of shield activations.  
• Generalization: zero‐shot success on novel object categories/text instructions.  
• Resource Utilization: fraction of tunable parameters, peak GPU memory.  

4. Expected Outcomes & Impact  
4.1 Performance Gains  
We anticipate that our Safe Adapter framework will:  
• Achieve ≥ 90% task success within 1 hour of fine‐tuning on a single GPU, using < 5% of VLPM parameters.  
• Reduce unsafe events (constraint violations) by > 80% compared to unconstrained adapter tuning.  
• Generalize to new objects/instructions with < 5% performance drop versus in‐distribution tasks.  

4.2 Theoretical Contributions  
• A novel contrastive pre‐training objective aligning semantic and control embeddings in parameter‐efficient adapters.  
• A rigorous safety‐constrained RL formulation combining CQL and action shielding with provable risk bounds.  
• Insights into the trade‐off between adapter size, data efficiency, and safety guarantees.  

4.3 Practical Implications  
Our approach democratizes the deployment of large VLPM backbones on resource‐limited robotic platforms, enabling rapid, safe adaptation for warehouse automation, assistive indoor robots, and field robotics. By open‐sourcing the adapter modules and pre‐training logs, we will catalyze community adoption and further research on safe, parameter‐efficient robot learning with large models.