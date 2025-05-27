Title  
Adaptive Compute Fabric for Co-Optimized Sparse Neural Network Training  

Introduction  

Background  
Deep neural networks (DNNs) have grown explosively in size and complexity, reaching billions of parameters to achieve state-of-the-art results in vision, language, and decision-making applications. However, training these over-parameterized models incurs enormous costs in compute, memory bandwidth, and energy. Recent studies estimate that training a single large model can emit hundreds of tons of CO₂, raising urgent sustainability concerns. Meanwhile, much of this cost stems from repeatedly processing zero or near-zero weights and activations in sparse models—wasted work that hardware like GPUs cannot efficiently bypass due to their optimization for dense, regular compute patterns.  

Model compression techniques—pruning, quantization, low-rank factorization—offer a path toward efficiency, but practical gains are limited by hardware bottlenecks. The community has demonstrated algorithmic methods for maintaining accuracy at high sparsity (e.g., dynamic sparse training, structured block pruning), yet existing hardware lacks support for the irregular memory accesses and control flow required to exploit these methods fully. Conversely, accelerators designed for sparse inference (e.g., Procrustes, TensorDash) target inference workloads or restrict to fixed sparsity patterns, leaving sparse training largely unsupported. A co-design of algorithms and hardware tailored to dynamic sparsity during training is needed to close this gap.  

Research Objectives  
This project aims to design, implement, and evaluate an Adaptive Compute Fabric (ACF) co-optimized with novel sparse training algorithms to enable efficient, large-scale sparse DNN training. Our specific objectives are:  
1. To architect a reconfigurable hardware fabric featuring zero-skipping compute units, index-aware memory controllers, and dynamic interconnects that adapt to evolving sparsity patterns.  
2. To develop a family of structured and dynamic sparse training algorithms—combining block-wise magnitude pruning, gradient-guided regrowth, and input-aware mask adaptation—that maximize hardware utilization on ACF.  
3. To integrate the hardware and algorithmic designs into a full prototype (cycle-level simulator and FPGA emulation) and demonstrate end-to-end training of vision and language models at high sparsity with significant speedup and energy reduction versus GPU baselines.  

Significance  
By bridging the algorithm‐hardware divide, ACF promises to reduce the computational and energy footprint of sparse training by factors of 3–5×, enabling sustainable deep learning on commodity and specialized platforms. The proposed research will provide:  
• A blueprint for next-generation sparse training accelerators and co-design methodologies.  
• New algorithmic insights into structured, dynamic sparsity that preserve accuracy while maximizing parallelism.  
• Empirical evidence quantifying trade-offs among sparsity, performance, and energy, informing both chip designers and practitioners.  

Methodology  

1. System Overview  
The proposed system comprises two tightly coupled components:  
A. Sparse Training Engine (STE): an algorithmic framework that maintains and adapts sparse connectivity during training using block-structured pruning and gradient-driven regrowth.  
B. Adaptive Compute Fabric (ACF): a hardware accelerator featuring  
   • Sparse Compute Units (SCUs) with zero-skipping pipelines,  
   • Index-aware Memory Controllers (IMCs) optimized for CSR/CSC and block-sparse formats,  
   • A Dynamic Interconnect Network (DIN) that reconfigures dataflows based on current sparsity masks.  

2. Sparse Training Algorithm Co-Design  
We adopt a block-structured dynamic sparse training algorithm, summarized in Algorithm 1.  

Algorithm 1: Block-Structured Dynamic Sparse Training  
Inputs:  
  • Initial dense weights $W^{(0)}\in\mathbb{R}^{M\times N}$  
  • Target sparsity ratio $s\in[0,1)$  
  • Block size $B\times B$  
  • Learning rate $\eta$  
  • Total iterations $T$  
  • Prune/regrowth schedule $\{t_k\}_{k=1}^K$  

Initialize mask $M^{(0)}_{i,j}=1$ for all blocks.  

For $t=1$ to $T$ do  
  1. Forward Pass: compute activations only for blocks where $M^{(t-1)}_{i,j}=1$.  
  2. Backward Pass: compute gradients $\nabla W^{(t-1)}$ only on non-zero blocks.  
  3. Weight Update:  
     $$W^{(t)} \;=\; W^{(t-1)} \;-\; \eta\,\nabla W^{(t-1)}\odot M^{(t-1)}$$  
  4. If $t\in\{t_k\}$ (prune/regrowth step):  
     a. Compute block ℓ₂ norms:  
        $$\ell_{i,j} = \|W^{(t)}_{(i,j)}\|_2$$  
     b. Determine threshold $\theta_t$ so that a fraction $s$ of blocks have $\ell_{i,j}<\theta_t$.  
     c. Prune:  
        $$M^{(t)}_{i,j} = 
          \begin{cases}
            0, & \ell_{i,j}<\theta_t,\\
            1, & \text{otherwise}.
          \end{cases}$$  
     d. Regrowth: select top-$k$ blocks by gradient magnitude $\|\nabla W^{(t)}_{(i,j)}\|_2$ among pruned blocks, set their mask to one to maintain overall sparsity ratio.  
  Else  
     $M^{(t)} = M^{(t-1)}$.  
End For  

Key properties:  
• Blocks ensure regularity for ACF PEs (processing elements).  
• Gradient-guided regrowth recovers capacity where needed.  
• Schedule $\{t_k\}$ (e.g., cosine-annealed) balances exploration and stability.  

3. Adaptive Compute Fabric Architecture  

3.1 Sparse Compute Units (SCUs)  
Each SCU contains a zero-skipping multiply-accumulate (MAC) pipeline: incoming activations $a$ and weights $w$ are checked; if $w=0$ or $a=0$, the MAC stage is bypassed to save dynamic power. The SCU pipeline is depth-one at full throughput when operating on non-zero data, achieving utilization  
$$U = \frac{\text{non-zero MACs}}{\text{total MAC slots}}.$$  

3.2 Index-Aware Memory Controllers (IMCs)  
IMCs manage loading of non-zero blocks from off-chip DRAM. We store weights in a block-CSR format:  

  • Row pointers $\text{rowPtr}[i]$  
  • Block column indices $\text{colIdx}[k]$  
  • Block data $\text{data}[k,B,B]$  

The IMC prefetches only needed blocks and broadcasts them to SCUs. A hardware accelerator for prefix-sum on rowPtr ensures minimal control overhead.  

3.3 Dynamic Interconnect Network (DIN)  
To maintain high utilization across varying sparsity patterns, the DIN is a reconfigurable, packet-switched mesh that routes blocks from IMCs to SCUs dynamically. At each prune/regrowth step, a small controller recomputes a routing table so that non-zero blocks are load-balanced across SCUs.  

4. Implementation and Integration  

4.1 Cycle-Level Simulator  
We extend the open-source Aladdin simulator to model SCUs, IMCs, and DIN cycles, measuring latency, throughput, and dynamic power.  

4.2 FPGA Prototype  
We implement a 512-SCU version of ACF on a Xilinx Virtex UltraScale+ board, integrating with a host CPU via PCIe. On-board power sensors measure energy.  

4.3 Software Stack  
A PyTorch extension orchestrates sparse training: at each iteration it updates masks $M^{(t)}$ and dispatches blocks to the FPGA via a CUDA-style API.  

5. Experimental Design  

5.1 Benchmarks  
  • Vision: ResNet-50 on ImageNet-1K (~1.2M images)  
  • Language: Transformer-Base on WMT’14 En↔De  
  • Reinforcement: DQN on Atari Pong  

5.2 Baseline Systems  
  1. NVIDIA V100 GPU with PyTorch dynamic sparse training  
  2. GPU + TensorDash-inspired sparse kernel (up to 2× speedup)  

5.3 Metrics  
  • Training Throughput: images or tokens processed/sec  
  • Time-to-Accuracy: wall-clock until target accuracy reached  
  • Energy per Iteration: Joules/step measured on board  
  • Final Model Quality: top-1/2 accuracy, BLEU score, game score  
  • Hardware Utilization: average SCU occupancy $U$  

5.4 Evaluation Protocol  
For each benchmark, we run 3 independent trials per system, report means ± standard deviation. We perform ablation studies on block size $B$, regrowth rate, and interconnect complexity. Scalability is measured by varying sparsity $s\in\{50\%,75\%,90\%\}$.  

Expected Outcomes & Impact  

Expected Outcomes  
1. Hardware Prototype Performance  
   • 3–5× training throughput speedup over dense GPU  
   • 2–4× reduction in energy per iteration versus GPU  
   • Sustained SCU utilization $U>80\%$ at $90\%$ sparsity  

2. Algorithmic Insights  
   • Identification of optimal block sizes $B$ and pruning/regrowth schedules for different model families.  
   • Quantitative analysis of trade-offs among sparsity, compute parallelism, and accuracy.  

3. Open-Source Deliverables  
   • RTL description of ACF (simulator-ready)  
   • PyTorch extension for block-sparse training  
   • Detailed benchmark scripts and datasets  

Impact  
• Sustainable Deep Learning: By substantially cutting energy and compute for sparse training, ACF paves the way for environmentally friendly large-scale models.  
• Hardware-Algorithm Co-Design: Demonstrates a practical methodology for co-optimizing dynamic algorithms with reconfigurable hardware, influencing future accelerator design.  
• Broader Adoption of Sparsity: Proving that high sparsity yields real speedups and energy gains in training could accelerate adoption in industry and research, reducing the carbon footprint of ML workloads worldwide.  

In summary, this research will bridge the current gap between algorithmic sparsity and hardware support, delivering a versatile compute fabric and co-designed algorithms that transform the efficiency of sparse neural network training.