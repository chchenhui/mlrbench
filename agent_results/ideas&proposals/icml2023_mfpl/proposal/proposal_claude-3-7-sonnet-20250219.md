# Multi-Objective Preference-Based Reinforcement Learning for Personalized Clinical Decision Support

## 1. Introduction

Healthcare decision-making represents one of the most complex domains for artificial intelligence applications, characterized by high-stakes outcomes, multiple competing objectives, and significant individual variation in both patient needs and provider approaches. Traditional reinforcement learning (RL) methods have shown promise in healthcare applications such as medication dosing (Yu et al., 2021), but they face significant limitations when applied to real clinical settings. Chief among these challenges is the difficulty in defining numerical reward functions that accurately capture the complex, multi-faceted nature of clinical objectives and the trade-offs between them.

Clinical decisions rarely optimize for a single outcome but instead balance multiple considerations such as treatment efficacy, adverse effects, cost, quality of life, and patient preferences. For example, when managing chronic conditions like diabetes or hypertension, clinicians must carefully balance glycemic control or blood pressure reduction against risks of hypoglycemia or hypotension, medication side effects, treatment burden, and financial considerations. The optimal balance varies substantially based on patient characteristics, comorbidities, and personal preferences.

Preference-based learning offers a promising alternative by allowing clinicians to express their expertise through pairwise comparisons of treatment trajectories rather than explicit numerical rewards. Recent advances in preference-based reinforcement learning have shown remarkable success in domains such as large language model alignment (Kim et al., 2023) and robotics (Li & Guo, 2024). However, most existing approaches assume a single underlying objective, which fundamentally limits their applicability to healthcare's inherently multi-objective nature.

This research aims to bridge this gap by developing a novel framework that combines multi-objective optimization with preference-based reinforcement learning for clinical decision support. Our proposed approach will maintain a Pareto front of policies representing different trade-offs between competing healthcare objectives and learn a distribution over objective weights that best explains clinical decision-making through pairwise preference elicitation. This approach aligns more naturally with how clinicians reason about complex cases and offers the potential for more personalized treatment policies that respect both individual patient priorities and physician expertise.

The significance of this research extends beyond technical innovation. By creating decision support systems that reason about healthcare decisions in ways that mirror clinical thinking, we can improve the interpretability, trustworthiness, and ultimate adoption of AI systems in healthcare. Furthermore, our approach addresses fairness concerns by explicitly modeling multiple objectives rather than collapsing them into a single reward function that may inadvertently prioritize certain outcomes over others.

## 2. Methodology

Our proposed methodology integrates multi-objective reinforcement learning with preference-based learning through a novel framework designed specifically for clinical decision support. The approach consists of four main components: (1) problem formulation as a multi-objective Markov Decision Process, (2) preference-based learning of objective weights, (3) Pareto-optimal policy computation, and (4) personalized policy recommendation.

### 2.1 Multi-Objective Markov Decision Process Formulation

We formulate the clinical decision-making problem as a Multi-Objective Markov Decision Process (MOMDP) defined by the tuple $\langle S, A, P, \mathbf{R}, \gamma \rangle$, where:

- $S$ is the state space representing patient characteristics (e.g., demographic information, disease markers, comorbidities)
- $A$ is the action space representing possible interventions (e.g., medication types and dosages)
- $P: S \times A \times S \rightarrow [0,1]$ is the transition function representing the probability of moving to a state given the current state and action
- $\mathbf{R} = [R_1, R_2, ..., R_m]: S \times A \times S \rightarrow \mathbb{R}^m$ is a vector of $m$ reward functions, each corresponding to a distinct clinical objective (e.g., efficacy, safety, cost)
- $\gamma \in [0,1)$ is the discount factor

The value function for a policy $\pi$ with respect to each objective $i$ is defined as:

$$V_i^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_i(s_t, a_t, s_{t+1}) \mid s_0 = s\right]$$

Instead of assuming a fixed weighting of objectives, we will learn a distribution over possible weights from clinical preferences. Given a weight vector $\mathbf{w} = [w_1, w_2, ..., w_m]$ where $w_i \geq 0$ and $\sum_{i=1}^m w_i = 1$, the scalarized value function is:

$$V_\mathbf{w}^\pi(s) = \mathbf{w} \cdot \mathbf{V}^\pi(s) = \sum_{i=1}^m w_i V_i^\pi(s)$$

### 2.2 Preference-Based Learning of Objective Weights

We will collect preference data from clinicians by presenting pairs of treatment trajectories $\tau_a$ and $\tau_b$, where each trajectory $\tau = (s_0, a_0, s_1, a_1, ..., s_T)$ represents a sequence of patient states and treatment decisions. The clinician indicates which trajectory they prefer, generating a dataset $\mathcal{D} = \{(\tau_a^j, \tau_b^j, y^j)\}_{j=1}^N$, where $y^j \in \{-1, 1\}$ indicates preference for $\tau_a^j$ or $\tau_b^j$.

We model clinician preferences using the Bradley-Terry model, assuming that the probability of preferring trajectory $\tau_a$ over $\tau_b$ is given by:

$$P(\tau_a \succ \tau_b | \mathbf{w}) = \frac{\exp(U(\tau_a | \mathbf{w}))}{\exp(U(\tau_a | \mathbf{w})) + \exp(U(\tau_b | \mathbf{w}))}$$

where $U(\tau | \mathbf{w})$ is the utility of trajectory $\tau$ under weights $\mathbf{w}$:

$$U(\tau | \mathbf{w}) = \sum_{t=0}^{T-1} \gamma^t \sum_{i=1}^m w_i R_i(s_t, a_t, s_{t+1})$$

To account for the uncertainty in clinicians' preferences and the fact that different clinicians may have different weightings, we will learn a distribution over weight vectors $p(\mathbf{w})$ rather than a point estimate. We adopt a Bayesian approach with a Dirichlet prior on weights:

$$p(\mathbf{w}) \sim \text{Dirichlet}(\alpha_1, \alpha_2, ..., \alpha_m)$$

We update this distribution using the clinical preference data:

$$p(\mathbf{w} | \mathcal{D}) \propto p(\mathcal{D} | \mathbf{w}) p(\mathbf{w})$$

where:

$$p(\mathcal{D} | \mathbf{w}) = \prod_{j=1}^N P(\tau_a^j \succ \tau_b^j | \mathbf{w})^{\mathbb{I}(y^j=1)} \cdot P(\tau_b^j \succ \tau_a^j | \mathbf{w})^{\mathbb{I}(y^j=-1)}$$

Since the posterior does not have a closed-form solution, we will approximate it using variational inference. We approximate $p(\mathbf{w} | \mathcal{D})$ with a parametric distribution $q_\phi(\mathbf{w})$ by minimizing the KL divergence:

$$\phi^* = \arg\min_\phi KL(q_\phi(\mathbf{w}) || p(\mathbf{w} | \mathcal{D}))$$

This is equivalent to maximizing the evidence lower bound (ELBO):

$$\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\mathbf{w})}[\log p(\mathcal{D} | \mathbf{w})] - KL(q_\phi(\mathbf{w}) || p(\mathbf{w}))$$

We represent $q_\phi(\mathbf{w})$ as another Dirichlet distribution with learnable parameters $\phi = [\phi_1, \phi_2, ..., \phi_m]$.

### 2.3 Pareto-Optimal Policy Computation

A key challenge in multi-objective RL is efficiently computing the Pareto front of non-dominated policies. We adopt an approach based on the Convex Hull Value Iteration (CHVI) algorithm (Barrett & Narayanan, 2008), extended to incorporate our learned weight distribution.

For each sampled weight vector $\mathbf{w} \sim q_\phi(\mathbf{w})$, we compute the corresponding optimal policy $\pi_\mathbf{w}$ using standard RL algorithms (e.g., Q-learning) with the scalarized reward function $R_\mathbf{w}(s, a, s') = \mathbf{w} \cdot \mathbf{R}(s, a, s')$.

To ensure we cover the relevant Pareto front, we will implement a sampling strategy that prioritizes regions of the weight space with high posterior probability under $q_\phi(\mathbf{w})$. This approach generates a set of diverse policies $\Pi = \{\pi_1, \pi_2, ..., \pi_K\}$ that cover the clinically relevant portion of the Pareto front.

For each policy $\pi_k \in \Pi$, we compute its value vector $\mathbf{V}^{\pi_k}(s)$ for all states $s \in S$, which represents the expected return for each objective. A policy $\pi_i$ dominates another policy $\pi_j$ if $V_l^{\pi_i}(s) \geq V_l^{\pi_j}(s)$ for all objectives $l \in \{1, 2, ..., m\}$ and there exists at least one objective $l'$ such that $V_{l'}^{\pi_i}(s) > V_{l'}^{\pi_j}(s)$. The Pareto front is the set of all non-dominated policies.

### 2.4 Personalized Policy Recommendation

The final component of our framework is a mechanism for recommending a personalized policy from the Pareto front based on individual patient characteristics and preferences. For a given patient with state $s$, we compute the expected value vector $\mathbf{V}^{\pi_k}(s)$ for each policy $\pi_k \in \Pi$.

We incorporate patient preferences through a lightweight preference elicitation process adapted to their capabilities. For patients who can express detailed preferences, we may use direct methods similar to those used with clinicians. For others, we may rely on simpler methods such as ranking objectives or proxy information from previously documented preferences.

The patient's preferences are used to update our belief about their personal weight vector $\mathbf{w}_{\text{patient}}$. The recommended policy is then:

$$\pi_{\text{recommended}} = \arg\max_{\pi_k \in \Pi} \mathbf{w}_{\text{patient}} \cdot \mathbf{V}^{\pi_k}(s)$$

This approach allows for personalized treatment recommendations that balance clinical expertise (encoded in the learned distribution over weights) with individual patient priorities.

### 2.5 Experimental Design

We will evaluate our approach on medication dosing for two chronic conditions: type 2 diabetes and hypertension, where balancing treatment efficacy against side effects is crucial. For each condition, we will:

1. **Data Collection and Preprocessing**:
   - Obtain de-identified electronic health record (EHR) data from partner healthcare institutions
   - Extract relevant patient features, treatments, and outcomes
   - Define state and action spaces based on clinical guidelines and expert input
   - Identify key objectives (e.g., for diabetes: glycemic control, hypoglycemia risk, medication side effects, treatment burden)

2. **Reward Function Definition**:
   - Define reward functions for each objective based on clinical outcomes
   - Validate these functions with clinical experts to ensure they capture medically meaningful outcomes

3. **Preference Data Collection**:
   - Recruit 20-30 clinical experts for preference elicitation
   - Generate realistic patient trajectories from historical data
   - Present pairs of trajectories and collect preference judgments
   - Conduct follow-up interviews to understand the reasoning behind preferences

4. **Model Training and Evaluation**:
   - Split data into training (70%), validation (15%), and test (15%) sets
   - Train the preference model and compute the Pareto front of policies
   - Evaluate performance using the following metrics:

     a. **Preference Prediction Accuracy**: Ability to predict clinical preferences on held-out trajectory pairs

     b. **Policy Performance**: For each objective, compare our approach against:
        - Single-objective RL policies optimizing each objective individually
        - Multi-objective RL with fixed weights
        - Standard of care (as implemented in historical data)

     c. **Clinical Evaluation**: Present generated treatment plans to clinicians for qualitative assessment of:
        - Clinical appropriateness
        - Treatment personalization
        - Alignment with clinical reasoning

     d. **Diversity of Recommendations**: Measure the coverage of the Pareto front and ability to generate meaningfully different recommendations for different patient profiles

5. **Robustness Analysis**:
   - Sensitivity to the number of preference samples
   - Robustness to variations in clinician preferences
   - Performance with limited or noisy data

## 3. Expected Outcomes & Impact

This research is expected to yield several significant outcomes with potential for transformative impact on clinical decision support systems:

### 3.1 Methodological Advances

1. **Novel Integration of Preference-Based Learning with Multi-Objective RL**: Our framework represents a significant advancement in how preference information can be incorporated into multi-objective reinforcement learning. By learning distributions over objective weights rather than point estimates, we capture the inherent uncertainty and variation in clinical decision-making.

2. **Personalized Treatment Recommendation Engine**: The proposed methodology will provide a systematic approach to generate personalized treatment recommendations that balance multiple clinical objectives while respecting both clinical expertise and individual patient preferences.

3. **Interpretable AI for Healthcare**: By explicitly modeling multiple objectives and learning how clinicians trade them off, our approach will produce more interpretable recommendations than black-box methods. Clinicians can understand not just what treatment is recommended but why it represents an appropriate balance of competing concerns.

### 3.2 Clinical Impact

1. **Improved Clinical Decision Support**: Our framework has the potential to enhance clinical decision-making by providing recommendations that more accurately reflect the complex trade-offs clinicians consider. This is particularly valuable for less experienced providers or in resource-constrained settings.

2. **Personalized Medicine**: By incorporating individual patient preferences, our approach advances the goal of truly personalized medicine, moving beyond one-size-fits-all guidelines toward treatments optimized for each patient's unique situation and values.

3. **Educational Tool for Clinicians**: The system can serve as an educational tool, helping clinicians understand the range of reasonable treatment options and the trade-offs involved. This may be especially valuable for trainees developing clinical judgment.

4. **Reduced Treatment Variation**: While respecting necessary personalization, our approach may help reduce unwarranted variation in care by encoding expert consensus on appropriate trade-offs between competing objectives.

### 3.3 Broader Impact

1. **Framework for Other Healthcare Domains**: Although our initial focus is on medication dosing for chronic conditions, the methodology can be extended to other healthcare domains involving complex trade-offs, such as oncology treatment planning, surgical decision-making, or intensive care management.

2. **Addressing AI Fairness in Healthcare**: By explicitly modeling multiple objectives rather than collapsing them into a single reward function, our approach may help address fairness concerns in healthcare AI by ensuring that important considerations for specific populations are not inadvertently weighted down.

3. **Bridge Between AI and Clinical Practice**: This research represents an important step in bridging the gap between technical AI approaches and clinical practice by developing methods that align with how clinicians actually think about and make decisions.

4. **Potential for Regulatory Consideration**: As healthcare AI becomes increasingly regulated, approaches like ours that emphasize interpretability, multi-objective evaluation, and preference alignment may be better positioned to meet emerging regulatory requirements.

In conclusion, this research proposes a novel framework that combines multi-objective optimization with preference-based reinforcement learning to create more personalized, interpretable, and clinically aligned decision support systems. By learning how clinicians balance competing healthcare objectives and incorporating individual patient preferences, our approach has the potential to significantly advance the practical application of AI in clinical settings, ultimately improving patient care through more personalized treatment recommendations.