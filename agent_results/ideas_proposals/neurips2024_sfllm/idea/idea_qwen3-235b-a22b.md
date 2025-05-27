```
1. Title: Robust Conformal Uncertainty Quantification for Black-Box Models Under Distributional Shift  
 
2. Motivation: Modern black-box foundation models often fail to provide reliable uncertainty estimates when deployed in dynamic environments with data distribution shifts. Traditional conformal prediction assumes exchangeability, making it brittle to real-world scenarios where input distributions evolve due to drift, adversarial perturbations, or domain mismatches. This gap creates operational risks in safety-critical applications like healthcare or autonomous systems, where confidence intervals must hold despite unseen data shifts. Developing methods to guarantee uncertainty validity under such shifts could transform how we audit and deploy foundation models responsibly.  

3. Main Idea: We propose an adaptive conformal prediction framework that explicitly models and mitigates distributional shifts. The method will:  
   - Use online drift detection algorithms (e.g., via maximum mean discrepancy) to monitor input streams and quantify the degree/currency of shifts.  
   - Dynamically adjust nonconformity scores using importance weighting or domain-adaptive normalization, trained on continuously updated calibration sets.  
   - Theoretically derive distributionally robust coverage guarantees under drift by leveraging time-decaying influence functions.  
   - Validate on vision and language models under synthetic/natural shifts (e.g., Brightness-ImageNet, dialectal shifts in NLP), benchmarking against standard conformal baselines.  
   Expected outcomes include rigorously certified uncertainty intervals that adapt in real-time to environmental changes, enabling safer deployment and auditability. This bridges a critical gap in the statistical accountability of foundation models.  
```