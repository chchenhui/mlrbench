Title: Gradient-Informed Fingerprinting for Scalable Foundation Model Attribution

Motivation:  
Accurately tracing a model’s output back to specific training examples is vital for legal compliance, transparency, and debugging. Existing attribution methods struggle with the scale of modern foundation models and their massive datasets, leading to high compute costs and slow response times.

Main Idea:  
We propose a two-stage attribution pipeline. First, during training, each data sample is assigned a lightweight fingerprint by combining its static embedding (e.g., via a pretrained encoder) with a gradient-based signature extracted from a small probe network. These fingerprints are indexed in an approximate nearest-neighbor (ANN) database. At inference, an output’s fingerprint is computed on the fly using the same probe, then matched against the ANN index to retrieve top-k candidate sources in sub-second time. Second, we refine these candidates by estimating influence scores through a fast approximation of influence functions, yielding a ranked list of likely origin samples. We will evaluate attribution precision, recall, and latency on large language and multimodal models.  
Expected outcomes include >80% precision at real-time speeds. This method empowers IP protection, audit trails, and accountability for foundation models at production scale.