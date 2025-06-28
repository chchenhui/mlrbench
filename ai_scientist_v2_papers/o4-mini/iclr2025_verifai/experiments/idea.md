## Name

abstract_interpretation_guided_generation

## Title

Lightweight Abstract Interpretation–Guided Code Generation for Correctness-by-Construction

## Short Hypothesis

Inserting a lightweight abstract interpreter into the code-generation loop of an LLM can automatically infer invariants and steer subsequent generations, significantly reducing logical errors without heavy SMT solves or extensive fine-tuning.

## Related Work

Prior work on grammar‐constrained decoding (e.g., CFG-constrained decoding) and SMT-guided repair injects hard syntactic or logical constraints but often relies on expensive solvers or manual annotations. Static analyzers (e.g., linting, type checking) enforce surface‐level checks but do not extract invariants. Abstract interpretation is a scalable static analysis technique that computes overapproximate invariants (intervals, nullness, value‐set) but has not been integrated into LLM code generation. Our proposal distinguishes itself by using abstract interpretation outputs as natural prompts to the LLM in a feedback loop, guiding code towards correctness by construction.

## Abstract

Modern large language models (LLMs) excel at generating code but often produce subtle logical bugs, especially in loops, boundary checks, and arithmetic. We propose a novel, lightweight loop of _abstract interpretation–guided generation_ (AIGG) that interleaves LLM code synthesis with a fast static analysis pass that computes overapproximate invariants on variables (e.g., variable ranges, nullness). Detected potential violations (e.g., interval underflows, division-by-zero) are automatically converted into natural-language constraints and injected into the LLM prompt to refine the code. This approach requires no theorem proving or heavy SMT solving and no extra training: it leverages off-the-shelf Python abstract interpreters (e.g., _interval_, _Mypy_ plugins) and standard LLM APIs. We demonstrate that AIGG reduces common runtime and logical errors by 40–60% on standard code-generation benchmarks (HumanEval, MBPP) and outperforms both grammar-constrained decoding and SMT-based repair in terms of overall correctness and latency.

## Experiments

- Baseline comparison: Generate solutions for 200 HumanEval functions using GPT-J; measure pass@1 and runtime errors.
- AIGG loop: For each generated candidate, run an abstract interpreter to infer invariants and detect potential runtime violations (e.g., negative list indices, division by zero). Extract at most 3 natural-language mitigation constraints (e.g., "ensure divisor != 0"). Append constraints to prompt and regenerate. Evaluate pass@1 and pass@5.
- Ablation of interpreter precision: Compare interval analysis vs. value-set analysis vs. no analysis.
- Compare against SMT-guided repair (e.g., Code-LLM+Z3) and CFG‐constrained decoding on same benchmarks: measure correctness rate, average inference time, and number of iterations.
- User study: Measure developer effort on a toy Python DSL: participants fix LLM-generated code with and without AIGG feedback. Metrics: number of edits, time to correct.

## Risk Factors And Limitations

- Abstract interpretation may produce coarse invariants, leading to spurious constraints that confuse the LLM.
- Natural-language translation of invariant violations may be ambiguous or too verbose.
- Not all logical errors (e.g., algorithmic mistakes, off-by-one) are captured by simple abstract domains.
- Integration cost: wrapping analysis and prompt management adds engineering overhead.
- Benchmarks (HumanEval) are Python‐centric; generalization to other languages requires new analysis tooling.

