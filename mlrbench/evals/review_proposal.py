import os
import os.path as osp
import numpy as np
import json
import re
from mlrbench.utils.utils import *
from mlrbench.llm.llm import *


RESEARCH_PROPOSAL_RUBRIC = """
You are an expert machine learning researcher!
You will be given a research proposal based on a task description, a research idea, and a literature review.
Your task is to evaluate the proposal on a scale of 1 to 10 across six key dimensions and finally give an overall assessment on a scale of 1 to 10.
Please be objective in your evaluation, and provide detailed justifications for each score you assign.
Do not hesitate to assign lower scores if the proposal does not fully meet the criteria. Avoid giving high scores by default.

## Evaluation Rubric

1. CONSISTENCY (1-10)
    How well does the proposal align with the requirements of the task description, the research idea, and the literature review?

    9-10 - Excellent: The proposal is perfectly aligned with the task description, research idea, and literature review. It addresses all aspects comprehensively and demonstrates a deep understanding of the context and prior work. There are no inconsistencies or gaps.
    7-8 - Good: The proposal is mostly aligned with the task description, research idea, and literature review. It addresses most aspects, with only minor omissions or inconsistencies that do not significantly affect the overall coherence.
    5-6 - Satisfactory: The proposal is somewhat aligned. It addresses some key aspects of the task description, research idea, and literature review, but misses several important points or contains moderate inconsistencies.
    3-4 - Needs Improvement: The proposal is poorly aligned. It addresses only a few aspects of the task description, research idea, or literature review, and contains significant inconsistencies or gaps that undermine its coherence.
    1-2 - Poor: The proposal is not aligned with the task description, research idea, or literature review. It fails to address the requirements or is completely irrelevant, with major inconsistencies or contradictions.

2. CLARITY (1-10)
    How clear and well-defined is the research proposal?

    9-10 - Excellent: The proposal is crystal clear, perfectly defined, and immediately understandable. All objectives, methods, and rationales are articulated concisely with no ambiguity. The structure is logical and easy to follow.
    7-8 - Good: The proposal is mostly clear and well-articulated, with only minor ambiguities or areas that could benefit from slight refinement. The main points are understandable and the structure is generally logical.
    5-6 - Satisfactory: The proposal is partially clear but has some ambiguities or unclear sections. Some aspects require further elaboration or clarification for a complete understanding. The structure may be somewhat disjointed.
    3-4 - Needs Improvement: The proposal is unclear with significant ambiguities or confusing sections. Major clarification is needed to make the objectives, methods, or rationale comprehensible. The structure is difficult to follow.
    1-2 - Poor: The proposal is extremely unclear, vague, or ambiguous with no proper explanation. It is difficult to understand or interpret, and the structure is disorganized or incoherent.

3. NOVELTY (1-10)
    How original and innovative is the research proposal?

    9-10 - Excellent: The proposal is highly original and innovative. It introduces a groundbreaking concept, method, or perspective that is significantly different from existing work in the literature. The novelty is clearly articulated and well-justified.
    7-8 - Good: The proposal demonstrates notable originality and innovation. It offers fresh perspectives or new combinations of existing concepts, with clear distinctions from prior work, though it may not be entirely groundbreaking.
    5-6 - Satisfactory: The proposal has some originality and innovation. It includes novel aspects but also shares similarities with existing approaches. The differences from prior work are present but not strongly emphasized.
    3-4 - Needs Improvement: The proposal has minimal originality. It largely resembles existing approaches in the literature, with only slight variations or incremental improvements.
    1-2 - Poor: The proposal lacks originality and innovation. It closely follows common or existing concepts without any new insights, and does not distinguish itself from prior work.

4. SOUNDNESS (1-10)
    How well-founded and rigorous is the research proposal?
    9-10 - Excellent: The proposal is highly sound and rigorous. It is based on solid theoretical foundations, well-established methods, and comprehensive literature review. The proposed methodology is robust and well-justified. Technical formulations are fully correct and clearly presented.
    7-8 - Good: The proposal is sound and mostly rigorous. It is based on solid foundations and established methods, though it may have minor gaps or areas that require further justification. The methodology is generally well-defined. Technical formulations are mostly correct, with minor errors or omissions.
    5-6 - Satisfactory: The proposal is somewhat sound but has some gaps or weaknesses in its theoretical foundations or methodology. It may rely on assumptions that are not fully justified. The proposed methods are acceptable but may lack rigor. Technical formulations have some errors or unclear aspects.
    3-4 - Needs Improvement: The proposal has significant weaknesses in its soundness or rigor. It may rely on questionable assumptions, poorly defined methods, or lack sufficient justification for its approach. Technical formulations are often incorrect or poorly presented.
    1-2 - Poor: The proposal is fundamentally unsound or lacks rigor. It is based on flawed assumptions, poorly defined methods, or lacks sufficient justification for its approach. Technical formulations are incorrect or absent.

5. FEASIBILITY (1-10)
    How practical and implementable is the research proposal?

    9-10 - Excellent: The proposal is highly practical and implementable with current resources, technology, and knowledge. The plan is realistic, and execution is straightforward with clearly defined steps and minimal risk.
    7-8 - Good: The proposal is largely feasible with existing technology and methods, though it may require moderate refinement, optimization, or additional resources. The plan is generally realistic, with manageable risks.
    5-6 - Satisfactory: The proposal is somewhat feasible but presents some implementation challenges. It may require considerable effort, resources, or further development to implement successfully. Some risks or uncertainties are present.
    3-4 - Needs Improvement: The proposal has significant implementation challenges. Major revisions, additional resources, or new methods would be needed to make it feasible. There are substantial risks or uncertainties that threaten successful execution.
    1-2 - Poor: The proposal is impractical or impossible to implement with current technology, knowledge, or constraints. The plan is unrealistic, and the likelihood of successful execution is extremely low.

6. SIGNIFICANCE (1-10)
    How important and impactful is the research proposal?

    9-10 - Excellent: The proposal is highly significant and impactful. It addresses a critical problem or gap in the field and has the potential to lead to major advancements or transformative change. The expected contributions are substantial and clearly articulated.
    7-8 - Good: The proposal is significant and has clear impact potential. It addresses an important issue and could lead to meaningful contributions or improvements in the field, though the impact may not be transformative.
    5-6 - Satisfactory: The proposal is somewhat significant. It addresses a relevant problem, but its impact may be moderate or limited to a specific area or community. The expected contributions are present but not far-reaching.
    3-4 - Needs Improvement: The proposal has limited significance. It addresses a minor issue or has a narrow scope, with minimal potential for impact or advancement in the field.
    1-2 - Poor: The proposal has little to no significance. It does not address a meaningful problem or offer any clear benefits, and its potential impact is negligible or absent.

7. Overall Assessment (1-10)
    How would you rate the research proposal overall, considering all five dimensions above?
    
    10 - Outstanding: The proposal is exceptional in every respect, with no significant weaknesses. It demonstrates excellence across all dimensions and has the potential for major impact.
    8-9 - Excellent: The proposal is very strong overall, with only minor weaknesses. It is well-balanced, highly promising, and likely to make a significant contribution.
    6-7 - Good: The proposal is solid, with a good balance of strengths and weaknesses. It is generally well-conceived and feasible, though some areas could be improved.
    4-5 - Satisfactory: The proposal is adequate but has notable weaknesses that limit its potential. It addresses the main requirements but lacks strength in one or more key areas.
    2-3 - Needs Improvement: The proposal has significant weaknesses that limit its potential for success or impact. Major revisions are needed to address critical issues.
    1 - Poor: The proposal is fundamentally flawed across most or all dimensions. It is unlikely to succeed or make a meaningful contribution without substantial reworking.

When assigning the Overall Assessment score, consider not just the average of the six dimensions, but also:
- Whether any single weakness is critical enough to lower the overall potential.
- The overall coherence and integration of the proposal.
- The likelihood of real-world impact if the proposal were pursued.
- The degree to which the proposal fulfills the task description, research idea, and literature review as a whole.
- Any unique strengths or fatal flaws that are not fully captured by the individual dimensions.
   
## Output Format

Please output a complete JSON object strictly following the format below, including all evaluation items (Consistency, Clarity, Novelty, Soundness, Feasibility, Significance, OverallAssessment). Do not output only a single item or partial content; you must output the entire JSON object. When writing mathematical formulas, you should avoid invalid escape JSON decode errors.

```json
{
    "Consistency": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the alignment with the task description, idea, and literature review>",
    },
    "Clarity": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the clarity of the proposal>",
    },
    "Novelty": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the originality and innovation of the proposal>",
    },
    "Soundness": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the technical foundations and rigor of the proposal>",
    },
    "Feasibility": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the practicality and implementability of the proposal>",
    },
    "Significance": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the importance and impact of the proposal>",
    },
    "OverallAssessment": {
        "score": <1-10>,
        "strengths": ["<strength 1>", "<strength 2>"],
        "weaknesses": ["<weakness 1>", "<weakness 2>"]
    }
}
```

Please make sure the answer is strictly in valid JSON format. 
- Do not include any text, comments, or explanations outside the JSON code block.
- Do not use trailing commas.
- Do not use single quotes; use double quotes for all keys and string values.
- Do not include comments inside the JSON.
- Do not use unescaped control characters (e.g., newlines inside strings).
- Do not use unquoted keys; all keys must be in double quotes.
- Do not use invalid values like NaN or Infinity.
- Ensure all brackets and braces are properly closed.
"""


def review_proposal(
    file_path,
    client,
    task_file,
    idea_file,
    lit_file,
    review_form=RESEARCH_PROPOSAL_RUBRIC,
):
    """
    Review the proposal: Consistency, Clarity, Novelty, Feasibility, Significance.

    Args:
        file_path (`str`): 
            The path to the proposal file.
        client ([`LLM`]):
            LLM engine of the evaluator.
        task_file (`str`): 
            The task file in the review.
        idea_file (`str`):
            The idea file in the review.
        lit_file (`str`):
            The literature review file in the review.
        review_form (`str`):
            The review form for the proposal.
    """
    proposal = read_text(file_path)
    task = read_text(task_file)
    idea = read_text(idea_file)
    related_work = read_text(lit_file)
    base_prompt = review_form
    base_prompt += f"""
    Here is the proposal you are asked to review:
    ```
    {proposal}
    ```
    Here is the task:
    ```
    {task}
    ```
    Here is the idea:
    ```
    {idea}
    ```
    Here is the literature review:
    ```
    {related_work}
    ```"""
    base_prompt += """
## Output Format

Please output a complete JSON object strictly following the format below, including all evaluation items (Consistency, Clarity, Novelty, Soundness, Feasibility, Significance, OverallAssessment). Do not output only a single item or partial content; you must output the entire JSON object. When writing mathematical formulas, you should avoid invalid escape JSON decode errors.

```json
{
    "Consistency": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the alignment with the task description, idea, and literature review>",
    },
    "Clarity": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the clarity of the proposal>",
    },
    "Novelty": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the originality and innovation of the proposal>",
    },
    "Soundness": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the technical foundations and rigor of the proposal>",
    },
    "Feasibility": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the practicality and implementability of the proposal>",
    },
    "Significance": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the importance and impact of the proposal>",
    },
    "OverallAssessment": {
        "score": <1-10>,
        "strengths": ["<strength 1>", "<strength 2>"],
        "weaknesses": ["<weakness 1>", "<weakness 2>"]
    }
}
```"""
    response, token_usage = client.generate(base_prompt)
    review = extract_json_between_markers(response)
    return review


if __name__ == "__main__":
    evaluator_name = "google/gemini-2.5-pro-preview-03-25"
    evaluator_folder = format_model_name(evaluator_name)
    llm_engine = create_client(model_name=evaluator_name, max_tokens=8192, judge_mode=True)
    agent_folder = format_model_name("mistral/ministral-8b")
    print(f"Reviewing proposal generated by {agent_folder} using {evaluator_name}...")
    tasklist = get_tasklist("tasks")
    # tasklist = tasklist[:1]
    for task in tasklist:
        task_file = osp.join("tasks", task+".md") 
        proposal_file = osp.join("results", task, "proposal", f"proposal_{agent_folder}.md")   
        if not os.path.exists(proposal_file):
            print(f"No such file {proposal_file}, please check again!") 
            continue
        idea_file = osp.join("results", task, "idea.md")
        lit_file = osp.join("results", task, "related_work.md")
        save_path = osp.join(f"reviews_{evaluator_folder}", task, "proposal")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        output_file = osp.join(save_path, f"proposal_{agent_folder}.json")
        if os.path.exists(output_file):
            print(f"""{output_file} already exists!""")
            continue
        result = review_proposal(file_path=proposal_file,
                                 client=llm_engine, 
                                 task_file=task_file,
                                 idea_file=idea_file,
                                 lit_file=lit_file,
                                 )
        if result is None:
            print(f"Error: reviews of {task_file} is not a valid JSON file.")
            continue
        if len(result) != 7:
            print(f"Error: reviews of {task_file} do not complete. Keys: {result.keys()}")
            continue
        save_json(result, output_file)