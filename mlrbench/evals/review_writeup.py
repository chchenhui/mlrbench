import os
import os.path as osp
import numpy as np
import json
import re
from mlrbench.utils.utils import *
from mlrbench.lmm.lmm import *


RESEARCH_PAPER_RUBRIC = """
You are an expert machine learning researcher and your task is to evaluate a paper.
You will be given a machine learning paper, which is based on a task description, a research idea, a literature review, a research proposal and experimental results.
You will evaluate the paper on a scale of 1 to 10 across four key dimensions: Consistency, Clarity, Completeness, Soundness and finally give an overall assessment on a scale of 1 to 10.
Please be objective in your evaluation, and provide detailed justifications for each score you assign.
Do not hesitate to assign lower scores if the paper does not fully meet the criteria. Avoid giving high scores by default.

## Evaluation Rubric

1. Consistency (1-10)
    - How well does the paper align with the task description, research idea, research proposal and experimental results?
    - Are there any contradictions or inconsistencies in the paper's arguments or findings?

    9-10 - Excellent: The paper is highly consistent, with no contradictions or inconsistencies. 
    7-8 - Good: The paper is mostly consistent, with minor contradictions or inconsistencies.
    5-6 - Fair: The paper has some inconsistencies, but they do not significantly detract from the overall quality.
    3-4 - Poor: The paper has several inconsistencies that detract from the overall quality.
    1-2 - Very Poor: The paper is highly inconsistent, with major contradictions or inconsistencies that significantly detract from the overall quality.

2. Clarity (1-10)
    - How clear and understandable is the paper's writing?
    - Are the arguments and findings presented in a logical and coherent manner?
    - Is the paper well-structured and easy to follow?

    9-10 - Excellent: The paper is very clear and easy to understand, with a logical structure and coherent arguments.
    7-8 - Good: The paper is mostly clear, with minor issues in structure or coherence.
    5-6 - Fair: The paper has some clarity issues, but they do not significantly detract from the overall quality.
    3-4 - Poor: The paper has several clarity issues that detract from the overall quality.
    1-2 - Very Poor: The paper is very unclear and difficult to understand, with major issues in structure or coherence.

3. Completeness (1-10)
    - How complete is the paper in terms of addressing the task description, research idea, research proposal and experimental results?
    - Are there any missing components or sections that should be included?

    9-10 - Excellent: The paper is very complete, addressing all components of the task description, research idea, research proposal and experimental results.
    7-8 - Good: The paper is mostly complete, with minor missing components or sections.
    5-6 - Fair: The paper has some missing components or sections, but they do not significantly detract from the overall quality.
    3-4 - Poor: The paper has several missing components or sections that detract from the overall quality.
    1-2 - Very Poor: The paper is very incomplete, with major missing components or sections that significantly detract from the overall quality.

4. Soundness (1-10)
    - Are the arguments and findings supported by evidence and reasoning?
    - Are there any flaws or weaknesses in the paper's methodology or approach?
    - Are the experimental results and analyses valid and reliable?

    9-10 - Excellent: The paper is very sound, with strong evidence and reasoning supporting the arguments and findings. The experimental results and analyses are fully valid and reliable.
    7-8 - Good: The paper is mostly sound, with minor flaws or weaknesses in methodology or approach. The experimental results and analyses are mostly valid and reliable.
    5-6 - Fair: The paper has some flaws or weaknesses in methodology or approach, but they do not significantly detract from the overall quality. The experimental results and analyses are somewhat valid and reliable.
    3-4 - Poor: The paper has several flaws or weaknesses in methodology or approach that detract from the overall quality. Many of the experimental results and analyses are not valid or reliable.
    1-2 - Very Poor: The paper is very unsound, with major flaws or weaknesses in methodology or approach that significantly detract from the overall quality. The experimental results and analyses are not valid or reliable.

5. Overall Assessment (1-10)
    - Based on your evaluation of the paper's consistency, clarity, completeness and soundness, what is your overall assessment of the paper?
    
    10 - Outstanding: The paper is of outstanding quality, with no issues in any of the key dimensions.
    8-9 - Excellent: The paper is of excellent quality, with minor issues in one or more of the key dimensions.
    6-7 - Good: The paper is of good quality, with some issues in one or more of the key dimensions.
    4-5 - Fair: The paper is of fair quality, with several issues in one or more of the key dimensions.
    2-3 - Poor: The paper is of poor quality, with major issues in one or more of the key dimensions.
    1 - Very Poor: The paper is of very poor quality, with major issues in all of the key dimensions.

When assigning the Overall Assessment score, consider not just the average of these dimensions, but also:
- The overall coherence and integration of the paper.
- The presence of any particularly outstanding strengths or critical weaknesses that may not be fully reflected in the individual scores.
- The potential impact or importance of the work as a whole.
- The degree to which the paper advances the field or opens new research directions.
- Any unique contributions, innovative aspects, or serious flaws that significantly affect the overall quality.
Your overall assessment should reflect a holistic judgment, taking into account both the quantitative scores and your qualitative evaluation of the paper.

## Output Format

Please evaluate the paper according to the rubric and output a complete JSON object strictly following the format below, including all evaluation items (Consistency, Clarity, Completeness, Soundness, OverallAssessment). Do not output only a single item or partial content; you must output the entire JSON object.

```json
{   
    "Consistency": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the consistency of the paper itself and between the paper and the task description, research idea, research proposal and experimental results>"
    },
    "Clarity": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the clarity of the paper writing, structure, and coherence>"
    },
    "Completeness": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the completeness of the paper in addressing the task description, research idea, research proposal and experimental results>"
    },
    "Soundness": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the soundness of the paper's arguments, findings, methodology, and experimental results>"
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


def review_writeup(
    file_path: str,
    client: LMM,
    task_file: str,
    idea_file: str,
    lit_file: str,
    proposal_file: str,
    exp_file: str,
    review_form: str = RESEARCH_PAPER_RUBRIC,
    max_retries: int =3,
):
    """
    Generate a review paper based on the provided files.
    
    Args:
        file_path (str): Path to the review paper
        client (LMM): The LMM client to use for generation
        task_file (str): Path to the task file
        idea_file (str): Path to the idea file
        lit_file (str): Path to the literature review file
        proposal_file (str): Path to the proposal file
        exp_file (str): Path to the experiment results file
        review_form (str): The review form to use for evaluation
    """
    paper, img_list = load_multimodal_content(file_path)
    task = read_text(task_file)
    idea = read_text(idea_file)
    related_work = read_text(lit_file)
    proposal = read_text(proposal_file)
    experiment_results = read_text(exp_file) #markdown
    base_prompt = review_form

    base_prompt += f"""
## Paper to Be Reviewed
    
```json
{paper}
```
## Task Description

```json
{task}
```
## Research Idea

```json
{idea}
```
## Literature Review

```json
{related_work}
```
## Research Proposal

```json
{proposal}
```
## Experimental Results

```json
{experiment_results}
```"""
    base_prompt += """
## Output Format

Please evaluate the paper according to the rubric and output a complete JSON object strictly following the format below, including all evaluation items (Consistency, Clarity, Completeness, Soundness, OverallAssessment). Do not output only a single item or partial content; you must output the entire JSON object.

JSON TEMPLATE TO COMPLETE:

```json
{   
    "Consistency": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the consistency of the paper itself and between the paper and the task description, research idea, research proposal and experimental results>"
    },
    "Clarity": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the clarity of the paper writing, structure, and coherence>"
    },
    "Completeness": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the completeness of the paper in addressing the task description, research idea, research proposal and experimental results>"
    },
    "Soundness": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the soundness of the paper's arguments, findings, methodology, and experimental results>"
    },
    "OverallAssessment": {
        "score": <1-10>,
        "strengths": ["<strength 1>", "<strength 2>"],
        "weaknesses": ["<weakness 1>", "<weakness 2>"]
    }
}
```

IMPORTANT: Your ENTIRE response must be ONLY this JSON object with no additional text.
"""
    for attempt in range(max_retries):
        try:
            response, token_usage = client.generate(prompt=base_prompt, media=img_list)
            review = extract_json_between_markers(response)
            
            # Validate JSON structure
            if not isinstance(review, dict):
                raise ValueError("Response is not a dictionary")
            
            required_keys = ["Consistency", "Clarity", "Completeness", "Soundness", "OverallAssessment"]
            if not all(key in review for key in required_keys):
                raise ValueError(f"Missing required keys. Found: {list(review.keys())}")
            
            return review
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                print("All retry attempts failed")
                return None
            continue
    
    return None

if __name__ == "__main__":
    evaluator_name = "google/gemini-2.5-pro-preview"
    # evaluator_name = "claude-3-7-sonnet-20250219"
    evaluator_folder = format_model_name(evaluator_name)
    lmm_engine = create_lmm_client(model_name=evaluator_name, max_tokens=8192*2, judge_mode=True)
    agent_folder = format_model_name("o4-mini-2025-04-16")
    print(f"Reviewing paper generated by {agent_folder} using {evaluator_name}...")
    tasklist = get_tasklist("claude_experiments")
    for task_name in tasklist:
        task = osp.join("claude_experiments", task_name)
        task_file = osp.join(task, "task.md")
        idea_file = osp.join(task, "idea.md")
        lit_file = osp.join(task, "related_work.md")
        proposal_file = osp.join(task, "proposal.md")
        exp_file = osp.join(task, "results", "results.md")
        paper_file = osp.join(task, "results", f"paper_{agent_folder}.md")
        save_path = osp.join(f"samples_{evaluator_folder}", task_name, "paper")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        output_file = osp.join(save_path, f"{agent_folder}.json")
        if os.path.exists(output_file):
            print(f"""{output_file} already exists!""")
            continue
        result = review_writeup(
            file_path=paper_file,
            client=lmm_engine,
            task_file=task_file,
            idea_file=idea_file,
            lit_file=lit_file,
            proposal_file=proposal_file,
            exp_file=exp_file,
        )
        if result is None:
            print(f"Error: reviews of {task} is not a valid JSON file.")
            continue
        save_json(result, output_file)