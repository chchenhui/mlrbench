import os
import os.path as osp
import numpy as np
import json
import re
from mlrbench.utils.utils import *
from mlrbench.llm.llm import *


RESEARCH_EXPERIMENT_RUBRIC = """
You are an expert machine learning researcher and your task is to evaluate a machine learning experimental document.
You will be given a document containing experimental excution records and experimental results, which is based on a task description, a research idea, a literature review and a research proposal.
You will first determine if the document contains any hallucinated content, then evaluate the document on a scale of 1 to 10 across six key dimensions: Consistency, Completeness, Novelty, Soundness, Insightfulness, Significance and finally give an overall assessment on a scale of 1 to 10.
Please be objective in your evaluation, and provide detailed justifications for each score you assign.
Do not hesitate to assign lower scores if the experimental document does not fully meet the criteria. Avoid giving high scores by default.

## Evaluation Rubric

1. Hallucination (True/False)
    Does the experimental document contain any hallucinated content? Hallucinated content refers to information that is fabricated or incorrect, and does not align with the provided task description, research idea, literature review, or research proposal. Fake data, results, or methods should be considered as hallucinated content.

    True - The experimental document contains hallucinated content.
    False - The experimental document does not contain any hallucinated content. 

2. Consistency (1-10)
    How well do the experimental document align with the task description, research idea, literature review and research proposal? Does the implementation match the proposed methods and ideas?
    
    9-10 - Excellent: The experimental document are fully consistent with the task description, research idea, literature review and research proposal. There are no discrepancies or contradictions. The implementation is a perfect match to the proposed methods and ideas.
    7-8  - Good: The experimental document are mostly consistent, with only minor discrepancies or contradictions. The main points are well-aligned with the task, idea, literature review and proposal. The implementation is largely aligned with the proposed methods and ideas.
    5-6  - Moderate: Some inconsistencies or unclear alignments exist, but overall the experimental document are still relevant to the task, idea, literature review and proposal. The implementation is somewhat aligned with the proposed methods and ideas.
    3-4  - Weak: Significant inconsistencies or contradictions are present, leading to confusion or misalignment with the task, idea, literature review and proposal. The implementation is poorly aligned with the proposed methods and ideas.
    1-2  - Poor: The experimental document are largely inconsistent with the task, idea, literature review and proposal, or there are major contradictions. The implementation is not aligned with the proposed methods and ideas at all.

3. Completeness (1-10)
    Are all necessary experiments, baselines, and ablation studies included in this experimental document? Are all relevant results reported, and is the experimental setup fully described?
    
    9-10 - Excellent: All necessary experiments, baselines, and ablations are included. The results are comprehensive and the setup is fully described.
    7-8  - Good: Most necessary experiments, baselines, and ablations are included, with only minor omissions. The setup is mostly described.
    5-6  - Moderate: Some important experiments, baselines, and ablations are missing, but the main points are covered. The setup is partially described.
    3-4  - Weak: Many key experiments are missing, making the evaluation incomplete. The setup is poorly described and lacks clarity.
    1-2  - Poor: The results are highly incomplete, with major omissions. The setup is missing.

4. Novelty (1-10)
    Does the experiment document demonstrate new findings, methods, or insights compared to existing work? Is the experimental design innovative or derivative?
    
    9-10 - Excellent: The experimental document presents highly novel findings, methods, or insights that significantly advance the field. The design is innovative and original.
    7-8  - Good: The experimental document shows some novel aspects, with a good level of innovation. The design is mostly original.
    5-6  - Moderate: The experimental document has some novel elements, but they are not particularly groundbreaking. The design is somewhat derivative.
    3-4  - Weak: The experimental document lacks novelty, with little to no new findings or insights. The design is largely derivative.
    1-2  - Poor: The experimental document is entirely derivative, with no new findings or insights. The design is unoriginal and lacks creativity.

5. Soundness (1-10)
    Are the experimental methods, analysis, and conclusions logically sound and scientifically rigorous? Are the results reproducible and well-supported?
    
    9-10 - Excellent: Methods and analysis are highly rigorous, logically sound, and statistically valid. Results are fully reproducible and well-supported.
    7-8  - Good: Methods and analysis are generally sound, with only minor issues. Results are mostly reproducible and well-supported.
    5-6  - Moderate: Some weaknesses in rigor or logic, but the main conclusions are still supported. Results are partially reproducible.
    3-4  - Weak: Significant flaws in methods or analysis, casting doubt on the conclusions. Results are not well-supported or reproducible.
    1-2  - Poor: Methods or analysis are fundamentally unsound or unscientific. Results are not reproducible and conclusions are unsupported.

6. Insightfulness (1-10)
    Do the results provide deep insights, meaningful interpretations, or valuable implications for the field? Are trends, patterns, and implications discussed thoughtfully?
    
    9-10 - Excellent: Results are analyzed in depth, with highly insightful interpretations and valuable implications for the field.
    7-8  - Good: Results are thoughtfully analyzed, with some meaningful insights.
    5-6  - Moderate: Some insights are provided, but analysis is relatively superficial.
    3-4  - Weak: Little meaningful analysis or interpretation is provided.
    1-2  - Poor: No insight or thoughtful analysis is present.

7. Significance (1-10)
    How important or impactful are the experiment results for the field? Do they address a critical problem or open new research directions?
    
    9-10 - Excellent: Results are highly significant, addressing critical problems or opening important new directions.
    7-8  - Good: Results are significant and make a clear contribution.
    5-6  - Moderate: Results are somewhat significant, but impact is limited.
    3-4  - Weak: Results have little significance or impact.
    1-2  - Poor: Results are insignificant or irrelevant.

8. Overall Assessment (1-10)
    Provide an overall assessment of the experimental work, considering all the above dimensions.
    
    10 - Outstanding: The experimental work is exemplary in every respect, demonstrating exceptional quality, rigor, and impact. All aspects are handled with great care and expertise, with no significant weaknesses. The work sets a high standard and is likely to have a major influence in its field.
    8-9 - Excellent: The work is very strong overall, with clear strengths across most or all dimensions. Any weaknesses are minor and do not detract from the overall quality or credibility. The experimental work is well-executed, insightful, and makes a significant contribution.
    6-7 - Good: The experimental work is solid and generally well-conceived, with a good balance of strengths and weaknesses. While there may be some areas for improvement, the work is credible, meaningful, and meets the main requirements for quality research.
    4-5 - Satisfactory: The work is adequate but has several notable weaknesses that limit its overall quality or impact. While it addresses the main objectives, shortcomings in design, execution, or analysis reduce its effectiveness and significance.
    2-3 - Needs Improvement: The experimental work has substantial weaknesses in multiple areas, which undermine its credibility or value. Major revisions and improvements are needed for the work to reach an acceptable standard.
    1 - Poor: The work is fundamentally flawed across most or all dimensions. It fails to meet essential standards for research quality and is unlikely to provide meaningful insights or contributions.

When assigning the Overall Assessment score, consider not just the average of these dimensions, but also:
- The overall coherence and integration of the experimental work.
- The presence of any particularly outstanding strengths or critical weaknesses that may not be fully reflected in the individual scores.
- The potential impact or importance of the work as a whole.
- The degree to which the experimental work advances the field or opens new research directions.
- Any unique contributions, innovative aspects, or serious flaws that significantly affect the overall quality.
Your overall assessment should reflect a holistic judgment, taking into account both the quantitative scores and your qualitative evaluation of the experimental work.

## Output Format

Please evaluate the experimental document according to the rubric and output a complete JSON object strictly following the format below, including all evaluation items (Hallucination, Consistency, Completeness, Novelty, Soundness, Insightfulness, Significance, OverallAssessment). Do not output only a single item or partial content; you must output the entire JSON object.

```json
{   
    "Hallucination": {
        "has_hallucination": <true/false>,
        "details": "<if has_hallucination is true, provide specific examples of hallucinated content; if false, explain why you believe there is no hallucination>"
    },
    "Consistency": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the alignment with the task description, idea, and literature review>"
    },
    "Completeness": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the inclusion of necessary experiments, baselines, and ablation studies>"
    },
    "Novelty": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the originality and innovation of the findings, methods and experimental design>"
    },
    "Soundness": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the logical soundness and scientific rigor of the experimental design, analysis, and conclusions>"
    },
    "Insightfulness": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the depth of insights, meaningful interpretations, and valuable implications for the field>"
    },
    "Significance": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the importance and impact of the experimental results for the field>"
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


def review_experiment(
    file_path,
    client,
    review_form=RESEARCH_EXPERIMENT_RUBRIC,
    max_retries=3,
):
    """
    Review the experimental document based on the task description, research idea, literature review and research proposal.
    Key dimensions to review: Consistency, Completeness, Novelty, Soundness, Insightfulness, Significance.
    """
    exp_log = load_json_without_image(file_path)
    base_prompt = review_form

    base_prompt += f"""
## Experimental Document
    
```json
{exp_log}
```"""
    base_prompt += """    
## Output Format

Based ONLY on the document above, provide your assessment according to the rubric criteria.
Your response must be EXCLUSIVELY a JSON object with the following structure.
Do not add any explanation, introduction, or conclusion outside the JSON.

IMPORTANT: Your response must be a valid JSON object that can be parsed by Python's json.loads().
- Do not include any text outside the JSON object
- Do not use comments
- Do not use trailing commas
- Use double quotes for all strings
- Ensure all brackets and braces are properly closed
- Do not use unescaped control characters

JSON TEMPLATE TO COMPLETE:

```json
{   
    "Hallucination": {
        "has_hallucination": <true/false>,
        "details": "<if has_hallucination is true, provide specific examples of hallucinated content; if false, explain why you believe there is no hallucination>"
    },
    "Consistency": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the alignment with the task description, idea, and literature review>"
    },
    "Completeness": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the inclusion of necessary experiments, baselines, and ablation studies>"
    },
    "Novelty": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the originality and innovation of the findings, methods and experimental design>"
    },
    "Soundness": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the logical soundness and scientific rigor of the experimental design, analysis, and conclusions>"
    },
    "Insightfulness": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the depth of insights, meaningful interpretations, and valuable implications for the field>"
    },
    "Significance": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the importance and impact of the experimental results for the field>"
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
            response, token_usage = client.generate(prompt=base_prompt)
            review = extract_json_between_markers(response)
            
            # Validate JSON structure
            if not isinstance(review, dict):
                raise ValueError("Response is not a dictionary")
            
            required_keys = ["Hallucination", "Consistency", "Completeness", "Novelty", "Soundness", 
                           "Insightfulness", "Significance", "OverallAssessment"]
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
    # evaluator_name = "google/gemini-2.5-pro-preview"
    evaluator_name = "claude-3-7-sonnet-20250219"
    evaluator_folder = format_model_name(evaluator_name)
    llm_engine = create_client(model_name=evaluator_name, max_tokens=8192*2, judge_mode=True) #gemini max tokens: 65535
    print(f"Reviewing experiments with {evaluator_name}...")
    tasklist = get_tasklist("claude_experiments")
    for task_name in tasklist:
        task = osp.join("claude_experiments", task_name)
        exp_log = osp.join(task, "claude_output.json")
        save_path = osp.join(f"samples_{evaluator_folder}", task_name, "experiments")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        output_file = osp.join(save_path, f"claude.json")
        if os.path.exists(output_file):
            print(f"""{output_file} already exists!""")
            continue
        result = review_experiment(file_path=exp_log,
                                    client=llm_engine,
                                    )
        if result is None:
            print(f"Error: reviews of {task} is not a valid JSON file.")
            continue
        save_json(result, output_file)