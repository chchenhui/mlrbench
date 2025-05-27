import os
import os.path as osp
import numpy as np
import json
import re
from mlrbench.utils.utils import *
from mlrbench.lmm.lmm import *


OVERALL_RUBRIC = """
You are an expert machine learning researcher!
You will be given a research paper which is based on a task description.
You might also be given the code of the paper to check the reproducibility of the paper. 
You task is to review the paper in terms of 4 key aspects - Clarity, Novelty, Soundness and Significance.
Please provide a score from 1 to 10 for each aspect and an overall assessment, where 1 is the lowest and 10 is the highest. Lastly, provide a confidence score from 1 to 5 for the overall assessment, where 1 is the lowest and 10 is the highest.

## Evaluation Rubric

1. Clarity (1-10)
    - Is the paper well-written and easy to understand?
    - Are the ideas and contributions clearly articulated?
    - Is the structure of the paper logical and coherent?

    9-10 - The paper is exceptionally well-written, with clear and concise language. The ideas are presented in a logical and coherent manner, making it easy to follow the author's arguments.
    7-8 - The paper is well-written, but there are some areas that could be improved for clarity. The ideas are mostly clear, but there may be some minor issues with the structure or language.
    5-6 - The paper is somewhat difficult to read, with several areas that are unclear or poorly articulated. The structure may be confusing, making it hard to follow the author's arguments.
    3-4 - The paper is poorly written, with many unclear or confusing sections. The ideas are not well-articulated, and the structure is disorganized.
    1-2 - The paper is extremely difficult to read, with numerous unclear or confusing sections. The ideas are poorly articulated, and the structure is completely disorganized.

2. Novelty (1-10)
    - Does the paper present new and original ideas and findings?
    - Are the experimental results and contributions original and novel?
    - Is the work a significant advance over existing research?

    9-10 - The paper presents groundbreaking ideas and findings that are highly original and significant. The contributions are a major advance over existing research and are likely to have a lasting impact on the field.
    7-8 - The paper presents some new and original ideas, and the contributions are significant. The work is a notable advance over existing research, but it may not be as groundbreaking as top-tier papers.
    5-6 - The paper presents some new ideas and findings, but they are not particularly original or significant. The contributions are somewhat incremental and do not represent a major advance over existing research.
    3-4 - The paper presents few new ideas or findings, and those that are presented are not original or significant. The contributions are minimal and do not advance the field.
    1-2 - The paper presents no new ideas, and the contributions are completely unoriginal. The work does not advance the field in any meaningful way.

3. Soundness (1-10)
    - Are the methods and techniques used in the paper sound and appropriate?
    - Are the results and conclusions supported by the data?
    - Are there any major flaws or weaknesses in the experimental design, results or analysis?
    - Are the experimental results reliable and consistent to the code of the paper? Are the experimental results real or fake?
    - Are the visualization and analysis figures based on real experimental results or based on fake data? 

    9-10 - The methods and techniques used in the paper are sound and appropriate. The results are well-supported by the data, and there are no major flaws or weaknesses in the experimental design, results or analysis. The experimental results are fully reliable and consistent with the code of the paper.
    7-8 - The methods and techniques used in the paper are mostly sound, but there may be some minor issues. The results are generally well-supported by the data, but there may be some areas that could be improved. The experimental design, results or analysis may have some minor flaws. The experimental results are mostly reliable.
    5-6 - The methods and techniques used in the paper are somewhat questionable, with several areas that could be improved. The results are not well-supported by the data, and there may be some significant flaws in the experimental design, results or analysis. Some experimental results are not reliable.
    3-4 - The methods and techniques used in the paper are flawed or inappropriate. The results are not well-supported by the data, and there are major flaws in the experimental design, results or analysis. Most of experimental results are not reliable.
    1-2 - The methods and techniques used in the paper are completely unsound. The results are not supported by the data, and there are numerous major flaws in the experimental design, results or analysis. The conclusions drawn from the paper are completely invalid. All experimental results are not reliable.

4. Significance (1-10)
    - Does the paper address an important problem or question?
    - Are the contributions significant to the field?
    - Are the experimental results reproducible and reliable? Do they have a significant impact?
    - Will the work have a lasting impact on the field?

    9-10 - The paper addresses a highly important problem or question, and the results and contributions are significant to the field. The work is likely to have a lasting impact on the field.
    7-8 - The paper addresses an important problem or question, and the results and contributions are significant. The work may have a lasting impact on the field, but it may not be as groundbreaking as top-tier papers.
    5-6 - The paper addresses a somewhat important problem or question, but the results and contributions are not particularly significant. The work may have some impact on the field, but it is unlikely to be lasting.
    3-4 - The paper addresses a minor problem or question, and the results and contributions are minimal. The work is unlikely to have any significant impact on the field.
    1-2 - The paper addresses an unimportant problem or question, and the results and contributions are completely insignificant. The work will have no impact on the field.

5. Overall Assessment (1-10)
    - Based on the above criteria, how would you rate the overall quality of the paper? Note that any single weakness can be critical to lower the overall assessment.
    - Is the paper suitable for publication in a top-tier conference or journal?
    - Would you recommend this paper to your colleagues?

    10 - The paper is of exceptional quality and is highly suitable for publication in a top-tier conference or journal. I would strongly recommend this paper.
    8-9 - The paper is of high quality and is suitable for publication in a top-tier conference or journal. I would recommend this paper.
    6-7 - The paper is of good quality and is suitable for publication in a reputable conference or journal. I would recommend this paper with some reservations.
    4-5 - The paper is of acceptable quality but may not be suitable for publication in a top-tier conference or journal. I would recommend this paper with significant reservations.
    2-3 - The paper is of poor quality and is not suitable for publication in a top-tier conference or journal. I would not recommend this paper.
    1 - The paper is of extremely poor quality and is not suitable for publication in any conference or journal. I would strongly advise against recommending this paper.

6. Confidence Score (1-5)
    - How confident are you in your overall assessment of the paper?

    5 - Extremely confident in the overall assessment.
    4 - Very confident in the overall assessment.
    3 - Moderately confident in the overall assessment.
    2 - Slightly confident in the overall assessment.
    1 - Not confident in the overall assessment.

Please provide a detailed review of the paper, including your scores for each aspect and an overall assessment. Be sure to justify your scores with specific examples from the paper.
Please do not include any personal opinions or biases in your review. Your review should be objective and based solely on the content of the paper. Please provide a confidence score from 1 to 5 for the overall assessment.
Do not hesitate to assign lower scores if the paper does not fully meet the criteria. Avoid giving high scores by default.

## Output Format

Please provide your review in the following format:

```json
{
    "Clarity": {
        "score": <1-10>,
        "justification": "<Your justification here>"
    },
    "Novelty": {
        "score": <1-10>,
        "justification": "<Your justification here>"
    },
    "Soundness": {
        "score": <1-10>,
        "justification": "<Your justification here>"
    },
    "Significance": {
        "score": <1-10>,
        "justification": "<Your justification here>"
    },
    "Overall": {
        "score": <1-10>,
        "strengths": ["<strength 1>", "<strength 2>"],
        "weaknesses": ["<weakness 1>", "<weakness 2>"]
    },
    "Confidence": <1-5>
}
```

Note that any single weakness can be critical to lower the overall assessment.
Please provide detailed justifications for each score, including specific examples from the paper. 
IMPORTANT: Please ensure that your output is a complete and valid JSON object and includes all the fields above. Do not output only a single item or partial content; you must output the entire JSON object.
"""


def overall_review(
    paper_path,
    client,
    task_file,
    code_path=None,
    review_form=OVERALL_RUBRIC,
    max_retries=3,
):
    paper, img_list = load_multimodal_content(paper_path)
    task = read_text(task_file)
    if code_path:
        code_content = read_combine_files(code_path)
    else:
        code_content = ""
    base_prompt = review_form

    base_prompt += f"""
## Task Description

```
{task}
```

## Paper to Be Reviewed
Note: The paper is generated by AI and may contain some errors. Please check the paper carefully and provide your review.
    
```json
{paper}
```

## Code of the Paper

```json
{code_content}
```
"""

    base_prompt += """
Please provide a detailed review of the paper, including your scores for each aspect and an overall assessment. Be sure to justify your scores with specific examples from the paper.
Please do not include any personal opinions or biases in your review. Your review should be objective and based solely on the content of the paper. Please provide a confidence score from 1 to 5 for the overall assessment.
Do not hesitate to assign lower scores if the paper does not fully meet the criteria. Avoid giving high scores by default.
    
## Output Format

Please provide your review in the following format:

```json
{
    "Clarity": {
        "score": <1-10>,
        "justification": "<Your justification here>"
    },
    "Novelty": {
        "score": <1-10>,
        "justification": "<Your justification here>"
    },
    "Soundness": {
        "score": <1-10>,
        "justification": "<Your justification here>"
    },
    "Significance": {
        "score": <1-10>,
        "justification": "<Your justification here>"
    },
    "Overall": {
        "score": <1-10>,
        "strengths": ["<strength 1>", "<strength 2>"],
        "weaknesses": ["<weakness 1>", "<weakness 2>"]
    },
    "Confidence": <1-5>
}
```

Note that any single weakness can be critical to lower the overall assessment.
Please provide detailed justifications for each score, including specific examples from the paper. 
IMPORTANT: Please ensure that your output is a complete and valid JSON object and includes all the fields above. Do not output only a single item or partial content; you must output the entire JSON object.
"""
    for attempt in range(max_retries):
        try:
            response, token_usage = client.generate(prompt=base_prompt, media=img_list)
            review = extract_json_between_markers(response)
            # Validate JSON structure
            if not isinstance(review, dict):
                raise ValueError("Response is not a dictionary")
            
            required_keys = ["Clarity", "Novelty", "Soundness", "Significance", "Overall", "Confidence"]
            if not all(key in review for key in required_keys):
                raise ValueError(f"Missing required keys. Found: {list(review.keys())}")
            
            return review, token_usage
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                print("All retry attempts failed")
                return None
            continue
    
    return None


if __name__ == "__main__":
    # evaluator_name = "claude-3-7-sonnet-20250219"
    evaluator_name = "google/gemini-2.5-pro-preview"
    evaluator_folder = format_model_name(evaluator_name)
    reviewer = create_lmm_client(model_name=evaluator_name, max_tokens=8192*2, judge_mode=True)
    agent_name = "o4-mini"
    print(f"Reviewing {agent_name} paper using {evaluator_name}...")
    # set up your task name and paths for the review
    task_name = "iclr2025_verifai"
    task_path = osp.join(f"pipeline_{agent_name}", task_name)
    task_file = osp.join("claude_experiments", task_name, "task.md")
    paper_path = osp.join(task_path, "results", "paper.md")
    output_path = osp.join(f"samples_{evaluator_folder}", task_name, f"review_o4-mini_no_code.json")
    if os.path.exists(output_path):
        print(f"Review file {output_path} already exists!")
    review, token_usage = overall_review(paper_path=paper_path, client=reviewer, task_file=task_file)
    save_json(review, output_path)
    # tasklist = get_tasklist(f"pipeline_{agent_name}")
    # for task_name in tasklist:
    #     task_path = osp.join(f"pipeline_{agent_name}", task_name)
    #     task_file = osp.join(task_path, "task.md")
    #     paper_path = osp.join(task_path, "results", "paper.md")
    #     code_path = osp.join(task_path, "claude_code")
    #     if not os.path.exists(paper_path):
    #         print(f"Paper file {paper_path} does not exist.")
    #         continue
    #     save_path = osp.join(f"samples_{evaluator_folder}", task_name)
    #     output_path = osp.join(save_path, f"review_{agent_name}.json")
    #     if os.path.exists(output_path):
    #         print(f"Review file {output_path} already exists!")
    #         continue
    #     print(f"Reviewing paper: {paper_path}")
    #     review, token_usage = overall_review(paper_path=paper_path, 
    #                                          client=reviewer, 
    #                                          task_file=task_file,
    #                                          code_path=code_path,
    #                                         )
    #     if review:
    #         save_json(review, output_path)
            # print(f"Token usage for {task_name}: {token_usage}")