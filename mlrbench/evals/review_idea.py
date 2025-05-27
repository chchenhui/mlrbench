import os
import os.path as osp
import numpy as np
import json
import re
from mlrbench.utils.utils import *
from mlrbench.llm.llm import *


RESEARCH_IDEA_RUBRIC = """
You are an expert machine learning researcher! 
You will be given a research idea and a task description. 
Your task is to evaluate a research idea on a scale of 1 to 10 across five key dimensions and finally give an overall assessment on a scale of 1 to 10.
Please be objective in your evaluation, and provide detailed justifications for each score you assign.
Do not hesitate to assign lower scores if the idea does not fully meet the criteria. Avoid giving high scores by default.

## Evaluation Rubric

1. CONSISTENCY (1-10)
    How well does the idea align with the requirements of the task description?

    9-10 - Excellent: The idea is perfectly aligned with the task description. It addresses all aspects of the task and is highly relevant.
    7-8 - Good: The idea is mostly aligned with the task description. It addresses most aspects but may miss some minor details.
    5-6 - Satisfactory: The idea is somewhat aligned with the task description. It addresses some aspects but misses several key points.
    3-4 - Needs Improvement: The idea is poorly aligned with the task description. It addresses only a few aspects and misses many key points.
    1-2 - Poor: The idea is not aligned with the task description. It does not address the task or is completely irrelevant.

2. CLARITY (1-10)
   How clear and well-defined is the research idea?

   9-10 - Excellent: The idea is crystal clear, perfectly defined, and immediately understandable. It is articulated concisely with no ambiguity.
   7-8 - Good: The idea is mostly clear and well-articulated with only minor ambiguities. Minor refinements would make it even more precise.
   5-6 - Satisfactory: The idea is partially clear but has some ambiguities. Some aspects need further elaboration for a complete understanding.
   3-4 - Needs Improvement: The idea is unclear with significant ambiguities. Major clarification is needed to make it comprehensible.
   1-2 - Poor: The idea is extremely unclear, vague, or ambiguous with no proper explanation. It is difficult to understand or interpret.

3. NOVELTY (1-10)
   How original and innovative is the research idea?

   9-10 - Excellent: The idea is highly original and innovative. It introduces a groundbreaking concept or approach that is significantly different from existing work.
   7-8 - Good: The idea has notable originality and innovation. It offers fresh perspectives or new combinations of existing concepts.
   5-6 - Satisfactory: The idea has some originality and innovation. It includes some novel aspects but also shares similarities with existing approaches.
   3-4 - Needs Improvement: The idea has minimal originality. It largely resembles existing approaches with only slight variations.
   1-2 - Poor: The idea lacks originality and innovation. It closely resembles common or existing concepts without any new insights.

4. FEASIBILITY (1-10)
   How practical and implementable is the research idea?

   9-10 - Excellent: The idea is highly practical and implementable with current resources, technology, and knowledge. Execution is straightforward.
   7-8 - Good: The idea is largely feasible with existing technology and methods, though it may require moderate refinement or optimization.
   5-6 - Satisfactory: The idea is somewhat feasible but has some implementation challenges. It may require considerable effort or resources to implement.
   3-4 - Needs Improvement: The idea has significant implementation challenges. Major revisions would be needed to make it feasible.
   1-2 - Poor: The idea is impractical or impossible to implement with current technology, knowledge, or constraints.

5. SIGNIFICANCE (1-10)
   How important and impactful is the research idea?

   9-10 - Excellent: The idea is highly significant and impactful. It addresses a critical problem and could lead to major advancements in the field.
   7-8 - Good: The idea is significant and has clear impact potential. It addresses an important issue and could lead to meaningful contributions.
   5-6 - Satisfactory: The idea is somewhat significant. It addresses a relevant problem but its impact may be moderate or limited to a specific area.
   3-4 - Needs Improvement: The idea has limited significance. It addresses a minor issue or has a narrow scope with minimal impact.
   1-2 - Poor: The idea has little to no significance. It does not address a meaningful problem or offer any clear benefits.

6. Overall Assessment (1-10)
    How would you rate the research idea overall, considering all five dimensions above?

    10 - Outstanding: The idea is exceptional in every respect, with no significant weaknesses.
    8-9 - Excellent: The idea is very strong overall, with only minor weaknesses.
    6-7 - Good: The idea is solid, with a good balance of strengths and weaknesses.
    4-5 - Satisfactory: The idea is adequate but has notable weaknesses.
    2-3 - Needs Improvement: The idea has significant weaknesses that limit its potential.
    1 - Poor: The idea is fundamentally flawed across most or all dimensions.

When assigning the Overall Assessment score, consider not just the average of the five dimensions, but also:
- Whether any single weakness is critical enough to lower the overall potential.
- The overall coherence and integration of the idea.
- The likelihood of real-world impact if the idea were pursued.
- The degree to which the idea fulfills the task description as a whole.
- Any unique strengths or fatal flaws that are not fully captured by the individual dimensions.
   
## Output Format

Please output a complete JSON object strictly following the format below, including all evaluation items (Consistency, Clarity, Novelty, Feasibility, Significance, OverallAssessment). Do not output only a single item or partial content; you must output the entire JSON object.

```json
{
    "Consistency": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the alignment with the task description>",
    },
    "Clarity": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the clarity of the idea>",
    },
    "Novelty": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the originality and innovation of the idea>",
    },
    "Feasibility": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the practicality and implementability of the idea>",
    },
    "Significance": {
        "score": <1-10>,
        "justification": "<detailed explanation of why the score was given, referencing the importance and impact of the idea>",
    },
    "OverallAssessment": {
        "score": <1-10>,
        "strengths": ["<strength 1>", "<strength 2>"],
        "weaknesses": ["<weakness 1>", "<weakness 2>"]
    }
}
```

Please make sure your output is a complete JSON object and includes all the fields above.
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


def review_idea(
    file_path,
    client,
    task_file,
    review_form=RESEARCH_IDEA_RUBRIC,
):
    """
    Review the idea: Consistency, Clarity, Novelty, Feasibility, Significance.

    Args:
        file_path (`str`): 
            The path to the idea file.
        client ([`LLM`]):
            LLM engine of the evaluator.
        task_file (`str`): 
            The task file in the review.
        review_form (`str`):
            The review form for the idea.
    """
    idea = read_text(file_path)
    task = read_text(task_file)
    base_prompt = review_form
    base_prompt += f"""
    Here is the idea you are asked to review:
    ```
    {idea}
    ```
    Here is the task:
    ```
    {task}
    ```"""
    response, token_usage = client.generate(base_prompt)
    review = extract_json_between_markers(response)
    return review


if __name__ == "__main__":
    evaluator_name = "claude-3-7-sonnet-20250219" # set up your judge model
    evaluator_folder = format_model_name(evaluator_name)
    llm_engine = create_client(model_name=evaluator_name, judge_mode=True)
    agent_folder = format_model_name("mistral/ministral-8b") # set up your agent model
    print(f"Reviewing idea generated by {agent_folder} using {evaluator_name}...")
    # get all tasks
    # if you want to review idea for a specific task, you can specify the task name
    tasklist = get_tasklist("tasks")
    for task in tasklist:
        task_file = osp.join("tasks", task+".md") 
        idea_file = osp.join("results", task, "idea", f"idea_{agent_folder}.md")
        save_path = osp.join(f"reviews_{evaluator_folder}", task, "idea")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        output_file = osp.join(save_path, f"idea_{agent_folder}.json")
        if os.path.exists(output_file):
            print(f"""{output_file} already exists!""")
            continue
        result = review_idea(file_path=idea_file,
                             client=llm_engine, 
                             task_file=task_file,
                             )
        if result is None:
            print(f"Error: reviews of {task_file} is not a valid JSON file.")
            continue
        if len(result) != 6:
            print(f"Error: reviews of {task_file} do not complete. Keys: {result.keys()}")
            continue
        save_json(result, output_file)