import os
import os.path as osp
import logging

from mlrbench.utils.utils import *
from mlrbench.llm.llm import *


def generate_proposal(
    client,
    task_file,
    idea_file,
    lit_file,
):  
    prompt = """
    You are an excellent machine learning researcher!
    Please generate a detailed research proposal based on a given task description, a research idea and their literature review.
    The proposal should be about 2000 words and include the following five sections:
    1. Title: a concise and descriptive title for the research proposal.
    2. Introduction: background, research objectives and significance.
    3. Methodology: detailed and precise research design (including data collection, full algorithmic steps and/or mathematical formulas where appropriate, and full details about experimental design to validate the method, with evaluation metrics).
    4. Expected Outcomes & Impact.
    The proposal should be well-structured and clearly articulate the research plan.
    When writing mathematical formulas, you should use LaTeX syntax. For inline formulas, use single dollar signs, for example: $x^2$ to represent x squared. For block equations, use double dollar signs at the beginning and end, for example: $$x^2$$.
    Please directly respond to the proposal.
    """
    task = read_text(task_file)
    idea = read_text(idea_file)
    related_work = read_text(lit_file)
    prompt += f"""
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
    response = client.generate(prompt)
    return response


def generate_proposal_for_step(model_name):
    print(f"Generating proposal using {model_name}...")
    model = format_model_name(model_name)
    llm_engine = create_client(model_name=model_name, max_tokens=8192*2)
    tasklist = get_tasklist("tasks")
    for task in tasklist:
        task_file = osp.join("tasks", task+".md") 
        task_path = osp.join("stepwise_results", task)
        if not os.path.exists(task_path):
            raise NotImplementedError("No such directory, please check again!")
        idea_file = osp.join(task_path, "idea.md")
        lit_file = osp.join(task_path, "related_work.md")
        file_name = f"proposal_{model}.md"
        output_path = osp.join(task_path, "proposal")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_file = osp.join(output_path, file_name)
        if os.path.exists(output_file):
            print(f"""{output_file} already exists!""")
            continue
        result = generate_proposal(client=llm_engine, task_file=task_file, idea_file=idea_file, lit_file=lit_file)
        save_text(result, output_file)


def generate_proposal_for_pipeline(
    model_name,
    task_path,
    max_retry=3
):
    logging.info(f"Generating proposal using {model_name}...")
    model = format_model_name(model_name)
    model = simplify_name(model)
    llm_engine = create_client(model_name=model_name, max_tokens=8192*2)
    task_file = osp.join(task_path, "task.md")
    idea_file = osp.join(task_path, "idea.md")
    lit_file = osp.join(task_path, "related_work.md")
    output_file = osp.join(task_path, "proposal.md")
    token_file = osp.join(task_path, "proposal_token_usage.json")
    if os.path.exists(output_file):
        logging.info(f"""{output_file} already exists!""")
        return

    for attempt in range(1, max_retry + 1):
        try:
            result, token_usage = generate_proposal(
                client=llm_engine, 
                task_file=task_file, 
                idea_file=idea_file, 
                lit_file=lit_file
            )
            if result:
                save_text(result, output_file)
                save_json(token_usage, token_file)
            # check if the proposal file is generated
            if os.path.exists(output_file):
                logging.info(f"Successfully generated proposal for {task_path} on attempt {attempt}.")
                return
            else:
                raise Exception("File not generated after save_text.")
        except Exception as e:
            logging.warning(f"Attempt {attempt} failed: {e}")
            if attempt == max_retry:
                raise NotImplementedError(f"Failed to generate proposal for {task_path} after {max_retry} attempts.")


if __name__ == "__main__":
    model_name = "claude-3-7-sonnet-20250219"
    generate_proposal_for_step(model_name=model_name)
    # generate_proposal_for_pipeline(model_name="claude-3-7-sonnet-20250219")
