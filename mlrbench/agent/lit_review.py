import os
import os.path as osp
import logging

from mlrbench.utils.utils import *
from mlrbench.llm.llm import *


def generate_lit_review(
    client,
    task_file,
    idea_file,
):  
    prompt = """
    You are an excellent machine learning researcher! 
    Please help me do a literature review for a given idea. 
    The idea is based on a given task description.
    The papers in the literature review should be extracted from arxiv and published between 2023 to 2025.
    The literature review should include the following:
    1. Related Papers: at least 10 academic papers most closely related to the current research idea, with a brief summary and the publication year of each one, organized logically.
    2. Key Challenges: A discussion of the main challenges and limitations in the current research. List no more than five key challenges.
    The paper should be in the format of:
    ```
    1. **Title**: <title> (<arxiv_id>)
        - **Authors**: <author1>, <author2>, ...
        - **Summary**: <summary>
        - **Year**: <year>
    ```
    Please directly respond to the literature review and do not include any additional comments.
    """
    task = read_text(task_file)
    idea = read_text(idea_file)
    prompt += f"""
    Here is the idea:
    ```
    {idea}
    ```
    Here is the task:
    ```
    {task}
    ```"""
    response, token_usage = client.generate(prompt)
    return response, token_usage


def generate_lit_review_for_step(engine_name):
    print(f"Generating literature review using {engine_name}...")
    llm_engine = create_client(model_name=engine_name)
    tasklist = get_tasklist("tasks")
    for task in tasklist:
        task_file = osp.join("tasks", task+".md") 
        save_path = osp.join("stepwise_results", task)
        if not os.path.exists(save_path):
            raise NotImplementedError("No such directory, please check again!")
        idea_file = osp.join(save_path, "idea.md")
        output_file = osp.join(save_path, "related_work.md")
        if os.path.exists(output_file):
            print(f"""Literature review of {save_path} already exists!""")
            continue
        result, token_usage = generate_lit_review(client=llm_engine, task_file=task_file, idea_file=idea_file)
        # print(result)
        save_text(result, output_file)


def generate_lit_review_for_pipeline(
    engine_name, 
    task_path,
    max_retry=3
):
    logging.info(f"Generating literature review using {engine_name}...")
    llm_engine = create_client(model_name=engine_name)
    task_file = osp.join(task_path, "task.md")
    idea_file = osp.join(task_path, "idea.md")
    output_file = osp.join(task_path, "related_work.md")
    token_file = osp.join(task_path, "lit_token_usage.json")
    if os.path.exists(output_file):
        logging.info(f"""{output_file} already exists!""")
        return

    for attempt in range(1, max_retry + 1):
        try:
            result, token_usage = generate_lit_review(
                client=llm_engine,
                task_file=task_file,
                idea_file=idea_file
            )
            if result:
                save_text(result, output_file)
                save_json(token_usage, token_file)
            # check if the literature review file is generated
            if os.path.exists(output_file):
                logging.info(f"Successfully generated literature review for {task_path} on attempt {attempt}.")
                return
            else:
                raise Exception("File not generated after save_text.")
        except Exception as e:
            logging.warning(f"Attempt {attempt} failed: {e}")
            if attempt == max_retry:
                raise NotImplementedError(f"Failed to generate literature review for {task_path} after {max_retry} attempts.")


if __name__ == "__main__":
    generate_lit_review_for_step(engine_name="gpt-4o-search-preview-2025-03-11")
    # generate_lit_review_for_pipeline(engine_name="gpt-4o-search-preview-2025-03-11")
