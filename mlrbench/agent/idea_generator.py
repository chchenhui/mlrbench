import os
import os.path as osp
import logging

from mlrbench.utils.utils import *
from mlrbench.llm.llm import *


def generate_idea(
    client,
    task_file,
):  
    prompt = """
    You are an excellent machine learning researcher. Please generate innovative and practical ideas based on a given task description.
    Note that there might be a couple of research topics in the task description, and you should focus on one of them.
    The idea should be no more than 200 words and include the following three sections:
    1. Title: A concise and descriptive title for the research idea.
    2. Motivation: A brief explanation of why this research is important and what problems it aims to solve.
    3. Main Idea: A clear and detailed description of the proposed research idea, including the methodology, expected outcomes, and potential impact.
    Please directly respond to the idea.
    """
    task = read_text(task_file)
    prompt += f"""
    Here is the task:
    ```
    {task}
    ```"""
    response, token_usage = client.generate(prompt)
    return response, token_usage


def generate_idea_for_step(model_name):
    print(f"Generating idea for {model_name}...")
    model = format_model_name(model_name)
    llm_engine = create_client(model_name=model_name)
    # get all tasks
    # if you want to generate idea for a specific task, you can specify the task name
    tasklist = get_tasklist("tasks")
    for task in tasklist:
        task_file = osp.join("tasks", task+".md") 
        task_path = osp.join("stepwise_results", task)
        if not os.path.exists(task_path):
            os.makedirs(task_path)
        # generate idea based on task description
        file_name = f"idea_{model}.md"
        output_path = osp.join(task_path, "idea")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_file = osp.join(output_path, file_name)
        if os.path.exists(output_file):
            print(f"""{output_file} already exists!""")
        else:
            result = generate_idea(client=llm_engine, task_file=task_file)
            save_text(result, output_file)


def generate_idea_for_pipeline(
    model_name,
    task_path,
    max_retry=3,
    ):
    logging.info(f"Generating idea for {model_name}...")
    model = format_model_name(model_name)
    model = simplify_name(model)
    llm_engine = create_client(model_name=model_name)
    task_file = osp.join(task_path, "task.md")
    output_file = osp.join(task_path, "idea.md")
    token_file = osp.join(task_path, "idea_token_usage.json")
    if os.path.exists(output_file):
        logging.info(f"""{output_file} already exists!""")
        return
    for attempt in range(1, max_retry + 1):
        try:
            result, token_usage = generate_idea(client=llm_engine, task_file=task_file)
            if result:
                save_text(result, output_file)
                save_json(token_usage, token_file)
            # check if the idea file is generated
            if os.path.exists(output_file):
                logging.info(f"Successfully generated idea for {task_path} on attempt {attempt}.")
                return
            else:
                raise Exception("File not generated after save_text.")
        except Exception as e:
            logging.warning(f"Attempt {attempt} failed: {e}")
            raise NotImplementedError(f"Failed to generate idea for {task_path} after {max_retry} attempts.")



if __name__ == "__main__":
    model_name = "claude-3-7-sonnet-20250219"
    generate_idea_for_step(model_name=model_name)
    # generate_idea_for_pipeline(model_name)
