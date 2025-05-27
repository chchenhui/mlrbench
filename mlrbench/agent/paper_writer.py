import os
import os.path as osp
import logging

from mlrbench.utils.utils import *
from mlrbench.lmm.lmm import *


def write_paper(
    client,
    task_file,
    idea_file,
    lit_file,
    proposal_file,
    exp_file,
):
    prompt = """
    You are an excellent machine learning researcher!
    Given the task, research idea, literature review, proposal and experiment results, please write a paper for the machine learning project.
    It should include the following sections:
    1. Title and Abstract: A concise title and a brief abstract summarizing the research.
    2. Introduction: An introduction to the problem, its significance, and the proposed solution.
    3. Related Work: A review of existing literature and how your work fits into the current landscape.
    4. Methodology: A detailed description of the proposed method, including any algorithms or models used.
    5. Experiment Setup: A description of the experimental setup, including datasets, metrics, and evaluation methods.
    6. Experiment Results: A presentation of the results obtained from the experiments, including tables and figures.
    7. Analysis: An analysis of the results, discussing their implications and any limitations.
    8. Conclusion: A summary of the findings and suggestions for future work.
    9. References: A list of references cited in the paper.
    
    Figures and tables should be included in the paper. You may refer to the figures and tables in the experiment results. If there is no image in the experiment results, please do not create or cite any fake figures. Please directly use the paths of the figures in the markdown file and do not use any placeholders.
    When writing mathematical formulas, you should use LaTeX syntax. For inline formulas, use single dollar signs, for example: $x^2$ to represent x squared. For block equations, use double dollar signs at the beginning and end, for example: $x^2$.
    If you need to write text in mathematical formulas, please avoid invalid escape characters.
    The paper should be well-structured and clearly present the research findings.
    """
    task = read_text(task_file)
    idea = read_text(idea_file)
    related_work = read_text(lit_file)
    proposal = read_text(proposal_file)
    exp_results, img_list = load_multimodal_content(exp_file)
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
    ```
    Here is the proposal:
    ```
    {proposal}
    ```
    Here is a summary of the experiment results:
    ```
    {exp_results}
    ```"""
    response, token_usage = client.generate(prompt=prompt, media=img_list)
    return response, token_usage


def write_paper_for_step(model_name):
    model_folder = format_model_name(model_name)
    lmm_engine = create_lmm_client(model_name=model_name, max_tokens=8192*2)
    tasklist = get_tasklist("claude_experiments")
    for task_name in tasklist:
        task = osp.join("claude_experiments", task_name) 
        task_file = osp.join(task, "task.md")
        idea_file = osp.join(task, "idea.md")
        lit_file = osp.join(task, "related_work.md")
        proposal_file = osp.join(task, "proposal.md")
        exp_file = osp.join(task, "results", "results.md")
        if not osp.exists(exp_file):
            raise NotImplementedError(f"Experiment results not found for {task_name}...")
        save_path = osp.join(task, "results")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        output_file = osp.join(save_path, f"paper_{model_folder}.md")
        if os.path.exists(output_file):
            print(f"""Paper of {save_path} already exists!""")
            continue
        result, token_usage= write_paper(client=lmm_engine, 
                             task_file=task_file, 
                             idea_file=idea_file, 
                             lit_file=lit_file, 
                             proposal_file=proposal_file, 
                             exp_file=exp_file
                            )
        # print(result)
        save_text(result, output_file)


def write_paper_for_pipeline(
    model_name,
    task_path,
    max_retry=3
):
    logging.info(f"Writing paper using {model_name}...")
    lmm_engine = create_lmm_client(model_name=model_name, max_tokens=8192*2)
    task_file = osp.join(task_path, "task.md")
    idea_file = osp.join(task_path, "idea.md")
    lit_file = osp.join(task_path, "related_work.md")
    proposal_file = osp.join(task_path, "proposal.md")
    exp_file = osp.join(task_path, "results", "results.md")
    output_file = osp.join(task_path, "results", "paper.md")
    token_file = osp.join(task_path, "paper_token_usage.json")
    if os.path.exists(output_file):
        logging.info(f"""{output_file} already exists!""")
        return

    for attempt in range(1, max_retry + 1):
        try:
            result, token_usage = write_paper(
                client=lmm_engine,
                task_file=task_file,
                idea_file=idea_file,
                lit_file=lit_file,
                proposal_file=proposal_file,
                exp_file=exp_file
            )
            if result:
                save_text(result, output_file)
                save_json(token_usage, token_file)
            # check if the paper file is generated
            if os.path.exists(output_file):
                logging.info(f"Successfully wrote paper for {task_path} on attempt {attempt}.")
                return
            else:
                raise Exception("File not generated after save_text.")
        except Exception as e:
            logging.error(f"Attempt {attempt} failed: {e}")
            if attempt == max_retry:
                raise NotImplementedError(f"Failed to write paper for {task_path} after {max_retry} attempts.")


if __name__ == "__main__":
    model_name = "claude-3-7-sonnet-20250219"
    write_paper_for_step(model_name=model_name)
    # write_paper_for_pipeline(model_name)
