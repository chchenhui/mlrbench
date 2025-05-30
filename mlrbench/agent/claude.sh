#!/bin/bash

# Check if claude command is available
if ! command -v claude &> /dev/null; then
    echo "Error: claude command not found. Please install it first by running 'npm install -g @anthropic/claude-code'"
    exit 1
fi

# Set base folder path, you can change this to your desired path
base_folder="claude_experiments/"

# Check if base folder exists
if [ ! -d "$base_folder" ]; then
    echo "Error: Base folder $base_folder does not exist."
    exit 1
fi

# Iterate over all subdirectories in the base folder
for folder in "$base_folder"*/; do
    # Remove trailing slash to get the folder name
    folder_name=$(basename "$folder")
    
    echo "Processing task: $folder_name"
    
    # Check if claude_code folder already exists in the target folder
    if [ -d "$folder/claude_code" ]; then
        echo "Warning: 'claude_code' folder already exists in $folder. Skipping this task."
        continue
    fi
    
    # Build prompt
    prompt="
    Based on task.md, idea.md, related_work.md and proposal.md in $folder, please design an experimental plan to test the hypothesis that the proposed method is effective. Then write code to implement the experimental plan, and run the experiments in a fully automated manner.
    The experimental process should be fully automated and include the following steps:
    1. Create a folder named 'claude_code' inside $folder to save the code and results.
    2. Write some Python scripts to run the experiment automatically. IMPORTANT: The script must be thoroughly debugged and tested until it can run successfully without any errors. The script should handle all necessary data processing, model training, and evaluation steps automatically. NOTE: If the environment has available GPUs, the script should utilize GPU acceleration where possible.
       - The scripts should be able to run all baseline methods automatically.
       - The scripts should be able to save the results in a structured format (e.g., CSV or JSON) for easy analysis.
       - The scripts should generate figures to visualize the results. Figures should be generated for the main results, including but not limited to:
            - Training and validation loss curves
            - Performance metrics (e.g., accuracy, F1 score) over time
            - Comparison of the proposed method with baseline methods
            - Any other relevant figures that help to understand the performance of the proposed method
        - Make sure that the figures are properly labeled and include legends, titles, and axis labels. There should be data points in the figures, and the figures should be easy to read and interpret.
    3. Write a README.md file to explain how to run the experiment.
    4. Run the experiment automatically, visualize the results, and save the results. Make sure the generated figure files are not empty.
    5. Save the experiment execution process in log.txt.
    6. Analyze and summarize the experiment results in results.md after all experiments (including all baselines) have been successfully completed. 
        - Tables should be generated for the main results, including but not limited to:
            - Summary of the experimental setup (e.g., hyperparameters, dataset splits)
            - Comparison of the proposed method with baseline methods
            - Any other relevant tables that help to understand the performance of the proposed method
    7. Finally, create a folder named 'results' under $folder and move results.md, log.txt, and all figures generated by the experiment into 'results'.

    IMPORTANT: 
    - Do not use synthetic results or generate any fake data. The results should be based on real experiments. If the experimental dataset is too large, please use a smaller subset of the dataset for testing purposes.
    - If the experiment is about large language models (LLMs), please confirm the models used are LLMs and not simple multi-layer neural networks such as LSTM.
    - You can download the datasets from Hugging Face datasets or other open-sourced datasets. Please do not use any closed-sourced datasets.
    - If the experiment requires using open-sourced large language models (LLMs) such as Meta Llama-3.1-8B, Llama-3.2-1B, Mistral Ministral-8B or Qwen Qwen3-0.6B and multimodal LLMs like Qwen2-VL-3B, please download the models from the Hugging Face model hub. Due to limited computing resources, please use the models no larger than 8B.
    - If the experiment requires using closed-sourced large language models (LLMs) such as OpenAI's GPT-4o-mini, o4-mini or Claude-3.7-sonnet, please directly use the API provided by the model provider. API keys have already been provided in the environment variables. Please use the API key to access the model and run the experiment.
    - Only save essential files that are necessary for the experiment. Do not create or save any unnecessary files, temporary files, or intermediate files. Keep the output minimal and focused on the core experiment requirements.
    - No need to generate any paper or academic document. Focus only on the experimental implementation and results.
    - Please remove checkpoints or datasets larger than 1MB after the experiment is completed.

    Remember to analyze and summarize the experiment results, create a folder named 'results' under $folder and move results.md, log.txt, and all figures generated by the experiment into 'results'.
    - Figures and tables generated by the experiment should be included in the results.md with clear explanations of what they show. Please make sure the generated figures are not repeated. Please do not use the same figures multiple times in results.md.
    - Make sure paths to the figures in results.md are correct and point to the right locations.
    - The results.md should also include a discussion of the results, including any insights gained from the experiment and how they relate to the hypothesis.
    - The results.md should include a summary of the main findings and conclusions drawn from the experiment.
    - The results.md should also include any limitations of the experiment and suggestions for future work.
    "
    
    echo "Executing Claude command for task: $folder_name"
    
    # Execute claude command
    if claude -p "$prompt" --output-format json --verbose --dangerously-skip-permissions > "${folder}claude_output.json"; then
        echo "Claude command executed successfully for task: $folder_name!"
    else
        echo "Error: Claude command executed failed for task: $folder_name."
        # Continue with the next task even if this one fails
        continue
    fi
done

echo "All tasks processed."