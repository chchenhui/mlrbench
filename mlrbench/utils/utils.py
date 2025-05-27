import os
import re
import json
import time
import logging
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Optional
import pymupdf
import pymupdf4llm
from typing import Union


def format_model_name(model_name: str) -> str:
    """
    Format the model name to be used in the path.
    """
    if "/" in model_name:
        model_folder = model_name.split("/")[-1]
        if ":free" in model_folder:
            model_folder = model_folder.replace(":free", "")
    else:
        model_folder = model_name
    return model_folder


def simplify_name(text: str) -> str:
    if "claude" in text:
        return "claude"
    elif "gemini" in text:
        return "gemini"
    elif "llama" in text:
        return "llama"
    elif "o4-mini" in text:
        return "o4-mini"

def extract_json_between_markers(llm_output):
    """
    Extracts JSON content from text between markdown code block markers.
    Handles LaTeX expressions and other special characters that might cause JSON parsing issues.
    
    Args:
        llm_output (str): Text containing JSON content
        
    Returns:
        dict or None: Parsed JSON as a Python dictionary, or None if extraction failed
    """
    # Extract content between markdown code block markers
    json_pattern = r"```json\s*(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)
    
    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}))*\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)
    
    for json_string in matches:
        # Clean up the string
        json_string = json_string.strip()
        
        # Pre-process LaTeX expressions - replace them with properly escaped versions
        def replace_latex(match):
            latex_content = match.group(1)
            # Double all backslashes to make them valid JSON escape sequences
            escaped_latex = latex_content.replace("\\", "\\\\")
            return escaped_latex
        
        # Replace LaTeX expressions ($...$) with their escaped content
        processed_json = re.sub(r'\$(.*?)\$', replace_latex, json_string)
        
        # Fix other common JSON issues
        # Fix trailing commas (valid in JS but not in JSON)
        processed_json = re.sub(r',(\s*[}\]])', r'\1', processed_json)
        
        # Remove invalid control characters
        processed_json = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', processed_json)
        
        try:
            # Try parsing the processed JSON
            parsed_json = json.loads(processed_json)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            # If normal JSON parsing fails, try a different strategy
            try:
                # Remove all LaTeX expressions for a second attempt
                sanitized = re.sub(r'\$.*?\$', '""', processed_json)
                parsed_json = json.loads(sanitized)
                return parsed_json
            except json.JSONDecodeError as e2:
                print(f"Second JSON decode error: {e2}")
                # If both attempts fail, continue to the next match if any
                continue
    
    # If we reach here, no valid JSON was found or parsed successfully
    print("llm output:", llm_output)
    return None


def save_json(data: Any, filename: str):
    """
    Save data to a JSON file
    """
    with open(filename, "w", encoding='utf8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False) 


def save_text(data: str, filename: str):
    """
    Save data to .txt or Markdown file
    """
    with open(filename, "w") as f:
        f.write(data)


def read_text(filename: str):
    """
    Read data from .txt or Markdown file
    """
    with open(filename, "r") as f:
        return f.read()


def load_pdf(path: str):
    """
    Load PDF file
    """
    try:
        md_text = pymupdf4llm.to_markdown(path)
    except Exception as e:
        print(f"Error with pymupdf, falling back to pypdf: {e}")
    return md_text


def get_tasklist(filepath: str = "tasks"):
    file_list = os.listdir(filepath)
    names = list()

    for name in file_list:
        path = name.split(".")[0]
        names.append(path)
    return sorted(names)


def load_json_without_image(json_file_path):
    """
    Read a JSON file and replace all base64 image data with empty strings.
    
    Args:
        json_file_path (str): Path to the input JSON file
    
    Returns:
        str: JSON string with base64 image data removed (formatted for readability)
    """
    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Recursively process the JSON object
    def process_item(item):
        if isinstance(item, dict):
            # If it's a dictionary, check if it's an image content
            if 'type' in item and item['type'] == 'image':
                if 'source' in item and 'type' in item['source'] and item['source']['type'] == 'base64':
                    # Replace base64 data with empty string
                    item['source']['data'] = ""
            
            # Recursively process all values in the dictionary
            for key, value in item.items():
                item[key] = process_item(value)
        
        elif isinstance(item, list):
            # If it's a list, recursively process each element
            return [process_item(element) for element in item]
        
        return item
    
    # Process the entire JSON data
    processed_data = process_item(data)
    
    # Convert processed data to a formatted JSON string
    json_string = json.dumps(processed_data, ensure_ascii=False, indent=2)
    
    return json_string

def load_multimodal_content(file_path: str) -> list:
    """
    Extract image paths from a markdown file.
    
    Args:
        markdown_file_path (str): Path to the markdown file
        
    Returns:
        list: List of image paths
    """
    if ".pdf" in file_path:
        content = pymupdf4llm.to_markdown(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    
    # Regex pattern to match image paths in markdown
    pattern = r'!\[.*?\]\((.*?)\)'
    matches = re.findall(pattern, content)
    image_links = [resolve_image_path(link.strip(), file_path) 
                  for link in matches if link.strip()]
    
    return content, image_links

def resolve_image_path(image_path: str, markdown_file_path: Union[str, Path] = None) -> str:
    """
    Resolve image path, convert relative path to absolute path.
    
    Args:
        image_path (str): Image path (can be relative or absolute)
        markdown_file_path (Union[str, Path], optional): Path of the markdown file, used for resolving relative paths
        
    Returns:
        str: Resolved absolute path
    """
    # If it's a URL, return as is
    if image_path.startswith(('http://', 'https://')):
        return image_path
        
    # If it's already an absolute path, return as is
    if os.path.isabs(image_path):
        return image_path
        
    # If markdown file path is provided, resolve based on its location
    if markdown_file_path:
        markdown_dir = os.path.dirname(os.path.abspath(markdown_file_path))
        return os.path.normpath(os.path.join(markdown_dir, image_path))
    
    # If no markdown file path provided, resolve based on current working directory
    return os.path.abspath(image_path)


def read_combine_files(directory, extensions=None, exclude_dirs=None):
    if extensions is None:
        extensions = ['.py', '.json', '.log', '.csv', '.txt', '.yaml', '.yml', '.sh', '.md']  
    if exclude_dirs is None:
        exclude_dirs = ['.git', 'venv', '__pycache__', 'outputs', 'data', 'experiment_results']  
    
    combined_text = ""
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if any(file.endswith(ext) for ext in extensions) and file != 'results.md' and file != 'results.json':
                file_path = os.path.join(root, file)
                combined_text += f"\n{'='*50}\n"
                combined_text += f"File: {file_path}\n"
                combined_text += f"{'='*50}\n\n"
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        combined_text += infile.read()
                        combined_text += "\n\n"
                except Exception as e:
                    combined_text += f"Error reading file: {str(e)}\n"
    
    return combined_text