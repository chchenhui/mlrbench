import re
from .utils import get_openai_client

class LLMHandler:
    def __init__(self, model_name="gpt-4o-mini"):
        self.client = get_openai_client()
        self.model_name = model_name

    def _extract_code(self, response_text):
        """
        Extracts the Python code from the LLM's response.
        """
        code_block_match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1)
        # Fallback for responses that might not use the markdown block
        return response_text

    def generate_code(self, prompt):
        """
        Generates code for a given prompt.
        """
        system_prompt = (
            "You are an expert Python programmer. "
            "Please provide a complete and correct Python function for the given prompt. "
            "The response should only contain the function definition. "
            "Do not include any import statements or example usage."
        )
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        
        raw_response = response.choices[0].message.content
        return self._extract_code(raw_response)

    def correct_code(self, original_prompt, code_to_correct, feedback):
        """
        Attempts to correct a piece of code based on feedback.
        """
        system_prompt = (
            "You are an expert Python programmer specializing in debugging. "
            "You will be given a Python function, the original prompt, and feedback on why the code is incorrect. "
            "Your task is to provide a corrected version of the function. "
            "The response should only contain the corrected function definition. "
            "Do not include any import statements or example usage."
        )
        
        user_prompt = (
            f"Original Prompt:\n---\n{original_prompt}\n---\n\n"
            f"Incorrect Code:\n---\n```python\n{code_to_correct}\n```\n---\n\n"
            f"Feedback:\n---\n{feedback}\n---\n\nPlease provide the corrected function."
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
        )
        
        raw_response = response.choices[0].message.content
        return self._extract_code(raw_response)

if __name__ == '__main__':
    handler = LLMHandler()
    test_prompt = "def add(a: int, b: int) -> int:\n    \"\"\"Adds two integers.\"\"\""
    generated_code = handler.generate_code(test_prompt)
    print("--- Generated Code ---")
    print(generated_code)
    print("-" * 22)

    feedback = "The code fails with an overflow when a is the maximum integer and b is 1."
    corrected_code = handler.correct_code(test_prompt, generated_code, feedback)
    print("--- Corrected Code ---")
    print(corrected_code)
    print("-" * 22)

