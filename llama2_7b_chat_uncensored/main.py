from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain import PromptTemplate

from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
import torch

import json
import textwrap



import time

def compute_runtime(func):
    """
    Decorator that computes the runtime of a function and prints the elapsed time in seconds.
    
    Usage Example:
    @compute_runtime
    def my_function():
        # code here
    
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        return result
    
    return wrapper


class LlamaRunner:
    # Wraps an example from https://huggingface.co/georgesung/llama2_7b_chat_uncensored
    def __init__(self, use_accelerator=False):
        """
        Initialize the LlamaRunner class.
    
        Args:
            use_accelerator (bool, optional): Whether to use an accelerator or not. Defaults to False.
        """
        
        model_id = "georgesung/open_llama_7b_qlora_uncensored"
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        if use_accelerator:
            model = LlamaForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=True)
        else:
            model = LlamaForCausalLM.from_pretrained(model_id)
        
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )

    @compute_runtime
    def ask(self, prompt):
        print(prompt)
        print(self.get_llm_response(prompt))
        print("\n--------")

    def get_prompt(self, human_prompt):
        """
        Generate a prompt for the conversation with the given human prompt.
        
        Args:
            human_prompt (str): The human prompt for the conversation.
        
        Returns:
            str: The generated prompt for the conversation.
        """
        prompt = f"### HUMAN:\n{human_prompt}\n\n### RESPONSE:\n"
        return prompt

    def get_response_text(self, data, wrap_text=True):
        """
        Extracts the response text from the generated output data.

        Args:
            data (list): The generated output data.
            wrap_text (bool, optional): Whether to wrap the text or not. Defaults to True.

        Returns:
            str: The extracted response text.
        """
        text = data[0]["generated_text"]

        response_start_index = text.find('### RESPONSE:')
        if response_start_index != -1:
            text = text[response_start_index+len('### RESPONSE:'):].strip()

        if wrap_text:
            text = textwrap.fill(text, width=100)

        return text

    def get_llm_response(self, prompt, wrap_text=True):
        """
        Generates a response text using the given prompt.

        Args:
            prompt (str): The prompt for the conversation.
            wrap_text (bool, optional): Whether to wrap the text or not. Defaults to True.

        Returns:
            str: The generated response text.
        """
        raw_output = self.pipe(self.get_prompt(prompt))
        text = self.get_response_text(raw_output, wrap_text=wrap_text)
        return text


