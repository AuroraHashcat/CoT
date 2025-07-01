# causal_cot/llm_handler.py
import os
import json
from abc import ABC, abstractmethod

class LLMHandler(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def query(self, prompt: str) -> str:
        pass

class APIHandler(LLMHandler):
    def __init__(self, config: dict):
        super().__init__(config)
        import openai
        
        api_key_env = config['api_key_info'].get('api_key_env')
        if not api_key_env:
            raise ValueError("'api_key_env' not specified in API config.")
            
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found. Please set the {api_key_env} environment variable.")
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=config['api_key_info']['api_url'],
        )
        self.model_name = config['model_info']['name']
        self.params = config.get('params', {})

    def query(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **self.params
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API Error: {e}")
            return f"API_ERROR: {e}"

class LocalHandler(LLMHandler):
    def __init__(self, config: dict):
        super().__init__(config)
        try:
            from transformers import pipeline
            import torch
        except ImportError:
            raise ImportError("Please install 'transformers' and 'torch' to use local models.")

        model_path = config['model_info']['path']
        print(f"Loading local model from: {model_path}")
        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.params = config.get('params', {})

    def query(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        terminators = [
            self.pipe.tokenizer.eos_token_id,
        ]
        if self.pipe.tokenizer.pad_token_id is None:
             self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id

        try:
            # Check for special tokens for some models like Llama 3
            if hasattr(self.pipe.tokenizer, 'convert_tokens_to_ids'):
                try:
                    terminators.append(self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
                except: pass # Ignore if token doesn't exist

            outputs = self.pipe(
                messages,
                eos_token_id=terminators,
                **self.params
            )
            # The output structure for pipelines is a list containing a dict
            return outputs[0]["generated_text"][-1]['content']
        except Exception as e:
            print(f"Local model inference error: {e}")
            return f"LOCAL_MODEL_ERROR: {e}"

def get_llm_handler(model_config_path: str) -> LLMHandler:
    with open(model_config_path, 'r') as f:
        config = json.load(f)
    
    mode = config['model_info']['mode']
    if mode == 'api':
        return APIHandler(config)
    elif mode == 'local':
        return LocalHandler(config)
    else:
        raise ValueError(f"Unsupported model mode: {mode}")