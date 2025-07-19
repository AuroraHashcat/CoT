import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class LLMHandlerError(Exception):
    """Custom exception for LLM handler errors"""
    pass

class LLMHandler(ABC):
    """Abstract base class for LLM handlers"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.total_queries = 0
        self.failed_queries = 0

    @abstractmethod
    def query(self, prompt: str) -> str:
        pass

    def query_batch(self, prompts: List[str]) -> List[str]:
        # 默认串行fallback
        results = []
        for p in prompts:
            try:
                results.append(self.query(p))
            except Exception as e:
                results.append(f"[ERROR] {e}")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics"""
        success_rate = (self.total_queries - self.failed_queries) / max(1, self.total_queries)
        return {
            "total_queries": self.total_queries,
            "failed_queries": self.failed_queries,
            "success_percentage": f"{success_rate:.1%}" 
        }

class APIHandler(LLMHandler):
    """Handler for API-based LLM services, with parameter filtering."""
    VALID_API_PARAMS = {"temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "seed", "stop"}

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import openai
        except ImportError:
            raise LLMHandlerError("Please install 'openai' package to use API handler")
        
        api_key_info = config.get('api_key_info', {})
        api_key_env = api_key_info.get('api_key_env')
        if not api_key_env:
            raise LLMHandlerError("'api_key_env' not specified in API config")
            
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise LLMHandlerError(f"API key not found. Please set the {api_key_env} environment variable")
        
        self.client = openai.OpenAI(api_key=api_key, base_url=api_key_info.get('api_url'))
        
        model_info = config.get('model_info', {})
        self.model_name = model_info.get('name')
        if not self.model_name:
            raise LLMHandlerError("Model name not specified in config")
        
        all_params = config.get('params', {})
        self.api_params = {k: v for k, v in all_params.items() if k in self.VALID_API_PARAMS}
        if 'max_output_tokens' in all_params and 'max_tokens' not in self.api_params:
            self.api_params['max_tokens'] = all_params['max_output_tokens']

        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.timeout = config.get('timeout', 60)

    def query(self, prompt: str) -> str:
        import time
        self.total_queries += 1
        for attempt in range(self.max_retries):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt.strip()}],
                    timeout=self.timeout,
                    **self.api_params
                )
                content = response.choices[0].message.content
                elapsed = time.time() - start
                print(f'[LLM] Query took {elapsed:.2f}s')
                if content is None:
                    raise LLMHandlerError("API returned empty response")
                return content.strip()
            except Exception as e:
                error_msg = f"API Error (attempt {attempt + 1}/{self.max_retries}): {str(e)}"
                if attempt < self.max_retries - 1:
                    print(f"⚠️ {error_msg}\n🔄 Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.failed_queries += 1
                    raise LLMHandlerError(error_msg) from e
        return ""

class LocalHandler(LLMHandler):
    """Handler for local LLM models using transformers, with batching."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            from transformers import pipeline
            import torch
        except ImportError:
            raise LLMHandlerError("Please install 'transformers' and 'torch' to use local models")

        model_info = config.get('model_info', {})
        model_path = model_info.get('path')
        if not model_path:
            raise LLMHandlerError("Model path not specified in local config")
        
        device_map = "auto" if torch.cuda.is_available() else None
        
        self.params = config.get('params', {})
        self.batch_size = self.params.pop('batch_size', 1)

        print(f"Loading local model: {model_path}")
        self.pipe = pipeline("text-generation", model=model_path, device_map=device_map, trust_remote_code=True)
        if self.pipe.tokenizer.pad_token_id is None:
            self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id

    def query(self, prompt: str) -> str:
        self.total_queries += 1
        try:
            outputs = self.pipe([prompt], batch_size=1, **self.params)
            result = outputs[0]["generated_text"][len(prompt):].strip()
            return result
        except Exception as e:
            self.failed_queries += 1
            raise LLMHandlerError(f"Local model inference error: {e}") from e

    def query_batch(self, prompts: List[str]) -> List[str]:
        self.total_queries += len(prompts)
        try:
            outputs = self.pipe(prompts, batch_size=self.batch_size, **self.params)
            results = [out[0]['generated_text'][len(prompts[i]):].strip() for i, out in enumerate(outputs)]
            return results
        except Exception as e:
            self.failed_queries += len(prompts)
            # fallback: 返回错误信息
            return [f"[ERROR] {e}" for _ in prompts]

def get_llm_handler(model_config_path: str) -> LLMHandler:
    """Factory function to create the appropriate LLM handler."""
    with open(model_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    mode = config.get('model_info', {}).get('mode')
    if mode == 'api':
        return APIHandler(config)
    elif mode == 'local':
        return LocalHandler(config)
    else:
        raise LLMHandlerError(f"Unsupported mode: '{mode}'. Must be 'api' or 'local'.")