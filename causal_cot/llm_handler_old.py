import os
import json
import time
from abc import ABC, abstractmethod
<<<<<<< HEAD
from typing import Dict, Any, List
=======
from typing import Dict, Any, Optional
>>>>>>> 7c4acd1832cd2789043a29b967b1a89b0b7f748e

class LLMHandlerError(Exception):
    """Custom exception for LLM handler errors"""
    pass

class LLMHandler(ABC):
    """Abstract base class for LLM handlers"""
<<<<<<< HEAD
=======
    
>>>>>>> 7c4acd1832cd2789043a29b967b1a89b0b7f748e
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.total_queries = 0
        self.failed_queries = 0

    @abstractmethod
    def query(self, prompt: str) -> str:
        """
        Send a query to the LLM and return the response.
        
        Args:
            prompt (str): The input prompt
            
        Returns:
            str: The LLM response
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics"""
        success_rate = (self.total_queries - self.failed_queries) / max(1, self.total_queries)
        return {
            "total_queries": self.total_queries,
            "failed_queries": self.failed_queries,
            "success_rate": success_rate,
            "success_percentage": f"{success_rate:.1%}"
        }

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
<<<<<<< HEAD
    """Handler for API-based LLM services, with parameter filtering."""
    VALID_API_PARAMS = {"temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "seed", "stop"}

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import openai
        except ImportError:
            raise LLMHandlerError("Please install 'openai' package to use API handler")
        
        api_key_info = config.get('api_key_info', {})
=======
    """Handler for API-based LLM services (OpenAI, Anthropic, etc.)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        try:
            import openai
        except ImportError:
            raise LLMHandlerError("Please install 'openai' package to use API handler")
        
        # Validate configuration
        api_key_info = config.get('api_key_info', {})
        if not api_key_info:
            raise LLMHandlerError("'api_key_info' section missing from API config")
            
>>>>>>> 7c4acd1832cd2789043a29b967b1a89b0b7f748e
        api_key_env = api_key_info.get('api_key_env')
        if not api_key_env:
            raise LLMHandlerError("'api_key_env' not specified in API config")
            
        api_key = os.environ.get(api_key_env)
        if not api_key:
<<<<<<< HEAD
            raise LLMHandlerError(f"API key not found. Please set the {api_key_env} environment variable")
        
        self.client = openai.OpenAI(api_key=api_key, base_url=api_key_info.get('api_url'))
=======
            raise LLMHandlerError(
                f"API key not found. Please set the {api_key_env} environment variable"
            )
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_key_info.get('api_url', 'https://api.openai.com/v1'),
        )
>>>>>>> 7c4acd1832cd2789043a29b967b1a89b0b7f748e
        
        model_info = config.get('model_info', {})
        self.model_name = model_info.get('name')
        if not self.model_name:
            raise LLMHandlerError("Model name not specified in config")
<<<<<<< HEAD
        
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
=======
            
        self.params = config.get('params', {})
        
        # Set reasonable defaults
        self.params.setdefault('temperature', 0.7)
        self.params.setdefault('max_tokens', 2048)
        
        # Retry configuration
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)

    def query(self, prompt: str) -> str:
        """
        Query the API with retry logic and better error handling.
        """
        if not prompt or not prompt.strip():
            raise LLMHandlerError("Empty prompt provided")
            
        self.total_queries += 1
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt.strip()}],
                    **self.params
                )
                
                content = response.choices[0].message.content
                if content is None:
                    raise LLMHandlerError("API returned empty response")
                    
                return content.strip()
                
            except Exception as e:
                error_msg = f"API Error (attempt {attempt + 1}/{self.max_retries}): {str(e)}"
                
                if attempt < self.max_retries - 1:
                    print(f"⚠️ {error_msg}")
                    print(f"🔄 Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"{error_msg}")
                    self.failed_queries += 1
                    return f"API_ERROR_FINAL: {str(e)}"
        
        # This should never be reached, but just in case
        self.failed_queries += 1
        return "API_ERROR: Maximum retries exceeded"

class LocalHandler(LLMHandler):
    """Handler for local LLM models using transformers"""
    
>>>>>>> 7c4acd1832cd2789043a29b967b1a89b0b7f748e
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        try:
            from transformers.pipelines import pipeline
            import torch
        except ImportError:
<<<<<<< HEAD
            raise LLMHandlerError("Please install 'transformers' and 'torch' to use local models")
=======
            raise LLMHandlerError(
                "Please install 'transformers' and 'torch' to use local models"
            )
>>>>>>> 7c4acd1832cd2789043a29b967b1a89b0b7f748e

        model_info = config.get('model_info', {})
        model_path = model_info.get('path')
        if not model_path:
<<<<<<< HEAD
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
=======
            raise LLMHandlerError("Model path not specified in config")
            
        print(f"🔄 Loading local model from: {model_path}")
        
        try:
            # Check if CUDA is available
            device_map = "auto" if torch.cuda.is_available() else None
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            self.pipe = pipeline(
                "text-generation",
                model=model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True  # Some models require this
            )
            
            # Set up tokenizer padding
            if self.pipe.tokenizer is not None and self.pipe.tokenizer.pad_token_id is None:
                if hasattr(self.pipe.tokenizer, 'eos_token_id'):
                    self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id
                
            print(f"Local model loaded successfully")
        except Exception as e:
            raise LLMHandlerError(f"Failed to load local model: {str(e)}")
        
        self.params = config.get('params', {})
        
        # Set reasonable defaults for local models
        self.params.setdefault('max_new_tokens', 1024)
        self.params.setdefault('do_sample', True)
        self.params.setdefault('temperature', 0.7)
        self.params.setdefault('top_p', 0.9)

    def query(self, prompt: str) -> str:
        """
        Query the local model with better error handling.
        """
        if not prompt or not prompt.strip():
            raise LLMHandlerError("Empty prompt provided")
            
        self.total_queries += 1
        
        try:
            messages = [{"role": "user", "content": prompt.strip()}]
            
            # Set up terminators
            terminators = []
            if self.pipe.tokenizer is not None and hasattr(self.pipe.tokenizer, 'eos_token_id'):
                terminators.append(self.pipe.tokenizer.eos_token_id)
            
            # Add model-specific terminators
            if self.pipe.tokenizer is not None and hasattr(self.pipe.tokenizer, 'convert_tokens_to_ids'):
                try:
                    # Common terminators for chat models
                    special_tokens = ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]
                    for token in special_tokens:
                        try:
                            token_id = self.pipe.tokenizer.convert_tokens_to_ids(token)
                            if token_id != self.pipe.tokenizer.unk_token_id:
                                terminators.append(token_id)
                        except:
                            continue
                except:
                    pass  # Ignore if special token handling fails

            outputs = self.pipe(
                messages,
                eos_token_id=terminators if terminators else None,
                pad_token_id=self.pipe.tokenizer.eos_token_id if self.pipe.tokenizer else None,
                **self.params
            )
            
            # Extract the generated text
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]["generated_text"]
                if isinstance(generated_text, list) and len(generated_text) > 0:
                    # Get the assistant's response (last message)
                    assistant_response = generated_text[-1].get('content', '')
                    return assistant_response.strip()
                else:
                    return str(generated_text).strip()
            else:
                raise LLMHandlerError("Model produced empty output")
                
        except Exception as e:
            error_msg = f"Local model inference error: {str(e)}"
            print(f"{error_msg}")
            self.failed_queries += 1
            return f"LOCAL_MODEL_ERROR: {str(e)}"

def get_llm_handler(model_config_path: str) -> LLMHandler:
    """
    Factory function to create the appropriate LLM handler based on configuration.
    """
    try:
        with open(model_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise LLMHandlerError(f"Configuration file not found: {model_config_path}")
    except json.JSONDecodeError as e:
        raise LLMHandlerError(f"Invalid JSON in configuration file: {str(e)}")
    except Exception as e:
        raise LLMHandlerError(f"Error reading configuration file: {str(e)}")
    
    model_info = config.get('model_info', {})
    mode = model_info.get('mode')
    
    if not mode:
        raise LLMHandlerError("Model mode not specified in configuration")
    
>>>>>>> 7c4acd1832cd2789043a29b967b1a89b0b7f748e
    if mode == 'api':
        return APIHandler(config)
    elif mode == 'local':
        return LocalHandler(config)
    else:
<<<<<<< HEAD
        raise LLMHandlerError(f"Unsupported mode: '{mode}'. Must be 'api' or 'local'.")
=======
        raise LLMHandlerError(f"Unsupported model mode: {mode}. Supported modes: 'api', 'local'")

def validate_config(config_path: str) -> Dict[str, Any]:
    """
    Validate a configuration file without creating a handler.
    """
    results = {
        "valid": False,
        "errors": [],
        "warnings": []
    }
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # Check required fields
        required_fields = ['model_info']
        for field in required_fields:
            if field not in config:
                results["errors"].append(f"Missing required field: {field}")
        
        model_info = config.get('model_info', {})
        if 'mode' not in model_info:
            results["errors"].append("Missing 'mode' in model_info")
        else:
            mode = model_info['mode']
            if mode == 'api':
                api_key_info = config.get('api_key_info', {})
                if not api_key_info.get('api_key_env'):
                    results["errors"].append("API mode requires 'api_key_env' in api_key_info")
                if not model_info.get('name'):
                    results["errors"].append("API mode requires 'name' in model_info")
            elif mode == 'local':
                if not model_info.get('path'):
                    results["errors"].append("Local mode requires 'path' in model_info")
            else:
                results["errors"].append(f"Unsupported mode: {mode}")
        
        # Check for potential issues
        params = config.get('params', {})
        if params.get('temperature', 0) > 2.0:
            results["warnings"].append("High temperature value may cause inconsistent outputs")
            
        results["valid"] = len(results["errors"]) == 0
        
    except Exception as e:
        results["errors"].append(f"Configuration validation error: {str(e)}")
    
    return results
>>>>>>> 7c4acd1832cd2789043a29b967b1a89b0b7f748e
