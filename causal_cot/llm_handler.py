import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

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
        """
        Send a query to the LLM and return the response.
        
        Args:
            prompt (str): The input prompt for the LLM
            
        Returns:
            str: The LLM's response
            
        Raises:
            LLMHandlerError: If the query fails
        """
        pass

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about LLM usage."""
        return {
            "total_queries": self.total_queries,
            "failed_queries": self.failed_queries,
            "success_rate": (self.total_queries - self.failed_queries) / max(self.total_queries, 1)
        }


class APILLMHandler(LLMHandler):
    """Handler for API-based LLM services (OpenAI, Anthropic, etc.)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.model = config.get("model")
        self.base_url = config.get("base_url")
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        
        # Initialize the appropriate client
        self._init_client()

    def _init_client(self):
        """Initialize the API client based on the provider."""
        provider = self.config.get("provider", "").lower()
        
        if provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise LLMHandlerError("OpenAI library not installed. Install with: pip install openai")
                
        elif provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise LLMHandlerError("Anthropic library not installed. Install with: pip install anthropic")
                
        elif provider == "google":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
            except ImportError:
                raise LLMHandlerError("Google AI library not installed. Install with: pip install google-generativeai")
                
        else:
            raise LLMHandlerError(f"Unsupported provider: {provider}")

    def query(self, prompt: str) -> str:
        """Send query to API-based LLM."""
        self.total_queries += 1
        
        for attempt in range(self.max_retries):
            try:
                return self._make_api_call(prompt)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"API call failed (attempt {attempt + 1}): {e}")
                    time.sleep(self.retry_delay)
                else:
                    self.failed_queries += 1
                    raise LLMHandlerError(f"API call failed after {self.max_retries} attempts: {e}")

    def _make_api_call(self, prompt: str) -> str:
        """Make the actual API call based on provider."""
        provider = self.config.get("provider", "").lower()
        
        if provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.get("max_tokens", 2000),
                temperature=self.config.get("temperature", 0.7)
            )
            return response.choices[0].message.content
            
        elif provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.config.get("max_tokens", 2000),
                temperature=self.config.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
            
        elif provider == "google":
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": self.config.get("temperature", 0.7),
                    "max_output_tokens": self.config.get("max_tokens", 2000)
                }
            )
            return response.text
            
        else:
            raise LLMHandlerError(f"Unsupported provider: {provider}")


class LocalLLMHandler(LLMHandler):
    """Handler for local LLM services (Ollama, Transformers, etc.)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model")
        self.endpoint = config.get("endpoint", "http://localhost:11434")
        self.provider = config.get("provider", "").lower()
        
        if self.provider == "ollama":
            self._init_ollama()
        elif self.provider == "transformers":
            self._init_transformers()
        else:
            raise LLMHandlerError(f"Unsupported local provider: {self.provider}")

    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            import requests
            self.session = requests.Session()
            # Test connection
            response = self.session.get(f"{self.endpoint}/api/tags")
            if response.status_code != 200:
                raise LLMHandlerError(f"Cannot connect to Ollama at {self.endpoint}")
        except ImportError:
            raise LLMHandlerError("Requests library not installed. Install with: pip install requests")

    def _init_transformers(self):
        """Initialize Transformers pipeline."""
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                device_map="auto" if self.config.get("use_gpu", True) else "cpu"
            )
        except ImportError:
            raise LLMHandlerError("Transformers library not installed. Install with: pip install transformers torch")

    def query(self, prompt: str) -> str:
        """Send query to local LLM."""
        self.total_queries += 1
        
        try:
            if self.provider == "ollama":
                return self._query_ollama(prompt)
            elif self.provider == "transformers":
                return self._query_transformers(prompt)
        except Exception as e:
            self.failed_queries += 1
            raise LLMHandlerError(f"Local LLM query failed: {e}")

    def _query_ollama(self, prompt: str) -> str:
        """Query Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.get("temperature", 0.7),
                "num_predict": self.config.get("max_tokens", 2000)
            }
        }
        
        response = self.session.post(
            f"{self.endpoint}/api/generate",
            json=payload,
            timeout=self.config.get("timeout", 300)
        )
        
        if response.status_code != 200:
            raise LLMHandlerError(f"Ollama API error: {response.status_code} - {response.text}")
        
        return response.json()["response"]

    def _query_transformers(self, prompt: str) -> str:
        """Query using Transformers pipeline."""
        result = self.pipeline(
            prompt,
            max_new_tokens=self.config.get("max_tokens", 2000),
            temperature=self.config.get("temperature", 0.7),
            do_sample=True,
            return_full_text=False
        )
        
        return result[0]["generated_text"]


def create_llm_handler(config: Dict[str, Any]) -> LLMHandler:
    """Factory function to create appropriate LLM handler."""
    handler_type = config.get("type", "").lower()
    
    if handler_type == "api":
        return APILLMHandler(config)
    elif handler_type == "local":
        return LocalLLMHandler(config)
    else:
        raise LLMHandlerError(f"Unknown handler type: {handler_type}")


def load_llm_config(config_path: str) -> Dict[str, Any]:
    """Load LLM configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Environment variable substitution for API keys
        if "api_key" in config and config["api_key"].startswith("$"):
            env_var = config["api_key"][1:]  # Remove $
            config["api_key"] = os.getenv(env_var)
            if not config["api_key"]:
                raise LLMHandlerError(f"Environment variable {env_var} not set")
        
        return config
    except FileNotFoundError:
        raise LLMHandlerError(f"Config file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise LLMHandlerError(f"Invalid JSON in config file: {e}")


# Backward compatibility aliases
OpenAIHandler = APILLMHandler
AnthropicHandler = APILLMHandler
OllamaHandler = LocalLLMHandler
