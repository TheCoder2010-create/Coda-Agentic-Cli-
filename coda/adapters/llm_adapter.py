"""
Multi-model LLM adapter supporting OpenAI, Anthropic, and Ollama
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
# import tiktoken  # Disabled due to dependency issues

# Integration references for OpenAI and Anthropic APIs
# The newest OpenAI model is "gpt-5" which was released August 7, 2025.
# Do not change this unless explicitly requested by the user
# The newest Anthropic model is "claude-sonnet-4-20250514"

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float = 0.0

@dataclass
class LLMResponse:
    content: str
    token_usage: TokenUsage
    model: str
    provider: str

class LLMAdapter:
    """Unified interface for multiple LLM providers"""
    
    def __init__(self):
        self.providers = {}
        self._setup_providers()
    
    def _setup_providers(self):
        """Initialize available LLM providers"""
        # OpenAI
        if os.getenv('OPENAI_API_KEY'):
            from openai import OpenAI
            self.providers['openai'] = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Anthropic
        if os.getenv('ANTHROPIC_API_KEY'):
            from anthropic import Anthropic
            self.providers['anthropic'] = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # Ollama (local)
        self.providers['ollama'] = 'http://localhost:11434'  # Default Ollama endpoint
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        available = []
        if 'openai' in self.providers:
            available.append('openai')
        if 'anthropic' in self.providers:
            available.append('anthropic')
        if self._check_ollama_available():
            available.append('ollama')
        return available
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running locally"""
        try:
            response = requests.get(f"{self.providers['ollama']}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _estimate_cost(self, provider: str, model: str, token_usage: TokenUsage) -> float:
        """Estimate cost based on provider and model"""
        # Cost estimates per 1k tokens (approximate as of 2025)
        cost_map = {
            'openai': {
                'gpt-5': {'input': 0.03, 'output': 0.06},
                'gpt-4o': {'input': 0.015, 'output': 0.03}
            },
            'anthropic': {
                'claude-sonnet-4-20250514': {'input': 0.015, 'output': 0.075},
                'claude-3-7-sonnet-20250219': {'input': 0.015, 'output': 0.075}
            },
            'ollama': {
                'default': {'input': 0.0, 'output': 0.0}  # Local models are free
            }
        }
        
        if provider not in cost_map:
            return 0.0
        
        model_costs = cost_map[provider].get(model, cost_map[provider].get('default', {'input': 0.0, 'output': 0.0}))
        
        input_cost = (token_usage.prompt_tokens / 1000) * model_costs['input']
        output_cost = (token_usage.completion_tokens / 1000) * model_costs['output']
        
        return input_cost + output_cost
    
    def _count_tokens(self, text: str, model: str = "gpt-5") -> int:
        """Count tokens in text using simple estimation"""
        # Simple token estimation: roughly 1.3 tokens per word
        # This is less accurate but avoids dependency issues
        return int(len(text.split()) * 1.3)
    
    def chat(self, messages: List[Dict[str, str]], provider: str = 'openai', 
             model: Optional[str] = None, **kwargs) -> LLMResponse:
        """Send chat messages to specified provider"""
        
        if provider not in self.get_available_providers():
            raise ValueError(f"Provider {provider} not available. Available: {self.get_available_providers()}")
        
        # Set default models
        if not model:
            if provider == 'openai':
                model = 'gpt-5'  # newest OpenAI model
            elif provider == 'anthropic':
                model = 'claude-sonnet-4-20250514'  # newest Anthropic model
            elif provider == 'ollama':
                model = 'llama3'  # Common local model
            else:
                model = 'gpt-5'  # fallback default
        
        # Count input tokens
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        prompt_tokens = self._count_tokens(input_text, model)
        
        if provider == 'openai':
            return self._chat_openai(messages, model, prompt_tokens, **kwargs)
        elif provider == 'anthropic':
            return self._chat_anthropic(messages, model, prompt_tokens, **kwargs)
        elif provider == 'ollama':
            return self._chat_ollama(messages, model, prompt_tokens, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _chat_openai(self, messages: List[Dict[str, str]], model: str, 
                     prompt_tokens: int, **kwargs) -> LLMResponse:
        """Chat with OpenAI API"""
        client = self.providers['openai']
        
        # Remove temperature for gpt-5 as it doesn't support it
        chat_kwargs = {k: v for k, v in kwargs.items() if k != 'temperature' or not model.startswith('gpt-5')}
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **chat_kwargs
        )
        
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=self._estimate_cost('openai', model, TokenUsage(prompt_tokens, completion_tokens, total_tokens))
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            token_usage=token_usage,
            model=model,
            provider='openai'
        )
    
    def _chat_anthropic(self, messages: List[Dict[str, str]], model: str, 
                        prompt_tokens: int, **kwargs) -> LLMResponse:
        """Chat with Anthropic API"""
        client = self.providers['anthropic']
        
        # Convert messages format for Anthropic
        system_message = ""
        chat_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message += msg['content'] + "\n"
            else:
                chat_messages.append(msg)
        
        response = client.messages.create(
            model=model,
            system=system_message.strip() if system_message else None,
            messages=chat_messages,
            max_tokens=kwargs.get('max_tokens', 4000),
            **{k: v for k, v in kwargs.items() if k not in ['max_tokens']}
        )
        
        completion_tokens = response.usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens
        
        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=self._estimate_cost('anthropic', model, TokenUsage(prompt_tokens, completion_tokens, total_tokens))
        )
        
        return LLMResponse(
            content=response.content[0].text,
            token_usage=token_usage,
            model=model,
            provider='anthropic'
        )
    
    def _chat_ollama(self, messages: List[Dict[str, str]], model: str, 
                     prompt_tokens: int, **kwargs) -> LLMResponse:
        """Chat with Ollama local API"""
        url = f"{self.providers['ollama']}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            content = result['message']['content']
            completion_tokens = self._count_tokens(content, model)
            total_tokens = prompt_tokens + completion_tokens
            
            token_usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost=0.0  # Local models are free
            )
            
            return LLMResponse(
                content=content,
                token_usage=token_usage,
                model=model,
                provider='ollama'
            )
        
        except Exception as e:
            raise Exception(f"Ollama API error: {e}")