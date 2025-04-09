import logging
import os
import json
import requests
import time
import random
from typing import Dict, Any, Optional, List, Union

from openai import OpenAI
import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__)) 
LOGS_DIR = os.path.join(PROJECT_ROOT, "data", "logs")
RAW_LOGS_DIR = os.path.join(PROJECT_ROOT, "data", "raw_llm_logs")

os.makedirs(RAW_LOGS_DIR, exist_ok=True)  # Ensure subdirectory exists

def log_llm_call(provider: str, model: str, request_data: Dict[str, Any], response_data: Dict[str, Any]) -> str:
    """
    Log LLM API requests and responses to a file.
    
    Args:
        provider: The LLM provider name
        model: The model name
        request_data: The API request data
        response_data: The API response data
        
    Returns:
        Path to the log file
    """
    safe_provider = provider.replace("/", "_")
    safe_model = model.replace("/", "_")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = os.path.join(RAW_LOGS_DIR, f"{safe_provider}_{safe_model}_{timestamp}.json")

    log_entry = {
        "timestamp": timestamp,
        "provider": provider,
        "model": model,
        "request": request_data,
        "response": response_data
    }

    try:
        with open(log_filename, "w") as log_file:
            json.dump(log_entry, log_file, indent=4, default=str)
        return log_filename
    except Exception as e:
        logger.error(f"Failed to write LLM log: {e}")
        return ""

class RetryStrategy:
    """
    Implements various retry strategies for API calls.
    """
    
    @staticmethod
    def exponential_backoff(retry_count: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """
        Exponential backoff strategy with jitter.
        
        Args:
            retry_count: The current retry attempt (0-based)
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            
        Returns:
            Delay time in seconds before next retry
        """
        delay = min(max_delay, base_delay * (2 ** retry_count))
        # Add jitter (±20%)
        jitter = delay * 0.2
        delay = delay + random.uniform(-jitter, jitter)
        return delay

class LLMClient:
    """
    Client for interfacing with various LLM providers.
    Handles API calls, retries, and error handling.
    """

    REASONING_MODELS={
        "o3-mini-2025-01-31"
        ,"o1-2024-12-17"
        ,"o1-mini-2024-09-12"
    }
    
    def __init__(
        self, 
        provider: str = "openai", 
        model: str = "gpt-4o", 
        max_tokens: int = 1000, 
        temperature: float = 1.0,
        max_retries: int = 3,
        timeout: int = 60
    ):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider name ("openai", "anthropic", "openrouter")
            model: Model name to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (higher = more random)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = self._initialize_client()

    def _initialize_client(self):
        """
        Initialize the API client based on provider.
        
        Returns:
            Provider-specific client
        
        Raises:
            ValueError: If provider is unsupported or missing API key
        """
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_GAMEBENCH_KEY")
            if not api_key:
                raise ValueError("Missing OpenAI API key. Set OPENAI_GAMEBENCH_KEY in your environment variables.")
            return OpenAI(api_key=api_key)
            
        elif self.provider == "anthropic":
            api_key = os.getenv("CLAUDE_GAMEBENCH_KEY")
            if not api_key:
                raise ValueError("Missing Claude API key. Set CLAUDE_GAMEBENCH_KEY in your environment variables.")
            return anthropic.Anthropic(api_key=api_key)
            
        elif self.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_GAMEBENCH_KEY")
            if not api_key:
                raise ValueError("Missing OpenRouter API key. Set OPENROUTER_GAMEBENCH_KEY in your environment variables.")
            return api_key  # OpenRouter does not require an instantiated client
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def call_llm(
        self, 
        developer_message: str, 
        user_message: str, 
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call the LLM with retry logic.
        
        Args:
            developer_message: Context for the model (not shown to end users)
            user_message: The user's message/query
            system_message: Optional system message to set behavior
            
        Returns:
            Response data from the LLM or None if all retries failed
        """
        request_data = {
            "developer_message": developer_message,
            "user_message": user_message,
            "system_message": system_message,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        # Implement retry logic
        for retry_attempt in range(self.max_retries + 1):
            try:
                if retry_attempt > 0:
                    # Calculate backoff time
                    delay = RetryStrategy.exponential_backoff(retry_attempt - 1)
                    logger.info(f"Retry attempt {retry_attempt}/{self.max_retries} after {delay:.2f}s delay")
                    time.sleep(delay)
                
                # Call appropriate provider
                if self.provider == "openai":
                    response = self._call_openai(developer_message, user_message, system_message)
                elif self.provider == "anthropic":
                    response = self._call_anthropic(developer_message, user_message, system_message)
                elif self.provider == "openrouter":
                    response = self._call_openrouter(developer_message, user_message, system_message)
                else:
                    logger.error(f"Unsupported provider: {self.provider}")
                    return None
                
                # Log the successful call
                log_llm_call(self.provider, self.model, request_data, response)
                return response
                
            except Exception as e:
                logger.warning(f"Call to {self.provider} failed (attempt {retry_attempt+1}/{self.max_retries+1}): {str(e)}")
                # If this was the last retry, log the failure and return None
                if retry_attempt == self.max_retries:
                    logger.error(f"All retry attempts to {self.provider} failed: {str(e)}")
                    return None
        
        # This should not be reached due to the return in the exception handler
        return None
    
    def _supports_reasoning(self) -> bool:
        """Return True if the current model is an OpenAI reasoning-capable model."""
        return self.provider == "openai" and self.model in self.REASONING_MODELS

    def _call_openai(
        self, 
        developer_message: str, 
        user_message: str, 
        system_message: Optional[str]
    ) -> Dict[str, Any]:
        if self._supports_reasoning():
            return self._call_openai_reasoning(developer_message, user_message)
        else:
            return self._call_openai_chat(developer_message, user_message, system_message)

    def _call_openai_chat(
        self,
        developer_message: str,
        user_message: str,
        system_message: Optional[str]
    ) -> Dict[str, Any]:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": f"Developer Context: {developer_message}\n\nUser Message: {user_message}"})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            return response.model_dump()
        except Exception as e:
            logger.error(f"Error calling OpenAI Chat API: {str(e)}")
            raise

    def _call_openai_reasoning(
        self,
        developer_message: str,
        user_message: str
    ) -> Dict[str, Any]:
        full_prompt = f"Developer Context: {developer_message}\n\nUser Message: {user_message}"
        try:
            response = self.client.responses.create(
                model=self.model,
                input=full_prompt,
                reasoning={"effort": "high"},
                max_output_tokens=None
            )
            return response.model_dump()
        except Exception as e:
            logger.error(f"Error calling OpenAI Reasoning API: {str(e)}")
            raise

    def _call_anthropic(
        self, 
        developer_message: str, 
        user_message: str, 
        system_message: Optional[str]
    ) -> Dict[str, Any]:
        """
        Call the Anthropic (Claude) API.
        
        Args:
            developer_message: Context for the model
            user_message: User's message
            system_message: Optional system message
            
        Returns:
            Anthropic API response
            
        Raises:
            Exception: On API error
        """
        try:
            # Use the system message if provided, otherwise create a default one
            system_prompt = system_message if system_message else "You are a helpful AI assistant playing a game."
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Developer Context: {developer_message}\n\nUser Message: {user_message}"
                            }
                        ]
                    }
                ]
            )
            
            # Convert response to a dict for logging
            response_dict = {
                "id": response.id,
                "model": response.model,
                "content": response.content,
                "type": response.type,
                "role": response.role,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "stop_reason": response.stop_reason
            }
            
            return response_dict
            
        except Exception as e:
            logger.error(f"Error calling Claude API: {str(e)}")
            raise

    def _call_openrouter(
        self, 
        developer_message: str, 
        user_message: str, 
        system_message: Optional[str]
    ) -> Dict[str, Any]:
        """
        Call the OpenRouter API.
        
        Args:
            developer_message: Context for the model
            user_message: User's message
            system_message: Optional system message
            
        Returns:
            OpenRouter API response
            
        Raises:
            Exception: On API error
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.client}",
                "HTTP-Referer": "https://gamebench.ai",
                "X-Title": "GameBench",
                "Content-Type": "application/json"
            }
            
            messages = []
            
            # Add system message if provided
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            # Add developer context and user message
            messages.append({"role": "user", "content": f"Developer Context: {developer_message}\n\nUser Message: {user_message}"})
            
            data = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": messages
            }
            
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            
            # Raise exception for HTTP errors
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling OpenRouter API: {str(e)}")
            raise


def parse_response_text(response_json: Dict[str, Any]) -> Optional[str]:
    """
    Safely parse the assistant's text content from different response formats.
    
    Args:
        response_json: The response JSON from the LLM
        
    Returns:
        Extracted text content or None if parsing fails
    """
    if not response_json:
        return None

    # Track parsing attempts for debugging
    parsing_attempts = []

    # 🆕 Reasoning API format: OpenAI /v1/responses output block
    if "output" in response_json and isinstance(response_json["output"], list):
        for block in response_json["output"]:
            if block.get("type") == "message":
                contents = block.get("content", [])
                for item in contents:
                    if isinstance(item, dict) and item.get("type") == "output_text":
                        parsing_attempts.append({"method": "reasoning_output_text", "success": True})
                        return item.get("text")

    # Check for OpenAI style: "choices" key with "message"
    if "choices" in response_json:
        try:
            # Standard OpenAI chat completion format
            content = response_json["choices"][0]["message"]["content"]
            parsing_attempts.append({"method": "openai_choices", "success": True})
            return content
        except (KeyError, IndexError, TypeError) as e:
            parsing_attempts.append({"method": "openai_choices", "error": str(e)})
            pass
    
    # Check for newer OpenAI style: "content" within "completion"
    if "completion" in response_json:
        try:
            content = response_json["completion"]
            parsing_attempts.append({"method": "openai_completion", "success": True})
            return content
        except (KeyError, TypeError) as e:
            parsing_attempts.append({"method": "openai_completion", "error": str(e)})
            pass
            
    # Check for OpenAI style: "output" key
    if "output" in response_json:
        try:
            # Attempt to parse text from the first output chunk
            content = response_json["output"][0]["content"][0]["text"]
            parsing_attempts.append({"method": "openai_output", "success": True})
            return content
        except (KeyError, IndexError, TypeError) as e:
            parsing_attempts.append({"method": "openai_output", "error": str(e)})
            pass

    # Check for Claude style: "content" with list of contents
    if "content" in response_json:
        try:
            # Handle both string content and content objects
            if isinstance(response_json["content"], list):
                # If content is a list of content blocks
                for content_block in response_json["content"]:
                    if hasattr(content_block, "model_dump"):
                        # Convert to a dict if it's a pydantic model
                        block_dict = content_block.model_dump()
                        if block_dict.get("type") == "text":
                            parsing_attempts.append({"method": "claude_content_list_model", "success": True})
                            return block_dict.get("text")
                    elif isinstance(content_block, dict):
                        # If it's already a dict
                        if content_block.get("type") == "text":
                            parsing_attempts.append({"method": "claude_content_list_dict", "success": True})
                            return content_block.get("text")
            else:
                # If content is directly accessible
                parsing_attempts.append({"method": "claude_content_direct", "success": True})
                return response_json["content"]
        except (KeyError, TypeError) as e:
            parsing_attempts.append({"method": "claude_content", "error": str(e)})
            pass
            
    # Log parsing failure with details of attempts
    logger.warning(f"Failed to parse response text. Attempts: {json.dumps(parsing_attempts)}")
    logger.debug(f"Response structure: {json.dumps(list(response_json.keys()))}")
    
    return None


class PromptTemplate:
    """
    Template system for generating prompts with variable substitution.
    """
    
    def __init__(self, template_text: str):
        """
        Initialize with a template string.
        
        Args:
            template_text: The template text with {placeholders}
        """
        self.template = template_text
        
    def format(self, **kwargs) -> str:
        """
        Fill the template with provided variables.
        
        Args:
            **kwargs: Key-value pairs for template variables
            
        Returns:
            Formatted prompt string
            
        Raises:
            KeyError: If a required template variable is missing
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise KeyError(f"Missing required template variable: {missing_key}")


# Example usage
if __name__ == "__main__":
    # Example with OpenRouter
    openrouter_client = LLMClient(
        provider="openrouter", 
        model="meta-llama/llama-3-8b-instruct",
        max_tokens=1000,
        temperature=0.7,
        max_retries=2
    )
    
    openrouter_response = openrouter_client.call_llm(
        developer_message="Let's play a game. I'm thinking of something, and you have 20 yes/no questions to figure out what it is.",
        user_message="What's your first question?"
    )
    
    print("OpenRouter response:", parse_response_text(openrouter_response))
    
    # Example with Claude
    claude_client = LLMClient(
        provider="anthropic", 
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.7
    )
    
    claude_response = claude_client.call_llm(
        developer_message="Let's play a game. I'm thinking of something, and you have 20 yes/no questions to figure out what it is.",
        user_message="What's your first question?",
        system_message="You are a world-class game player. Ask strategic yes/no questions."
    )
    
    print("Claude response:", parse_response_text(claude_response))
    
    # Example with OpenAI
    openai_client = LLMClient(
        provider="openai", 
        model="gpt-4o-mini",
        max_tokens=1000,
        temperature=0.7
    )
    
    openai_response = openai_client.call_llm(
        developer_message="Let's play a game. I'm thinking of something, and you have 20 yes/no questions to figure out what it is.",
        user_message="What's your first question?"
    )
    
    print("OpenAI response:", parse_response_text(openai_response))
    
    # Example of prompt template usage
    game_template = PromptTemplate("""
    You are playing {game_name}.
    
    Current state:
    {game_state}
    
    Your options:
    {valid_actions}
    
    What would you like to do?
    """)
    
    formatted_prompt = game_template.format(
        game_name="Poker",
        game_state="You have a pair of aces. There's $100 in the pot.",
        valid_actions="1. Check\n2. Bet\n3. Fold"
    )
    
    print("\nTemplate example:", formatted_prompt)