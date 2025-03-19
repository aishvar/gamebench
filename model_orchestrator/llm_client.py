import logging
import os
import json
import requests
from openai import OpenAI
import anthropic
import time

# Base logs directory
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "logs")

# Subdirectory for raw LLM logs
RAW_LOGS_DIR = os.path.join(LOGS_DIR, "raw_llm_logs")

os.makedirs(RAW_LOGS_DIR, exist_ok=True)  # Ensure subdirectory exists

def log_llm_call(provider, model, request_data, response_data):
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
    except Exception as e:
        logging.error(f"Failed to write LLM log: {e}")

class LLMClient:
    def __init__(self, provider="openai", model="gpt-4o", max_tokens=1000, temperature=1):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = self._initialize_client()

    def _initialize_client(self):
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

    def call_llm(self, developer_message, user_message, system_message=None):
        request_data = {
            "developer_message": developer_message,
            "user_message": user_message,
            "system_message": system_message,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        # OpenAI Provider
        if self.provider == "openai":
            try:
                response = self.client.responses.create(
                    model=self.model,
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                    input=[
                        {"role": "developer", "content": developer_message},
                        {"role": "user", "content": user_message}
                    ]
                )
                response_json = response.model_dump()  # Full response as JSON
                log_llm_call(self.provider, self.model, request_data, response_json)
                return response_json
            except Exception as e:
                logging.error(f"Error calling OpenAI API: {e}")
                return None
                
        # Anthropic (Claude) Provider
        elif self.provider == "anthropic":
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
                response_json = {
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
                log_llm_call(self.provider, self.model, request_data, response_json)
                return response_json
            except Exception as e:
                logging.error(f"Error calling Claude API: {e}")
                return None

        # OpenRouter Provider
        elif self.provider == "openrouter":
            try:
                headers = {
                    "Authorization": f"Bearer {self.client}",
                    "HTTP-Referer": "https://your-site.com",
                    "X-Title": "GameBench",
                }
                data = json.dumps({
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": [
                        {"role": "developer", "content": developer_message},
                        {"role": "user", "content": user_message}
                    ]
                })
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    data=data
                )
                response_json = response.json()
                log_llm_call(self.provider, self.model, request_data, response_json)
                return response_json
            except Exception as e:
                logging.error(f"Error calling OpenRouter API: {e}")
                return None

        else:
            logging.error(f"LLM provider {self.provider} is not implemented.")
            return None

def parse_response_text(response_json):
    """
    Safely parse the assistant's text content from
    either an OpenAI-style response, OpenRouter-style response, or Claude-style response.
    """
    if not response_json:
        return None

    # Check for OpenAI style: "output" key
    if "output" in response_json:
        # Attempt to parse text from the first output chunk
        try:
            return response_json["output"][0]["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            pass

    # Check for OpenRouter style: "choices" key
    if "choices" in response_json:
        try:
            return response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
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
                            return block_dict.get("text")
                    elif isinstance(content_block, dict):
                        # If it's already a dict
                        if content_block.get("type") == "text":
                            return content_block.get("text")
            else:
                # If content is directly accessible
                return response_json["content"]
        except (KeyError, TypeError):
            pass

    return None

# Example usage
if __name__ == "__main__":
    # Example with OpenRouter
    openrouter_client = LLMClient(
        provider="openrouter", 
        model="deepseek/deepseek-r1:free",
        max_tokens=500,
        temperature=0.7
    )
    openrouter_response = openrouter_client.call_llm(
        developer_message="Let's play a game. I'm thinking of something, and you have 20 yes/no questions to figure out what it is.",
        user_message="What's your first question?"
    )
    print("OpenRouter response:", parse_response_text(openrouter_response))
    
    # Example with Claude
    claude_client = LLMClient(
        provider="anthropic", 
        model="claude-3-5-haiku-20241022",
        max_tokens=500,
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
        max_tokens=500,
        temperature=0.7
    )
    openai_response = openai_client.call_llm(
        developer_message="Let's play a game. I'm thinking of something, and you have 20 yes/no questions to figure out what it is.",
        user_message="What's your first question?"
    )
    print("OpenAI response:", parse_response_text(openai_response))