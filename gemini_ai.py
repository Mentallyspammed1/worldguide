
"""
Enhanced Gemini AI Integration Module
With advanced features, colorful output, and robust error handling
"""
from colorama import init, Fore, Style, Back
import google.generativeai as genai
import os
import json
import asyncio
from typing import Optional, Dict, List, Union, Any
from datetime import datetime

# Initialize colorama with autoreset
init(autoreset=True)

class GeminiAI:
    def __init__(self, model_name: str = 'gemini-pro'):
        """Initialize Gemini AI with enhanced configuration"""
        self._setup_logging()
        self.api_key = self._get_api_key()
        self.model_name = model_name
        self.history: List[Dict[str, Any]] = []
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
            print(f"{Fore.GREEN}# Gemini AI initialized with model: {Style.BRIGHT}{model_name}{Style.RESET_ALL}")
        else:
            raise ValueError(f"{Fore.RED}Error: GEMINI_API_KEY not found in environment variables{Style.RESET_ALL}")

    def _setup_logging(self) -> None:
        """Configure enhanced logging with timestamps"""
        self.log_file = "gemini_ai.log"
        print(f"{Fore.CYAN}# Arcane logs will be channeled to: {self.log_file}{Style.RESET_ALL}")

    def _get_api_key(self) -> Optional[str]:
        """Retrieve API key with enhanced security checks"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print(f"{Back.RED}{Fore.WHITE}Warning: GEMINI_API_KEY not found!{Style.RESET_ALL}")
        return api_key

    async def generate_text(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Generate text with enhanced error handling and retries"""
        for attempt in range(max_retries):
            try:
                print(f"{Fore.CYAN}# Channeling AI wisdom... {Style.DIM}(Attempt {attempt + 1}/{max_retries}){Style.RESET_ALL}")
                response = await self.model.generate_content_async(prompt)
                
                # Log successful generation
                self._log_interaction("generate_text", prompt, response.text)
                
                print(f"{Fore.GREEN}# Response successfully crystallized{Style.RESET_ALL}")
                return response.text

            except Exception as e:
                print(f"{Fore.RED}Error on attempt {attempt + 1}: {str(e)}{Style.RESET_ALL}")
                if attempt == max_retries - 1:
                    self._log_error("generate_text", str(e), prompt)
                    return None
                await asyncio.sleep(1)  # Backoff between retries

    async def chat(self, 
                  messages: Union[str, List[Dict[str, str]]], 
                  temperature: float = 0.7,
                  max_retries: int = 3) -> Optional[str]:
        """Enhanced chat interface with temperature control and history"""
        try:
            print(f"{Fore.CYAN}# Initiating mystical dialogue... {Style.DIM}(Temperature: {temperature}){Style.RESET_ALL}")
            
            # Handle both string and structured messages
            if isinstance(messages, str):
                chat_messages = [{"role": "user", "content": messages}]
            else:
                chat_messages = messages

            chat = self.model.start_chat(history=self.history)
            
            for attempt in range(max_retries):
                try:
                    response = await chat.send_message_async(
                        chat_messages,
                        generation_config={"temperature": temperature}
                    )
                    
                    # Update chat history
                    self.history.extend([
                        {"role": "user", "content": str(chat_messages)},
                        {"role": "assistant", "content": response.text}
                    ])
                    
                    print(f"{Fore.GREEN}# Mystical response manifested{Style.RESET_ALL}")
                    return response.text

                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(1)

        except Exception as e:
            self._log_error("chat", str(e), str(messages))
            print(f"{Fore.RED}Error in mystical dialogue: {str(e)}{Style.RESET_ALL}")
            return None

    def _log_interaction(self, 
                        interaction_type: str, 
                        prompt: str, 
                        response: str) -> None:
        """Log successful interactions with timestamp"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "type": interaction_type,
            "prompt": prompt,
            "response": response
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _log_error(self, 
                   operation: str, 
                   error: str, 
                   context: str) -> None:
        """Log errors with detailed context"""
        timestamp = datetime.now().isoformat()
        error_entry = {
            "timestamp": timestamp,
            "operation": operation,
            "error": error,
            "context": context
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(error_entry) + "\n")

    async def clear_history(self) -> None:
        """Clear chat history"""
        self.history = []
        print(f"{Fore.YELLOW}# Chat history cleared{Style.RESET_ALL}")

# Example usage
async def example():
    ai = GeminiAI()
    
    # Generate text example
    response = await ai.generate_text("Explain quantum computing in one paragraph")
    print(f"{Fore.YELLOW}Generated Response:{Style.RESET_ALL}\n{response}\n")
    
    # Chat example
    chat_response = await ai.chat("What's the best way to learn Python?")
    print(f"{Fore.YELLOW}Chat Response:{Style.RESET_ALL}\n{chat_response}")

if __name__ == "__main__":
    asyncio.run(example())
