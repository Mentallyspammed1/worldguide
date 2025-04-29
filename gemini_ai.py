
"""
Gemini AI Integration Module
Enhanced with Colorama for vibrant output styling
"""
from colorama import init, Fore, Style
import google.generativeai as genai
import os

init(autoreset=True)

class GeminiAI:
    def __init__(self):
        """Initialize Gemini AI with API key"""
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            print(Fore.RED + "Error: GEMINI_API_KEY not found in environment variables")
            return
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        print(Fore.GREEN + "# Gemini AI initialized successfully" + Style.RESET_ALL)

    async def generate_text(self, prompt):
        """Generate text using Gemini AI"""
        try:
            print(Fore.CYAN + "# Channeling the AI wisdom..." + Style.RESET_ALL)
            response = await self.model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            print(Fore.RED + f"Error generating response: {e}" + Style.RESET_ALL)
            return None

    async def chat(self, messages):
        """Have a chat conversation with Gemini AI"""
        try:
            print(Fore.CYAN + "# Initiating mystical dialogue..." + Style.RESET_ALL)
            chat = self.model.start_chat()
            response = await chat.send_message_async(messages)
            return response.text
        except Exception as e:
            print(Fore.RED + f"Error in chat: {e}" + Style.RESET_ALL)
            return None
