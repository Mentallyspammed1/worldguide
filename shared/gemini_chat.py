
import os
import google.generativeai as genai
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

def main():
    """Main function to run the interactive Gemini chat."""
    # Retrieve the API key from memory/environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Fallback to the key I have in memory if env var is not set
        api_key = "AIzaSyAZjOB8vDjiTHtkNJRXNreWNovBaiuX0qw"

    if not api_key:
        print(Fore.RED + "Error: GEMINI_API_KEY not found. Please set it as an environment variable.")
        return

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        chat = model.start_chat(history=[])

        print(Fore.CYAN + Style.BRIGHT + "Initiating communion with Gemini. Type 'exit' or 'quit' to sever the connection.")

        while True:
            user_input = input(Fore.GREEN + Style.BRIGHT + "\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print(Fore.MAGENTA + "The connection is severed. Farewell!")
                break

            print(Fore.YELLOW + Style.BRIGHT + "Gemini: ", end="")
            
            response_stream = chat.send_message(user_input, stream=True)

            for chunk in response_stream:
                # Ensure the chunk has text content before printing
                if hasattr(chunk, 'text') and chunk.text:
                    print(Fore.YELLOW + chunk.text, end="", flush=True)
            
            print() # for a new line after the assistant's response

    except KeyboardInterrupt:
        print(Fore.MAGENTA + "\n\nThe connection was abruptly severed. Farewell!")
    except Exception as e:
        # Catch potential configuration or API errors
        print(Fore.RED + f"\nA disturbance in the ether has occurred: {e}")

if __name__ == "__main__":
    main()
