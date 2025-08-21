## Complete Guide: Creating a New AIChat Agent with Custom Tools

This guide will walk you through the process of setting up a new AI agent in AIChat, configuring it, and integrating custom tools to extend its capabilities.

### 1. Introduction to AIChat Agents and Tools

*   **What is an AIChat Agent?**
    An AIChat agent is a specialized AI assistant configured with specific instructions, a chosen language model, and a set of tools. It allows you to tailor the AI's persona and capabilities for particular tasks, such as coding assistance, research, or file management.

*   **What are Tools (Function Calling)?**
    Tools, also known as function calling, enable your AIChat agent to interact with external systems, execute commands, or access information beyond its core knowledge. These tools are typically shell scripts (Bash, Python, JavaScript) that the agent can invoke based on your natural language requests.

*   **Prerequisites:**
    *   A working Termux environment (or other Linux/macOS/Windows terminal).
    *   Basic familiarity with command-line operations.
    *   A Google AI Studio account (or other LLM provider) to obtain an API key for your chosen model (e.g., Gemini).

---

### 2. Step 1: Install AIChat (if not already installed)

If you don't have AIChat installed, you can typically install it using your system's package manager.

**For Termux (or other Linux/macOS with Homebrew):**

```bash
pkg install aichat # For Termux
brew install aichat # For Homebrew
```

**For other systems:** Refer to the official AIChat GitHub repository for installation instructions.

---

### 3. Step 2: Initial AIChat Configuration

The first time you run `aichat`, it will guide you through an initial setup and create its configuration directory and `config.yaml` file.

1.  **Run `aichat`:**
    ```bash
    aichat
    ```
    Follow the prompts to create a new config file.

2.  **Locate `config.yaml`:**
    The main configuration file is usually located at `~/.config/aichat/config.yaml` (on Linux/Termux). You can confirm its location by running:
    ```bash
    aichat --info | grep config_file
    ```

3.  **Add your LLM API Key:**
    Open `config.yaml` in a text editor and add your API key for your chosen LLM (e.g., Gemini).

    **Example for Gemini:**
    ```yaml
    # ~/.config/aichat/config.yaml
    model: gemini:gemini-2.5-flash # Set your preferred default model

    clients:
      - type: gemini
        api_key: YOUR_GEMINI_API_KEY_HERE
    ```
    Replace `YOUR_GEMINI_API_KEY_HERE` with your actual API key.

---

### 4. Step 3: Set Up LLM Functions Repository

AIChat uses the `llm-functions` repository to manage its tools. You need to clone this repository and link it to AIChat's functions directory.

1.  **Clone `llm-functions`:**
    Navigate to your home directory and clone the repository:
    ```bash
    cd ~
    git clone https://github.com/sigoden/llm-functions
    ```

2.  **Link `llm-functions` to AIChat's `functions_dir`:**
    This command creates a symbolic link from the cloned `llm-functions` directory into AIChat's designated functions directory, making the tools accessible.
    ```bash
    ln -s "$(pwd)/llm-functions" "$(aichat --info | sed -n 's/^functions_dir\s\+//p')"
    ```
    *(Note: If you cloned `llm-functions` into a different path, adjust `$(pwd)/llm-functions` accordingly.)*

---

### 5. Step 4: Create Your New Agent's Directory Structure

Each AIChat agent resides in its own directory within the `agents` folder of your AIChat configuration.

1.  **Navigate to AIChat's config directory:**
    ```bash
    cd "$(aichat --info | grep config_dir | awk '{print $2}')"
    ```

2.  **Create your agent's directory:**
    Replace `<your_agent_name>` with a unique name for your agent (e.g., `my_coding_agent`).
    ```bash
    mkdir -p agents/<your_agent_name>
    cd agents/<your_agent_name>
    ```

---

### 6. Step 5: Define Your Agent's `index.yaml`

The `index.yaml` file defines the core identity and instructions for your agent.

1.  **Create `index.yaml`** in your agent's directory (`~/.config/aichat/agents/<your_agent_name>/`).
    ```bash
    touch index.yaml
    ```

2.  **Add content to `index.yaml`:**
    Define your agent's `name`, `description`, `version`, `instructions` (its persona and primary directives), and `conversation_starters`.

    **Example `index.yaml`:**
    ```yaml
    # ~/.config/aichat/agents/<your_agent_name>/index.yaml
    name: MyCodingAgent
    description: A helpful AI assistant specialized in coding tasks.
    version: 1.0.0
    instructions: |
      You are a highly skilled coding assistant.
      Your primary goal is to help users write, debug, and understand code.
      You have access to various tools to interact with the file system and execute code.
      Always be precise, provide clear explanations, and suggest best practices.
      When asked to perform a task that requires a tool, use the appropriate tool.
    conversation_starters:
      - "How can you help me with coding today?"
      - "Can you write a Python script for me?"
      - "What files are in this directory?"
      - "Help me debug my JavaScript code."
    ```

---

### 7. Step 6: Configure Your Agent's `config.yaml`

This `config.yaml` file (within your agent's directory) overrides or extends the main AIChat `config.yaml` for this specific agent. It's where you enable tools.

1.  **Create `config.yaml`** in your agent's directory.
    ```bash
    touch config.yaml
    ```

2.  **Add content to `config.yaml`:**
    Specify the `model` this agent will use, `temperature`, `top_p`, enable `function_calling`, list the `use_tools` (tool groups or individual tools), and define `mapping_tools` for tool groups.

    **Example `config.yaml`:**
    ```yaml
    # ~/.config/aichat/agents/<your_agent_name>/config.yaml
    model: gemini:gemini-2.5-flash # Or your preferred model
    temperature: 0.7
    top_p: 0.9

    # Enable function calling for this agent
    function_calling: true

    # Specify which tool groups or individual tools this agent can use
    # 'fs' is a common group for file system operations
    # 'web_search' is for web searching
    # 'execute_command' for running shell commands
    # 'execute_py_code' for running Python code
    # 'execute_js_code' for running JavaScript code
    use_tools: fs,web_search,execute_command,execute_py_code,execute_js_code

    # Define what functions belong to a tool group (like 'fs')
    # These names correspond to the tool scripts in llm-functions/tools/
    mapping_tools:
      fs: 'fs_cat,fs_ls,fs_mkdir,fs_rm,fs_write'
    ```

---

### 8. Step 7: Create Custom Tools (Optional but Recommended)

You can create your own custom tools using Bash, Python, or JavaScript. These tools must follow the `argc` tool format.

1.  **Navigate to the `llm-functions/tools/` directory:**
    ```bash
    cd ~/llm-functions/tools/
    ```

2.  **Create your tool file:**
    For example, `my_python_tool.py`:

    ```python
    #!/usr/bin/env python3
    # my_python_tool.py
    import sys
    import json

    # @cmd Echoes a message.
    # @arg message!: The message to echo.
    def run(message: str):
        """
        Echoes a given message back to the user.
        """
        print(json.dumps({"output": f"You said: {message}"}))

    if __name__ == "__main__":
        # This part is for argc to execute the tool
        # It typically involves parsing arguments and calling the run function
        # For simple tools, argc handles much of this automatically
        # You might need to add argc's boilerplate if not using its direct decorators
        pass
    ```
    *(Note: For Python tools, `argc` expects a `run` function. For more complex argument parsing, you'd use `argc`'s decorators and boilerplate. The example above is simplified.)*

3.  **Make your tool executable:**
    ```bash
    chmod +x my_python_tool.py
    ```

---

### 9. Step 8: Update `llm-functions/tools.txt`

The `tools.txt` file in the `llm-functions` repository tells `argc build` which tools to compile into executables and include in `functions.json`.

1.  **Navigate to the `llm-functions` root directory:**
    ```bash
    cd ~/llm-functions/
    ```

2.  **Edit `tools.txt`:**
    Open `tools.txt` and add the filename of your new tool (e.g., `my_python_tool.py`) to the list. Ensure all tools you want your agent to use are listed here.

    **Example `tools.txt` content:**
    ```
    fs_cat.sh
    fs_ls.sh
    fs_mkdir.sh
    fs_rm.sh
    fs_write.sh
    web_search.sh
    execute_command.sh
    execute_py_code.py
    execute_js_code.js
    my_python_tool.py # Your new tool
    ```

---

### 10. Step 9: Build the Agent's Tools (`argc build`)

After making changes to `tools.txt` or any tool files, you must run `argc build` to generate the `functions.json` file and the executable tool binaries.

1.  **Navigate to the `llm-functions` root directory:**
    ```bash
    cd ~/llm-functions/
    ```

2.  **Run `argc build`:**
    ```bash
    argc build
    ```
    This command will:
    *   Create a `bin/` directory containing executable versions of your tools.
    *   Generate a `functions.json` file, which is a JSON schema describing your tools that AIChat (and the LLM) uses to understand how to call them.

---

### 11. Step 10: Test Your New Agent

Now that your agent is configured and its tools are built, it's time to test it!

1.  **Launch your agent:**
    ```bash
    aichat -a <your_agent_name>
    ```
    You will enter an interactive REPL session with your agent.

2.  **Interact with your agent to trigger tool calls:**
    *   **To test `fs_ls`:** Ask, "List files in the current directory."
    *   **To test `web_search`:** Ask, "Search for the latest news on AI."
    *   **To test `my_python_tool` (if you created it):** Ask, "Echo the message 'Hello Agent!'"

    Observe the agent's responses. If a tool is called, AIChat will often show a "Tool Call:" message before the agent's response.

---

### 12. Troubleshooting Common Issues

*   **`error: unexpected argument '--test' found`**:
    This means you're trying to use `--test` directly with `aichat -a <agent_name>`. The `--test` flag is for testing the main `aichat` configuration or specific tools, not for agents directly. Test agents by interacting with them in the REPL.

*   **`command not found` errors (e.g., `web_search.sh: command not found`)**:
    This usually means the tool's executable is not in the system's PATH or not correctly linked to AIChat's `functions_dir`.
    *   Ensure you ran `argc build` successfully in the `llm-functions` directory.
    *   Verify that the `llm-functions` directory is correctly symlinked into `~/.config/aichat/functions/`.
    *   Check that the tool's filename is correctly listed in `~/llm-functions/tools.txt`.

*   **`argc build` errors (e.g., `@describe(line X) shouldn't be here, @cmd is missing?`)**:
    This indicates a syntax error or incorrect formatting within one of your tool files (e.g., a Python script not conforming to `argc`'s expected tool structure). Review the error message to identify the problematic file and line number, then correct the tool's code.

---

By following these steps, you can successfully create and configure new AIChat agents with custom tools, empowering your AI assistant to perform a wide range of tasks directly from your terminal.