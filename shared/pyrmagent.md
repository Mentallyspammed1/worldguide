Based on my research, I'll create a complete tutorial for setting up and using an agent with gemini-2.5-flash in aichat by sigoden.

# Complete Tutorial: Setting Up Gemini 2.5 Flash Agent in AIChat

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Basic Configuration](#basic-configuration)
5. [Setting Up Gemini 2.5 Flash](#setting-up-gemini-25-flash)
6. [Creating Your First Agent](#creating-your-first-agent)
7. [Advanced Agent Configuration](#advanced-agent-configuration)
8. [Using Tools and Functions](#using-tools-and-functions)
9. [RAG Integration](#rag-integration)
10. [Practical Examples](#practical-examples)
11. [Troubleshooting](#troubleshooting)

## Introduction

AIChat is an all-in-one LLM CLI tool featuring Shell Assistant, Chat-REPL, RAG, AI Tools & Agents, with access to OpenAI, Claude, Gemini, Ollama, Groq, and more. AI Agent = Instructions (Prompt) + Tools (Function Callings) + Documents (RAG).

Gemini 2.5 Flash is our workhorse thinking model, ideal for fast performance on everyday tasks. This tutorial will guide you through setting up and using Gemini 2.5 Flash with AIChat's agent system.

## Prerequisites

Before starting, ensure you have:

1. **Operating System**: macOS, Linux, or Windows
2. **Google AI Studio Account**: For obtaining Gemini API key
3. **Terminal/Command Line**: Basic familiarity with command-line operations
4. **Git** (optional): For cloning repositories

## Installation

### Step 1: Install AIChat

Download pre-built binaries for macOS, Linux, and Windows from GitHub Releases, extract them, and add the aichat binary to your $PATH.

```bash
# For macOS using Homebrew
brew install sigoden/tap/aichat

# For Linux/macOS manual installation
curl -LsSf https://github.com/sigoden/aichat/releases/latest/download/aichat-linux-x64.tar.gz | tar xz
sudo mv aichat /usr/local/bin/

# For Windows
# Download the Windows binary from GitHub releases and add to PATH
```

### Step 2: Verify Installation

```bash
aichat --version
aichat --info
```

## Basic Configuration

### Step 1: Initialize Configuration

First run of aichat will create the configuration directory:

```bash
aichat
```

This creates the configuration file at:
- Linux/macOS: `~/.config/aichat/config.yaml`
- Windows: `%APPDATA%\aichat\config.yaml`

### Step 2: Get Configuration Directory

```bash
aichat --info | grep config_dir
```

## Setting Up Gemini 2.5 Flash

### Step 1: Obtain Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google Account
3. Navigate to "Get API Key"
4. Create a new API key
5. Copy and save the key securely

### Step 2: Configure Gemini in AIChat

Edit your `config.yaml` file:

```yaml
model: gemini:gemini-2.5-flash
clients:
  # See https://ai.google.dev/docs
  - type: gemini
    api_key: YOUR_GEMINI_API_KEY_HERE
    models:
      - name: gemini-2.5-flash
        max_input_tokens: 1048576  # 1M tokens
      - name: gemini-2.5-pro
        max_input_tokens: 2097152  # 2M tokens
    # Optional: Configure safety settings
    patch:
      chat_completions:
        '.*':
          body:
            safetySettings:
              - category: HARM_CATEGORY_HARASSMENT
                threshold: BLOCK_NONE
              - category: HARM_CATEGORY_HATE_SPEECH
                threshold: BLOCK_NONE
              - category: HARM_CATEGORY_SEXUALLY_EXPLICIT
                threshold: BLOCK_NONE
              - category: HARM_CATEGORY_DANGEROUS_CONTENT
                threshold: BLOCK_NONE
```

### Step 3: Test Gemini Connection

```bash
aichat -m gemini:gemini-2.5-flash "Hello, can you introduce yourself?"
```

## Creating Your First Agent

### Step 1: Create Agent Directory Structure

└── agents └── myagent ├── functions.json # JSON declarations for functions (Auto-generated) ├── index.yaml # Agent definition ├── tools.txt # Shared tools └── tools.

```bash
# Navigate to aichat config directory
cd $(aichat --info | grep config_dir | awk '{print $2}')

# Create agents directory
mkdir -p agents/myassistant

# Create agent configuration
cd agents/myassistant
```

### Step 2: Create Agent Definition (index.yaml)

Create `index.yaml` with the following content:

```yaml
name: MyAssistant
description: A helpful AI assistant powered by Gemini 2.5 Flash
version: 1.0.0
instructions: |
  You are a helpful AI assistant powered by Gemini 2.5 Flash.
  You excel at:
  - Code generation and debugging
  - Technical explanations
  - Creative problem solving
  - Data analysis
  
  Always be concise, accurate, and helpful.
  If you're unsure about something, say so.
conversation_starters:
  - "What can you help me with?"
  - "Show me your capabilities"
  - "Help me write some code"
  - "Explain a complex concept"
```

### Step 3: Create Agent-Specific Configuration

Create `config.yaml` in the agent directory:

# Agent-specific configuration
# Location `<aichat-config-dir>/agents/<agent-name>/config.yaml`

model: openai:gpt-4o             # Specify the LLM to use
temperature: null                # Set default temperature parameter, range (0, 1)
top_p: null                      # Set default top-p parameter, with a range of (0, 1) or (0, 2) depending on the model
use_tools: null                  # Which additional tools to use by agent.

```yaml
model: gemini:gemini-2.5-flash
temperature: 0.7
top_p: 0.95
use_tools: null  # Will be configured later
agent_prelude: null
```

### Step 4: Register the Agent

Create or update the `agents.txt` file in the config directory:

```bash
cd $(aichat --info | grep config_dir | awk '{print $2}')
echo "myassistant" >> agents.txt
```

### Step 5: Test Your Agent

```bash
# List all agents
aichat --list-agents

# Use your agent
aichat -a myassistant "Hello!"

# Get agent info
aichat -a myassistant --info
```

## Advanced Agent Configuration

### Adding Variables to Your Agent

Update `index.yaml` to include variables:

```yaml
variables:
  - name: project_type
    description: Type of project (web, cli, api, etc.)
    default: web
  - name: language
    description: Programming language preference
    default: python
```

Use variables when calling the agent:

```bash
aichat -a myassistant --agent-variable project_type=api --agent-variable language=go "Create a REST API boilerplate"
```

### Dynamic Instructions

AIChat will load index.yaml and check if the dynamic_instructions field is set to true. If it is true, it will execute <agent-name> _instructions and return its value as instructions.

Add to `index.yaml`:

```yaml
dynamic_instructions: true
```

Create a script `myassistant_instructions.sh`:

```bash
#!/bin/bash
echo "Current time: $(date)"
echo "Working directory: $(pwd)"
echo "You are MyAssistant. Help the user with their tasks."
```

Make it executable:

```bash
chmod +x myassistant_instructions.sh
```

## Using Tools and Functions

### Step 1: Install LLM Functions

AIChat expects LLM-functions to be placed in AIChat's functions_dir so that AIChat can use the tools and agents that LLM-functions provides.

```bash
# Clone llm-functions repository
git clone https://github.com/sigoden/llm-functions
cd llm-functions

# Link to AIChat's functions directory
ln -s "$(pwd)" "$(aichat --info | sed -n 's/^functions_dir\s\+//p')"
```

### Step 2: Configure Tools for Your Agent

Update your agent's `config.yaml`:

```yaml
model: gemini:gemini-2.5-flash
temperature: 0.7
use_tools: "fs,web_search,execute_command"
```

### Step 3: Update Agent Instructions

Add tool usage instructions to `index.yaml`:

```yaml
instructions: |
  You are a helpful AI assistant with access to:
  - File system operations (read, write, list files)
  - Web search capabilities
  - Command execution
  
  Use these tools when appropriate to help users.
```

### Step 4: Test Tool Usage

```bash
aichat -a myassistant "What files are in the current directory?"
aichat -a myassistant "Search for the latest news about AI"
```

## RAG Integration

### Step 1: Prepare Documents

Create a documents directory for your agent:

```bash
cd $(aichat --info | grep config_dir | awk '{print $2}')/agents/myassistant
mkdir documents
```

### Step 2: Update Agent Definition

Add documents to `index.yaml`:

```yaml
documents:
  - documents/guide.md
  - documents/api-reference.pdf
  - https://example.com/documentation
```

### Step 3: Create a RAG

```bash
# Create RAG from documents
aichat --rag myassistant-docs -f documents/

# Use agent with RAG
aichat -a myassistant --rag myassistant-docs "Explain the API endpoints"
```

## Practical Examples

### Example 1: Code Assistant Agent

Create `agents/codehelper/index.yaml`:

```yaml
name: CodeHelper
description: Expert programming assistant with Gemini 2.5 Flash
version: 1.0.0
instructions: |
  You are an expert programmer specializing in:
  - Writing clean, efficient code
  - Debugging and optimization
  - Best practices and design patterns
  - Code reviews and refactoring
  
  Always:
  - Include comments in code
  - Consider error handling
  - Follow language-specific conventions
  - Explain your reasoning
conversation_starters:
  - "Help me debug this code"
  - "Write a function that..."
  - "Optimize this algorithm"
  - "Review my code"
variables:
  - name: language
    description: Programming language
    default: python
```

Create `agents/codehelper/config.yaml`:

```yaml
model: gemini:gemini-2.5-flash
temperature: 0.3  # Lower for more consistent code
use_tools: "fs,execute_py_code"
```

### Example 2: Research Assistant Agent

Create `agents/researcher/index.yaml`:

```yaml
name: Researcher
description: Research assistant for gathering and analyzing information
version: 1.0.0
instructions: |
  You are a research assistant that:
  - Searches for reliable information
  - Summarizes findings clearly
  - Cites sources properly
  - Provides balanced perspectives
  
  Always verify information and indicate confidence levels.
conversation_starters:
  - "Research the latest developments in..."
  - "Summarize information about..."
  - "Compare different viewpoints on..."
documents:
  - research_papers/
  - notes.md
```

## Using the Agent with Sessions

### Create a Session

aichat -s session1 # Use session 'session1'

```bash
# Start a new session with your agent
aichat -a myassistant -s work-session

# Continue previous session
aichat -a myassistant -s work-session "Continue where we left off"

# Save session automatically
aichat -a myassistant --save-session
```

## Integration with Other Tools

### Shell Integration

```bash
# Execute commands suggested by agent
aichat -a myassistant -e "create a Python virtual environment"

# Code generation
aichat -a myassistant -c "fibonacci function in rust" > fib.rs
```

### API Server Mode

AIChat includes a lightweight built-in HTTP server for easy deployment. $ aichat --serve Chat Completions API: http://127.0.0.1:8000/v1/chat/completions Embeddings API: http://127.0.0.1:8000/v1/embeddings Rerank API: http://127.0.0.1:8000/v1/rerank LLM Playground: http://127.0.0.1:8000/playground LLM Arena: http://127.0.0.1:8000/arena?num=2

```bash
# Start API server
aichat --serve

# Use your agent via API
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini:gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

## Troubleshooting

### Common Issues

1. **API Key Issues**
   ```bash
   # Check if API key is set
   aichat --info | grep gemini
   
   # Test connection
   aichat -m gemini:gemini-2.5-flash --dry-run "test"
   ```

2. **Agent Not Found**
   - Ensure agent name is in `agents.txt`
   - Check directory structure is correct
   - Verify `index.yaml` syntax

3. **Tool Errors**
   - Ensure llm-functions is properly linked
   - Check tool permissions
   - Verify tool names in `use_tools`

### Debug Mode

```bash
# Enable debug logging
export AICHAT_LOG_LEVEL=debug
aichat -a myassistant "test query"
```

## Best Practices

1. **Model Selection**: Gemini 2.5 Flash is our workhorse thinking model, ideal for fast performance on everyday tasks. Use it for:
   - General assistance
   - Code generation
   - Quick responses
   - Cost-effective operations

2. **Temperature Settings**:
   - Lower (0.1-0.3): Consistent, focused responses
   - Medium (0.5-0.7): Balanced creativity
   - Higher (0.8-1.0): Creative tasks

3. **Context Management**:
   - Use sessions for long conversations
   - Clear context when switching topics
   - Leverage RAG for domain-specific knowledge

4. **Tool Usage**:
   - Only enable needed tools
   - Provide clear instructions for tool use
   - Monitor tool execution

## Conclusion

You now have a fully functional Gemini 2.5 Flash agent in AIChat! This setup provides:

- Fast, intelligent responses with Gemini 2.5 Flash
- Customizable agent behavior
- Tool integration for enhanced capabilities
- RAG support for domain-specific knowledge
- Session management for context retention

Experiment with different configurations, add more tools, and customize your agents for specific use cases. The combination of AIChat's powerful framework and Gemini 2.5 Flash's capabilities creates a versatile AI assistant for any task.
Of course, here is a complete tutorial and walkthrough for setting up an agent using `gemini-2.5-flash` in `aichat sigoden`.

## Unleash the Power of Gemini 2.5 Flash: A Complete Guide to Setting Up Your `aichat` Agent

This tutorial will guide you through the process of installing and configuring `aichat`, a powerful, all-in-one command-line interface (CLI) for interacting with large language models (LLMs). By the end of this guide, you will have a fully functional AI agent powered by Google's `gemini-2.5-flash` model, ready to assist you directly from your terminal.

`aichat`, developed by sigoden, stands out for its extensive features, including support for over 20 LLM providers like OpenAI, Claude, and Gemini, a chat-REPL, Retrieval-Augmented Generation (RAG), and the ability to create custom AI agents. An AI agent in `aichat` combines specific instructions, optional tools (for function calling), and documents (for RAG) to create a specialized assistant tailored to your needs.

### Prerequisites

Before we begin, you will need:

*   **A Google Account:** To access Google's AI services.
*   **A Gemini API Key:** `aichat` connects to Gemini through Google AI Studio. You can obtain a free API key from the Google AI Studio website.
*   **A Command-Line Terminal:** Access to a terminal on macOS, Linux, or Windows.

### Step 1: Install `aichat`

`aichat` is a single, cross-platform executable file, making installation straightforward. Here are the recommended installation methods for various operating systems:

**macOS (using Homebrew):**
```bash
brew install aichat
```

**Linux (using a package manager, e.g., Pacman on Arch Linux):**
```bash
pacman -S aichat
```

**Windows (using Scoop):**
```bash
scoop install aichat
```

**Other Systems (Manual Installation):**
You can also download pre-built binaries directly from the official GitHub Releases page. Download the appropriate file for your system, extract it, and add the `aichat` binary to your system's `$PATH`.

### Step 2: Initial Configuration and Adding Your Gemini API Key

The first time you run `aichat`, it will guide you through an initial setup process and create a configuration file.

1.  Open your terminal and run:
    ```bash
    aichat
    ```

2.  You will be prompted to create a new config file. Answer `Yes`. You can skip the initial prompts for other API keys if you only intend to use Gemini for now.

3.  The main configuration file, `config.yaml`, is where you'll store your API keys and set default behaviors. You need to add your Gemini API key to this file.

4.  Locate your `config.yaml` file. The default locations are:
    *   **Linux:** `/home/your-username/.config/aichat/config.yaml`
    *   **macOS:** `/Users/your-username/Library/Application Support/aichat/config.yaml`
    *   **Windows:** `C:\Users\your-username\AppData\Roaming\aichat\config.yaml`

    You can also find the exact path by running `aichat --info | grep config_file`.

5.  Open `config.yaml` in a text editor and add the following section for the Gemini client, replacing `<YOUR_GEMINI_API_KEY>` with the key you obtained from Google AI Studio:

    ```yaml
    clients:
      - type: gemini
        api_key: <YOUR_GEMINI_API_KEY>
    ```

    Your `config.yaml` might look something like this after adding the key:

    ```yaml
    # Specify the default LLM to use
    model: gemini:gemini-2.5-flash

    # Controls whether to use the stream-style API
    stream: true

    # A list of LLM clients
    clients:
      - type: gemini
        api_key: <YOUR_GEMINI_API_KEY>
    ```

### Step 3: Creating Your `gemini-2.5-flash` Agent

Now for the exciting part: creating a dedicated agent. An agent has its own configuration file where you define its specific model, instructions, and capabilities.

1.  **Create the Agent's Directory:** `aichat` looks for agent configurations in an `agents` sub-directory within your main `aichat` config folder. Create a new folder for your agent. Let's name our agent `flash-assistant`.

    *   **Linux/macOS:**
        ```bash
        mkdir -p ~/.config/aichat/agents/flash-assistant
        ```
    *   **Windows:**
        ```powershell
        mkdir "$env:APPDATA\aichat\agents\flash-assistant"
        ```

2.  **Create the Agent's Configuration File:** Inside the new directory, create a file named `config.yaml`.

    *   **Linux/macOS:**
        ```bash
        touch ~/.config/aichat/agents/flash-assistant/config.yaml
        ```
    *   **Windows:**
        ```powershell
        New-Item -Path "$env:APPDATA\aichat\agents\flash-assistant\config.yaml" -ItemType File
        ```

3.  **Configure the Agent:** Open the newly created agent `config.yaml` and define its properties. The most important are `model` and `instructions`. The `instructions` field contains the prompt that tells the agent how to behave.

    Here is a sample configuration for a helpful, concise assistant that uses `gemini-2.5-flash`:

    ```yaml
    # Specify the LLM for this agent
    model: gemini:gemini-2.5-flash

    # The instructions that define the agent's persona and task
    instructions: |
      You are a helpful and concise assistant.
      Your primary goal is to provide accurate and to-the-point answers.
      You should be friendly and efficient.
    ```

### Step 4: Interacting with Your New Agent

With your agent configured, you can now start a conversation with it using the `aichat` REPL (Read-Eval-Print Loop).

1.  Launch `aichat` in your terminal:
    ```bash
    aichat
    ```

2.  Once inside the REPL, you can use the `.agent` command to activate your newly created agent.

    ```
    .agent flash-assistant
    ```

3.  `aichat` will load your agent's configuration, and you'll be ready to chat. The prompt might change to indicate that you are now interacting with an agent.

4.  Start asking questions!

    ```
    〉.agent flash-assistant
    Welcome to agent `flash-assistant`
    flash-assistant〉What is the main advantage of the gemini-2.5-flash model?

    The main advantage of the Gemini 2.5 Flash model is its balance of price and performance, offering strong capabilities for a wide range of tasks at a lower cost compared to larger models. [13, 14]
    ```

To exit the agent and return to the normal chat mode, you can use the `.exit agent` command.

### Advanced Possibilities

This tutorial covers the basic setup, but `aichat` agents can be much more powerful. You can enhance your agent by:

*   **Adding Tools (Function Calling):** Allow your agent to interact with external tools and APIs.
*   **Using RAG:** Provide your agent with a set of documents to draw information from, creating a knowledgeable expert on a specific topic.
*   **Customizing Prompts:** Experiment with different instructions in your agent's `config.yaml` to tailor its personality and expertise.

You have now successfully set up a custom AI agent with `gemini-2.5-flash` in `aichat`. Enjoy the speed and efficiency of having a powerful language model directly in your terminal
Of course. Here is a complete guide on how to create and implement a custom coding toolset for your `aichat` agent.

## Supercharge Your AI: Building a Custom Coding Toolset for Your `aichat` Agent

This tutorial will guide you through creating and implementing a practical, file-system-based toolset for your `aichat` agent. By enabling your agent to interact with your local files, you can transform it from a simple conversationalist into a powerful coding assistant capable of reading, writing, and listing files on your command.

We will build a toolset with three essential functions:

1.  **`list_files`**: To see the contents of the current directory.
2.  **`read_file`**: To read the contents of a specific file.
3.  **`write_file`**: To write or overwrite a file with new content.

This functionality, known as "Function Calling" or "Tool Use," allows the LLM to request the execution of a command-line tool to gather information or perform an action. `aichat` seamlessly integrates this, allowing your agent to leverage the full power of your shell.

### Prerequisites

*   You have already completed the initial setup of `aichat` and your `gemini-2.5-flash` agent as described in the previous tutorial.
*   You have a working `aichat` installation and a configured agent.
*   You are comfortable with basic command-line operations.

### Step 1: Define the Tools in Your Agent's Configuration

The core of creating a toolset is defining it within your agent's `config.yaml` file. `aichat` uses a `tools` array to declare the available functions. Each tool is an object with a `name`, a `description`, and a `command` to execute.

1.  **Navigate to your agent's configuration directory.** If you followed the previous tutorial, this will be:
    *   **Linux/macOS:** `~/.config/aichat/agents/flash-assistant/`
    *   **Windows:** `%APPDATA%\aichat\agents\flash-assistant\`

2.  **Open the `config.yaml` file** in your text editor.

3.  **Add the `tools` section** to the file. We will define our three tools using standard shell commands (`ls`, `cat`, and `tee`). These commands are simple, powerful, and available on most systems (macOS/Linux). For Windows, you can use PowerShell equivalents.

    **For macOS/Linux Users:**
    Append the following `tools` block to your `config.yaml`.

    ```yaml
    # In ~/.config/aichat/agents/flash-assistant/config.yaml

    # Specify the LLM for this agent
    model: gemini:gemini-2.5-flash

    # The instructions that define the agent's persona and task
    instructions: |
      You are a helpful and concise coding assistant.
      When asked to perform a file operation, you must use the available tools.
      - To see what files are in the current directory, use the `list_files` tool.
      - To read a file, use the `read_file` tool with the `path` argument.
      - To write to a file, use the `write_file` tool with the `path` and `content` arguments.
      Always confirm which file you are about to write to before using the `write_file` tool.

    # Define the available tools for the agent
    tools:
      - name: list_files
        description: List all files and directories in the current working directory.
        command: ls -F

      - name: read_file
        description: Read the entire content of a specified file.
        command: cat {path}
        args:
          - name: path
            description: The path to the file to be read.

      - name: write_file
        description: Write content to a specified file, overwriting it if it exists.
        command: echo "{content}" | tee {path} > /dev/null
        args:
          - name: path
            description: The path to the file to be written.
          - name: content
            description: The new content to write to the file.
    ```

    **For Windows (PowerShell) Users:**
    Use these commands in your `tools` block instead.

    ```yaml
    tools:
      - name: list_files
        description: List all files and directories in the current working directory.
        command: Get-ChildItem | ForEach-Object { $_.Name }

      - name: read_file
        description: Read the entire content of a specified file.
        command: Get-Content -Path {path} -Raw
        args:
          - name: path
            description: The path to the file to be read.

      - name: write_file
        description: Write content to a specified file, overwriting it if it exists.
        command: Set-Content -Path {path} -Value "{content}"
        args:
          - name: path
            description: The path to the file to be written.
          - name: content
            description: The new content to write to the file.
    ```

### How It Works: A Breakdown of the Tool Definition

*   **`name`**: A simple, descriptive name for the tool. The LLM will use this name to request the tool's execution.
*   **`description`**: This is crucial. The LLM uses the description to understand what the tool does and when to use it. A good description leads to more reliable tool use.
*   **`command`**: The actual shell command that `aichat` will execute.
*   **`{path}` and `{content}`**: These are placeholders for the arguments (`args`) that the LLM will provide. When the agent decides to use the `read_file` tool, it will determine a value for `path`, and `aichat` will substitute it into the `cat {path}` command before running it.
*   **`args`**: An array defining the parameters the command accepts. Each argument has a `name` and a `description` to help the LLM understand what kind of information to provide.

### Step 2: Using Your Tool-Equipped Agent

Now that you've equipped your agent with tools, let's put them to work.

1.  **Launch `aichat` and activate your agent:**
    ```bash
    aichat
    ```
    Inside the REPL:
    ```
    〉.agent flash-assistant
    Welcome to agent `flash-assistant`
    ```

2.  **Run a command from your terminal** to create a dummy file for testing.
    ```bash
    echo "Hello World" > hello.txt
    ```

3.  **Ask the agent to list the files.** The agent should recognize the request and use the `list_files` tool.

    ```
    flash-assistant〉What files are in this directory?
    ```

    `aichat` will show that the tool is being called and then return the output to the LLM, which then presents it to you. The output will look something like this:

    ```
    Tool Call: `list_files`
    Tool Output:
    hello.txt

    I see one file in this directory: `hello.txt`.
    ```

4.  **Ask the agent to read the file.**

    ```
    flash-assistant〉Can you read the contents of hello.txt for me?
    ```

    Again, the agent will invoke the correct tool:

    ```
    Tool Call: `read_file` with `path`="hello.txt"
    Tool Output:
    Hello World

    The content of `hello.txt` is: "Hello World".
    ```

5.  **Ask the agent to write a new file.** This demonstrates using a tool with multiple arguments.

    ```
    flash-assistant〉Please create a new file named `example.py` and add a simple python print statement to it.
    ```

    The agent will confirm and use the `write_file` tool:

    ```
    Tool Call: `write_file` with `path`="example.py" and `content`="print('Hello from aichat!')"
    Tool Output:
    (No output from tee)

    I have created the file `example.py` with the requested content.
    ```

6.  **Verify the result.** You can ask the agent to list the files again or use the `cat` command in your terminal to see the new file's content.

    ```
    flash-assistant〉Show me the files again.
    ```
    ```
    Tool Call: `list_files`
    Tool Output:
    example.py
    hello.txt

    The files in the directory are now `example.py` and `hello.txt`.
    ```

You have successfully created a custom toolset that allows your `aichat` agent to interact with the local file system. This foundation can be extended with more complex tools, such as those for running code, interacting with version control (Git), or even calling web APIs, turning your CLI into a truly interactive and intelligent development environment.
Of course. Here is a complete guide on how to create and implement a custom coding toolset for your `aichat` agent.

## Supercharge Your AI: Building a Custom Coding Toolset for Your `aichat` Agent

This tutorial will guide you through creating and implementing a practical, file-system-based toolset for your `aichat` agent. By enabling your agent to interact with your local files, you can transform it from a simple conversationalist into a powerful coding assistant capable of reading, writing, and listing files on your command.

We will build a toolset with three essential functions:

1.  **`list_files`**: To see the contents of the current directory.
2.  **`read_file`**: To read the contents of a specific file.
3.  **`write_file`**: To write or overwrite a file with new content.

This functionality, known as "Function Calling" or "Tool Use," allows the LLM to request the execution of a command-line tool to gather information or perform an action. `aichat` seamlessly integrates this, allowing your agent to leverage the full power of your shell.

### Prerequisites

*   You have already completed the initial setup of `aichat` and your `gemini-2.5-flash` agent as described in the previous tutorial.
*   You have a working `aichat` installation and a configured agent.
*   You are comfortable with basic command-line operations.

### Step 1: Define the Tools in Your Agent's Configuration

The core of creating a toolset is defining it within your agent's `config.yaml` file. `aichat` uses a `tools` array to declare the available functions. Each tool is an object with a `name`, a `description`, and a `command` to execute.

1.  **Navigate to your agent's configuration directory.** If you followed the previous tutorial, this will be:
    *   **Linux/macOS:** `~/.config/aichat/agents/flash-assistant/`
    *   **Windows:** `%APPDATA%\aichat\agents\flash-assistant\`

2.  **Open the `config.yaml` file** in your text editor.

3.  **Add the `tools` section** to the file. We will define our three tools using standard shell commands (`ls`, `cat`, and `tee`). These commands are simple, powerful, and available on most systems (macOS/Linux). For Windows, you can use PowerShell equivalents.

    **For macOS/Linux Users:**
    Append the following `tools` block to your `config.yaml`.

    ```yaml
    # In ~/.config/aichat/agents/flash-assistant/config.yaml

    # Specify the LLM for this agent
    model: gemini:gemini-2.5-flash

    # The instructions that define the agent's persona and task
    instructions: |
      You are a helpful and concise coding assistant.
      When asked to perform a file operation, you must use the available tools.
      - To see what files are in the current directory, use the `list_files` tool.
      - To read a file, use the `read_file` tool with the `path` argument.
      - To write to a file, use the `write_file` tool with the `path` and `content` arguments.
      Always confirm which file you are about to write to before using the `write_file` tool.

    # Define the available tools for the agent
    tools:
      - name: list_files
        description: List all files and directories in the current working directory.
        command: ls -F

      - name: read_file
        description: Read the entire content of a specified file.
        command: cat {path}
        args:
          - name: path
            description: The path to the file to be read.

      - name: write_file
        description: Write content to a specified file, overwriting it if it exists.
        command: echo "{content}" | tee {path} > /dev/null
        args:
          - name: path
            description: The path to the file to be written.
          - name: content
            description: The new content to write to the file.
    ```

    **For Windows (PowerShell) Users:**
    Use these commands in your `tools` block instead.

    ```yaml
    tools:
      - name: list_files
        description: List all files and directories in the current working directory.
        command: Get-ChildItem | ForEach-Object { $_.Name }

      - name: read_file
        description: Read the entire content of a specified file.
        command: Get-Content -Path {path} -Raw
        args:
          - name: path
            description: The path to the file to be read.

      - name: write_file
        description: Write content to a specified file, overwriting it if it exists.
        command: Set-Content -Path {path} -Value "{content}"
        args:
          - name: path
            description: The path to the file to be written.
          - name: content
            description: The new content to write to the file.
    ```

### How It Works: A Breakdown of the Tool Definition

*   **`name`**: A simple, descriptive name for the tool. The LLM will use this name to request the tool's execution.
*   **`description`**: This is crucial. The LLM uses the description to understand what the tool does and when to use it. A good description leads to more reliable tool use.
*   **`command`**: The actual shell command that `aichat` will execute.
*   **`{path}` and `{content}`**: These are placeholders for the arguments (`args`) that the LLM will provide. When the agent decides to use the `read_file` tool, it will determine a value for `path`, and `aichat` will substitute it into the `cat {path}` command before running it.
*   **`args`**: An array defining the parameters the command accepts. Each argument has a `name` and a `description` to help the LLM understand what kind of information to provide.

### Step 2: Using Your Tool-Equipped Agent

Now that you've equipped your agent with tools, let's put them to work.

1.  **Launch `aichat` and activate your agent:**
    ```bash
    aichat
    ```
    Inside the REPL:
    ```
    〉.agent flash-assistant
    Welcome to agent `flash-assistant`
    ```

2.  **Run a command from your terminal** to create a dummy file for testing.
    ```bash
    echo "Hello World" > hello.txt
    ```

3.  **Ask the agent to list the files.** The agent should recognize the request and use the `list_files` tool.

    ```
    flash-assistant〉What files are in this directory?
    ```

    `aichat` will show that the tool is being called and then return the output to the LLM, which then presents it to you. The output will look something like this:

    ```
    Tool Call: `list_files`
    Tool Output:
    hello.txt

    I see one file in this directory: `hello.txt`.
    ```

4.  **Ask the agent to read the file.**

    ```
    flash-assistant〉Can you read the contents of hello.txt for me?
    ```

    Again, the agent will invoke the correct tool:

    ```
    Tool Call: `read_file` with `path`="hello.txt"
    Tool Output:
    Hello World

    The content of `hello.txt` is: "Hello World".
    ```

5.  **Ask the agent to write a new file.** This demonstrates using a tool with multiple arguments.

    ```
    flash-assistant〉Please create a new file named `example.py` and add a simple python print statement to it.
    ```

    The agent will confirm and use the `write_file` tool:

    ```
    Tool Call: `write_file` with `path`="example.py" and `content`="print('Hello from aichat!')"
    Tool Output:
    (No output from tee)

    I have created the file `example.py` with the requested content.
    ```

6.  **Verify the result.** You can ask the agent to list the files again or use the `cat` command in your terminal to see the new file's content.

    ```
    flash-assistant〉Show me the files again.
    ```
    ```
    Tool Call: `list_files`
    Tool Output:
    example.py
    hello.txt

    The files in the directory are now `example.py` and `hello.txt`.
    ```

You have successfully created a custom toolset that allows your `aichat` agent to interact with the local file system. This foundation can be extended with more complex tools, such as those for running code, interacting with version control (Git), or even calling web APIs, turning your CLI into a truly interactive and intelligent development environment.# Creating a Custom Coding Toolset for AIChat Agent

## Overview

I'll help you create a comprehensive custom toolset specifically designed for coding tasks. This toolset will include various utilities that your Gemini 2.5 Flash agent can use to assist with programming tasks.

## Toolset Architecture

```
└── functions/
    └── coding_tools/
        ├── code_analyzer.py
        ├── code_generator.py
        ├── test_runner.py
        ├── doc_generator.py
        ├── git_helper.py
        ├── package_manager.py
        ├── code_formatter.py
        ├── project_scaffolder.py
        └── functions.json
```

## Step 1: Create the Functions Directory

```bash
# Navigate to AIChat functions directory
cd $(aichat --info | grep functions_dir | awk '{print $2}')

# Create coding tools directory
mkdir -p coding_tools
cd coding_tools
```

## Step 2: Implement Custom Tools

### Tool 1: Code Analyzer (code_analyzer.py)

```python
#!/usr/bin/env python3
"""
Analyze code for complexity, issues, and suggestions.
"""

import sys
import json
import ast
import re
from pathlib import Path

def analyze_python_code(code):
    """Analyze Python code for various metrics."""
    try:
        tree = ast.parse(code)
        
        # Count various elements
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        
        # Calculate cyclomatic complexity (simplified)
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        # Check for common issues
        issues = []
        
        # Check for long functions
        for func in functions:
            func_lines = func.end_lineno - func.lineno
            if func_lines > 50:
                issues.append(f"Function '{func.name}' is {func_lines} lines long (consider breaking it down)")
        
        # Check for missing docstrings
        for func in functions:
            if not ast.get_docstring(func):
                issues.append(f"Function '{func.name}' is missing a docstring")
        
        return {
            "language": "python",
            "metrics": {
                "lines": len(code.splitlines()),
                "functions": len(functions),
                "classes": len(classes),
                "imports": len(imports),
                "complexity": complexity
            },
            "issues": issues,
            "suggestions": generate_suggestions(tree, code)
        }
    except SyntaxError as e:
        return {
            "error": f"Syntax error: {str(e)}",
            "line": e.lineno,
            "offset": e.offset
        }

def analyze_javascript_code(code):
    """Analyze JavaScript code."""
    # Basic analysis for JavaScript
    lines = code.splitlines()
    
    metrics = {
        "lines": len(lines),
        "functions": len(re.findall(r'function\s+\w+|const\s+\w+\s*=\s*\([^)]*\)\s*=>', code)),
        "classes": len(re.findall(r'class\s+\w+', code)),
        "imports": len(re.findall(r'import\s+.*from|require\(', code))
    }
    
    issues = []
    # Check for console.log statements
    console_logs = len(re.findall(r'console\.log', code))
    if console_logs > 0:
        issues.append(f"Found {console_logs} console.log statements (consider removing in production)")
    
    # Check for var usage
    var_usage = len(re.findall(r'\bvar\s+', code))
    if var_usage > 0:
        issues.append(f"Found {var_usage} 'var' declarations (consider using 'let' or 'const')")
    
    return {
        "language": "javascript",
        "metrics": metrics,
        "issues": issues
    }

def generate_suggestions(tree, code):
    """Generate improvement suggestions."""
    suggestions = []
    
    # Check for list comprehension opportunities
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            # Simple pattern matching for append operations
            if hasattr(node, 'body') and len(node.body) == 1:
                if isinstance(node.body[0], ast.Expr):
                    if isinstance(node.body[0].value, ast.Call):
                        if hasattr(node.body[0].value.func, 'attr'):
                            if node.body[0].value.func.attr == 'append':
                                suggestions.append("Consider using list comprehension instead of append in loop")
    
    return suggestions

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input provided"}))
        return
    
    input_data = json.loads(sys.argv[1])
    
    code = input_data.get("code", "")
    language = input_data.get("language", "auto")
    file_path = input_data.get("file_path", "")
    
    # Read from file if provided
    if file_path and Path(file_path).exists():
        with open(file_path, 'r') as f:
            code = f.read()
        if language == "auto":
            if file_path.endswith('.py'):
                language = "python"
            elif file_path.endswith(('.js', '.jsx')):
                language = "javascript"
    
    # Detect language if auto
    if language == "auto":
        if "def " in code or "import " in code or "class " in code:
            language = "python"
        elif "function " in code or "const " in code or "let " in code:
            language = "javascript"
    
    # Analyze based on language
    if language == "python":
        result = analyze_python_code(code)
    elif language == "javascript":
        result = analyze_javascript_code(code)
    else:
        result = {"error": f"Unsupported language: {language}"}
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
```

### Tool 2: Code Generator (code_generator.py)

```python
#!/usr/bin/env python3
"""
Generate code snippets and boilerplate.
"""

import sys
import json
from datetime import datetime

TEMPLATES = {
    "python": {
        "class": '''class {name}:
    """
    {description}
    """
    
    def __init__(self{params}):
        """Initialize {name}."""
        {init_body}
    
    def __repr__(self):
        return f"{{self.__class__.__name__}}({repr_params})"
''',
        "function": '''def {name}({params}){return_type}:
    """
    {description}
    
    Args:
        {args_doc}
    
    Returns:
        {returns_doc}
    """
    {body}
''',
        "test": '''import unittest
{imports}

class Test{name}(unittest.TestCase):
    """Test cases for {name}."""
    
    def setUp(self):
        """Set up test fixtures."""
        {setup}
    
    def test_{test_name}(self):
        """Test {test_description}."""
        {test_body}
    
    def tearDown(self):
        """Clean up after tests."""
        {teardown}

if __name__ == "__main__":
    unittest.main()
''',
        "async_function": '''async def {name}({params}){return_type}:
    """
    {description}
    
    Args:
        {args_doc}
    
    Returns:
        {returns_doc}
    """
    {body}
''',
        "dataclass": '''from dataclasses import dataclass
from typing import {imports}

@dataclass
class {name}:
    """
    {description}
    """
    {fields}
''',
        "api_endpoint": '''from fastapi import APIRouter, HTTPException, Depends
from typing import {imports}

router = APIRouter()

@router.{method}("/{path}")
async def {name}({params}){return_type}:
    """
    {description}
    """
    try:
        {body}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
    },
    "javascript": {
        "class": '''class {name} {{
    /**
     * {description}
     */
    constructor({params}) {{
        {init_body}
    }}
    
    {methods}
}}
''',
        "function": '''/**
 * {description}
 * @param {params_doc}
 * @returns {returns_doc}
 */
{async}function {name}({params}) {{
    {body}
}}
''',
        "react_component": '''import React{imports} from 'react';

/**
 * {description}
 */
{component_type} {name} = ({props}) => {{
    {hooks}
    
    {logic}
    
    return (
        {jsx}
    );
}};

export default {name};
''',
        "express_route": '''/**
 * {description}
 */
router.{method}('/{path}', {middleware}async (req, res) => {{
    try {{
        {validation}
        
        {body}
        
        res.json({{ success: true, data: result }});
    }} catch (error) {{
        res.status(500).json({{ success: false, error: error.message }});
    }}
}});
'''
    }
}

def generate_code(language, template_type, config):
    """Generate code based on template and configuration."""
    if language not in TEMPLATES:
        return {"error": f"Unsupported language: {language}"}
    
    if template_type not in TEMPLATES[language]:
        return {"error": f"Unsupported template type: {template_type} for {language}"}
    
    template = TEMPLATES[language][template_type]
    
    # Process the template with provided configuration
    try:
        code = template.format(**config)
        return {
            "code": code,
            "language": language,
            "template": template_type,
            "timestamp": datetime.now().isoformat()
        }
    except KeyError as e:
        return {"error": f"Missing required parameter: {str(e)}"}

def generate_snippet(language, snippet_type):
    """Generate common code snippets."""
    snippets = {
        "python": {
            "error_handling": '''try:
    # Your code here
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Specific error occurred: {e}")
    # Handle specific exception
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
finally:
    # Cleanup code
    cleanup_resources()
''',
            "logging_setup": '''import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
''',
            "context_manager": '''from contextlib import contextmanager

@contextmanager
def my_context():
    """Custom context manager."""
    # Setup
    resource = acquire_resource()
    try:
        yield resource
    finally:
        # Cleanup
        release_resource(resource)
''',
            "decorator": '''from functools import wraps
import time

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper
'''
        },
        "javascript": {
            "error_handling": '''try {
    // Your code here
    const result = await riskyOperation();
} catch (error) {
    if (error instanceof SpecificError) {
        console.error('Specific error:', error.message);
        // Handle specific error
    } else {
        console.error('Unexpected error:', error);
        throw error;
    }
} finally {
    // Cleanup code
    await cleanup();
}
''',
            "promise_chain": '''fetchData()
    .then(data => processData(data))
    .then(result => saveResult(result))
    .catch(error => {
        console.error('Error in promise chain:', error);
        // Handle error
    })
    .finally(() => {
        // Cleanup
    });
''',
            "async_await": '''async function handleAsync() {
    try {
        const data = await fetchData();
        const processed = await processData(data);
        const result = await saveResult(processed);
        return result;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}
''',
            "event_handler": '''element.addEventListener('click', (event) => {
    event.preventDefault();
    
    // Handle the event
    const target = event.target;
    const data = target.dataset;
    
    // Your logic here
});
'''
        }
    }
    
    if language in snippets and snippet_type in snippets[language]:
        return {
            "code": snippets[language][snippet_type],
            "language": language,
            "snippet_type": snippet_type
        }
    else:
        return {"error": f"Snippet '{snippet_type}' not found for {language}"}

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input provided"}))
        return
    
    input_data = json.loads(sys.argv[1])
    
    action = input_data.get("action", "generate")
    language = input_data.get("language", "python")
    
    if action == "generate":
        template_type = input_data.get("template_type")
        config = input_data.get("config", {})
        result = generate_code(language, template_type, config)
    elif action == "snippet":
        snippet_type = input_data.get("snippet_type")
        result = generate_snippet(language, snippet_type)
    else:
        result = {"error": f"Unknown action: {action}"}
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
```

### Tool 3: Test Runner (test_runner.py)

```python
#!/usr/bin/env python3
"""
Run tests and analyze test coverage.
"""

import sys
import json
import subprocess
import tempfile
import os
from pathlib import Path

def run_python_tests(test_path, coverage=False):
    """Run Python tests using pytest."""
    cmd = ["python", "-m", "pytest", "-v"]
    
    if coverage:
        cmd.extend(["--cov", "--cov-report", "json"])
    
    cmd.append(test_path)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        output = {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
        
        # Parse test results
        if "passed" in result.stdout or "failed" in result.stdout:
            lines = result.stdout.split('\n')
            for line in lines:
                if "passed" in line and "failed" in line:
                    # Extract test statistics
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            output["passed"] = int(parts[i-1])
                        elif part == "failed":
                            output["failed"] = int(parts[i-1])
        
        # Parse coverage if enabled
        if coverage and os.path.exists("coverage.json"):
            with open("coverage.json", "r") as f:
                coverage_data = json.load(f)
                output["coverage"] = {
                    "percent": coverage_data.get("totals", {}).get("percent_covered", 0),
                    "lines": coverage_data.get("totals", {}).get("num_statements", 0),
                    "covered": coverage_data.get("totals", {}).get("covered_lines", 0)
                }
            os.remove("coverage.json")
        
        return output
    except Exception as e:
        return {"error": str(e)}

def run_javascript_tests(test_path):
    """Run JavaScript tests using Jest."""
    # Check if jest is available
    jest_cmd = ["npx", "jest", "--json", test_path]
    
    try:
        result = subprocess.run(jest_cmd, capture_output=True, text=True)
        
        if result.stdout:
            try:
                jest_output = json.loads(result.stdout)
                return {
                    "success": jest_output.get("success", False),
                    "passed": jest_output.get("numPassedTests", 0),
                    "failed": jest_output.get("numFailedTests", 0),
                    "total": jest_output.get("numTotalTests", 0),
                    "coverage": jest_output.get("coverageMap", {})
                }
            except json.JSONDecodeError:
                pass
        
        # Fallback to basic output
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    except Exception as e:
        return {"error": str(e)}

def generate_test_template(language, code_to_test):
    """Generate test template for given code."""
    if language == "python":
        # Simple analysis to generate test template
        import ast
        try:
            tree = ast.parse(code_to_test)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            test_template = "import unittest\n\n"
            test_template += "class TestGeneratedTests(unittest.TestCase):\n"
            
            for func in functions:
                test_template += f'''
    def test_{func.name}_basic(self):
        """Test {func.name} with basic inputs."""
        # TODO: Add test implementation
        pass
    
    def test_{func.name}_edge_cases(self):
        """Test {func.name} with edge cases."""
        # TODO: Add edge case tests
        pass
'''
            
            test_template += "\nif __name__ == '__main__':\n    unittest.main()\n"
            
            return {"test_code": test_template, "functions_found": len(functions)}
        except:
            return {"error": "Failed to parse Python code"}
    
    elif language == "javascript":
        # Basic JavaScript test template
        test_template = '''describe('Generated Tests', () => {
    beforeEach(() => {
        // Setup
    });
    
    afterEach(() => {
        // Cleanup
    });
    
    test('should work with basic inputs', () => {
        // TODO: Add test implementation
        expect(true).toBe(true);
    });
    
    test('should handle edge cases', () => {
        // TODO: Add edge case tests
        expect(true).toBe(true);
    });
});
'''
        return {"test_code": test_template}
    
    return {"error": f"Unsupported language: {language}"}

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input provided"}))
        return
    
    input_data = json.loads(sys.argv[1])
    
    action = input_data.get("action", "run")
    
    if action == "run":
        test_path = input_data.get("test_path", ".")
        language = input_data.get("language", "python")
        coverage = input_data.get("coverage", False)
        
        if language == "python":
            result = run_python_tests(test_path, coverage)
        elif language == "javascript":
            result = run_javascript_tests(test_path)
        else:
            result = {"error": f"Unsupported language: {language}"}
    
    elif action == "generate":
        language = input_data.get("language", "python")
        code = input_data.get("code", "")
        result = generate_test_template(language, code)
    
    else:
        result = {"error": f"Unknown action: {action}"}
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
```

### Tool 4: Documentation Generator (doc_generator.py)

```python
#!/usr/bin/env python3
"""
Generate documentation from code.
"""

import sys
import json
import ast
import re
from pathlib import Path

def extract_python_docs(code):
    """Extract documentation from Python code."""
    try:
        tree = ast.parse(code)
        docs = {
            "module": ast.get_docstring(tree),
            "classes": [],
            "functions": [],
            "summary": {}
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_doc = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "methods": [],
                    "line": node.lineno
                }
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_doc = {
                            "name": item.name,
                            "docstring": ast.get_docstring(item),
                            "args": [arg.arg for arg in item.args.args],
                            "line": item.lineno
                        }
                        class_doc["methods"].append(method_doc)
                
                docs["classes"].append(class_doc)
            
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                func_doc = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "args": [arg.arg for arg in node.args.args],
                    "line": node.lineno,
                    "return_annotation": ast.unparse(node.returns) if node.returns else None
                }
                docs["functions"].append(func_doc)
        
        # Generate summary
        docs["summary"] = {
            "total_classes": len(docs["classes"]),
            "total_functions": len(docs["functions"]),
            "documented_classes": len([c for c in docs["classes"] if c["docstring"]]),
            "documented_functions": len([f for f in docs["functions"] if f["docstring"]])
        }
        
        return docs
    except SyntaxError as e:
        return {"error": f"Syntax error: {str(e)}"}

def extract_javascript_docs(code):
    """Extract JSDoc comments from JavaScript code."""
    docs = {
        "functions": [],
        "classes": [],
        "summary": {}
    }
    
    # Extract JSDoc comments
    jsdoc_pattern = r'/\*\*(.*?)\*/'
    jsdoc_comments = re.findall(jsdoc_pattern, code, re.DOTALL)
    
    # Extract function declarations
    func_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)'
    functions = re.findall(func_pattern, code)
    
    # Extract class declarations
    class_pattern = r'class\s+(\w+)'
    classes = re.findall(class_pattern, code)
    
    docs["summary"] = {
        "total_functions": len(functions),
        "total_classes": len(classes),
        "jsdoc_comments": len(jsdoc_comments)
    }
    
    return docs

def generate_markdown_docs(docs, language):
    """Generate markdown documentation from extracted docs."""
    md = f"# Documentation\n\n"
    
    if language == "python":
        if docs.get("module"):
            md += f"## Module Description\n\n{docs['module']}\n\n"
        
        if docs.get("classes"):
            md += "## Classes\n\n"
            for cls in docs["classes"]:
                md += f"### {cls['name']}\n\n"
                if cls["docstring"]:
                    md += f"{cls['docstring']}\n\n"
                
                if cls["methods"]:
                    md += "#### Methods\n\n"
                    for method in cls["methods"]:
                        md += f"##### `{method['name']}({', '.join(method['args'])})`\n\n"
                        if method["docstring"]:
                            md += f"{method['docstring']}\n\n"
        
        if docs.get("functions"):
            md += "## Functions\n\n"
            for func in docs["functions"]:
                args_str = ', '.join(func['args'])
                return_str = f" -> {func['return_annotation']}" if func.get('return_annotation') else ""
                md += f"### `{func['name']}({args_str}){return_str}`\n\n"
                if func["docstring"]:
                    md += f"{func['docstring']}\n\n"
    
    # Add summary
    if docs.get("summary"):
        md += "## Summary\n\n"
        for key, value in docs["summary"].items():
            md += f"- **{key.replace('_', ' ').title()}**: {value}\n"
    
    return md

def generate_api_docs(code, style="openapi"):
    """Generate API documentation."""
    # This is a simplified example - real implementation would parse actual API definitions
    api_doc = {
        "openapi": "3.0.0",
        "info": {
            "title": "Generated API Documentation",
            "version": "1.0.0",
            "description": "Auto-generated API documentation"
        },
        "paths": {}
    }
    
    # Extract route patterns (simplified)
    route_patterns = [
        r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']\)',  # Flask/FastAPI
        r'router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']\)',  # Express
    ]
    
    for pattern in route_patterns:
        matches = re.findall(pattern, code, re.IGNORECASE)
        for method, path in matches:
            if path not in api_doc["paths"]:
                api_doc["paths"][path] = {}
            
            api_doc["paths"][path][method.lower()] = {
                "summary": f"{method.upper()} {path}",
                "responses": {
                    "200": {"description": "Successful response"},
                    "400": {"description": "Bad request"},
                    "500": {"description": "Internal server error"}
                }
            }
    
    return api_doc

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input provided"}))
        return
    
    input_data = json.loads(sys.argv[1])
    
    action = input_data.get("action", "extract")
    code = input_data.get("code", "")
    file_path = input_data.get("file_path", "")
    language = input_data.get("language", "python")
    
    # Read from file if provided
    if file_path and Path(file_path).exists():
        with open(file_path, 'r') as f:
            code = f.read()
    
    if action == "extract":
        if language == "python":
            docs = extract_python_docs(code)
        elif language == "javascript":
            docs = extract_javascript_docs(code)
        else:
            docs = {"error": f"Unsupported language: {language}"}
        result = docs
    
    elif action == "generate_markdown":
        if language == "python":
            docs = extract_python_docs(code)
        elif language == "javascript":
            docs = extract_javascript_docs(code)
        else:
            docs = {"error": f"Unsupported language: {language}"}
        
        if "error" not in docs:
            markdown = generate_markdown_docs(docs, language)
            result = {"markdown": markdown, "docs": docs}
        else:
            result = docs
    
    elif action == "generate_api":
        api_docs = generate_api_docs(code)
        result = {"api_documentation": api_docs}
    
    else:
        result = {"error": f"Unknown action: {action}"}
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
```

### Tool 5: Git Helper (git_helper.py)

```python
#!/usr/bin/env python3
"""
Git operations helper tool.
"""

import sys
import json
import subprocess
import os
from datetime import datetime
from pathlib import Path

def run_git_command(cmd, cwd=None):
    """Run a git command and return the output."""
    try:
        result = subprocess.run(
            ["git"] + cmd,
            capture_output=True,
            text=True,
            cwd=cwd or os.getcwd()
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_repo_status(path="."):
    """Get current repository status."""
    status_result = run_git_command(["status", "--porcelain"], path)
    
    if not status_result["success"]:
        return {"error": "Not a git repository or git not available"}
    
    # Parse status
    status_lines = status_result["stdout"].split('\n') if status_result["stdout"] else []
    
    files = {
        "modified": [],
        "added": [],
        "deleted": [],
        "untracked": [],
        "renamed": []
    }
    
    for line in status_lines:
        if line:
            status = line[:2]
            filename = line[3:]
            
            if status == "??":
                files["untracked"].append(filename)
            elif "M" in status:
                files["modified"].append(filename)
            elif "A" in status:
                files["added"].append(filename)
            elif "D" in status:
                files["deleted"].append(filename)
            elif "R" in status:
                files["renamed"].append(filename)
    
    # Get branch info
    branch_result = run_git_command(["branch", "--show-current"], path)
    current_branch = branch_result["stdout"] if branch_result["success"] else "unknown"
    
    # Get last commit
    log_result = run_git_command(["log", "-1", "--pretty=format:%H|%an|%ae|%ad|%s"], path)
    
    last_commit = None
    if log_result["success"] and log_result["stdout"]:
        parts = log_result["stdout"].split('|')
        if len(parts) >= 5:
            last_commit = {
                "hash": parts[0],
                "author": parts[1],
                "email": parts[2],
                "date": parts[3],
                "message": parts[4]
            }
    
    return {
        "branch": current_branch,
        "files": files,
        "last_commit": last_commit,
        "clean": all(len(v) == 0 for v in files.values())
    }

def create_commit(message, files=None, path="."):
    """Create a commit with given message."""
    if files:
        # Add specific files
        for file in files:
            add_result = run_git_command(["add", file], path)
            if not add_result["success"]:
                return {"error": f"Failed to add {file}: {add_result['stderr']}"}
    else:
        # Add all changes
        add_result = run_git_command(["add", "-A"], path)
        if not add_result["success"]:
            return {"error": f"Failed to add files: {add_result['stderr']}"}
    
    # Create commit
    commit_result = run_git_command(["commit", "-m", message], path)
    
    if commit_result["success"]:
        # Get commit info
        log_result = run_git_command(["log", "-1", "--pretty=format:%H"], path)
        commit_hash = log_result["stdout"] if log_result["success"] else "unknown"
        
        return {
            "success": True,
            "commit_hash": commit_hash,
            "message": message
        }
    else:
        return {
            "success": False,
            "error": commit_result["stderr"]
        }

def analyze_commit_history(path=".", limit=10):
    """Analyze commit history."""
    log_format = "%H|%an|%ae|%ad|%s"
    log_result = run_git_command(["log", f"-{limit}", f"--pretty=format:{log_format}"], path)
    
    if not log_result["success"]:
        return {"error": "Failed to get commit history"}
    
    commits = []
    for line in log_result["stdout"].split('\n'):
        if line:
            parts = line.split('|')
            if len(parts) >= 5:
                commits.append({
                    "hash": parts[0][:8],  # Short hash
                    "author": parts[1],
                    "email": parts[2],
                    "date": parts[3],
                    "message": parts[4]
                })
    
    # Get contribution stats
    shortlog_result = run_git_command(["shortlog", "-sn", "--no-merges"], path)
    
    contributors = []
    if shortlog_result["success"]:
        for line in shortlog_result["stdout"].split('\n'):
            if line:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    contributors.append({
                        "commits": int(parts[0]),
                        "author": parts[1]
                    })
    
    return {
        "recent_commits": commits,
        "contributors": contributors,
        "total_commits": len(commits)
    }

def create_branch(branch_name, from_branch=None, path="."):
    """Create a new branch."""
    if from_branch:
        # Checkout the base branch first
        checkout_result = run_git_command(["checkout", from_branch], path)
        if not checkout_result["success"]:
            return {"error": f"Failed to checkout {from_branch}: {checkout_result['stderr']}"}
    
    # Create and checkout new branch
    create_result = run_git_command(["checkout", "-b", branch_name], path)
    
    if create_result["success"]:
        return {
            "success": True,
            "branch": branch_name,
            "message": f"Created and switched to branch '{branch_name}'"
        }
    else:
        return {
            "success": False,
            "error": create_result["stderr"]
        }

def generate_commit_message(files_changed, change_type="feat"):
    """Generate a conventional commit message."""
    # Conventional commit types
    commit_types = {
        "feat": "A new feature",
        "fix": "A bug fix",
        "docs": "Documentation only changes",
        "style": "Changes that do not affect the meaning of the code",
        "refactor": "A code change that neither fixes a bug nor adds a feature",
        "perf": "A code change that improves performance",
        "test": "Adding missing tests or correcting existing tests",
        "chore": "Changes to the build process or auxiliary tools"
    }
    
    # Analyze files to determine scope
    scopes = set()
    for file in files_changed:
        parts = Path(file).parts
        if len(parts) > 1:
            scopes.add(parts[0])
        else:
            scopes.add(Path(file).stem)
    
    scope = ', '.join(list(scopes)[:2]) if scopes else "general"
    
    # Generate message
    message = f"{change_type}({scope}): update {len(files_changed)} file(s)"
    
    return {
        "message": message,
        "type": change_type,
        "scope": scope,
        "description": commit_types.get(change_type, "Unknown type")
    }

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input provided"}))
        return
    
    input_data = json.loads(sys.argv[1])
    
    action = input_data.get("action", "status")
    path = input_data.get("path", ".")
    
    if action == "status":
        result = get_repo_status(path)
    
    elif action == "commit":
        message = input_data.get("message", "")
        files = input_data.get("files", None)
        
        if not message:
            # Generate commit message if not provided
            status = get_repo_status(path)
            all_changed = []
            for file_list in status["files"].values():
                all_changed.extend(file_list)
            
            if all_changed:
                msg_result = generate_commit_message(all_changed)
                message = msg_result["message"]
            else:
                message = "Update files"
        
        result = create_commit(message, files, path)
    
    elif action == "history":
        limit = input_data.get("limit", 10)
        result = analyze_commit_history(path, limit)
    
    elif action == "branch":
        branch_name = input_data.get("name", "")
        from_branch = input_data.get("from", None)
        
        if not branch_name:
            result = {"error": "Branch name is required"}
        else:
            result = create_branch(branch_name, from_branch, path)
    
    elif action == "generate_message":
        change_type = input_data.get("type", "feat")
        status = get_repo_status(path)
        all_changed = []
        for file_list in status["files"].values():
            all_changed.extend(file_list)
        
        result = generate_commit_message(all_changed, change_type)
    
    else:
        result = {"error": f"Unknown action: {action}"}
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
```

### Tool 6: Project Scaffolder (project_scaffolder.py)

```python
#!/usr/bin/env python3
"""
Create project scaffolds and boilerplate.
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

PROJECT_TEMPLATES = {
    "python": {
        "fastapi": {
            "files": {
                "main.py": '''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from routers import api

app = FastAPI(
    title="{project_name}",
    description="{description}",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to {project_name}"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
''',
                "config.py": '''from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    APP_NAME: str = "{project_name}"
    DEBUG: bool = True
    API_KEY: str = "your-secret-key"
    DATABASE_URL: str = "sqlite:///./test.db"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"

settings = Settings()
''',
                "requirements.txt": '''fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
sqlalchemy==2.0.23
python-dotenv==1.0.0
pytest==7.4.3
httpx==0.25.2
''',
                ".env.example": '''APP_NAME={project_name}
DEBUG=True
API_KEY=your-secret-key-here
DATABASE_URL=sqlite:///./test.db
''',
                "routers/__init__.py": '',
                "routers/api.py": '''from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter()

class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    price: float

# In-memory storage for demo
items = []

@router.get("/items", response_model=List[Item])
async def get_items():
    """Get all items."""
    return items

@router.post("/items", response_model=Item)
async def create_item(item: Item):
    """Create a new item."""
    item.id = len(items) + 1
    items.append(item)
    return item

@router.get("/items/{item_id}", response_model=Item)
async def get_item(item_id: int):
    """Get item by ID."""
    for item in items:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")
''',
                "tests/__init__.py": '',
                "tests/test_api.py": '''from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_create_item():
    response = client.post(
        "/api/v1/items",
        json={"name": "Test Item", "price": 9.99}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Test Item"
''',
                "README.md": '''# {project_name}

{description}

## Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and configure

4. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

## API Documentation

Visit http://localhost:8000/docs for interactive API documentation.

## Testing

```bash
pytest
```
''',
                ".gitignore": '''__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
.env
.pytest_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
.idea/
.vscode/
*.log
'''
            }
        },
        "cli": {
            "files": {
                "cli.py": '''#!/usr/bin/env python3
"""
{project_name} - {description}
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class {project_name_class}:
    """Main application class."""
    
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration."""
        # TODO: Implement configuration loading
        return {}
    
    def run(self, args):
        """Run the application."""
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Running {project_name} with args: {args}")
        
        # TODO: Implement main logic
        print(f"Hello from {project_name}!")
        
        return 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="{description}",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '-c', '--config',
        type=Path,
        help='Configuration file path'
    )
    
    # TODO: Add more arguments
    
    args = parser.parse_args()
    
    app = {project_name_class}()
    return app.run(args)

if __name__ == '__main__':
    sys.exit(main())
''',
                "setup.py": '''from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="{project_name}",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="{description}",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/{project_name}",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        # Add your dependencies here
    ],
    entry_points={{
        "console_scripts": [
            "{project_name}={project_name}.cli:main",
        ],
    }},
)
''',
                "requirements.txt": '''pytest==7.4.3
black==23.11.0
flake8==6.1.0
mypy==1.7.1
''',
                "README.md": '''# {project_name}

{description}

## Installation

```bash
pip install {project_name}
```

## Usage

```bash
{project_name} --help
```

## Development

1. Clone the repository
2. Create virtual environment
3. Install in development mode:
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

## Testing

```bash
pytest
```
'''
            }
        }
    },
    "javascript": {
        "react": {
            "files": {
                "package.json": '''{
  "name": "{project_name}",
  "version": "0.1.0",
  "private": true,
  "description": "{description}",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "test": "jest --watch",
    "test:ci": "jest --ci"
  },
  "dependencies": {
    "next": "14.0.3",
    "react": "^18",
    "react-dom": "^18"
  },
  "devDependencies": {
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "autoprefixer": "^10.0.1",
    "eslint": "^8",
    "eslint-config-next": "14.0.3",
    "jest": "^29.7.0",
    "postcss": "^8",
    "tailwindcss": "^3.3.0",
    "typescript": "^5"
  }
}
''',
                "app/layout.tsx": '''import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: '{project_name}',
  description: '{description}',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  )
}
''',
                "app/page.tsx": '''export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm">
        <h1 className="text-4xl font-bold mb-8">{project_name}</h1>
        <p className="text-xl mb-4">{description}</p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
          <div className="p-6 border rounded-lg">
            <h2 className="text-2xl font-semibold mb-2">Feature 1</h2>
            <p>Description of feature 1</p>
          </div>
          
          <div className="p-6 border rounded-lg">
            <h2 className="text-2xl font-semibold mb-2">Feature 2</h2>
            <p>Description of feature 2</p>
          </div>
          
          <div className="p-6 border rounded-lg">
            <h2 className="text-2xl font-semibold mb-2">Feature 3</h2>
            <p>Description of feature 3</p>
          </div>
        </div>
      </div>
    </main>
  )
}
''',
                "app/globals.css": '''@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --foreground-rgb: 0, 0, 0;
  --background-start-rgb: 214, 219, 220;
  --background-end-rgb: 255, 255, 255;
}

@media (prefers-color-scheme: dark) {
  :root {
    --foreground-rgb: 255, 255, 255;
    --background-start-rgb: 0, 0, 0;
    --background-end-rgb: 0, 0, 0;
  }
}

body {
  color: rgb(var(--foreground-rgb));
  background: linear-gradient(
      to bottom,
      transparent,
      rgb(var(--background-end-rgb))
    )
    rgb(var(--background-start-rgb));
}
''',
                "components/Button.tsx": '''import React from 'react'

interface ButtonProps {
  children: React.ReactNode
  onClick?: () => void
  variant?: 'primary' | 'secondary'
  disabled?: boolean
}

export const Button: React.FC<ButtonProps> = ({
  children,
  onClick,
  variant = 'primary',
  disabled = false
}) => {
  const baseClasses = 'px-4 py-2 rounded font-semibold transition-colors'
  const variantClasses = {
    primary: 'bg-blue-500 text-white hover:bg-blue-600',
    secondary: 'bg-gray-200 text-gray-800 hover:bg-gray-300'
  }
  
  return (
    <button
      className={`${baseClasses} ${variantClasses[variant]} ${
        disabled ? 'opacity-50 cursor-not-allowed' : ''
      }`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  )
}
''',
                "tailwind.config.ts": '''import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic':
          'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
    },
  },
  plugins: [],
}
export default config
''',
                "tsconfig.json": '''{
  "compilerOptions": {
    "target": "es5",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ],
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
''',
                ".gitignore": '''# dependencies
/node_modules
/.pnp
.pnp.js
.yarn/install-state.gz

# testing
/coverage

# next.js
/.next/
/out/

# production
/build

# misc
.DS_Store
*.pem

# debug
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# local env files
.env*.local

# vercel
.vercel

# typescript
*.tsbuildinfo
next-env.d.ts
''',
                "README.md": '''# {project_name}

{description}

## Getting Started

First, install dependencies:

```bash
npm install
# or
yarn install
# or
pnpm install
```

Then, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev)

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com).
'''
            }
        },
        "express": {
            "files": {
                "package.json": '''{
  "name": "{project_name}",
  "version": "1.0.0",
  "description": "{description}",
  "main": "src/index.js",
  "scripts": {
    "start": "node src/index.js",
    "dev": "nodemon src/index.js",
    "test": "jest",
    "lint": "eslint src/"
  },
  "dependencies": {
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "dotenv": "^16.3.1",
    "helmet": "^7.1.0",
    "morgan": "^1.10.0"
  },
  "devDependencies": {
    "nodemon": "^3.0.1",
    "jest": "^29.7.0",
    "supertest": "^6.3.3",
    "eslint": "^8.54.0"
  }
}
''',
                "src/index.js": '''const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
require('dotenv').config();

const routes = require('./routes');
const errorHandler = require('./middleware/errorHandler');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api', routes);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// Error handling
app.use(errorHandler);

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

module.exports = app;
''',
                "src/routes/index.js": '''const express = require('express');
const router = express.Router();

// Example route
router.get('/', (req, res) => {
  res.json({
    message: 'Welcome to {project_name} API',
    version: '1.0.0'
  });
});

// Example resource routes
router.get('/items', (req, res) => {
  // TODO: Implement get all items
  res.json({ items: [] });
});

router.post('/items', (req, res) => {
  // TODO: Implement create item
  const { name, description } = req.body;
  res.status(201).json({
    id: Date.now(),
    name,
    description,
    createdAt: new Date().toISOString()
  });
});

router.get('/items/:id', (req, res) => {
  // TODO: Implement get item by id
  const { id } = req.params;
  res.json({
    id,
    name: 'Example Item',
    description: 'This is an example item'
  });
});

module.exports = router;
''',
                "src/middleware/errorHandler.js": '''const errorHandler = (err, req, res, next) => {
  console.error(err.stack);
  
  const status = err.status || 500;
  const message = err.message || 'Something went wrong!';
  
  res.status(status).json({
    error: {
      message,
      status,
      timestamp: new Date().toISOString()
    }
  });
};

module.exports = errorHandler;
''',
                ".env.example": '''PORT=3000
NODE_ENV=development
''',
                ".gitignore": '''node_modules/
.env
.DS_Store
*.log
dist/
coverage/
.vscode/
.idea/
''',
                "README.md": '''# {project_name}

{description}

## Installation

```bash
npm install
```

## Development

```bash
npm run dev
```

## Production

```bash
npm start
```

## Testing

```bash
npm test
```

## API Endpoints

- `GET /health` - Health check
- `GET /api` - API info
- `GET /api/items` - Get all items
- `POST /api/items` - Create new item
- `GET /api/items/:id` - Get item by ID

## Environment Variables

Copy `.env.example` to `.env` and configure as needed.
'''
            }
        }
    }
}

def create_project(project_type, template_name, project_name, description, path="."):
    """Create a new project from template."""
    if project_type not in PROJECT_TEMPLATES:
        return {"error": f"Unknown project type: {project_type}"}
    
    if template_name not in PROJECT_TEMPLATES[project_type]:
        return {"error": f"Unknown template: {template_name} for {project_type}"}
    
    template = PROJECT_TEMPLATES[project_type][template_name]
    project_path = Path(path) / project_name
    
    # Create project directory
    project_path.mkdir(parents=True, exist_ok=True)
    
    # Create files from template
    created_files = []
    
    # Convert project name to different formats
    project_name_snake = project_name.replace("-", "_").replace(" ", "_").lower()
    project_name_class = ''.join(word.capitalize() for word in project_name.replace("-", " ").replace("_", " ").split())
    
    for file_path, content in template["files"].items
    
# Creating a Custom Coding Toolset for AIChat Agent - In-Depth Guide

## Continuing Tool Implementation

### Tool 5: Project Scaffolder (project_scaffolder.py) - Complete Implementation

```python
#!/usr/bin/env python3
"""
Create project scaffolds and boilerplate.
"""

import sys
import json
import os
import shutil
from pathlib import Path
from datetime import datetime

# ... (previous PROJECT_TEMPLATES definition continues) ...

def create_project(project_type, template_name, project_name, description, path="."):
    """Create a new project from template."""
    if project_type not in PROJECT_TEMPLATES:
        return {"error": f"Unknown project type: {project_type}"}
    
    if template_name not in PROJECT_TEMPLATES[project_type]:
        return {"error": f"Unknown template: {template_name} for {project_type}"}
    
    template = PROJECT_TEMPLATES[project_type][template_name]
    project_path = Path(path) / project_name
    
    # Check if project already exists
    if project_path.exists():
        return {"error": f"Project directory already exists: {project_path}"}
    
    # Create project directory
    project_path.mkdir(parents=True, exist_ok=True)
    
    # Create files from template
    created_files = []
    
    # Convert project name to different formats
    project_name_snake = project_name.replace("-", "_").replace(" ", "_").lower()
    project_name_class = ''.join(word.capitalize() for word in project_name.replace("-", " ").replace("_", " ").split())
    
    replacements = {
        "{project_name}": project_name,
        "{project_name_snake}": project_name_snake,
        "{project_name_class}": project_name_class,
        "{description}": description,
        "{timestamp}": datetime.now().isoformat(),
        "{year}": str(datetime.now().year)
    }
    
    try:
        for file_path, content in template["files"].items():
            # Create file path
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Replace placeholders in content
            for placeholder, value in replacements.items():
                content = content.replace(placeholder, value)
            
            # Write file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            created_files.append(str(file_path))
        
        # Initialize git repository if requested
        if template.get("init_git", True):
            os.system(f"cd {project_path} && git init -q")
        
        # Create virtual environment for Python projects
        if project_type == "python":
            os.system(f"cd {project_path} && python -m venv venv")
        
        return {
            "success": True,
            "project_path": str(project_path),
            "created_files": created_files,
            "next_steps": get_next_steps(project_type, template_name),
            "commands": get_setup_commands(project_type, template_name, project_name)
        }
    
    except Exception as e:
        # Cleanup on error
        if project_path.exists():
            shutil.rmtree(project_path)
        return {"error": f"Failed to create project: {str(e)}"}

def get_next_steps(project_type, template_name):
    """Get next steps for the created project."""
    steps = {
        "python": {
            "fastapi": [
                "Activate virtual environment",
                "Install dependencies: pip install -r requirements.txt",
                "Copy .env.example to .env and configure",
                "Run the server: uvicorn main:app --reload",
                "Visit http://localhost:8000/docs for API documentation"
            ],
            "cli": [
                "Activate virtual environment",
                "Install in development mode: pip install -e .",
                "Install dev dependencies: pip install -r requirements.txt",
                "Run tests: pytest",
                "Run the CLI: python cli.py --help"
            ]
        },
        "javascript": {
            "react": [
                "Install dependencies: npm install",
                "Run development server: npm run dev",
                "Visit http://localhost:3000",
                "Start editing app/page.tsx"
            ],
            "express": [
                "Install dependencies: npm install",
                "Copy .env.example to .env",
                "Run development server: npm run dev",
                "API will be available at http://localhost:3000/api"
            ]
        }
    }
    
    return steps.get(project_type, {}).get(template_name, [])

def get_setup_commands(project_type, template_name, project_name):
    """Get setup commands for the project."""
    commands = {
        "python": {
            "fastapi": [
                f"cd {project_name}",
                "python -m venv venv",
                "source venv/bin/activate  # On Windows: venv\\Scripts\\activate",
                "pip install -r requirements.txt",
                "cp .env.example .env",
                "uvicorn main:app --reload"
            ],
            "cli": [
                f"cd {project_name}",
                "python -m venv venv",
                "source venv/bin/activate  # On Windows: venv\\Scripts\\activate",
                "pip install -e .",
                "pip install -r requirements.txt",
                "python cli.py --help"
            ]
        },
        "javascript": {
            "react": [
                f"cd {project_name}",
                "npm install",
                "npm run dev"
            ],
            "express": [
                f"cd {project_name}",
                "npm install",
                "cp .env.example .env",
                "npm run dev"
            ]
        }
    }
    
    return commands.get(project_type, {}).get(template_name, [])

def add_feature(project_path, feature_type, feature_config):
    """Add a feature to an existing project."""
    features = {
        "python": {
            "database": {
                "files": {
                    "database.py": '''from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import settings

SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
''',
                    "models.py": '''from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
'''
                },
                "dependencies": [
                    "sqlalchemy==2.0.23",
                    "alembic==1.12.1"
                ]
            },
            "auth": {
                "files": {
                    "auth.py": '''from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    # TODO: Get user from database
    return username
'''
                },
                "dependencies": [
                    "python-jose[cryptography]==3.3.0",
                    "passlib[bcrypt]==1.7.4",
                    "python-multipart==0.0.6"
                ]
            }
        },
        "javascript": {
            "database": {
                "files": {
                    "src/db/index.js": '''const mongoose = require('mongoose');

const connectDB = async () => {
    try {
        await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/myapp', {
            useNewUrlParser: true,
            useUnifiedTopology: true,
        });
        console.log('MongoDB connected successfully');
    } catch (error) {
        console.error('MongoDB connection error:', error);
        process.exit(1);
    }
};

module.exports = { connectDB };
''',
                    "src/models/User.js": '''const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const UserSchema = new mongoose.Schema({
    username: {
        type: String,
        required: true,
        unique: true,
        trim: true,
        minlength: 3
    },
    email: {
        type: String,
        required: true,
        unique: true,
        lowercase: true,
        trim: true
    },
    password: {
        type: String,
        required: true,
        minlength: 6
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

UserSchema.pre('save', async function(next) {
    if (!this.isModified('password')) return next();
    
    try {
        const salt = await bcrypt.genSalt(10);
        this.password = await bcrypt.hash(this.password, salt);
        next();
    } catch (error) {
        next(error);
    }
});

UserSchema.methods.comparePassword = async function(candidatePassword) {
    return await bcrypt.compare(candidatePassword, this.password);
};

module.exports = mongoose.model('User', UserSchema);
'''
                },
                "dependencies": [
                    "mongoose",
                    "bcryptjs"
                ]
            }
        }
    }
    
    project_path = Path(project_path)
    if not project_path.exists():
        return {"error": "Project path does not exist"}
    
    # Detect project type
    if (project_path / "package.json").exists():
        project_type = "javascript"
    elif (project_path / "requirements.txt").exists():
        project_type = "python"
    else:
        return {"error": "Could not detect project type"}
    
    if feature_type not in features.get(project_type, {}):
        return {"error": f"Unknown feature type: {feature_type} for {project_type}"}
    
    feature = features[project_type][feature_type]
    created_files = []
    
    try:
        # Create feature files
        for file_path, content in feature["files"].items():
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Apply any custom configurations
            if feature_config:
                for key, value in feature_config.items():
                    content = content.replace(f"{{{key}}}", str(value))
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            created_files.append(file_path)
        
        # Update dependencies
        if project_type == "python" and "dependencies" in feature:
            req_file = project_path / "requirements.txt"
            if req_file.exists():
                with open(req_file, 'a') as f:
                    f.write("\n# " + feature_type + " dependencies\n")
                    for dep in feature["dependencies"]:
                        f.write(dep + "\n")
        
        elif project_type == "javascript" and "dependencies" in feature:
            # For JavaScript, we'll return the npm install command
            pass
        
        return {
            "success": True,
            "feature": feature_type,
            "created_files": created_files,
            "dependencies": feature.get("dependencies", []),
            "next_steps": get_feature_next_steps(project_type, feature_type)
        }
    
    except Exception as e:
        return {"error": f"Failed to add feature: {str(e)}"}

def get_feature_next_steps(project_type, feature_type):
    """Get next steps after adding a feature."""
    steps = {
        "python": {
            "database": [
                "Install new dependencies: pip install -r requirements.txt",
                "Update config with DATABASE_URL",
                "Run migrations: alembic init alembic",
                "Create initial migration: alembic revision --autogenerate -m 'Initial migration'"
            ],
            "auth": [
                "Install new dependencies: pip install -r requirements.txt",
                "Add SECRET_KEY to your .env file",
                "Update your routes to use authentication",
                "Test with POST /token endpoint"
            ]
        },
        "javascript": {
            "database": [
                "Install dependencies: npm install mongoose bcryptjs",
                "Add MONGODB_URI to your .env file",
                "Import and call connectDB() in your main file",
                "Use the User model in your routes"
            ]
        }
    }
    
    return steps.get(project_type, {}).get(feature_type, [])

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input provided"}))
        return
    
    input_data = json.loads(sys.argv[1])
    
    action = input_data.get("action", "create")
    
    if action == "create":
        project_type = input_data.get("type", "python")
        template = input_data.get("template", "fastapi" if project_type == "python" else "express")
        name = input_data.get("name", "my-project")
        description = input_data.get("description", "A new project")
        path = input_data.get("path", ".")
        
        result = create_project(project_type, template, name, description, path)
    
    elif action == "add_feature":
        project_path = input_data.get("project_path", ".")
        feature_type = input_data.get("feature", "database")
        config = input_data.get("config", {})
        
        result = add_feature(project_path, feature_type, config)
    
    elif action == "list_templates":
        templates = {}
        for lang, temps in PROJECT_TEMPLATES.items():
            templates[lang] = list(temps.keys())
        result = {"templates": templates}
    
    else:
        result = {"error": f"Unknown action: {action}"}
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
```

### Tool 6: Package Manager (package_manager.py)

```python
#!/usr/bin/env python3
"""
Manage project dependencies and packages.
"""

import sys
import json
import subprocess
import os
from pathlib import Path
import re
import requests
from packaging import version

def detect_package_manager(path="."):
    """Detect which package manager is being used."""
    path = Path(path)
    
    # Python
    if (path / "requirements.txt").exists():
        return "pip"
    elif (path / "Pipfile").exists():
        return "pipenv"
    elif (path / "poetry.lock").exists() or (path / "pyproject.toml").exists():
        # Check if it's a poetry project
        if (path / "pyproject.toml").exists():
            with open(path / "pyproject.toml", 'r') as f:
                if "[tool.poetry]" in f.read():
                    return "poetry"
    
    # JavaScript
    if (path / "package-lock.json").exists():
        return "npm"
    elif (path / "yarn.lock").exists():
        return "yarn"
    elif (path / "pnpm-lock.yaml").exists():
        return "pnpm"
    
    # Ruby
    if (path / "Gemfile").exists():
        return "bundler"
    
    # Rust
    if (path / "Cargo.toml").exists():
        return "cargo"
    
    # Go
    if (path / "go.mod").exists():
        return "go"
    
    return None

def get_installed_packages(package_manager, path="."):
    """Get list of installed packages."""
    try:
        if package_manager == "pip":
            # Parse requirements.txt
            req_file = Path(path) / "requirements.txt"
            if req_file.exists():
                packages = []
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Parse package name and version
                            match = re.match(r'^([a-zA-Z0-9\-_]+)([=<>]+)?(.*)$', line)
                            if match:
                                packages.append({
                                    "name": match.group(1),
                                    "version": match.group(3) if match.group(3) else "latest",
                                    "constraint": match.group(2) if match.group(2) else "=="
                                })
                return packages
        
        elif package_manager == "npm":
            # Parse package.json
            pkg_file = Path(path) / "package.json"
            if pkg_file.exists():
                with open(pkg_file, 'r') as f:
                    pkg_data = json.load(f)
                
                packages = []
                for dep_type in ["dependencies", "devDependencies"]:
                    if dep_type in pkg_data:
                        for name, version_spec in pkg_data[dep_type].items():
                            packages.append({
                                "name": name,
                                "version": version_spec,
                                "dev": dep_type == "devDependencies"
                            })
                return packages
        
        elif package_manager == "poetry":
            # Run poetry show
            result = subprocess.run(
                ["poetry", "show", "--format", "json"],
                capture_output=True,
                text=True,
                cwd=path
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        
    except Exception as e:
        return {"error": str(e)}
    
    return []

def add_package(package_manager, package_name, version_spec=None, dev=False, path="."):
    """Add a package to the project."""
    commands = {
        "pip": {
            "install": ["pip", "install", f"{package_name}" + (f"{version_spec}" if version_spec else "")],
            "save": lambda: update_requirements_txt(package_name, version_spec, path)
        },
        "npm": {
            "install": ["npm", "install", "--save" + ("-dev" if dev else ""), 
                       f"{package_name}" + (f"@{version_spec}" if version_spec else "")]
        },
        "yarn": {
            "install": ["yarn", "add"] + (["-D"] if dev else []) + 
                      [f"{package_name}" + (f"@{version_spec}" if version_spec else "")]
        },
        "poetry": {
            "install": ["poetry", "add"] + (["-D"] if dev else []) + 
                      [f"{package_name}" + (f"@{version_spec}" if version_spec else "")]
        },
        "pipenv": {
            "install": ["pipenv", "install"] + (["--dev"] if dev else []) + 
                      [f"{package_name}" + (f"{version_spec}" if version_spec else "")]
        }
    }
    
    if package_manager not in commands:
        return {"error": f"Unsupported package manager: {package_manager}"}
    
    try:
        # Run install command
        cmd = commands[package_manager]["install"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=path)
        
        if result.returncode == 0:
            # For pip, update requirements.txt
            if package_manager == "pip" and "save" in commands[package_manager]:
                commands[package_manager]["save"]()
            
            return {
                "success": True,
                "package": package_name,
                "version": version_spec or "latest",
                "output": result.stdout
            }
        else:
            return {
                "success": False,
                "error": result.stderr
            }
    
    except Exception as e:
        return {"error": str(e)}

def update_requirements_txt(package_name, version_spec, path):
    """Update requirements.txt file."""
    req_file = Path(path) / "requirements.txt"
    
    # Get current version if no version specified
    if not version_spec:
        try:
            result = subprocess.run(
                ["pip", "show", package_name],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith("Version:"):
                        version_spec = "==" + line.split(":")[1].strip()
                        break
        except:
            version_spec = ""
    
    # Read existing requirements
    lines = []
    if req_file.exists():
        with open(req_file, 'r') as f:
            lines = f.readlines()
    
    # Check if package already exists
    package_found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(package_name):
            lines[i] = f"{package_name}{version_spec}\n"
            package_found = True
            break
    
    # Add package if not found
    if not package_found:
        lines.append(f"{package_name}{version_spec}\n")
    
    # Write back
    with open(req_file, 'w') as f:
        f.writelines(lines)

def remove_package(package_manager, package_name, path="."):
    """Remove a package from the project."""
    commands = {
        "pip": ["pip", "uninstall", "-y", package_name],
        "npm": ["npm", "uninstall", package_name],
        "yarn": ["yarn", "remove", package_name],
        "poetry": ["poetry", "remove", package_name],
        "pipenv": ["pipenv", "uninstall", package_name]
    }
    
    if package_manager not in commands:
        return {"error": f"Unsupported package manager: {package_manager}"}
    
    try:
        result = subprocess.run(
            commands[package_manager],
            capture_output=True,
            text=True,
            cwd=path
        )
        
        if result.returncode == 0:
            # For pip, update requirements.txt
            if package_manager == "pip":
                remove_from_requirements_txt(package_name, path)
            
            return {
                "success": True,
                "package": package_name,
                "output": result.stdout
            }
        else:
            return {
                "success": False,
                "error": result.stderr
            }
    
    except Exception as e:
        return {"error": str(e)}

def remove_from_requirements_txt(package_name, path):
    """Remove package from requirements.txt."""
    req_file = Path(path) / "requirements.txt"
    
    if req_file.exists():
        lines = []
        with open(req_file, 'r') as f:
            for line in f:
                if not line.strip().startswith(package_name):
                    lines.append(line)
        
        with open(req_file, 'w') as f:
            f.writelines(lines)

def check_outdated(package_manager, path="."):
    """Check for outdated packages."""
    commands = {
        "pip": ["pip", "list", "--outdated", "--format", "json"],
        "npm": ["npm", "outdated", "--json"],
        "yarn": ["yarn", "outdated", "--json"],
        "poetry": ["poetry", "show", "--outdated"]
    }
    
    if package_manager not in commands:
        return {"error": f"Unsupported package manager: {package_manager}"}
    
    try:
        result = subprocess.run(
            commands[package_manager],
            capture_output=True,
            text=True,
            cwd=path
        )
        
        outdated = []
        
        if package_manager == "pip" and result.stdout:
            outdated = json.loads(result.stdout)
        elif package_manager == "npm" and result.stdout:
            npm_data = json.loads(result.stdout)
            for pkg, info in npm_data.items():
                outdated.append({
                    "name": pkg,
                    "current": info.get("current", "unknown"),
                    "wanted": info.get("wanted", "unknown"),
                    "latest": info.get("latest", "unknown")
                })
        
        return {
            "outdated": outdated,
            "count": len(outdated)
        }
    
    except Exception as e:
        return {"error": str(e)}

def search_package(package_name, package_manager=None):
    """Search for a package in package registries."""
    results = []
    
    # Search PyPI
    try:
        response = requests.get(
            f"https://pypi.org/pypi/{package_name}/json",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            results.append({
                "registry": "PyPI",
                "name": data["info"]["name"],
                "version": data["info"]["version"],
                "description": data["info"]["summary"],
                "homepage": data["info"]["home_page"],
                "language": "Python"
            })
    except:
        pass
    
    # Search NPM
    try:
        response = requests.get(
            f"https://registry.npmjs.org/{package_name}",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            latest_version = data.get("dist-tags", {}).get("latest", "unknown")
            results.append({
                "registry": "NPM",
                "name": data["name"],
                "version": latest_version,
                "description": data.get("description", ""),
                "homepage": data.get("homepage", ""),
                "language": "JavaScript"
            })
    except:
        pass
    
    return results

def analyze_dependencies(package_manager, path="."):
    """Analyze project dependencies."""
    packages = get_installed_packages(package_manager, path)
    
    if isinstance(packages, dict) and "error" in packages:
        return packages
    
    analysis = {
        "total_packages": len(packages),
        "package_manager": package_manager,
        "packages": packages,
        "statistics": {}
    }
    
    # Analyze version constraints
    if package_manager in ["pip", "npm"]:
        exact_versions = 0
        range_versions = 0
        any_versions = 0
        
        for pkg in packages:
            version_spec = pkg.get("version", "")
            if version_spec in ["*", "latest", ""]:
                any_versions += 1
            elif "==" in version_spec or version_spec.startswith("^"):
                exact_versions += 1
            else:
                range_versions += 1
        
        analysis["statistics"] = {
            "exact_versions": exact_versions,
            "range_versions": range_versions,
            "any_versions": any_versions
        }
    
    # Check for security vulnerabilities (simplified)
    vulnerable_packages = check_vulnerabilities(packages, package_manager)
    analysis["vulnerabilities"] = vulnerable_packages
    
    return analysis

def check_vulnerabilities(packages, package_manager):
    """Check for known vulnerabilities (simplified version)."""
    # In a real implementation, this would query vulnerability databases
    # like Safety for Python or npm audit for JavaScript
    vulnerabilities = []
    
    # Example: Check for commonly vulnerable packages
    vulnerable_list = {
        "requests": {"<2.20.0": "Security vulnerability in versions before 2.20.0"},
        "django": {"<2.2": "Security updates required"},
        "lodash": {"<4.17.11": "Prototype pollution vulnerability"}
    }
    
    for pkg in packages:
        pkg_name = pkg.get("name", "").lower()
        if pkg_name in vulnerable_list:
            vulnerabilities.append({
                "package": pkg_name,
                "current_version": pkg.get("version", "unknown"),
                "vulnerability": list(vulnerable_list[pkg_name].values())[0]
            })
    
    return vulnerabilities

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input provided"}))
        return
    
    input_data = json.loads(sys.argv[1])
    
    action = input_data.get("action", "detect")
    path = input_data.get("path", ".")
    
    # Auto-detect package manager if not specified
    package_manager = input_data.get("package_manager")
    if not package_manager:
        package_manager = detect_package_manager(path)
        if not package_manager:
            print(json.dumps({"error": "Could not detect package manager"}))
            return
    
    if action == "detect":
        result = {"package_manager": package_manager}
    
    elif action == "list":
        packages = get_installed_packages(package_manager, path)
        result = {"packages": packages, "package_manager": package_manager}
    
    elif action == "add":
        package_name = input_data.get("package")
        version = input_data.get("version")
        dev = input_data.get("dev", False)
        
        if not package_name:
            result = {"error": "Package name is required"}
        else:
            result = add_package(package_manager, package_name, version, dev, path)
    
    elif action == "remove":
        package_name = input_data.get("package")
        
        if not package_name:
            result = {"error": "Package name is required"}
        else:
            result = remove_package(package_manager, package_name, path)
    
    elif action == "outdated":
        result = check_outdated(package_manager, path)
    
    elif action == "search":
        package_name = input_data.get("package")
        
        if not package_name:
            result = {"error": "Package name is required"}
        else:
            results = search_package(package_name, package_manager)
            result = {"results": results, "count": len(results)}
    
    elif action == "analyze":
        result = analyze_dependencies(package_manager, path)
    
    else:
        result = {"error": f"Unknown action: {action}"}
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
```

### Tool 7: Code Formatter (code_formatter.py)

```python
#!/usr/bin/env python3
"""
Format and lint code across different languages.
"""

import sys
import json
import subprocess
import tempfile
import os
from pathlib import Path

# Formatter configurations
FORMATTERS = {
    "python": {
        "black": {
            "command": ["black", "--quiet"],
            "check_command": ["black", "--check", "--quiet"],
            "config_file": "pyproject.toml",
            "file_extensions": [".py"],
            "config_template": '''[tool.black]
line-length = {line_length}
target-version = ['py38']
include = '\\.pyi?$'
extend-exclude = '''
    /(
      # directories
      \\.eggs
      | \\.git
      | \\.hg
      | \\.mypy_cache
      | \\.tox
      | \\.venv
      | build
      | dist
    )/
'''
'''
        },
        "autopep8": {
            "command": ["autopep8", "--in-place"],
            "check_command": ["autopep8", "--diff"],
            "file_extensions": [".py"],
            "options": {
                "max_line_length": "--max-line-length",
                "aggressive": "--aggressive"
            }
        },
        "isort": {
            "command": ["isort"],
            "check_command": ["isort", "--check-only"],
            "file_extensions": [".py"],
            "config_file": ".isort.cfg",
            "config_template": '''[settings]
profile = black
line_length = {line_length}
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
'''
        }
    },
    "javascript": {
        "prettier": {
            "command": ["prettier", "--write"],
            "check_command": ["prettier", "--check"],
            "config_file": ".prettierrc",
            "file_extensions": [".js", ".jsx", ".ts", ".tsx", ".json", ".css", ".scss"],
            "config_template": '''{
  "semi": {semi},
  "trailingComma": "{trailing_comma}",
  "singleQuote": {single_quote},
  "printWidth": {print_width},
  "tabWidth": {tab_width},
  "useTabs": {use_tabs},
  "bracketSpacing": {bracket_spacing},
  "arrowParens": "{arrow_parens}",
  "endOfLine": "{end_of_line}"
}'''
        },
        "eslint": {
            "command": ["eslint", "--fix"],
            "check_command": ["eslint"],
            "file_extensions": [".js", ".jsx", ".ts", ".tsx"],
            "config_file": ".eslintrc.json",
            "config_template": '''{
  "env": {
    "browser": true,
    "es2021": true,
    "node": true
  },
  "extends": [
    "eslint:recommended"
  ],
  "parserOptions": {
    "ecmaVersion": "latest",
    "sourceType": "module"
  },
  "rules": {
    "indent": ["error", {indent_size}],
    "linebreak-style": ["error", "{linebreak_style}"],
    "quotes": ["error", "{quote_style}"],
    "semi": ["error", "{semi_style}"]
  }
}'''
        }
    },
    "css": {
        "prettier": {
            "command": ["prettier", "--write"],
            "check_command": ["prettier", "--check"],
            "file_extensions": [".css", ".scss", ".sass", ".less"]
        },
        "stylelint": {
            "command": ["stylelint", "--fix"],
            "check_command": ["stylelint"],
            "file_extensions": [".css", ".scss", ".sass", ".less"],
            "config_file": ".stylelintrc.json",
            "config_template": '''{
  "extends": "stylelint-config-standard",
  "rules": {
    "indentation": {indent_size},
    "string-quotes": "{quote_style}",
    "max-line-length": {max_line_length}
  }
}'''
        }
    },
    "markdown": {
        "prettier": {
            "command": ["prettier", "--write"],
            "check_command": ["prettier", "--check"],
            "file_extensions": [".md", ".mdx"]
        },
        "markdownlint": {
            "command": ["markdownlint", "--fix"],
            "check_command": ["markdownlint"],
            "file_extensions": [".md"],
            "config_file": ".markdownlint.json",
            "config_template": '''{
  "default": true,
  "MD013": {
    "line_length": {line_length}
  },
  "MD033": false,
  "MD041": false
}'''
        }
    },
    "yaml": {
        "prettier": {
            "command": ["prettier", "--write"],
            "check_command": ["prettier", "--check"],
            "file_extensions": [".yml", ".yaml"]
        }
    },
    "json": {
        "prettier": {
            "command": ["prettier", "--write"],
            "check_command": ["prettier", "--check"],
            "file_extensions": [".json"]
        }
    }
}

def detect_language(file_path):
    """Detect language from file extension."""
    ext = Path(file_path).suffix.lower()
    
    for lang, formatters in FORMATTERS.items():
        for formatter_name, formatter_config in formatters.items():
            if ext in formatter_config.get("file_extensions", []):
                return lang
    
    return None

def format_code(code, language, formatter=None, options=None):
    """Format code using specified formatter."""
    if language not in FORMATTERS:
        return {"error": f"Unsupported language: {language}"}
    
    # Select formatter
    if not formatter:
        formatter = list(FORMATTERS[language].keys())[0]
    
    if formatter not in FORMATTERS[language]:
        return {"error": f"Unsupported formatter: {formatter} for {language}"}
    
    formatter_config = FORMATTERS[language][formatter]
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=formatter_config["file_extensions"][0], delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        
        # Build command
        cmd = formatter_config["command"].copy()
        
        # Add options if provided
        if options and "options" in formatter_config:
            for opt_key, opt_value in options.items():
                if opt_key in formatter_config["options"]:
                    cmd.extend([formatter_config["options"][opt_key], str(opt_value)])
        
        cmd.append(tmp_path)
        
        # Run formatter
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Read formatted code
            with open(tmp_path, 'r') as f:
                formatted_code = f.read()
            
            return {
                "success": True,
                "formatted_code": formatted_code,
                "formatter": formatter,
                "language": language
            }
        else:
            return {
                "success": False,
                "error": result.stderr or result.stdout,
                "formatter": formatter
            }
    
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        # Cleanup
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def format_file(file_path, formatter=None, options=None):
    """Format a file in place."""
    if not Path(file_path).exists():
        return {"error": f"File not found: {file_path}"}
    
    # Detect language
    language = detect_language(file_path)
    if not language:
        return {"error": f"Could not detect language for file: {file_path}"}
    
    # Read file
    with open(file_path, 'r') as f:
        original_code = f.read()
    
    # Format code
    result = format_code(original_code, language, formatter, options)
    
    if result.get("success"):
        # Write back to file
        with open(file_path, 'w') as f:
            f.write(result["formatted_code"])
        
        # Check if code changed
        result["changed"] = original_code != result["formatted_code"]
        result["file_path"] = file_path
    
    return result

def format_directory(directory_path, languages=None, formatter=None, options=None, exclude=None):
    """Format all files in a directory."""
    directory = Path(directory_path)
    if not directory.exists():
        return {"error": f"Directory not found: {directory_path}"}
    
    # Default excludes
    default_excludes = {
        "node_modules", "__pycache__", ".git", ".venv", "venv", 
        "build", "dist", ".next", ".cache"
    }
    
    if exclude:
        default_excludes.update(exclude)
    
    results = {
        "total_files": 0,
        "formatted_files": 0,
        "failed_files": 0,
        "unchanged_files": 0,
        "files": []
    }
    
    # Find all files
    for file_path in directory.rglob("*"):
        # Skip excluded directories
        if any(excl in file_path.parts for excl in default_excludes):
            continue
        
        if file_path.is_file():
            language = detect_language(file_path)
            
            # Skip if language not specified or not detected
            if languages and language not in languages:
                continue
            
            if not language:
                continue
            
            results["total_files"] += 1
            
            # Format file
            file_result = format_file(str(file_path), formatter, options)
            
            if file_result.get("success"):
                if file_result.get("changed"):
                    results["formatted_files"] += 1
                else:
                    results["unchanged_files"] += 1
                
                results["files"].append({
                    "path": str(file_path),
                    "changed": file_result.get("changed", False),
                    "formatter": file_result.get("formatter")
                })
            else:
                results["failed_files"] += 1
                results["files"].append({
                    "path": str(file_path),
                    "error": file_result.get("error", "Unknown error")
                })
    
    return results

def check_format(code_or_path, language=None, formatter=None):
    """Check if code or file is properly formatted."""
    # Determine if input is code or file path
    is_file = Path(code_or_path).exists() if len(code_or_path) < 260 else False
    
    if is_file:
        file_path = code_or_path
        with open(file_path, 'r') as f:
            code = f.read()
        
        if not language:
            language = detect_language(file_path)
    else:
        code = code_or_path
        if not language:
            return {"error": "Language must be specified for code input"}
    
    if not language or language not in FORMATTERS:
        return {"error": f"Unsupported language: {language}"}
    
    # Select formatter
    if not formatter:
        formatter = list(FORMATTERS[language].keys())[0]
    
    formatter_config = FORMATTERS[language][formatter]
    
    if "check_command" not in formatter_config:
        return {"error": f"Formatter {formatter} does not support format checking"}
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=formatter_config["file_extensions"][0], delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        
        # Run check command
        cmd = formatter_config["check_command"].copy()
        cmd.append(tmp_path)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            "formatted": result.returncode == 0,
            "formatter": formatter,
            "language": language,
            "output": result.stdout or result.stderr
        }
    
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        # Cleanup
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def generate_formatter_config(language, formatter, options=None):
    """Generate configuration file for formatter."""
    if language not in FORMATTERS:
        return {"error": f"Unsupported language: {language}"}
    
    if formatter not in FORMATTERS[language]:
        return {"error": f"Unsupported formatter: {formatter} for {language}"}
    
    formatter_config = FORMATTERS[language][formatter]
    
    if "config_template" not in formatter_config:
        return {"error": f"No configuration template available for {formatter}"}
    
    # Default options
    default_options = {
        "line_length": 88,
        "indent_size": 4,
        "tab_width": 4,
        "use_tabs": "false",
        "semi": "true",
        "single_quote": "true",
        "trailing_comma": "es5",
        "print_width": 80,
        "bracket_spacing": "true",
        "arrow_parens": "avoid",
        "end_of_line": "lf",
        "quote_style": "single",
        "semi_style": "always",
        "linebreak_style": "unix",
        "max_line_length": 120
    }
    
    # Merge with provided options
    if options:
        default_options.update(options)
    
    # Generate config
    config_content = formatter_config["config_template"].format(**default_options)
    
    return {
        "formatter": formatter,
        "language": language,
        "config_file": formatter_config.get("config_file", f".{formatter}rc"),
        "config_content": config_content
    }

def analyze_code_style(code_or_path, language=None):
    """Analyze code style and suggest formatters."""
    # Determine if input is code or file path
    is_file = Path(code_or_path).exists() if len(code_or_path) < 260 else False
    
    if is_file:
        with open(code_or_path, 'r') as f:
            code = f.read()
        
        if not language:
            language = detect_language(code_or_path)
    else:
        code = code_or_path
    
    if not language:
        return {"error": "Could not detect language"}
    
    analysis = {
        "language": language,
        "style_metrics": {},
        "available_formatters": list(FORMATTERS.get(language, {}).keys()),
        "recommendations": []
    }
    
    # Analyze style metrics
    lines = code.split('\n')
    
    # Line length analysis
    line_lengths = [len(line) for line in lines]
    if line_lengths:
        analysis["style_metrics"]["max_line_length"] = max(line_lengths)
        analysis["style_metrics"]["avg_line_length"] = sum(line_lengths) / len(line_lengths)
        
        if max(line_lengths) > 120:
            analysis["recommendations"].append("Consider setting a maximum line length (e.g., 88 or 120 characters)")
    
    # Indentation analysis
    indent_sizes = []
    for line in lines:
        if line and line[0] in ' \t':
            indent = len(line) - len(line.lstrip())
            if line[0] == ' ':
                indent_sizes.append(indent)
    
    if indent_sizes:
        # Find most common indent size
        from collections import Counter
        indent_counter = Counter(indent_sizes)
        common_indent = indent_counter.most_common(1)[0][0] if indent_counter else 4
        analysis["style_metrics"]["common_indent_size"] = common_indent
        
        # Check for mixed indentation
        has_tabs = any('\t' in line for line in lines)
        has_spaces = any(line.startswith(' ') for line in lines)
        
        if has_tabs and has_spaces:
            analysis["recommendations"].append("Mixed tabs and spaces detected - choose one style")
    
    # Language-specific analysis
    if language == "python":
        # Check for PEP 8 style
        import_lines = [i for i, line in enumerate(lines) if line.strip().startswith(('import ', 'from '))]
        if import_lines and any(i > 0 and lines[i-1].strip() and not lines[i-1].strip().startswith(('import ', 'from ')) for i in import_lines):
            analysis["recommendations"].append("Consider grouping imports at the top of the file")
    
    elif language == "javascript":
        # Check semicolon usage
        semicolon_lines = sum(1 for line in lines if line.rstrip().endswith(';'))
        total_statement_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith('//'))
        
        if total_statement_lines > 0:
            semicolon_ratio = semicolon_lines / total_statement_lines
            if 0 < semicolon_ratio < 0.5:
                analysis["recommendations"].append("Inconsistent semicolon usage detected")
    
    return analysis

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input provided"}))
        return
    
    input_data = json.loads(sys.argv[1])
    
    action = input_data.get("action", "format")
    
    if action == "format":
        code = input_data.get("code")
        file_path = input_data.get("file_path")
        language = input_data.get("language")
        formatter = input_data.get("formatter")
        options = input_data.get("options", {})
        
        if code:
            result = format_code(code, language, formatter, options)
        elif file_path:
            result = format_file(file_path, formatter, options)
        else:
            result = {"error": "Either 'code' or 'file_path' must be provided"}
    
    elif action == "format_directory":
        directory = input_data.get("directory", ".")
        languages = input_data.get("languages")
        formatter = input_data.get("formatter")
        options = input_data.get("options", {})
        exclude = input_data.get("exclude", [])
        
        result = format_directory(directory, languages, formatter, options, exclude)
    
    elif action == "check":
        code_or_path = input_data.get("code") or input_data.get("file_path")
        language = input_data.get("language")
        formatter = input_data.get("formatter")
        
        if not code_or_path:
            result = {"error": "Either 'code' or 'file_path' must be provided"}
        else:
            result = check_format(code_or_path, language, formatter)
    
    elif action == "generate_config":
        language = input_data.get("language")
        formatter = input_data.get("formatter")
        options = input_data.get("options", {})
        
        if not language or not formatter:
            result = {"error": "Both 'language' and 'formatter' are required"}
        else:
            result = generate_formatter_config(language, formatter, options)
    
    elif action == "analyze":
        code_or_path = input_data.get("code") or input_data.get("file_path")
        language = input_data.get("language")
        
        if not code_or_path:
            result = {"error": "Either 'code' or 'file_path' must be provided"}
        else:
            result = analyze_code_style(code_or_path, language)
    
    elif action == "list_formatters":
        language = input_data.get("language")
        
        if language:
            result = {
                "language": language,
                "formatters": list(FORMATTERS.get(language, {}).keys())
            }
        else:
            result = {"formatters": FORMATTERS}
    
    else:
        result = {"error": f"Unknown action: {action}"}
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
```

## Step 3: Create functions.json

Now we need to create the `functions.json` file that declares all our tools to AIChat:

```json
{
  "name": "coding_tools",
  "description": "Comprehensive coding toolset for development tasks",
  "version": "1.0.0",
  "functions": [
    {
      "name": "analyze_code",
      "description": "Analyze code for complexity, issues, and improvement suggestions",
      "parameters": {
        "type": "object",
        "properties": {
          "code": {
            "type": "string",
            "description": "The code to analyze"
          },
          "file_path": {
            "type": "string",
            "description": "Path to the file to analyze"
          },
          "language": {
            "type": "string",
            "enum": ["python", "javascript", "auto"],
            "description": "Programming language (auto-detect if not specified)"
          }
        },
        "required": []
      },
      "command": "python code_analyzer.py"
    },
    {
      "name": "generate_code",
      "description": "Generate code snippets, boilerplate, and templates",
      "parameters": {
        "type": "object",
        "properties": {
          "action": {
            "type": "string",
            "enum": ["generate", "snippet"],
            "description": "Action to perform"
          },
          "language": {
            "type": "string",
            "enum": ["python", "javascript"],
            "description": "Programming language"
          },
          "template_type": {
            "type": "string",
            "description": "Type of template to generate (class, function, test, etc.)"
          },
          "snippet_type": {
            "type": "string",
            "description": "Type of snippet to generate"
          },
          "config": {
            "type": "object",
            "description": "Configuration for the template"
          }
        },
        "required": ["language"]
      },
      "command": "python code_generator.py"
    },
    {
      "name": "run_tests",
      "description": "Run tests and analyze test coverage",
      "parameters": {
        "type": "object",
        "properties": {
          "action": {
            "type": "string",
            "enum": ["run", "generate"],
            "description": "Action to perform"
          },
          "test_path": {
            "type": "string",
            "description": "Path to test file or directory"
          },
          "language": {
            "type": "string",
            "enum": ["python", "javascript"],
            "description": "Programming language"
          },
          "coverage": {
            "type": "boolean",
            "description": "Enable coverage analysis"
          },
          "code": {
            "type": "string",
            "description": "Code to generate tests for"
          }
        },
        "required": ["action"]
      },
      "command": "python test_runner.py"
    },
    {
      "name": "generate_docs",
      "description": "Generate documentation from code",
      "parameters": {
        "type": "object",
        "properties": {
          "action": {
            "type": "string",
            "enum": ["extract", "generate_markdown", "generate_api"],
            "description": "Action to perform"
          },
          "code": {
            "type": "string",
            "description": "Code to document"
          },
          "file_path": {
            "type": "string",
            "description": "Path to file to document"
          },
          "language": {
            "type": "string",
            "enum": ["python", "javascript"],
            "description": "Programming language"
          }
        },
        "required": ["action"]
      },
      "command": "python doc_generator.py"
    },
    {
      "name": "git_operations",
      "description": "Perform Git operations and repository management",
      "parameters": {
        "type": "object",
        "properties": {
          "action": {
            "type": "string",
            "enum": ["status", "commit", "history", "branch", "generate_message"],
            "description": "Git action to perform"
          },
          "path": {
            "type": "string",
            "description": "Repository path"
          },
          "message": {
            "type": "string",
            "description": "Commit message"
          },
          "files": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Files to commit"
          },
          "name": {
            "type": "string",
            "description": "Branch name"
          },
          "from": {
            "type": "string",
            "description": "Base branch for new branch"
          },
          "limit": {
            "type": "integer",
            "description": "Number of commits to show in history"
          },
          "type": {
            "type": "string",
            "enum": ["feat", "fix", "docs", "style", "refactor", "perf", "test", "chore"],
            "description": "Commit type for conventional commits"
          }
        },
        "required": ["action"]
      },
      "command": "python git_helper.py"
    },
    {
      "name": "scaffold_project",
      "description": "Create new projects or add features to existing projects",
      "parameters": {
        "type": "object",
        "properties": {
          "action": {
            "type": "string",
            "enum": ["create", "add_feature", "list_templates"],
            "description": "Action to perform"
          },
          "type": {
            "type": "string",
            "enum": ["python", "javascript"],
            "description": "Project type"
          },
          "template": {
            "type": "string",
            "description": "Template name (fastapi, cli, react, express)"
          },
          "name": {
            "type": "string",
            "description": "Project name"
          },
          "description": {
            "type": "string",
            "description": "Project description"
          },
          "path": {
            "type": "string",
            "description": "Path where to create project"
          },
          "project_path": {
            "type": "string",
            "description": "Path to existing project"
          },
          "feature": {
            "type": "string",
            "enum": ["database", "auth"],
            "description": "Feature to add"
          },
          "config": {
            "type": "object",
            "description": "Feature configuration"
          }
        },
        "required": ["action"]
      },
      "command": "python project_scaffolder.py"
    },
    {
      "name": "manage_packages",
      "description": "Manage project dependencies and packages",
      "parameters": {
        "type": "object",
        "properties": {
          "action": {
            "type": "string",
            "enum": ["detect", "list", "add", "remove", "outdated", "search", "analyze"],
            "description": "Package management action"
          },
          "path": {
            "type": "string",
            "description": "Project path"
          },
          "package_manager": {
            "type": "string",
            "enum": ["pip", "npm", "yarn", "poetry", "pipenv"],
            "description": "Package manager to use"
          },
          "package": {
            "type": "string",
            "description": "Package name"
          },
          "version": {
            "type": "string",
            "description": "Package version"
          },
          "dev": {
            "type": "boolean",
            "description": "Install as development dependency"
          }
        },
        "required": ["action"]
      },
      "command": "python package_manager.py"
    },
    {
      "name": "format_code",
      "description": "Format and lint code across different languages",
      "parameters": {
      Of course. Let's go deeper.

This in-depth guide moves beyond the initial setup and explores the advanced concepts, best practices, and security considerations necessary to transform your `aichat` agent into a truly sophisticated and reliable coding collaborator.

## From Assistant to Collaborator: An In-Depth Guide to Advanced `aichat` Tooling

In the previous tutorial, we built a basic toolset for file I/O. Now, we will elevate that foundation by exploring the core mechanics of tool use, building a more advanced and practical set of coding tools, and implementing robust error handling and security measures.

### The Core Mechanism: How Tool Use *Really* Works

Understanding the conversation flow between you, `aichat`, the LLM, and your tools is key to mastering them. It's not magic; it's a well-defined, multi-step process:

1.  **User Prompt:** You give the agent a command, like "Run the test suite and tell me if it passes."
2.  **LLM Reasoning:** `aichat` sends your prompt, the conversation history, and the list of available tools (including their names and descriptions) to the Gemini model. The LLM analyzes your request and determines that it cannot answer directly. It recognizes that the task "run the test suite" matches the description of one of its available tools.
3.  **Tool Request (Function Call):** The LLM decides to use a tool. It formulates a "tool request" object, which includes the `name` of the tool to use and the `arguments` it requires. For example: `{ "name": "run_tests", "arguments": { "test_file": "tests/test_main.py" } }`. It sends this object back to `aichat`.
4.  **`aichat` Execution:** `aichat` receives the tool request. It does **not** execute it blindly. It first prompts you for confirmation, showing you the exact command it's about to run. This is a critical security checkpoint.
5.  **User Confirmation:** You see the prompt: `Call `run_tests` with `test_file`="tests/test_main.py", execute `pytest tests/test_main.py`? (Y/n)`. You press `Y`.
6.  **Shell Execution & Output Capture:** `aichat` executes the command (`pytest tests/test_main.py`) in your local shell. It captures the command's output (`stdout`) and any errors (`stderr`).
7.  **LLM Analysis of Output:** `aichat` sends the captured output back to the LLM as the result of the tool call.
8.  **Final Response:** The LLM receives the tool's output (e.g., "3 passed, 1 failed"). It now has the information it needs to fulfill your original request. It formulates a final, human-readable answer, such as "The test suite ran, but one test failed." and sends this to `aichat` to be displayed to you.

This loop is what makes the agent interactive and powerful. It can reason, act, and observe the results of its actions.

### Building an Advanced Coding Toolset

Let's upgrade our `flash-assistant` with a more practical set of tools for a typical development workflow. We will add tools to run code, search for content within files, and check the Git status.

Navigate to your agent's `config.yaml` and replace the previous `tools` section with this more advanced set.

**Agent `config.yaml` for Advanced Coding:**

```yaml
# In ~/.config/aichat/agents/flash-assistant/config.yaml

model: gemini:gemini-2.5-flash

instructions: |
  You are an expert AI programming assistant.
  Your capabilities include reading, writing, searching, and executing code.
  - Before writing or modifying a file, always confirm the file path with the user.
  - When executing code, you must use the `run_code` tool. Analyze both the output and any errors to debug effectively.
  - To find specific functions or text in the codebase, use the `search_in_files` tool.
  - Always check the version control status with `git_status` before making significant changes.
  - Be concise and clear in your responses.

tools:
  - name: list_files
    description: List all files and directories in the current working directory, including hidden files.
    command: ls -aF

  - name: read_file
    description: Read the entire content of a specified file to understand its code or text.
    command: cat {path}
    args:
      - name: path
        description: The relative path to the file to be read.

  - name: write_file
    description: Write or overwrite a file with new content. Use with caution.
    command: echo "{content}" | tee {path} > /dev/null
    args:
      - name: path
        description: The path to the file to be written.
      - name: content
        description: The new content to write to the file.

  - name: run_code
    description: Execute a script (e.g., Python, Bash) and capture its output and errors.
    command: python3 {file_path} 2>&1
    args:
      - name: file_path
        description: The path to the script to execute.

  - name: search_in_files
    description: Search for a specific text pattern within files in the current directory.
    command: grep -r "{pattern}" .
    args:
      - name: pattern
        description: The text or regex pattern to search for.

  - name: git_status
    description: Check the current status of the git repository (untracked files, changes).
    command: git status --short
```

#### Key Improvements in This Toolset:

*   **`run_code`:** This is the cornerstone of our coding assistant. The crucial part is `2>&1`. This shell redirection sends the standard error (`stderr`, channel 2) to the same place as standard output (`stdout`, channel 1). This means the LLM will see **both the output and the error messages**, which is essential for debugging.
*   **`search_in_files`:** Using `grep` allows the agent to find information without having to read every single file, making it much more efficient.
*   **`git_status`:** Integrating with version control is a professional workflow staple. This read-only command gives the agent context about the state of the project.
*   **Enhanced Instructions:** The `instructions` prompt is now more detailed, guiding the agent on *how* and *when* to use its new tools, promoting safer and more effective behavior.

### Best Practices for Robust and Secure Agents

Giving an LLM the ability to execute shell commands is powerful but carries inherent risks.

#### 1. The Critical Importance of Security

**You are giving an AI control of your terminal.** A poorly constructed command or a compromised LLM could theoretically execute malicious code.

*   **The Human in the Loop:** `aichat`'s default behavior of asking for confirmation before executing any tool is your most important safety feature. **Never disable it.** Always read the command it proposes before typing `Y`.
*   **Principle of Least Privilege:** Do not give the agent tools it doesn't need. Avoid creating tools that use destructive commands like `rm`, `mv`, or `sed -i`. Favor read-only commands (`ls`, `cat`, `grep`, `git status`) whenever possible.
*   **Sandboxing:** For maximum security, run `aichat` inside a containerized environment like Docker. This isolates the agent's actions from your main operating system. If it misbehaves, only the container is affected.

#### 2. The Art of Instruction (Prompt Engineering)

The `instructions` block is your primary way of controlling the agent's behavior.

*   **Be Specific:** Instead of "You are a coding assistant," write "You are an expert Python developer. Your goal is to write clean, efficient, and well-documented code."
*   **Define Constraints:** Add rules to prevent undesirable actions. For example: `"Never write a file without first using read_file to ensure it doesn't contain important information."`
*   **Shape the Personality:** You can guide its communication style: `"Provide explanations in a clear, step-by-step format. Use markdown for code blocks."`

### A Practical Workflow Example: Finding and Fixing a Bug

Let's walk through a complete scenario using our advanced agent.

**1. Setup:** Create a buggy Python file named `main.py`.

```bash
echo 'def greet(name)
  print(f"Hello, {name}!")

greet("Alice")' > main.py
```
*(Note the missing colon `:` in the function definition)*

**2. Interaction with the Agent:**

```bash
aichat
〉.agent flash-assistant
Welcome to agent `flash-assistant`

flash-assistant〉Please run the python script in this directory.
```

**Agent uses `run_code`:**
```
Tool Call: `run_code` with `file_path`="main.py"
Tool Output:
  File "/path/to/your/project/main.py", line 1
    def greet(name)
                   ^
SyntaxError: expected ':'

The script `main.py` failed to execute due to a `SyntaxError`. It seems a colon `:` is missing at the end of the function definition on line 1.
```

**You continue the conversation:**
```
flash-assistant〉You are correct. Please fix the file.
```

**Agent uses `write_file` (after reasoning what the correct code should be):**
```
Tool Call: `write_file` with `path`="main.py" and `content`="def greet(name):
  print(f\"Hello, {name}!\")

greet(\"Alice\")"

I have corrected the syntax error in `main.py`.
```

**You verify the fix:**
```
flash-assistant〉Great. Run it again to confirm it works.
```

**Agent uses `run_code` again:**
```
Tool Call: `run_code` with `file_path`="main.py"
Tool Output:
Hello, Alice!

The script now runs successfully and prints "Hello, Alice!".
```

This workflow demonstrates a complete "thought-act-observe" loop, showcasing how a well-designed toolset and clear instructions empower the agent to be a genuine collaborator in the development process. By going in-depth, you unlock the true potential of command-line AI.
