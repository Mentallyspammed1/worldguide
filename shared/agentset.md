This is an excellent and comprehensive guide to setting up AIChat agents with custom tools and functions. The structure is logical, and the explanations are clear. Here's an enhanced version focusing on clarity, conciseness, and flow, along with some minor improvements and suggestions for even greater impact.

---

# **Mastering AIChat: A Complete Guide to Custom Agent Setup**

This guide provides a thorough walkthrough for setting up AIChat agents, integrating custom tools, and leveraging function calling. We'll cover everything from initial installation to advanced configuration.

## Table of Contents

1.  **Introduction & Architecture**
    *   AI Agent Formula: Instructions + Tools + Documents
    *   Key Components
2.  **Getting Started: Prerequisites & Installation**
    *   Installing AIChat
    *   Initial Configuration
    *   Setting Up the LLM Functions Repository
    *   Linking LLM Functions to AIChat
3.  **Crafting Custom Tools**
    *   Tool Structure & Declaration (Argc)
    *   Bash Tool Example
    *   Python Tool Example
    *   JavaScript Tool Example
4.  **Building Your First Agent**
    *   Agent Directory Structure
    *   Agent Definition (`index.yaml`)
    *   Agent Configuration (`config.yaml`)
    *   Dynamic Instructions
5.  **Configuration Deep Dive**
    *   Main AIChat Configuration (`config.yaml`)
    *   Tool Configuration (`tools.txt`)
    *   Agent Enablement (`agents.txt`)
6.  **Running & Testing Your Agent**
    *   Using Tools Directly
    *   Interacting with Agents
    *   REPL Mode with Agents
    *   Testing Tools with Argc
7.  **Advanced Features & Best Practices**
    *   Tool Mapping & Grouping
    *   Session Management (Agent Preludes)
    *   RAG Integration
    *   MCP Bridge Support
    *   Custom Function Groups
    *   Best Practices (Tool Design, Instructions, Security, Performance)
8.  **Troubleshooting Common Issues**
9.  **Conclusion**

---

## 1. Introduction & Architecture

### AI Agent Formula

An AI agent's capability is built upon a simple yet powerful formula:

**AI Agent = Instructions (Prompt) + Tools (Function Callings) + Documents (RAG)**

*   **Instructions**: Define the agent's behavior, persona, and goals through system prompts.
*   **Tools**: Enable agents to interact with the external world by executing pre-defined functions.
*   **Documents**: Provide contextual knowledge through Retrieval-Augmented Generation (RAG).

### Key Components

*   **Instructions**: The core directives that shape the agent's responses and actions.
*   **Tools**: Reusable functions that extend the agent's capabilities (e.g., fetching data, executing commands).
*   **Documents**: External knowledge sources that the agent can query for information.
*   **Variables**: Dynamic parameters that can customize agent behavior.

---

## 2. Getting Started: Prerequisites & Installation

### Installing AIChat

AIChat is distributed as a command-line tool. Download the pre-built binaries for macOS, Linux, and Windows from the [GitHub Releases](https://github.com/sigoden/aichat/releases) page. Extract the archive and ensure the `aichat` binary is in your system's `$PATH`.

### Initial Configuration

The first time you launch `aichat`, it will guide you through an initial setup. You'll be prompted to select your preferred AI Platform (e.g., OpenAI, Gemini, Ollama) and provide your API Key.

### Setting Up the LLM Functions Repository

AIChat relies on the `llm-functions` repository for managing custom tools.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sigoden/llm-functions
    cd llm-functions
    ```

2.  **Install Argc:**
    `argc` is a powerful framework for building command-line tools, used here to manage functions. Ensure it's installed. If not, follow instructions from its repository.

### Linking LLM Functions to AIChat

AIChat needs to know where to find your custom functions. You have two main options:

1.  **Symlink the Directory**:
    Create a symbolic link from the `llm-functions` directory to AIChat's designated functions directory.
    ```bash
    # Get AIChat's functions directory and create a symlink
    ln -s "$(pwd)" "$(aichat --info | sed -n 's/^functions_dir\s\+//p')"
    ```
    Alternatively, use the convenience command:
    ```bash
    argc link-to-aichat
    ```

2.  **Environment Variable**:
    Set the `AICHAT_FUNCTIONS_DIR` environment variable to point to your `llm-functions` directory:
    ```bash
    export AICHAT_FUNCTIONS_DIR=/path/to/your/llm-functions
    ```
    *(Replace `/path/to/your/llm-functions` with the actual path.)*

---

## 3. Crafting Custom Tools

LLM Functions uses a convention-based approach to define tools, primarily through comments in your script files. `argc` parses these comments to generate tool metadata.

### Tool Structure & Declaration

Tools are typically placed in the `llm-functions/tools/` directory. Each tool script should include special comments:

*   `# @describe <description>`: A brief explanation of what the tool does.
*   `# @option --<name>!<type>=<default> <description>`: Defines a parameter.
    *   `!` makes the parameter required.
    *   `<type>` specifies the data type (e.g., `string`, `int`, `bool`).
    *   `<default>` sets a default value if the parameter is optional.

### Bash Tool Example

Create a file like `llm-functions/tools/get_current_weather.sh`:

```bash
#!/usr/bin/env bash
set -e

# @describe Get current weather for a specified location.
# @option --location! The city name or location (e.g., "London").

main() {
    # Use argc_location to access the --location argument
    # LLM_OUTPUT is a special variable where tool results are written
    curl -s "https://wttr.in/${argc_location}?format=j1" | jq -r '"Location: \(.nearest_area[0].areaName[0].value), Temperature: \(.current_condition[0].temp_C)°C, Condition: \(.current_condition[0].weatherDesc[0].value)"' >> "$LLM_OUTPUT"
}

# This line is crucial for argc to parse the script
eval "$(argc --argc-eval "$0" "$@")"
```

### Python Tool Example

Create `llm-functions/tools/custom_calculator.py`:

```python
#!/usr/bin/env python3
import sys
import json
import math

# @describe Perform advanced mathematical calculations.
# @option --expression! str The mathematical expression to evaluate.
# @option --precision int=2 Number of decimal places for the result.

def main():
    # Arguments are passed as a JSON string in sys.argv[1]
    args_json = sys.argv[1]
    args = json.loads(args_json)
    
    expression = args.get('expression')
    precision = args.get('precision', 2)
    
    try:
        # Safely evaluate the expression
        # __builtins__ is restricted for security
        result = eval(expression, {"__builtins__": {}}, {"math": math, "abs": abs, "round": round})
        result = round(result, precision)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### JavaScript Tool Example

Create `llm-functions/tools/web_scraper.js`:

```javascript
#!/usr/bin/env node

// @describe Scrape web content from a URL.
// @option --url! The URL to scrape.
// @option --selector CSS selector for specific content (optional).

const puppeteer = require('puppeteer');

async function main() {
    const args = JSON.parse(process.argv[2]);
    const url = args.url;
    const selector = args.selector || 'body'; // Default to body if no selector provided
    
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.goto(url, { waitUntil: 'networkidle2' }); // Wait until network is idle
    
    const content = await page.evaluate((sel) => {
        const element = document.querySelector(sel);
        return element ? element.textContent.trim() : "Selector not found.";
    }, selector);
    
    console.log(content);
    await browser.close();
}

main().catch(console.error);
```

**Note**: For Node.js tools, ensure you have `puppeteer` installed (`npm install puppeteer`).

---

## 4. Building Your First Agent

### Agent Directory Structure

An agent's configuration and custom logic reside in its own directory within AIChat's agent configuration path.

```
~/.config/aichat/
└── agents/
    ├── my_custom_agent/
    │   ├── config.yaml     # Agent-specific settings (model, temp, etc.)
    │   ├── index.yaml      # Core agent definition (name, description, instructions, tools)
    │   ├── tools.txt       # List of tools this agent can use
    │   └── docs/           # Directory for RAG documents
    │       └── coding_standards.md
    │       └── api_reference.md
    └── another_agent/
        └── ...
```

### Agent Definition (`index.yaml`)

This file defines the agent's core properties. Example: `~/.config/aichat/agents/developer/index.yaml`:

```yaml
name: Developer Assistant
description: AI agent for software development tasks
version: 1.0.0
instructions: |
  You are an expert software developer assistant. You help with:
  - Writing and reviewing code
  - Debugging and optimization
  - System design and architecture
  - Best practices and design patterns
  
  Always consider security, performance, and maintainability.
  Provide clear explanations with your code suggestions.

tools:
  - execute_command  # Refers to tools managed by llm-functions
  - fs_cat
  - fs_write
  - fs_ls
  - web_search
  - custom_calculator # Example of a custom tool

documents:
  - ./docs/coding-standards.md
  - ./docs/api-reference.md

variables:
  language: python
  framework: django
  
dynamic_instructions: false # Set to true for dynamic instructions (see below)
```

### Agent Configuration (`config.yaml`)

This file allows for overrides and specific settings for an agent. Example: `~/.config/aichat/agents/developer/config.yaml`:

```yaml
model: openai:gpt-4o             # Specify the LLM to use for this agent
temperature: 0.7                 # Override default temperature
top_p: 0.9                       # Override default top-p
use_tools: 'fs,web_search'       # Additional tools to enable for this agent
agent_prelude: default           # Session to use when starting this agent
instructions: null               # Override instructions from index.yaml if needed
variables:                       # Custom default values for agent variables
  debug_mode: true
  output_format: markdown
```

### Dynamic Instructions

For agents that need context-aware instructions, set `dynamic_instructions: true` in `index.yaml` and provide a script that generates instructions.

Example: `~/.config/aichat/agents/dynamic_agent/dynamic_agent.sh`:

```bash
#!/usr/bin/env bash
# agents/dynamic_agent/dynamic_agent.sh

# This function generates dynamic instructions based on current context
_instructions() {
    cat <<EOF
Current time: $(date)
System: $(uname -s)
User: $USER
Working directory: $(pwd)

You are a context-aware assistant with access to real-time system information.
EOF
}

# Other tool implementations for this agent would go here...
# e.g., _execute_command() { ... }
```

---

## 5. Configuration Deep Dive

### Main AIChat Configuration (`~/.config/aichat/config.yaml`)

This is your global configuration file.

```yaml
model: openai:gpt-4o             # Default LLM
stream: true                     # Default streaming behavior
save: true                       # Default session saving

keybindings: emacs               # Editor keybindings (e.g., emacs, vim)
highlight: true                  # Code highlighting

# Function calling configuration
function_calling: true         # Enable function calling globally
mapping_tools:                   # Define tool aliases and groups
  fs: 'fs_cat,fs_ls,fs_mkdir,fs_rm,fs_write'
  dev: 'execute_command,web_search,custom_calculator'
use_tools: null                  # Default tools to use globally (can be overridden by agents)

# Session configuration
save_session: true
compress_threshold: 4000         # Compress history if it exceeds this token count

# RAG configuration
rag_embedding_model: openai:text-embedding-3-small
rag_top_k: 4                     # Number of documents to retrieve for RAG
rag_chunk_size: 1000
rag_chunk_overlap: 200

# API configuration
serve_addr: 127.0.0.1:8000       # Address for AIChat server (if applicable)

# Client configuration (add multiple LLM providers)
clients:
  - type: openai
    api_key: sk-xxx
  - type: openai-compatible
    name: ollama
    api_base: http://localhost:11434/v1
    models:
      - name: llama3.1
        max_input_tokens: 128000
        supports_function_calling: true
```

### Tool Configuration (`tools.txt`)

List all available tools in `llm-functions/tools.txt`. AIChat will load tools listed here.

```
execute_command.sh
fs_cat.sh
fs_ls.sh
fs_write.sh
get_current_weather.sh
web_search.sh
custom_calculator.py
web_scraper.js
```

### Agent Enablement (`agents.txt`)

List the names of agents you want AIChat to recognize. This file should be in your AIChat config directory (e.g., `~/.config/aichat/agents.txt`).

```
developer
todo
data_analyst
```

---

## 6. Running & Testing Your Agent

### Using Tools Directly

You can invoke tools directly using the `%functions%` role:

```bash
aichat --role %functions% "what is the weather in Paris?"
```

### Interacting with Agents

Launch an agent using the `--agent` flag:

```bash
aichat --agent developer "create a Python web scraper"
aichat -a todo "list all my todos"
```

### REPL Mode with Agents

Enter interactive REPL mode and switch to an agent:

```bash
aichat
> .agent developer
developer> Help me refactor this code for better performance
```

### Testing Tools with Argc

Argc provides utilities for testing and running your tools:

```bash
# Test all tools defined in llm-functions
argc test

# Check a specific tool for correctness and documentation
argc check@tool custom_calculator.py

# Run a tool directly with arguments
argc run@tool custom_calculator.py '{"expression":"2**10","precision":0}'
```

---

## 7. Advanced Features & Best Practices

### Tool Mapping & Grouping

Use `mapping_tools` in `config.yaml` to create logical groups of tools, making it easier for agents to select them.

```yaml
mapping_tools:
  dev: 'execute_command,fs_write,fs_cat,web_search'
  data: 'custom_calculator,data_processor,csv_reader'
```

You can then refer to these groups (e.g., `dev`, `data`) in agent configurations or when manually setting tools.

### Session Management (Agent Preludes)

The `agent_prelude` setting in an agent's `config.yaml` allows you to automatically load a specific session when entering an agent. This preserves conversation history and context.

```yaml
# In agent's config.yaml
agent_prelude: default  # Auto-load the 'default' session
```

### RAG Integration

Enhance your agent's knowledge by pointing to documents or directories in the `index.yaml`:

```yaml
# In agent's index.yaml
documents:
  - path: ./knowledge_base/       # Can be a file or directory
    recursive: true               # For directories, scan recursively
    extensions: [md, txt, pdf]    # File extensions to include
```

### MCP Bridge Support

AIChat supports the Model Context Protocol (MCP). You can use MCP tools through the bridge:

1.  **Start MCP Server**:
    ```bash
    argc mcp start
    ```
2.  **Use MCP Tools**:
    ```bash
    aichat --agent mcp_agent "use the MCP tools"
    ```

### Custom Function Groups

Define specialized tool groups for agents that require specific sets of functionalities.

```yaml
# In config.yaml
mapping_tools:
  analysis: 'data_analyzer,chart_generator,statistics'
  automation: 'task_scheduler,email_sender,notification'
```

Then, specify these groups when enabling tools:

```bash
# In agent's config.yaml or via .set command
use_tools: analysis,automation
```

### Best Practices

1.  **Tool Design**:
    *   **Single Responsibility**: Each tool should perform one specific task.
    *   **Clear Naming & Docs**: Use descriptive names and `@describe` comments.
    *   **Robust Error Handling**: Gracefully handle errors and provide informative messages.
    *   **Structured Output**: Return data in a predictable format (JSON is ideal).

2.  **Agent Instructions**:
    *   **Specificity**: Clearly define capabilities, limitations, and expected behavior.
    *   **Examples**: Include examples of desired interactions.
    *   **Boundaries**: Define the agent's scope and responsibilities.

3.  **Security**:
    *   **Input Validation**: Sanitize all inputs to custom tools to prevent injection attacks.
    *   **Permissions**: Limit file system access and execution privileges for tools.
    *   **Sandboxing**: Consider sandboxed environments for executing untrusted code.
    *   **Review**: Regularly audit tool permissions and code.

4.  **Performance**:
    *   **Caching**: Implement caching for frequently accessed data or tool results.
    *   **Optimization**: Ensure tool execution is efficient.
    *   **Model Choice**: Select models appropriate for the task's complexity and cost.
    *   **Rate Limiting**: Implement limits for external API calls.

---

## 8. Troubleshooting Common Issues

*   **Tools Not Found**:
    *   Ensure `function_calling: true` is set in your configuration.
    *   Verify tools are listed in `tools.txt` and built with `argc build`.
    *   Check that the `llm-functions` directory is correctly linked or referenced via `AICHAT_FUNCTIONS_DIR`.
*   **Agent Not Loading**:
    *   Confirm the agent's directory structure is correct (`~/.config/aichat/agents/<agent-name>/`).
    *   Check `index.yaml` and `config.yaml` for YAML syntax errors.
    *   Ensure the agent name is listed in `agents.txt` (if used).
*   **Permission Errors**:
    *   Make sure tool scripts are executable (`chmod +x tool.sh`).
    *   Check file permissions for configuration files (`.env`, `config.yaml`).
*   **API Key Issues**:
    *   Verify API keys are correctly entered in `~/.config/aichat/.env` or client configurations.
    *   Ensure the correct client type and model are specified.

---

## 9. Conclusion

By following this guide, you can establish a robust and versatile AIChat agent tailored to your specific needs. The modular design of AIChat, combined with the power of `llm-functions` and `argc`, allows for extensive customization and integration. Experiment with new tools, refine your agent instructions, and unlock the full potential of AI-powered automation.
