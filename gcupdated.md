This document provides a comprehensive guide to creating and using extensions with Gemini CLI on Termux. It covers everything from initial setup to advanced development techniques and production deployment.

---

# Complete In-Depth Guide to Creating and Using Extensions with Gemini CLI on Termux

## Table of Contents

### Part I: Foundation & Architecture
1.  [Deep Dive into Gemini CLI Architecture](#deep-dive-into-gemini-cli-architecture)
2.  [Complete Termux Setup & Optimization](#complete-termux-setup--optimization)
3.  [Authentication Deep Dive](#authentication-deep-dive)
4.  [Extension System Internals](#extension-system-internals)

### Part II: Advanced Extension Development
5.  [MCP Server Architecture & Implementation](#mcp-server-architecture--implementation)
6.  [Complex Command Hierarchies](#complex-command-hierarchies)
7.  [Context Management System](#context-management-system)
8.  [Tool Integration & Security](#tool-integration--security)

### Part III: Production Implementation
9.  [Building Production-Ready Extensions](#building-production-ready-extensions)
10. [Performance Optimization Strategies](#performance-optimization-strategies)
11. [Debugging & Troubleshooting](#debugging--troubleshooting)
12. [Real-World Case Studies](#real-world-case-studies)

---

## Part I: Foundation & Architecture

### Deep Dive into Gemini CLI Architecture

#### Core Components

Gemini CLI is an open-source AI agent that brings the power of Gemini directly into your terminal. It provides lightweight access to Gemini, giving you the most direct path from your prompt to our model.

The architecture consists of several key layers:

```
┌─────────────────────────────────────────┐
│         User Interface Layer            │
│    (REPL, Commands, Shell Integration)  │
├─────────────────────────────────────────┤
│         Extension System                │
│  (Extensions, Commands, Context Files)  │
├─────────────────────────────────────────┤
│         Tool Orchestration              │
│    (Built-in Tools, MCP Servers)        │
├─────────────────────────────────────────┤
│         Core Engine                     │
│    (ReAct Loop, Tool Discovery)         │
├─────────────────────────────────────────┤
│         Model Interface                 │
│    (Gemini 2.5 Pro, API Integration)    │
└─────────────────────────────────────────┘
```

#### ReAct Loop Implementation

The Gemini command line interface (CLI) is an open-source AI agent that provides access to Gemini directly in your terminal. The Gemini CLI uses a reason and act (ReAct) loop with your built-in tools and local or remote MCP servers to complete complex use cases like fixing bugs, creating new

The ReAct (Reason and Act) loop is the core execution model:

```javascript
// Conceptual ReAct loop implementation
class ReActLoop {
  constructor(model, tools, context) {
    this.model = model;
    this.tools = tools;
    this.context = context;
    this.maxIterations = 10;
  }

  async execute(prompt) {
    let iteration = 0;
    let thought = "";
    let observations = [];
    
    while (iteration < this.maxIterations) {
      // Reasoning phase
      thought = await this.model.reason(prompt, observations, this.context);
      
      // Action phase
      if (thought.requiresTool) {
        const toolResult = await this.executeTool(thought.tool, thought.params);
        observations.push(toolResult);
      } else if (thought.isComplete) {
        return thought.finalAnswer;
      }
      
      iteration++;
    }
  }
  
  async executeTool(toolName, params) {
    const tool = this.tools.get(toolName);
    if (!tool) throw new Error(`Tool ${toolName} not found`);
    
    // Permission check for Termux environment
    if (this.requiresPermission(tool)) {
      const granted = await this.requestPermission(tool);
      if (!granted) return { error: "Permission denied" };
    }
    
    return await tool.execute(params);
  }
}
```

#### Token Management & Context Window

Gemini 2.5 Pro offers a massive 1 million token context window, requiring sophisticated management.

```javascript
class ContextManager {
  constructor(maxTokens = 1000000) {
    this.maxTokens = maxTokens;
    this.contextStack = [];
    this.tokenCounter = new TokenCounter();
  }
  
  addContext(content, priority = 0) {
    const tokens = this.tokenCounter.count(content);
    this.contextStack.push({ content, tokens, priority });
    this.optimizeContext();
  }
  
  optimizeContext() {
    // Sort by priority
    this.contextStack.sort((a, b) => b.priority - a.priority);
    
    // Trim to fit token limit
    let totalTokens = 0;
    const optimized = [];
    
    for (const item of this.contextStack) {
      if (totalTokens + item.tokens <= this.maxTokens) {
        optimized.push(item);
        totalTokens += item.tokens;
      }
    }
    
    this.contextStack = optimized;
  }
}
```

### Complete Termux Setup & Optimization

#### Advanced Installation Process

The installation process is similar to Linux. Ensure you have Termux from F-Droid or GitHub (Google Play version is discontinued).

##### Step 1: Termux Environment Preparation

```bash
# Update Termux repositories
pkg update && pkg upgrade -y

# Install essential development tools
pkg install -y \
  nodejs-lts \
  python \
  git \
  build-essential \
  termux-api \
  termux-tools \
  openssh \
  vim \
  curl \
  wget

# Set up storage access for Termux
termux-setup-storage

# Configure Node.js environment for global packages
npm config set prefix ~/.npm-global
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify Node.js and npm versions
node --version
npm --version
```

##### Step 2: Gemini CLI Installation with Error Handling

```bash
# Function to install Gemini CLI with retry logic
install_gemini_cli() {
  local max_attempts=3
  local attempt=1
  
  while [ $attempt -le $max_attempts ]; do
    echo "Attempt $attempt: Installing Gemini CLI..."
    
    if npm install -g @google/gemini-cli; then
      echo "✓ Gemini CLI installed successfully"
      return 0
    else
      echo "✗ Installation failed, attempt $attempt of $max_attempts"
      attempt=$((attempt + 1))
      sleep 2
    fi
  done
  
  echo "Failed to install Gemini CLI after $max_attempts attempts"
  return 1
}

# Execute the installation function
install_gemini_cli

# Verify Gemini CLI installation
gemini --version
```

### Authentication Deep Dive

Gemini CLI offers several authentication methods for seamless integration.

#### Method 1: Debug Authentication (Manual)

Run Gemini CLI with the `--debug` flag, choose "Google Login," copy the displayed URL into your browser, authenticate, and return to Termux.

```bash
#!/bin/bash
# auth-debug.sh - Helper script for manual debug authentication

echo "Starting Gemini CLI authentication in debug mode..."
echo "=================================================="

# Run gemini in debug mode, capturing output to a log file
gemini --debug 2>&1 | tee auth.log &
GEMINI_PID=$!

# Wait for the authentication URL to appear in the log
echo "Waiting for authentication URL..."
while ! grep -q "https://accounts.google.com" auth.log 2>/dev/null; do
  sleep 1
done

# Extract and display the URL
AUTH_URL=$(grep -o "https://accounts.google.com[^\"]*" auth.log | head -1)
echo ""
echo "Authentication URL found!"
echo "=================================================="
echo "$AUTH_URL"
echo "=================================================="
echo ""
echo "1. Copy the above URL."
echo "2. Open it in your browser and complete the authentication."
echo "3. Return here and press Enter to continue."
read -r

# Clean up the log file
rm -f auth.log
```

#### Method 2: Termux:API Integration (Automated)

For a smoother experience, use Termux:API to automatically open the login URL in your browser.

```bash
#!/bin/bash
# auth-api.sh - Automated authentication using Termux:API

# Ensure termux-api package is installed
if ! command -v termux-open-url &> /dev/null; then
  echo "Installing termux-api package..."
  pkg install termux-api -y
fi

# Create a wrapper script for Gemini CLI
cat > ~/gemini-auth-wrapper.sh << 'EOF'
#!/bin/bash

# Intercept authentication URL and open it automatically in the browser
gemini "$@" 2>&1 | while IFS= read -r line; do
  echo "$line"
  
  # Check if the line contains the Google authentication URL
  if [[ "$line" =~ https://accounts.google.com ]]; then
    URL=$(echo "$line" | grep -o 'https://[^"]*' | head -1)
    if [ -n "$URL" ]; then
      echo "Opening authentication URL in browser..."
      termux-open-url "$URL"
      
      # Provide user feedback with vibration and notification
      termux-vibrate -d 500
      termux-notification \
        --title "Gemini CLI Authentication" \
        --content "Please complete authentication in your browser." \
        --action "termux-open-url $URL"
    fi
  fi
done
EOF

# Make the wrapper script executable
chmod +x ~/gemini-auth-wrapper.sh

# Create an alias for easy use
echo 'alias gemini-auth="~/gemini-auth-wrapper.sh"' >> ~/.bashrc
source ~/.bashrc

echo "✓ Automated authentication setup complete."
echo "Run 'gemini-auth' to start Gemini CLI with automatic browser opening."
```

#### Method 3: API Key Authentication

Obtain an API key from Google AI Studio and set it as an environment variable. For persistent use, add it to your shell's profile file (e.g., `~/.bashrc`).

```bash
#!/bin/bash
# setup-api-key.sh - Securely set up Gemini API Key

setup_api_key() {
  echo "Setting up Gemini API key..."
  
  # Create a secure directory for the API key
  mkdir -p ~/.gemini/secure
  chmod 700 ~/.gemini/secure
  
  # Prompt the user for their API key
  echo -n "Enter your Gemini API key: "
  read -rs API_KEY
  echo
  
  # Basic validation of the API key format
  if [[ ! "$API_KEY" =~ ^[A-Za-z0-9_-]{39}$ ]]; then
    echo "Invalid API key format. Please ensure it's a 39-character string."
    return 1
  fi
  
  # Store the API key encrypted (using base64 for simplicity in Termux)
  echo "$API_KEY" | base64 > ~/.gemini/secure/api_key.enc
  chmod 600 ~/.gemini/secure/api_key.enc
  
  # Create a script to load the API key into the environment
  cat > ~/.gemini/load_api_key.sh << 'EOF'
#!/bin/bash
if [ -f ~/.gemini/secure/api_key.enc ]; then
  export GEMINI_API_KEY=$(base64 -d < ~/.gemini/secure/api_key.enc)
fi
EOF
  
  chmod +x ~/.gemini/load_api_key.sh
  
  # Add the loader script to bashrc for automatic loading
  echo 'source ~/.gemini/load_api_key.sh' >> ~/.bashrc
  
  echo "✓ API key configured successfully."
  echo "Run 'source ~/.bashrc' to load the key into your current session."
}

setup_api_key
```

### Extension System Internals

#### Extension Discovery and Loading Process

Extensions are discovered in specific locations. The `name` field in `gemini-extension.json` is crucial for unique identification and conflict resolution. `mcpServers` are loaded similarly to those in `settings.json`, with `settings.json` taking precedence in case of name conflicts.

```javascript
// Extension loader implementation concept
class ExtensionLoader {
  constructor() {
    this.extensions = new Map();
    this.loadOrder = [];
    this.conflicts = [];
  }
  
  async discoverExtensions() {
    const locations = [
      path.join(os.homedir(), '.gemini', 'extensions'),  // Global extensions
      path.join(process.cwd(), '.gemini', 'extensions')   // Project-specific extensions
    ];
    
    for (const location of locations) {
      if (await this.directoryExists(location)) {
        await this.loadExtensionsFromDirectory(location);
      }
    }
    
    this.resolveConflicts();
    return this.extensions;
  }
  
  async loadExtensionsFromDirectory(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      if (entry.isDirectory()) {
        const extPath = path.join(dir, entry.name);
        const configPath = path.join(extPath, 'gemini-extension.json');
        
        if (await this.fileExists(configPath)) {
          try {
            const config = await this.loadExtensionConfig(configPath);
            const extension = new Extension(config, extPath);
            
            // Check for conflicts with already loaded extensions
            if (this.extensions.has(config.name)) {
              this.conflicts.push({
                name: config.name,
                existing: this.extensions.get(config.name).path,
                new: extPath
              });
            }
            
            this.extensions.set(config.name, extension);
            this.loadOrder.push(config.name);
          } catch (error) {
            console.error(`Failed to load extension from ${extPath}:`, error);
          }
        }
      }
    }
  }
  
  resolveConflicts() {
    // Prioritize project extensions over global extensions
    for (const conflict of this.conflicts) {
      const projectPath = path.join(process.cwd(), '.gemini', 'extensions');
      
      if (conflict.new.startsWith(projectPath)) {
        // Keep the project version if it's newer or preferred
        console.log(`Extension conflict resolved: ${conflict.name} (using project version)`);
      } else {
        // Revert to the existing (global) version if project version is not preferred
        const existing = this.extensions.get(conflict.name);
        existing.path = conflict.existing; // Ensure the correct path is used
      }
    }
  }
}
```

#### Extension Configuration Schema

The `gemini-extension.json` file defines the extension's metadata and configuration.

```typescript
interface ExtensionConfig {
  name: string; // Unique identifier for the extension
  version: string; // Extension version
  description?: string; // Optional description
  author?: string; // Optional author information
  license?: string; // Optional license information
  
  // MCP Server configuration: defines how to run external services
  mcpServers?: {
    [serverName: string]: {
      command: string; // The command to execute (e.g., 'node', 'python')
      args?: string[]; // Arguments for the command
      env?: Record<string, string>; // Environment variables for the server
      cwd?: string; // Working directory for the server process
      timeout?: number; // Timeout for server connection
      trust?: boolean; // Whether to trust the MCP server
      includeTools?: string[]; // List of tools to include from this server
      excludeTools?: string[]; // List of tools to exclude
    };
  };
  
  // Context configuration: specifies context files to load
  contextFileName?: string; // Primary context file for the extension
  additionalContexts?: string[]; // Other context files to load
  
  // Tool restrictions: globally exclude specific tools or commands
  excludeTools?: string[]; // Tools to exclude (e.g., "run_shell_command")
  includeTools?: string[]; // Tools to explicitly include
  
  // Dependencies: lists extensions, packages, or Termux packages required
  dependencies?: {
    extensions?: string[]; // Other Gemini extensions this one depends on
    packages?: string[]; // Node.js/Python packages (e.g., "@modelcontextprotocol/sdk")
    termuxPackages?: string[]; // Termux packages (e.g., "nodejs-lts")
  };
  
  // Hooks: scripts to run at specific lifecycle events
  hooks?: {
    onLoad?: string; // Script to run when the extension is loaded
    onUnload?: string; // Script to run when the extension is unloaded
    beforeCommand?: string; // Script to run before a command is executed
    afterCommand?: string; // Script to run after a command is executed
  };
}
```

## Part II: Advanced Extension Development

### MCP Server Architecture & Implementation

#### Understanding MCP Protocol

An MCP server is an application that exposes tools and resources to the Gemini CLI through the Model Context Protocol. This allows the CLI to interact with external systems and data sources, acting as a bridge between the Gemini model and your local environment or other services.

```javascript
// Complete MCP Server implementation for Termux (Node.js example)
const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;
const path = require('path');

const execAsync = promisify(exec);

class TermuxMCPServer {
  constructor() {
    this.server = new Server(
      {
        name: 'termux-advanced', // Name of this MCP server
        version: '2.0.0',
      },
      {
        capabilities: { // Define what this server can do
          tools: {}, // Exposes callable functions
          resources: {}, // Exposes data that can be read
        },
      }
    );
    
    this.setupTools(); // Register available tools
    this.setupResources(); // Register available resources
  }
  
  setupTools() {
    // Registering tools with their descriptions and input schemas
    this.server.setRequestHandler('tools/list', async () => ({
      tools: [
        {
          name: 'battery_status',
          description: 'Get Android battery status via Termux',
          inputSchema: { type: 'object', properties: {} }, // No arguments needed
        },
        {
          name: 'termux_notification',
          description: 'Send an Android notification',
          inputSchema: { // Schema for arguments
            type: 'object',
            properties: {
              title: { type: 'string', description: 'Notification title' },
              content: { type: 'string', description: 'Notification content' },
              priority: { type: 'string', enum: ['min', 'low', 'default', 'high', 'max'], default: 'default' },
              vibrate: { type: 'boolean', default: false },
            },
            required: ['title', 'content'], // Mandatory fields
          },
        },
        {
          name: 'storage_info',
          description: 'Get Android storage information',
          inputSchema: {
            type: 'object',
            properties: {
              path: { type: 'string', description: 'Storage path to check', default: '/storage/emulated/0' },
            },
          },
        },
        {
          name: 'termux_tts',
          description: 'Text-to-speech using Android TTS',
          inputSchema: {
            type: 'object',
            properties: {
              text: { type: 'string', description: 'Text to speak' },
              language: { type: 'string', description: 'Language code (e.g., en-US)', default: 'en-US' },
              rate: { type: 'number', description: 'Speech rate (0.5 - 2.0)', minimum: 0.5, maximum: 2.0, default: 1.0 },
            },
            required: ['text'],
          },
        },
        {
          name: 'clipboard_manager',
          description: 'Manage Android clipboard',
          inputSchema: {
            type: 'object',
            properties: {
              action: { type: 'string', enum: ['get', 'set'], description: 'Clipboard action' },
              text: { type: 'string', description: 'Text to set (required for set action)' },
            },
            required: ['action'],
          },
        },
      ],
    }));
    
    // Handler for tool execution requests
    this.server.setRequestHandler('tools/call', async (request) => {
      const { name, arguments: args } = request.params;
      
      try {
        switch (name) {
          case 'battery_status': return await this.getBatteryStatus();
          case 'termux_notification': return await this.sendNotification(args);
          case 'storage_info': return await this.getStorageInfo(args.path);
          case 'termux_tts': return await this.textToSpeech(args);
          case 'clipboard_manager': return await this.manageClipboard(args);
          default: throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        // Return error content in a structured way
        return { content: [{ type: 'text', text: `Error executing ${name}: ${error.message}` }] };
      }
    });
  }
  
  async getBatteryStatus() {
    try {
      const { stdout } = await execAsync('termux-battery-status');
      const battery = JSON.parse(stdout);
      // Format output for the LLM
      return { content: [{ type: 'text', text: `Battery Status:\n- Level: ${battery.percentage}%\n- Status: ${battery.status}\n- Health: ${battery.health}\n- Temperature: ${battery.temperature}°C\n- Plugged: ${battery.plugged}` }] };
    } catch (error) { throw new Error(`Failed to get battery status: ${error.message}`); }
  }
  
  async sendNotification(args) {
    const { title, content, priority, vibrate } = args;
    let command = `termux-notification --title "${title}" --content "${content}"`;
    if (priority && priority !== 'default') command += ` --priority ${priority}`;
    if (vibrate) command += ' --vibrate 200,100,200';
    
    try {
      await execAsync(command);
      return { content: [{ type: 'text', text: `Notification sent: ${title}` }] };
    } catch (error) { throw new Error(`Failed to send notification: ${error.message}`); }
  }
  
  async getStorageInfo(storagePath = '/storage/emulated/0') {
    try {
      const { stdout } = await execAsync(`df -h ${storagePath}`);
      const data = stdout.trim().split('\n')[1].split(/\s+/);
      return { content: [{ type: 'text', text: `Storage Information for ${storagePath}:\n- Total: ${data[1]}\n- Used: ${data[2]}\n- Available: ${data[3]}\n- Usage: ${data[4]}` }] };
    } catch (error) { throw new Error(`Failed to get storage info: ${error.message}`); }
  }
  
  async textToSpeech(args) {
    const { text, language, rate } = args;
    let command = `termux-tts-speak "${text}"`;
    if (language) command += ` -l ${language}`;
    if (rate) command += ` -r ${rate}`;
    
    try {
      await execAsync(command);
      return { content: [{ type: 'text', text: `Speaking: "${text}" in ${language || 'default language'}` }] };
    } catch (error) { throw new Error(`Failed to speak text: ${error.message}`); }
  }
  
  async manageClipboard(args) {
    const { action, text } = args;
    try {
      if (action === 'get') {
        const { stdout } = await execAsync('termux-clipboard-get');
        return { content: [{ type: 'text', text: `Clipboard content: ${stdout}` }] };
      } else if (action === 'set') {
        if (!text) throw new Error('Text is required for set action');
        await execAsync(`termux-clipboard-set "${text}"`);
        return { content: [{ type: 'text', text: `Clipboard set to: ${text}` }] };
      }
    } catch (error) { throw new Error(`Clipboard operation failed: ${error.message}`); }
  }
  
  setupResources() {
    // Registering resources that the LLM can read
    this.server.setRequestHandler('resources/list', async () => ({
      resources: [
        { uri: 'termux://system/info', name: 'System Information', description: 'Termux and Android system information', mimeType: 'application/json' },
        { uri: 'termux://contacts/list', name: 'Contact List', description: 'Android contacts (requires permission)', mimeType: 'application/json' },
      ],
    }));
    
    // Handler for resource read requests
    this.server.setRequestHandler('resources/read', async (request) => {
      const { uri } = request.params;
      
      if (uri === 'termux://system/info') return await this.getSystemInfo();
      else if (uri === 'termux://contacts/list') return await this.getContactList();
      
      throw new Error(`Unknown resource: ${uri}`);
    });
  }
  
  async getSystemInfo() {
    try {
      const [deviceInfo, termuxInfo] = await Promise.all([execAsync('termux-info'), execAsync('uname -a')]);
      return { contents: [{ uri: 'termux://system/info', mimeType: 'application/json', text: JSON.stringify({ device: deviceInfo.stdout, system: termuxInfo.stdout, termuxHome: process.env.HOME, prefix: process.env.PREFIX }, null, 2) }] };
    } catch (error) { throw new Error(`Failed to get system info: ${error.message}`); }
  }
  
  async getContactList() {
    try {
      const { stdout } = await execAsync('termux-contact-list');
      const contacts = JSON.parse(stdout);
      return { contents: [{ uri: 'termux://contacts/list', mimeType: 'application/json', text: JSON.stringify(contacts, null, 2) }] };
    } catch (error) { throw new Error(`Failed to get contacts: ${error.message}`); }
  }
  
  async start() {
    const transport = new StdioServerTransport(); // Use stdio for communication
    await this.server.connect(transport);
    console.error('Termux MCP Server started'); // Log server start
  }
}

// Instantiate and start the server
const server = new TermuxMCPServer();
server.start().catch(console.error);
```

#### Configuring MCP Servers in Extensions

MCP servers are configured in `settings.json` files. They can be defined globally (`~/.gemini/settings.json`) or per project (`<project>/.gemini/settings.json`).

```json
{
  "mcpServers": {
    "termux-advanced": { // Name of the MCP server configuration
      "command": "node", // Command to run the server
      "args": ["./mcp-servers/termux-advanced.js"], // Arguments for the command
      "env": { // Environment variables for the server process
        "TERMUX_HOME": "/data/data/com.termux/files/home",
        "ANDROID_DATA": "/storage/emulated/0",
        "NODE_ENV": "production"
      },
      "cwd": "~/.gemini/extensions/termux-suite", // Working directory for the server
      "timeout": 30000, // Connection timeout in milliseconds
      "trust": false, // Whether to trust this server (important for security)
      "includeTools": [ // Specify which tools from this server are allowed
        "battery_status",
        "termux_notification",
        "storage_info",
        "termux_tts",
        "clipboard_manager"
      ]
    },
    "github": { // Example of another MCP server
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}" // Using env variable substitution
      }
    },
    "sqlite": { // Example of an MCP server for SQLite databases
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "~/databases/"], // Path to databases
      "trust": true // Trusting this server
    }
  }
}
```

### Complex Command Hierarchies

#### Namespace Architecture

Organize commands into sub-directories to create namespaces. The path separator ( `/` or `\` ) is converted to a colon (`:`). For example, a file at `<project>/.gemini/commands/git/commit.toml` becomes the namespaced command `/git:commit`.

Create a structured command hierarchy:

```bash
# Create directories for organizing commands
mkdir -p ~/.gemini/extensions/termux-suite/commands/{android,dev,security,network,data}
```

#### Android Integration Commands

Define commands for interacting with Android features via Termux:API.

```toml
# commands/android/intent.toml
description = "Create and execute Android intents"
prompt = """
Create an Android intent to {{args}}.

Provide:
1. The complete termux-open or am start command.
2. Explanation of intent components: Action, Data URI, Package/Component, Flags, and Extras.
3. Alternative methods using termux-open.
4. Security considerations.

Include examples for common scenarios: opening URLs, launching apps, sharing content, starting activities.
"""

# commands/android/permissions.toml
description = "Check and request Android permissions"
prompt = """
For the operation: {{args}}

Analyze and provide:
1. Required Android permissions.
2. How to check current permissions (e.g., using termux-info).
3. Commands to request permissions.
4. Fallback strategies if permissions are denied.
5. Security best practices.

Current permission status:
!{pm list permissions -g | grep -A 10 "permission:"}
"""

# commands/android/sensors.toml
description = "Access Android sensor data"
prompt = """
Access sensor data for: {{args}}

Available sensors:
!{termux-sensor -l}

Provide:
1. Command to access the requested sensor.
2. Data parsing strategy.
3. Real-time monitoring setup.
4. Battery impact considerations.
5. Example processing script.
"""
```

#### Development Workflow Commands

Commands to assist with development tasks within Termux.

```toml
# commands/dev/setup.toml
description = "Set up complete development environment"
prompt = """
Set up a development environment for {{args}} in Termux.

System info:
!{uname -a}
!{node --version}
!{python --version}

Create:
1. Project structure with proper directories.
2. Package installation commands (pkg and language-specific).
3. Configuration files (.bashrc additions, env vars).
4. Git setup with proper .gitignore.
5. Editor configuration (vim/neovim).
6. Testing framework setup.
7. Build scripts adapted for Termux.
8. Debugging setup.

Consider Termux limitations: No systemd, modified FHS, limited syscalls, Android storage restrictions.
"""

# commands/dev/debug.toml
description = "Advanced debugging assistance"
prompt = """
Debug this issue: {{args}}

Current environment:
!{env | grep -E "TERMUX|ANDROID|PREFIX"}
!{pwd}
!{ls -la}

Perform:
1. Analyze error symptoms.
2. Check common Termux-specific issues: Shebang paths, permission problems, missing dependencies, path issues.
3. Provide diagnostic commands.
4. Suggest fixes with explanations.
5. Create test cases to verify fix.
"""

# commands/dev/optimize.toml
description = "Optimize code for Termux constraints"
prompt = """
Optimize this code/process for Termux: {{args}}

Current resource usage:
!{free -h}
!{df -h /data}
!{top -n 1 -b | head -20}

Optimization strategies: Memory, storage, battery efficiency, network usage, process management, caching, background tasks.

Provide optimized version with benchmarks.
"""
```

#### Security Commands

Commands focused on enhancing the security of the Termux environment.

```toml
# commands/security/audit.toml
description = "Security audit for Termux environment"
prompt = """
Perform security audit for: {{args}}

Current security status:
!{ls -la ~/.ssh 2>/dev/null || echo "No SSH directory"}
!{find ~ -name "*.key" -o -name "*.pem" 2>/dev/null | head -10}
!{ps aux | grep -E "ssh|vpn|tunnel" | grep -v grep}

Audit:
1. File permissions check.
2. Exposed credentials scan.
3. Network connections review.
4. Running processes analysis.
5. Package vulnerabilities.
6. Configuration weaknesses.
7. Encryption status.

Provide remediation steps for any issues found.
"""

# commands/security/encrypt.toml
description = "Implement encryption for sensitive data"
prompt = """
Set up encryption for: {{args}}

Available tools: gpg, openssl, crypt.

Implement:
1. Choose appropriate encryption method.
2. Key generation commands.
3. Encryption/decryption scripts.
4. Secure key storage strategy.
5. Integration with Termux-keyring if applicable.
6. Backup and recovery procedures.

Create working examples with proper error handling.
"""
```

#### Network Commands

Commands related to network configuration and management within Termux.

```toml
# commands/network/tunnel.toml
description = "Set up network tunnels and proxies"
prompt = """
Create network tunnel for: {{args}}

Network status:
!{ip addr show}
!{netstat -tuln | head -20}

Configure:
1. SSH tunnel setup.
2. VPN configuration.
3. Proxy settings.
4. Port forwarding rules.
5. DNS configuration.
6. Firewall rules (iptables).
7. Connection persistence.
8. Auto-reconnect scripts.

Handle Termux networking limitations appropriately.
"""

# commands/network/api.toml
description = "Create API client/server in Termux"
prompt = """
Build API {{args}} for Termux.

Requirements analysis based on available ports, network interfaces, and security constraints.

Implement:
1. Server setup (if applicable).
2. Client configuration.
3. Authentication mechanism.
4. Rate limiting.
5. Error handling.
6. Logging system.
7. Testing endpoints.
8. Documentation.

Use Termux-compatible libraries and consider mobile constraints.
"""
```

### Context Management System

#### Hierarchical Context Loading

Gemini CLI loads `GEMINI.md` files hierarchically:
1.  **Global Context:** `~/.gemini/GEMINI.md`
2.  **Project/Ancestor Context:** Searches from the current directory up to the project root.
3.  **Sub-directory Context:** Scans subdirectories for component-specific instructions.

Create a comprehensive context system:

```markdown
# ~/.gemini/extensions/termux-suite/GEMINI.md

# Termux Development Context

## System Environment
Operating within a Termux environment on Android presents unique constraints and characteristics.

### System Paths
- Home: `/data/data/com.termux/files/home`
- Prefix: `/data/data/com.termux/files/usr`
- Temp: `/data/data/com.termux/files/usr/tmp`
- Android Storage: `~/storage/` (after `termux-setup-storage`)
  - Shared: `~/storage/shared` (maps to `/storage/emulated/0`)
  - Downloads: `~/storage/downloads`
  - DCIM: `~/storage/dcim`
  - Pictures: `~/storage/pictures`
  - Music: `~/storage/music`

### Available Package Managers
- Primary: `pkg` (wrapper around apt)
- Python: `pip` (use `pip install --user` for user packages)
- Node.js: `npm` (configure prefix to avoid permission issues)
- Ruby: `gem` (may require special flags)

### Shell Environment
- Default shell: bash
- Shebang for scripts: `#!/data/data/com.termux/files/usr/bin/bash`
- Alternative: `#!/usr/bin/env bash`

## Import Specialized Contexts
@./contexts/android-integration.md
@./contexts/security-policies.md
@./contexts/performance-guidelines.md
@./contexts/networking-rules.md

## Development Standards

### Code Style
```bash
# Always use POSIX-compliant shell scripts
set -euo pipefail  # Safe script settings
IFS=$'\n\t'        # Safe IFS

# Function template
function_name() {
    local arg1="${1:-default}"
    local arg2="${2:-}"
    
    # Validate inputs
    [[ -z "$arg1" ]] && echo "Error: arg1 required" && return 1
    
    # Process
    echo "Processing: $arg1"
    
    # Return
    return 0
}
```

### Error Handling
Always implement comprehensive error handling:
1. Check command availability before use.
2. Validate all inputs.
3. Use `trap` for cleanup.
4. Provide meaningful error messages.
5. Log errors to: `~/.gemini/logs/`.

### Testing Requirements
- Test all scripts in an actual Termux environment.
- Check compatibility with different Android versions.
- Verify `termux-api` calls work correctly.
- Test with limited permissions.
- Validate storage access.

## Security Policies

### Forbidden Operations
NEVER attempt or suggest:
- Rooting the device.
- Modifying system files outside Termux.
- Accessing other apps' private data.
- Running commands with `su` or `sudo`.
- Disabling Android security features.

### Credential Management
- Use `termux-keyring` when available.
- Never store plaintext passwords.
- Use environment variables from encrypted sources.
- Implement proper session management.
- Regular credential rotation.

### Network Security
- Always use HTTPS when possible.
- Validate SSL certificates.
- Implement rate limiting.
- Use SSH keys instead of passwords.
- Configure `fail2ban` equivalents if possible.

## Performance Optimization

### Resource Constraints
Mobile devices have limited:
- RAM (typically 2-8GB, shared with Android).
- CPU (thermal throttling is common).
- Battery (optimize for power efficiency).
- Storage (internal is faster but limited).

### Optimization Strategies
1.  **Memory Management:** Use streaming, implement aggressive garbage collection, monitor memory with `free -h`.
2.  **CPU Optimization:** Use `nice` values for background tasks, implement task queuing, avoid CPU-intensive operations during peak hours.
3.  **Battery Optimization:** Use wake locks sparingly, batch network requests, implement exponential backoff, use Termux:Boot for scheduled tasks.

## Integration Guidelines

### Termux:API Integration
When using Termux:API, always:
1.  Check if `termux-api` package is installed.
2.  Verify Termux:API app is installed.
3.  Handle permission requests gracefully.
4.  Provide fallbacks for missing permissions.
5.  Test on different Android versions.

### Android Integration
- Use intents for app interaction.
- Respect Android's permission model.
- Handle storage access framework properly.
- Work with content providers when needed.
- Implement proper broadcast receivers.

## Project Structure Templates

### Standard Project Layout
```
project/
├── .gemini/
│   ├── commands/      # Project-specific commands
│   ├── extensions/    # Project extensions
│   └── GEMINI.md     # Project context
├── src/              # Source code
├── tests/            # Test files
├── docs/             # Documentation
├── scripts/          # Utility scripts
│   ├── setup.sh     # Setup script
│   ├── build.sh     # Build script
│   └── deploy.sh    # Deployment script
├── .env.example      # Environment template
├── .gitignore        # Git ignore rules
└── README.md         # Project documentation
```

## Error Messages and Solutions

### Common Issues Database
When encountering errors, check:

1.  **"Permission denied"**: Check file permissions (`ls -la`), verify storage access (`termux-setup-storage`), check SELinux context if applicable.
2.  **"Command not found"**: Install missing package (`pkg install <package>`), check `PATH` (`echo $PATH`), verify shebang path.
3.  **"No such file or directory"**: Check `PREFIX` paths, verify symbolic links, check case sensitivity.
4.  **"Cannot allocate memory"**: Check available memory (`free -h`), kill unnecessary processes, increase swap if possible.

## Workflow Automation

### Task Automation Rules
1.  Use Termux:Boot for startup tasks.
2.  Implement proper logging.
3.  Handle network connectivity changes.
4.  Respect Doze mode and battery optimization.
5.  Use Termux:Widget for quick actions.

### CI/CD in Termux
- Use local Git hooks.
- Implement testing pipelines.
- Automate builds with `make` or `npm` scripts.
- Deploy using `rsync` or `scp`.
- Monitor with custom scripts.

## Communication Protocols

### User Interaction
- Always explain Termux-specific considerations.
- Provide alternative solutions for limitations.
- Include installation commands for dependencies.
- Warn about battery/performance impact.
- Suggest optimization opportunities.

### Code Generation
When generating code:
1.  Include proper error handling.
2.  Add comprehensive comments.
3.  Provide usage examples.
4.  Include dependency checks.
5.  Add performance considerations.

## Maintenance Guidelines

### Regular Maintenance Tasks
```bash
# Weekly maintenance script
#!/data/data/com.termux/files/usr/bin/bash

# Update packages
pkg update && pkg upgrade -y

# Clean package cache
apt autoremove -y
apt clean

# Clear temporary files (older than 7 days)
find /data/data/com.termux/files/usr/tmp -type f -mtime +7 -delete

# Rotate logs (delete logs older than 30 days)
find ~/.gemini/logs -name "*.log" -mtime +30 -delete

# Check disk usage
df -h
du -sh ~/.gemini/*
```

## Advanced Features

### Custom Tool Integration
When integrating new tools:
1.  Check Termux compatibility.
2.  Verify architecture support (arm64, etc.).
3.  Test resource consumption.
4.  Document installation process.
5.  Create wrapper scripts if needed.

### Extension Development
For new extensions:
1.  Follow modular design principles.
2.  Implement proper error handling.
3.  Include comprehensive tests.
4.  Document all features clearly.
5.  Provide migration guides if applicable.
```

#### Context File Imports

Create specialized context files and import them using the `@./path/to/file.md` syntax.

```markdown
# contexts/android-integration.md

# Android Integration Context

## Available Termux:API Commands

### Device Information
- `termux-battery-status`: Battery information.
- `termux-brightness`: Screen brightness control.
- `termux-call-log`: Call history.
- `termux-camera-info`: Camera information.
- `termux-contact-list`: Access contacts.
- `termux-infrared-frequencies`: IR capabilities.
- `termux-location`: GPS location.
- `termux-sensor`: Sensor data.
- `termux-telephony-deviceinfo`: Device info.
- `termux-wifi-connectioninfo`: WiFi status.
- `termux-wifi-scaninfo`: WiFi networks.

### System Interaction
- `termux-clipboard-get/set`: Clipboard access.
- `termux-dialog`: UI dialogs.
- `termux-download`: Download manager.
- `termux-fingerprint`: Biometric authentication.
- `termux-keystore`: Android keystore.
- `termux-media-player`: Media control.
- `termux-media-scan`: Media scanner.
- `termux-microphone-record`: Audio recording.
- `termux-notification`: Display notifications.
- `termux-notification-remove`: Clear notifications.
- `termux-open`: Open files/URLs.
- `termux-open-url`: Open URLs in the default browser.
- `termux-share`: Share content.
- `termux-sms-list`: SMS history.
- `termux-sms-send`: Send SMS messages.
- `termux-storage-get`: Access storage.
- `termux-toast`: Display toast messages.
- `termux-torch`: Flashlight control.
- `termux-tts-engines`: List Text-to-Speech engines.
- `termux-tts-speak`: Speak text using TTS.
- `termux-usb`: USB device access.
- `termux-vibrate`: Vibration control.
- `termux-volume`: Volume control.
- `termux-wallpaper`: Wallpaper control.
- `termux-wake-lock`: Acquire a wake lock.
- `termux-wake-unlock`: Release a wake lock.

## Permission Requirements

### Critical Permissions
These require explicit user consent:
- Location access
- Contact access
- SMS access
- Call log access
- Microphone access
- Camera access
- Storage access

### Best Practices
1.  Always check permissions before using a feature.
2.  Provide graceful fallbacks if permissions are denied.
3.  Explain why a permission is needed to the user.
4.  Do not request unnecessary permissions.
5.  Cache permission status to avoid repeated requests.

## Intent Examples

### Common Intent Patterns
```bash
# Open URL in browser
termux-open-url "https://example.com"

# Share text via other apps
echo "Hello" | termux-share -a send

# Open specific app (e.g., Settings)
am start -n com.android.settings/.Settings

# Send a system broadcast (e.g., after boot)
am broadcast -a android.intent.action.BOOT_COMPLETED

# Start a specific service in an app
am startservice -n com.example/.MyService
```

## Storage Access

### Storage Paths After Setup
```bash
# Run this command first to grant storage access:
termux-setup-storage

# Then access storage via these paths:
~/storage/shared/          # Main internal storage (accessible by other apps)
~/storage/downloads/       # Downloads folder
~/storage/dcim/           # Camera folder
~/storage/pictures/       # Pictures folder
~/storage/music/          # Music folder
~/storage/movies/         # Videos folder
~/storage/external-1/     # SD card (if present)
```

### File Access Patterns
- Use `~/storage/shared/` for user-accessible files.
- Keep app-specific data within Termux's private directories (e.g., `~/.local/share/`).
- Use `$PREFIX/tmp/` for temporary files.
- Cache data in `~/.cache/`.
```

### Tool Integration & Security

#### Advanced Tool Restrictions

`excludeTools` can be used to block specific tools or commands, enhancing security.

```json
{
  "name": "termux-secure",
  "version": "1.0.0",
  "excludeTools": [
    "run_shell_command(rm -rf)", // Block specific dangerous commands
    "run_shell_command(su)",
    "run_shell_command(sudo)",
    "run_shell_command(chmod 777)",
    "run_shell_command(kill -9)",
    "run_shell_command(dd if=)",
    "run_shell_command(mkfs)",
    "file_delete(/data/data/com.termux/files/home/.bashrc)", // Protect critical config files
    "file_delete(/data/data/com.termux/files/home/.profile)",
    "file_write(/system)", // Prevent writing to system directories
    "file_write(/proc)",
    "file_write(/sys)"
  ],
  "includeTools": [ // Explicitly allow only necessary tools
    "read_file",
    "read_many_files",
    "file_write",
    "run_shell_command",
    "google_web_search",
    "save_memory",
    "web_fetch"
  ],
  "toolRestrictions": { // Fine-grained control over specific tools
    "run_shell_command": {
      "allowedCommands": [ // Whitelist specific commands
        "ls", "pwd", "echo", "cat", "grep", "find",
        "node", "python", "git", "npm", "pip",
        "termux-*", "pkg", "apt"
      ],
      "blockedPatterns": [ // Block commands matching these patterns
        "*/etc/*",
        "*/system/*",
        "*password*",
        "*secret*",
        "*private*key*"
      ],
      "requireConfirmation": true, // Prompt user before executing
      "logCommands": true // Log all executed commands for auditing
    },
    "file_write": {
      "allowedPaths": [ // Restrict file writing to specific directories
        "~/projects/*",
        "~/.gemini/*",
        "/data/data/com.termux/files/home/*"
      ],
      "maxFileSize": "10MB", // Limit file size
      "allowedExtensions": [ // Allow only specific file extensions
        ".js", ".py", ".sh", ".json", ".md",
        ".txt", ".toml", ".yaml", ".yml"
      ]
    }
  }
}
```

#### Security Middleware Implementation

A custom middleware can intercept tool calls to enforce security policies.

```javascript
// security-middleware.js
class SecurityMiddleware {
  constructor(config) {
    this.config = config; // Load security configuration
    this.auditLog = []; // Keep a log of all tool interactions
  }
  
  async validateToolCall(tool, params) {
    const restriction = this.config.toolRestrictions[tool];
    if (!restriction) return true; // No restrictions defined for this tool
    
    // Log the tool call attempt for auditing
    this.auditLog.push({
      timestamp: new Date(),
      tool,
      params,
      user: process.env.USER // User initiating the action
    });
    
    // Check allowed commands for shell execution
    if (tool === 'run_shell_command' && restriction.allowedCommands) {
      const command = params.command.split(' ')[0]; // Get the command name
      if (!restriction.allowedCommands.includes(command)) {
        throw new Error(`Command '${command}' is not allowed.`);
      }
    }
    
    // Check for blocked patterns in parameters or command strings
    if (restriction.blockedPatterns) {
      for (const pattern of restriction.blockedPatterns) {
        const regex = new RegExp(pattern.replace('*', '.*')); // Convert wildcard to regex
        if (regex.test(JSON.stringify(params))) { // Test against stringified parameters
          throw new Error(`Blocked pattern detected: ${pattern}`);
        }
      }
    }
    
    // Check file paths for write operations
    if (tool === 'file_write' && restriction.allowedPaths) {
      const filePath = params.path;
      const allowed = restriction.allowedPaths.some(allowedPath => {
        const regex = new RegExp('^' + allowedPath.replace('*', '.*') + '$'); // Regex for path matching
        return regex.test(filePath);
      });
      
      if (!allowed) {
        throw new Error(`Path '${filePath}' is not in the allowed paths.`);
      }
    }
    
    // Check file size limits for write operations
    if (tool === 'file_write' && restriction.maxFileSize) {
      const maxSize = this.parseSize(restriction.maxFileSize);
      const content = params.content || '';
      if (content.length > maxSize) {
        throw new Error(`File size exceeds the maximum limit of ${restriction.maxFileSize}.`);
      }
    }
    
    // Require user confirmation if specified in the restrictions
    if (restriction.requireConfirmation) {
      return await this.requestConfirmation(tool, params);
    }
    
    return true; // Tool call is validated
  }
  
  parseSize(sizeStr) {
    const units = { KB: 1024, MB: 1024*1024, GB: 1024*1024*1024 };
    const match = sizeStr.match(/^(\d+)(KB|MB|GB)$/i);
    if (!match) return parseInt(sizeStr); // Handle cases without units (bytes)
    return parseInt(match[1]) * units[match[2].toUpperCase()];
  }
  
  async requestConfirmation(tool, params) {
    // In a real-world scenario, this would involve user interaction (e.g., prompt)
    console.log(`Confirmation required for ${tool}:`, params);
    // For this example, we auto-approve. Replace with user prompt logic.
    return true;
  }
  
  getAuditLog() {
    return this.auditLog; // Return the collected audit log
  }
}

module.exports = SecurityMiddleware;
```

## Part III: Production Implementation

### Building Production-Ready Extensions

#### Complete Extension Package Structure

A well-organized extension follows a standard directory structure for maintainability and distribution.

```bash
termux-ultimate-extension/
├── gemini-extension.json       # Main extension configuration
├── package.json                # Node.js package manifest
├── README.md                   # Extension description and usage
├── LICENSE                     # License file
├── CHANGELOG.md                # Version history
├── .github/                    # GitHub-specific files (e.g., CI workflows)
│   └── workflows/
│       └── test.yml            # Continuous integration tests
├── commands/                   # Custom slash commands (TOML files)
│   ├── android/
│   │   ├── intent.toml
│   │   ├── permissions.toml
│   │   └── sensors.toml
│   ├── dev/
│   │   ├── setup.toml
│   │   ├── debug.toml
│   │   └── optimize.toml
│   ├── security/
│   │   ├── audit.toml
│   │   └── encrypt.toml
│   └── network/
│       ├── tunnel.toml
│       └── api.toml
├── contexts/                   # Context files (GEMINI.md, etc.)
│   ├── GEMINI.md               # Primary context for the extension
│   ├── android-integration.md
│   ├── security-policies.md
│   └── performance-guidelines.md
├── mcp-servers/                # MCP server implementations
│   ├── termux-advanced/        # Directory for a specific MCP server
│   │   ├── index.js            # Main server file
│   │   ├── package.json
│   │   └── test/               # Tests for this server
│   ├── android-bridge/
│   │   ├── index.js
│   │   └── package.json
│   └── security-monitor/
│       ├── index.js
│       └── package.json
├── scripts/                    # Utility scripts for installation, testing, etc.
│   ├── install.sh
│   ├── uninstall.sh
│   ├── update.sh
│   └── test.sh
├── tests/                      # Automated tests for the extension
│   ├── unit/
│   ├── integration/
│   └── e2e/                    # End-to-end tests
└── docs/                       # Detailed documentation
    ├── installation.md
    ├── configuration.md
    ├── commands.md
    └── troubleshooting.md
```

#### Production `gemini-extension.json`

A production-ready configuration includes thorough metadata, dependencies, and robust settings.

```json
{
  "name": "termux-ultimate", // Unique name of the extension
  "version": "3.0.0", // Semantic versioning
  "description": "Comprehensive Termux integration for Gemini CLI",
  "author": "Your Name",
  "license": "MIT",
  "homepage": "https://github.com/yourusername/termux-ultimate", // Project homepage
  "repository": { // Git repository details
    "type": "git",
    "url": "https://github.com/yourusername/termux-ultimate.git"
  },
  "bugs": { // Link to issue tracker
    "url": "https://github.com/yourusername/termux-ultimate/issues"
  },
  "engines": { // Specify required engine versions
    "node": ">=18.0.0",
    "gemini-cli": ">=1.0.0"
  },
  "mcpServers": { // Configuration for MCP servers
    "termux-advanced": {
      "command": "node",
      "args": ["./mcp-servers/termux-advanced/index.js"],
      "env": {
        "NODE_ENV": "production", // Set environment to production
        "LOG_LEVEL": "${LOG_LEVEL:-info}" // Use env var for log level, default to info
      },
      "timeout": 30000,
      "trust": false,
      "includeTools": [ // Explicitly include tools
        "battery_status",
        "termux_notification",
        "storage_info",
        "termux_tts",
        "clipboard_manager"
      ]
    },
    "android-bridge": { // Another MCP server example
      "command": "node",
      "args": ["./mcp-servers/android-bridge/index.js"],
      "timeout": 60000
    },
    "security-monitor": { // Security monitoring MCP server
      "command": "node", 
      "args": ["./mcp-servers/security-monitor/index.js"],
      "trust": true
    }
  },
  "contextFileName": "contexts/GEMINI.md", // Primary context file
  "additionalContexts": [ // Additional context files
    "contexts/android-integration.md",
    "contexts/security-policies.md",
    "contexts/performance-guidelines.md"
  ],
  "excludeTools": [ // Global tool exclusions for this extension
    "run_shell_command(su)",
    "run_shell_command(sudo)",
    "run_shell_command(rm -rf /)",
    "file_delete(/system)",
    "file_write(/proc)",
    "file_write(/sys)"
  ],
  "dependencies": { // Declare dependencies
    "extensions": ["base-gemini-ext"], // Depends on a base extension
    "packages": [ // Node.js/Python packages
      "@modelcontextprotocol/sdk@^0.5.0",
      "sqlite3@^5.1.6"
    ],
    "termuxPackages": [ // Required Termux packages
      "nodejs-lts",
      "python",
      "git",
      "termux-api"
    ]
  },
  "hooks": { // Define scripts for lifecycle events
    "onLoad": "./scripts/on-load.js",
    "onUnload": "./scripts/on-unload.js",
    "beforeCommand": "./scripts/before-command.js",
    "afterCommand": "./scripts/after-command.js"
  },
  "configuration": { // Define custom configuration options for the extension
    "properties": {
      "termux-ultimate.enableAdvancedFeatures": {
        "type": "boolean",
        "default": false,
        "description": "Enable advanced experimental features."
      },
      "termux-ultimate.logLevel": {
        "type": "string",
        "enum": ["debug", "info", "warn", "error"],
        "default": "info",
        "description": "Logging level for the extension."
      }
    }
  }
}
```

#### Installation Script

A robust installation script automates setup and dependency management.

```bash
#!/data/data/com.termux/files/usr/bin/bash
# install.sh - Production installation script for the extension

set -euo pipefail # Exit on error, unset variable, or pipe failure
IFS=$'\n\t' # Set Internal Field Separator for safer parsing

# ANSI Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Extension metadata
EXTENSION_NAME="termux-ultimate"
EXTENSION_VERSION="3.0.0"
INSTALL_DIR="$HOME/.gemini/extensions/$EXTENSION_NAME" # Target directory for installation
LOG_FILE="$HOME/.gemini/logs/install-$(date +%Y%m%d-%H%M%S).log" # Log file path

# Logging helper functions
log() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1 # Exit script on critical error
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Display header information
echo "=================================" | tee -a "$LOG_FILE"
echo "Termux Ultimate Extension Installer" | tee -a "$LOG_FILE"
echo "Version: $EXTENSION_VERSION" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Node.js installation and version
    if ! command -v node &> /dev/null; then
        error "Node.js is not installed. Please run: pkg install nodejs-lts"
    fi
    
    NODE_VERSION=$(node -v | cut -d'v' -f2)
    REQUIRED_VERSION="18.0.0"
    # Compare Node.js version using sort -V for proper version comparison
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$NODE_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        error "Node.js version $NODE_VERSION is too old. Required: v$REQUIRED_VERSION+"
    fi
    
    # Check Gemini CLI installation
    if ! command -v gemini &> /dev/null; then
        error "Gemini CLI is not installed. Please run: npm install -g @google/gemini-cli"
    fi
    
    # Check and install required Termux packages
    REQUIRED_PACKAGES=("git" "python" "termux-api")
    for pkg in "${REQUIRED_PACKAGES[@]}"; do
        if ! pkg list-installed 2>/dev/null | grep -q "^$pkg/"; then
            warn "Package '$pkg' is not installed. Attempting to install..."
            pkg install -y "$pkg" || error "Failed to install required package '$pkg'."
        fi
    done
    
    # Check if Termux:API app is likely installed
    if ! termux-api-start 2>/dev/null; then
        warn "Termux:API app might not be installed. Some features may not function correctly."
    fi
    
    log "Prerequisites check completed successfully."
}

# Function to back up existing installation
backup_existing() {
    if [ -d "$INSTALL_DIR" ]; then
        log "Backing up existing installation..."
        BACKUP_DIR="$INSTALL_DIR.backup.$(date +%Y%m%d-%H%M%S)"
        mv "$INSTALL_DIR" "$BACKUP_DIR" # Move the existing directory to a backup location
        log "Backup created at: $BACKUP_DIR"
    fi
}

# Function to install the extension files
install_extension() {
    log "Installing extension files..."
    
    # Create the target installation directory
    mkdir -p "$INSTALL_DIR"
    
    # Copy all files from the current directory to the installation directory
    cp -r ./* "$INSTALL_DIR/" 2>/dev/null || true # Use || true to ignore errors if cp fails (e.g., self-copy)
    
    # Install Node.js dependencies for MCP servers if package.json exists
    if [ -f "$INSTALL_DIR/package.json" ]; then
        log "Installing Node.js dependencies..."
        (cd "$INSTALL_DIR" && npm install) || warn "Failed to install Node.js dependencies. Check logs."
    fi
    
    log "Extension files copied successfully."
}

# --- Main installation process ---
check_prerequisites
backup_existing
install_extension

log "Installation of '$EXTENSION_NAME' v$EXTENSION_VERSION completed."
echo ""
echo "You can now use the extension's commands and features."
echo "Refer to the documentation in '$INSTALL_DIR/docs/' for details."
```

### Performance Optimization Strategies

#### Resource Constraints and Optimization

Mobile devices have limited resources (RAM, CPU, battery). Optimize extensions accordingly:

*   **Memory:** Use streaming data processing, implement efficient garbage collection, and monitor memory usage.
*   **CPU:** Utilize `nice` for background tasks, implement task queuing, and avoid heavy computations during peak usage.
*   **Battery:** Minimize wake lock usage, batch network requests, use exponential backoff for retries, and leverage Termux:Boot for scheduled operations.

#### Extension-Specific Settings

Configure extension behavior through `gemini-extension.json` or `settings.json` for performance tuning.

```json
{
  "name": "termux-optimized",
  "excludeTools": [
    "web_fetch(https://very-large-file.com)" // Exclude resource-intensive web fetches
  ],
  "settings": { // Extension-specific settings
    "maxFileSize": "1MB", // Limit file sizes for processing
    "timeout": 30000 // Set a reasonable timeout for operations
  }
}
```

### Debugging & Troubleshooting

#### Common Issues and Solutions

*   **Extension Not Loading:**
    *   Check `gemini --debug` output for loading errors.
    *   Verify `gemini-extension.json` syntax (use `python -m json.tool` for validation).
    *   Ensure correct file permissions for extension directories and files.
    *   Check for extension name conflicts.
*   **Command Conflicts:** Workspace extensions (`<project>/.gemini/extensions/`) take precedence over global ones (`~/.gemini/extensions/`).
*   **MCP Server Connection Issues:**
    *   Test the MCP server script independently (`python server.py`).
    *   Verify environment variables (e.g., API keys, paths).
    *   Check network connectivity if the server relies on external resources.
    *   Ensure the `command` and `args` in `settings.json` correctly point to the server script.
    *   Confirm the `transport` method (`stdio` is common for local servers).
*   **`termux-api` Errors:**
    *   Ensure the Termux:API app is installed from F-Droid.
    *   Verify the `termux-api` package is installed in Termux (`pkg install termux-api`).
    *   Check if necessary Android permissions have been granted.

#### Debugging Workflow

1.  **Use `gemini --debug`:** Provides verbose output on extension loading, command execution, and tool usage.
2.  **Inspect MCP Server Logs:** Check the console output of your running MCP server for errors.
3.  **Test Components Individually:** Run MCP server scripts or custom command TOML files in isolation.
4.  **Use `/memory show`:** Within `gemini-cli`, this command displays the full context being sent to the model, helping to diagnose prompt issues.
5.  **Use `/help`:** Lists all available commands, including custom ones from extensions.

### Real-World Case Studies

*   **Termux Automation Extension:** An extension that uses `termux-api` to automate tasks like checking battery status, sending notifications, and managing clipboard content. This leverages MCP servers for device interaction and custom commands for simple prompts.
*   **Code Review and Refactoring Assistant:** An extension with MCP servers that analyze code complexity, suggest refactoring, and generate unit tests using external libraries or LLM APIs. Commands can be defined for specific languages or frameworks.
*   **System Monitoring Extension:** An extension that provides commands to monitor Termux resource usage (CPU, memory, disk) and network activity, potentially integrating with MCP servers that query system metrics.

---

This comprehensive guide provides the foundation and advanced techniques necessary to build powerful, customized extensions for Gemini CLI on Termux, transforming your mobile terminal into a highly capable AI-powered development and automation environment.
