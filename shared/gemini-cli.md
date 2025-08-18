

# Complete In-Depth Guide to Creating and Using Extensions with Gemini CLI on Termux

## Table of Contents

### Part I: Foundation & Architecture
1. [Deep Dive into Gemini CLI Architecture](#deep-dive-into-gemini-cli-architecture)
2. [Complete Termux Setup & Optimization](#complete-termux-setup--optimization)
3. [Authentication Deep Dive](#authentication-deep-dive)
4. [Extension System Internals](#extension-system-internals)

### Part II: Advanced Extension Development
5. [MCP Server Architecture & Implementation](#mcp-server-architecture--implementation)
6. [Complex Command Hierarchies](#complex-command-hierarchies)
7. [Context Management System](#context-management-system)
8. [Tool Integration & Security](#tool-integration--security)

### Part III: Production Implementation
9. [Building Production-Ready Extensions](#building-production-ready-extensions)
10. [Performance Optimization Strategies](#performance-optimization-strategies)
11. [Debugging & Troubleshooting](#debugging--troubleshooting)
12. [Real-World Case Studies](#real-world-case-studies)

---

## Part I: Foundation & Architecture

## Deep Dive into Gemini CLI Architecture

### Core Components

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

### ReAct Loop Implementation

The Gemini command line interface (CLI) is an open source AI agent that provides access to Gemini directly in your terminal. The Gemini CLI uses a reason and act (ReAct) loop with your built-in tools and local or remote MCP servers to complete complex use cases like fixing bugs, creating new

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

### Token Management & Context Window

That free license gets you access to Gemini 2.5 Pro and its massive 1 million token context window.

The 1M token context window requires sophisticated management:

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

## Complete Termux Setup & Optimization

### Advanced Installation Process

The installation process is identical to Linux. You'll need Termux or a similar terminal emulator. I prefer Termux, make sure to download it from F-Droid store or GitHub. Version on Google Play is discontinued.

#### Step 1: Termux Environment Preparation

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

# Set up storage access
termux-setup-storage

# Configure Node.js environment
npm config set prefix ~/.npm-global
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify Node.js version (should be 18+)
node --version
npm --version
```

#### Step 2: Gemini CLI Installation with Error Handling

```bash
# Install Gemini CLI with retry logic
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

install_gemini_cli

# Verify installation
gemini --version
```

### Authentication Deep Dive

This is the simplest method. Run Gemini CLI with the --debug flag: ... Choose Google Login. The CLI will display a login URL in the terminal. Copy this link into your browser, authenticate with your Google account, and return to Termux. Gemini CLI will now be successfully authenticated.

#### Method 1: Debug Authentication (Manual)

```bash
#!/bin/bash
# auth-debug.sh - Debug authentication helper

echo "Starting Gemini CLI authentication in debug mode..."
echo "="*50

# Run in debug mode and capture output
gemini --debug 2>&1 | tee auth.log &
GEMINI_PID=$!

# Wait for authentication URL
echo "Waiting for authentication URL..."
while ! grep -q "https://accounts.google.com" auth.log 2>/dev/null; do
  sleep 1
done

# Extract and display URL
AUTH_URL=$(grep -o "https://accounts.google.com[^\"]*" auth.log | head -1)
echo ""
echo "Authentication URL found!"
echo "="*50
echo "$AUTH_URL"
echo "="*50
echo ""
echo "1. Copy the above URL"
echo "2. Open it in your browser"
echo "3. Complete authentication"
echo "4. Return here and press Enter"
read -r

# Clean up
rm -f auth.log
```

#### Method 2: Termux:API Integration (Automated)

For a more seamless experience, you can use Termux:API to open the login URL directly in your browser. Termux-api allows you to send commands to the Android system and send command to open a browser. This means that Termux would trigger Google authentication automatically by opening a browser, just like it would behave on desktop system. For that you need to install: Install Termux:API app from F-Droid. This will open the browser automatically, mimicking desktop behavior. Once authenticated, return to Termux and continue using Gemini CLI.

```bash
#!/bin/bash
# auth-api.sh - Automated authentication with Termux:API

# Check if Termux:API is installed
if ! command -v termux-open-url &> /dev/null; then
  echo "Installing termux-api package..."
  pkg install termux-api -y
fi

# Create authentication wrapper
cat > ~/gemini-auth-wrapper.sh << 'EOF'
#!/bin/bash

# Intercept authentication URL and open automatically
gemini "$@" 2>&1 | while IFS= read -r line; do
  echo "$line"
  
  # Check for Google auth URL
  if [[ "$line" =~ https://accounts.google.com ]]; then
    URL=$(echo "$line" | grep -o 'https://[^"]*' | head -1)
    if [ -n "$URL" ]; then
      echo "Opening authentication URL in browser..."
      termux-open-url "$URL"
      
      # Vibrate to notify user
      termux-vibrate -d 500
      
      # Show notification
      termux-notification \
        --title "Gemini CLI Authentication" \
        --content "Please complete authentication in browser" \
        --action "termux-open-url $URL"
    fi
  fi
done
EOF

chmod +x ~/gemini-auth-wrapper.sh

# Create alias for easy use
echo 'alias gemini-auth="~/gemini-auth-wrapper.sh"' >> ~/.bashrc
source ~/.bashrc

echo "✓ Automated authentication setup complete"
echo "Run 'gemini-auth' to start Gemini CLI with automatic browser opening"
```

#### Method 3: API Key Authentication

Get Your Key: Get an API key from Google AI Studio. Set Your Key: Make the key available to the CLI with one of these methods. Method 1: Shell Environment Variable Set the GEMINI_API_KEY environment variable. To use it across terminal sessions, add this line to your shell's profile (e.g., ~/.bashrc, ~/.zshrc).

```bash
#!/bin/bash
# setup-api-key.sh - Secure API key setup

# Secure API key storage with encryption
setup_api_key() {
  echo "Setting up Gemini API key..."
  
  # Create secure directory
  mkdir -p ~/.gemini/secure
  chmod 700 ~/.gemini/secure
  
  # Prompt for API key
  echo -n "Enter your Gemini API key: "
  read -rs API_KEY
  echo
  
  # Validate key format
  if [[ ! "$API_KEY" =~ ^[A-Za-z0-9_-]{39}$ ]]; then
    echo "Invalid API key format"
    return 1
  fi
  
  # Store encrypted (using simple base64 for Termux compatibility)
  echo "$API_KEY" | base64 > ~/.gemini/secure/api_key.enc
  chmod 600 ~/.gemini/secure/api_key.enc
  
  # Create loader script
  cat > ~/.gemini/load_api_key.sh << 'EOF'
#!/bin/bash
if [ -f ~/.gemini/secure/api_key.enc ]; then
  export GEMINI_API_KEY=$(base64 -d < ~/.gemini/secure/api_key.enc)
fi
EOF
  
  chmod +x ~/.gemini/load_api_key.sh
  
  # Add to bashrc
  echo 'source ~/.gemini/load_api_key.sh' >> ~/.bashrc
  
  echo "✓ API key configured successfully"
  echo "Run 'source ~/.bashrc' to load the key"
}

setup_api_key
```

## Extension System Internals

### Extension Discovery and Loading Process

name: The name of the extension. This is used to uniquely identify the extension and for conflict resolution when extension commands have the same name as user or project commands. ... mcpServers: A map of MCP servers to configure. The key is the name of the server, and the value is the server configuration. These servers will be loaded on startup just like MCP servers configured in a settings.json file. If both an extension and a settings.json file configure an MCP server with the same name, the server defined in the settings.json file takes precedence.

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
      path.join(os.homedir(), '.gemini', 'extensions'),  // Global
      path.join(process.cwd(), '.gemini', 'extensions')   // Project
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
            
            // Check for conflicts
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
    // Project extensions take precedence over global
    for (const conflict of this.conflicts) {
      const projectPath = path.join(process.cwd(), '.gemini', 'extensions');
      
      if (conflict.new.startsWith(projectPath)) {
        // Keep project extension
        console.log(`Extension conflict resolved: ${conflict.name} (using project version)`);
      } else {
        // Revert to existing
        const existing = this.extensions.get(conflict.name);
        existing.path = conflict.existing;
      }
    }
  }
}
```

### Extension Configuration Schema

```typescript
interface ExtensionConfig {
  name: string;
  version: string;
  description?: string;
  author?: string;
  license?: string;
  
  // MCP Server configuration
  mcpServers?: {
    [serverName: string]: {
      command: string;
      args?: string[];
      env?: Record<string, string>;
      cwd?: string;
      timeout?: number;
      trust?: boolean;
      includeTools?: string[];
      excludeTools?: string[];
    };
  };
  
  // Context configuration
  contextFileName?: string;
  additionalContexts?: string[];
  
  // Tool restrictions
  excludeTools?: string[];
  includeTools?: string[];
  
  // Dependencies
  dependencies?: {
    extensions?: string[];
    packages?: string[];
    termuxPackages?: string[];
  };
  
  // Hooks
  hooks?: {
    onLoad?: string;
    onUnload?: string;
    beforeCommand?: string;
    afterCommand?: string;
  };
}
```

## Part II: Advanced Extension Development

## MCP Server Architecture & Implementation

### Understanding MCP Protocol

An MCP server is an application that exposes tools and resources to the Gemini CLI through the Model Context Protocol, allowing it to interact with external systems and data sources. MCP servers act as a bridge between the Gemini model and your local environment or other services like APIs.

```javascript
// Complete MCP Server implementation for Termux
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
        name: 'termux-advanced',
        version: '2.0.0',
      },
      {
        capabilities: {
          tools: {},
          resources: {},
        },
      }
    );
    
    this.setupTools();
    this.setupResources();
  }
  
  setupTools() {
    // Battery status tool
    this.server.setRequestHandler('tools/list', async () => ({
      tools: [
        {
          name: 'battery_status',
          description: 'Get Android battery status via Termux',
          inputSchema: {
            type: 'object',
            properties: {},
          },
        },
        {
          name: 'termux_notification',
          description: 'Send Android notification',
          inputSchema: {
            type: 'object',
            properties: {
              title: {
                type: 'string',
                description: 'Notification title',
              },
              content: {
                type: 'string',
                description: 'Notification content',
              },
              priority: {
                type: 'string',
                enum: ['min', 'low', 'default', 'high', 'max'],
                default: 'default',
              },
              vibrate: {
                type: 'boolean',
                default: false,
              },
            },
            required: ['title', 'content'],
          },
        },
        {
          name: 'storage_info',
          description: 'Get Android storage information',
          inputSchema: {
            type: 'object',
            properties: {
              path: {
                type: 'string',
                description: 'Storage path to check',
                default: '/storage/emulated/0',
              },
            },
          },
        },
        {
          name: 'termux_tts',
          description: 'Text-to-speech using Android TTS',
          inputSchema: {
            type: 'object',
            properties: {
              text: {
                type: 'string',
                description: 'Text to speak',
              },
              language: {
                type: 'string',
                description: 'Language code (e.g., en-US)',
                default: 'en-US',
              },
              rate: {
                type: 'number',
                description: 'Speech rate (0.5 - 2.0)',
                minimum: 0.5,
                maximum: 2.0,
                default: 1.0,
              },
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
              action: {
                type: 'string',
                enum: ['get', 'set'],
                description: 'Clipboard action',
              },
              text: {
                type: 'string',
                description: 'Text to set (required for set action)',
              },
            },
            required: ['action'],
          },
        },
      ],
    }));
    
    // Tool execution handler
    this.server.setRequestHandler('tools/call', async (request) => {
      const { name, arguments: args } = request.params;
      
      try {
        switch (name) {
          case 'battery_status':
            return await this.getBatteryStatus();
            
          case 'termux_notification':
            return await this.sendNotification(args);
            
          case 'storage_info':
            return await this.getStorageInfo(args.path);
            
          case 'termux_tts':
            return await this.textToSpeech(args);
            
          case 'clipboard_manager':
            return await this.manageClipboard(args);
            
          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `Error executing ${name}: ${error.message}`,
            },
          ],
        };
      }
    });
  }
  
  async getBatteryStatus() {
    try {
      const { stdout } = await execAsync('termux-battery-status');
      const battery = JSON.parse(stdout);
      
      return {
        content: [
          {
            type: 'text',
            text: `Battery Status:
- Level: ${battery.percentage}%
- Status: ${battery.status}
- Health: ${battery.health}
- Temperature: ${battery.temperature}°C
- Plugged: ${battery.plugged}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to get battery status: ${error.message}`);
    }
  }
  
  async sendNotification(args) {
    const { title, content, priority, vibrate } = args;
    
    let command = `termux-notification --title "${title}" --content "${content}"`;
    
    if (priority && priority !== 'default') {
      command += ` --priority ${priority}`;
    }
    
    if (vibrate) {
      command += ' --vibrate 200,100,200';
    }
    
    try {
      await execAsync(command);
      return {
        content: [
          {
            type: 'text',
            text: `Notification sent: ${title}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to send notification: ${error.message}`);
    }
  }
  
  async getStorageInfo(storagePath = '/storage/emulated/0') {
    try {
      const { stdout } = await execAsync(`df -h ${storagePath}`);
      const lines = stdout.trim().split('\n');
      const data = lines[1].split(/\s+/);
      
      return {
        content: [
          {
            type: 'text',
            text: `Storage Information for ${storagePath}:
- Total: ${data[1]}
- Used: ${data[2]}
- Available: ${data[3]}
- Usage: ${data[4]}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to get storage info: ${error.message}`);
    }
  }
  
  async textToSpeech(args) {
    const { text, language, rate } = args;
    
    let command = `termux-tts-speak "${text}"`;
    
    if (language) {
      command += ` -l ${language}`;
    }
    
    if (rate) {
      command += ` -r ${rate}`;
    }
    
    try {
      await execAsync(command);
      return {
        content: [
          {
            type: 'text',
            text: `Speaking: "${text}" in ${language || 'default language'}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to speak text: ${error.message}`);
    }
  }
  
  async manageClipboard(args) {
    const { action, text } = args;
    
    try {
      if (action === 'get') {
        const { stdout } = await execAsync('termux-clipboard-get');
        return {
          content: [
            {
              type: 'text',
              text: `Clipboard content: ${stdout}`,
            },
          ],
        };
      } else if (action === 'set') {
        if (!text) {
          throw new Error('Text is required for set action');
        }
        await execAsync(`termux-clipboard-set "${text}"`);
        return {
          content: [
            {
              type: 'text',
              text: `Clipboard set to: ${text}`,
            },
          ],
        };
      }
    } catch (error) {
      throw new Error(`Clipboard operation failed: ${error.message}`);
    }
  }
  
  setupResources() {
    // Resource discovery
    this.server.setRequestHandler('resources/list', async () => ({
      resources: [
        {
          uri: 'termux://system/info',
          name: 'System Information',
          description: 'Termux and Android system information',
          mimeType: 'application/json',
        },
        {
          uri: 'termux://contacts/list',
          name: 'Contact List',
          description: 'Android contacts (requires permission)',
          mimeType: 'application/json',
        },
      ],
    }));
    
    // Resource reading
    this.server.setRequestHandler('resources/read', async (request) => {
      const { uri } = request.params;
      
      if (uri === 'termux://system/info') {
        return await this.getSystemInfo();
      } else if (uri === 'termux://contacts/list') {
        return await this.getContactList();
      }
      
      throw new Error(`Unknown resource: ${uri}`);
    });
  }
  
  async getSystemInfo() {
    try {
      const [deviceInfo, termuxInfo] = await Promise.all([
        execAsync('termux-info'),
        execAsync('uname -a'),
      ]);
      
      return {
        contents: [
          {
            uri: 'termux://system/info',
            mimeType: 'application/json',
            text: JSON.stringify({
              device: deviceInfo.stdout,
              system: termuxInfo.stdout,
              termuxHome: process.env.HOME,
              prefix: process.env.PREFIX,
            }, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to get system info: ${error.message}`);
    }
  }
  
  async getContactList() {
    try {
      const { stdout } = await execAsync('termux-contact-list');
      const contacts = JSON.parse(stdout);
      
      return {
        contents: [
          {
            uri: 'termux://contacts/list',
            mimeType: 'application/json',
            text: JSON.stringify(contacts, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to get contacts: ${error.message}`);
    }
  }
  
  async start() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Termux MCP Server started');
  }
}

// Start the server
const server = new TermuxMCPServer();
server.start().catch(console.error);
```

### Configuring MCP Servers in Extensions

The Gemini CLI uses the mcpServers configuration in your settings.json file to locate and connect to MCP servers. This configuration supports multiple servers with different transport mechanisms. You can configure MCP servers at the global level in the ~/.gemini/settings.json file or in your project's root directory, create or open the .gemini/settings.json file. Within the file, add the mcpServers configuration block. Add an mcpServers object to your settings.json file:

```json
{
  "mcpServers": {
    "termux-advanced": {
      "command": "node",
      "args": ["./mcp-servers/termux-advanced.js"],
      "env": {
        "TERMUX_HOME": "/data/data/com.termux/files/home",
        "ANDROID_DATA": "/storage/emulated/0",
        "NODE_ENV": "production"
      },
      "cwd": "~/.gemini/extensions/termux-suite",
      "timeout": 30000,
      "trust": false,
      "includeTools": [
        "battery_status",
        "termux_notification",
        "storage_info",
        "termux_tts",
        "clipboard_manager"
      ]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "~/databases/"],
      "trust": true
    }
  }
}
```

## Complex Command Hierarchies

### Namespace Architecture

Sub-directories are used to create namespaced commands, with the path separator (/ or \) being converted to a colon (:). A file at <project>/.gemini/commands/test.toml becomes the command /test. A file at <project>/.gemini/commands/git/commit.toml becomes the namespaced command /git:commit.

Create a sophisticated command structure:

```bash
# Create complex command hierarchy
mkdir -p ~/.gemini/extensions/termux-suite/commands/{android,dev,security,network,data}
```

### Android Integration Commands

```toml
# commands/android/intent.toml
description = "Create and execute Android intents"
prompt = """
Create an Android intent to {{args}}.

Provide:
1. The complete termux-open or am start command
2. Explanation of intent components:
   - Action (e.g., android.intent.action.VIEW)
   - Data URI if applicable
   - Package/Component if targeting specific app
   - Flags and extras
3. Alternative methods using termux-open
4. Security considerations

Include examples for common scenarios:
- Opening URLs
- Launching apps
- Sharing content
- Starting activities
"""

# commands/android/permissions.toml
description = "Check and request Android permissions"
prompt = """
For the operation: {{args}}

Analyze and provide:
1. Required Android permissions
2. How to check current permissions: !{termux-info}
3. Commands to request permissions
4. Fallback strategies if permissions are denied
5. Security best practices

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
1. Command to access the requested sensor
2. Data parsing strategy
3. Real-time monitoring setup
4. Battery impact considerations
5. Example processing script
"""
```

### Development Workflow Commands

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
1. Project structure with proper directories
2. Package installation commands (pkg and language-specific)
3. Configuration files (.bashrc additions, env vars)
4. Git setup with proper .gitignore
5. Editor configuration (vim/neovim)
6. Testing framework setup
7. Build scripts adapted for Termux
8. Debugging setup

Consider Termux limitations:
- No systemd (use termux-services)
- Modified FHS (PREFIX=/data/data/com.termux/files/usr)
- Limited syscalls
- Android storage restrictions
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
1. Analyze error symptoms
2. Check common Termux-specific issues:
   - Shebang paths (#!/data/data/com.termux/files/usr/bin/bash)
   - Permission problems
   - Missing dependencies
   - Path issues
3. Provide diagnostic commands
4. Suggest fixes with explanations
5. Create test cases to verify fix
"""

# commands/dev/optimize.toml
description = "Optimize code for Termux constraints"
prompt = """
Optimize this code/process for Termux: {{args}}

Current resource usage:
!{free -h}
!{df -h /data}
!{top -n 1 -b | head -20}

Optimization strategies:
1. Memory optimization (limited RAM on mobile)
2. Storage optimization (internal vs SD card)
3. Battery efficiency
4. Network usage reduction
5. Process management
6. Caching strategies
7. Background task handling

Provide optimized version with benchmarks.
"""
```

### Security Commands

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
1. File permissions check
2. Exposed credentials scan
3. Network connections review
4. Running processes analysis
5. Package vulnerabilities
6. Configuration weaknesses
7. Encryption status

Provide remediation steps for any issues found.
"""

# commands/security/encrypt.toml
description = "Implement encryption for sensitive data"
prompt = """
Set up encryption for: {{args}}

Available tools:
!{pkg list-installed | grep -E "gpg|openssl|crypt"}

Implement:
1. Choose appropriate encryption method
2. Key generation commands
3. Encryption/decryption scripts
4. Secure key storage strategy
5. Integration with Termux-keyring if applicable
6. Backup and recovery procedures

Create working examples with proper error handling.
"""
```

### Network Commands

```toml
# commands/network/tunnel.toml
description = "Set up network tunnels and proxies"
prompt = """
Create network tunnel for: {{args}}

Network status:
!{ip addr show}
!{netstat -tuln | head -20}

Configure:
1. SSH tunnel setup (if applicable)
2. VPN configuration
3. Proxy settings
4. Port forwarding rules
5. DNS configuration
6. Firewall rules (iptables if available)
7. Connection persistence
8. Auto-reconnect scripts

Handle Termux networking limitations appropriately.
"""

# commands/network/api.toml
description = "Create API client/server in Termux"
prompt = """
Build API {{args}} for Termux.

Requirements analysis based on:
- Available ports
- Network interfaces
- Security constraints

Implement:
1. Server setup (if server)
2. Client configuration (if client)
3. Authentication mechanism
4. Rate limiting
5. Error handling
6. Logging system
7. Testing endpoints
8. Documentation

Use Termux-compatible libraries and consider mobile constraints.
"""
```

## Context Management System

### Hierarchical Context Loading

Hierarchical Loading: The CLI combines GEMINI.md files from multiple locations. More specific files override general ones. The loading order is: Global Context: ~/.gemini/GEMINI.md (for instructions that apply to all your projects). Project/Ancestor Context: The CLI searches from your current directory up to the project root for GEMINI.md files. Sub-directory Context: The CLI also scans subdirectories for GEMINI.md files, allowing for component-specific instructions.

Create a comprehensive context system:

```markdown
# ~/.gemini/extensions/termux-suite/GEMINI.md

# Termux Development Context

## System Environment
You are operating in a Termux environment on Android. This is a Linux environment with significant constraints and unique characteristics.

### System Paths
- Home: /data/data/com.termux/files/home
- Prefix: /data/data/com.termux/files/usr
- Temp: /data/data/com.termux/files/usr/tmp
- Android Storage: ~/storage/ (after termux-setup-storage)
  - Shared: ~/storage/shared (maps to /storage/emulated/0)
  - Downloads: ~/storage/downloads
  - DCIM: ~/storage/dcim
  - Pictures: ~/storage/pictures
  - Music: ~/storage/music

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
1. Check command availability before use
2. Validate all inputs
3. Use trap for cleanup
4. Provide meaningful error messages
5. Log errors to: ~/.gemini/logs/

### Testing Requirements
- Test all scripts in actual Termux environment
- Check compatibility with different Android versions
- Verify termux-api calls work correctly
- Test with limited permissions
- Validate storage access

## Security Policies

### Forbidden Operations
NEVER attempt or suggest:
- Rooting device
- Modifying system files outside Termux
- Accessing other app's private data
- Running commands with `su` or `sudo`
- Disabling Android security features

### Credential Management
- Use termux-keyring when available
- Never store plaintext passwords
- Use environment variables from encrypted sources
- Implement proper session management
- Regular credential rotation

### Network Security
- Always use HTTPS when possible
- Validate SSL certificates
- Implement rate limiting
- Use SSH keys instead of passwords
- Configure fail2ban equivalents

## Performance Optimization

### Resource Constraints
Mobile devices have limited:
- RAM (typically 2-8GB, shared with Android)
- CPU (thermal throttling is common)
- Battery (optimize for power efficiency)
- Storage (internal is faster but limited)

### Optimization Strategies
1. **Memory Management**
   - Use streaming instead of loading entire files
   - Implement aggressive garbage collection
   - Monitor memory usage with `free -h`
   
2. **CPU Optimization**
   - Use nice values for background tasks
   - Implement task queuing
   - Avoid CPU-intensive operations during peak hours
   
3. **Battery Optimization**
   - Use wake locks sparingly
   - Batch network requests
   - Implement exponential backoff
   - Use Termux:Boot for scheduled tasks

## Integration Guidelines

### Termux:API Integration
When using Termux:API, always:
1. Check if termux-api package is installed
2. Verify Termux:API app is installed
3. Handle permission requests gracefully
4. Provide fallbacks for missing permissions
5. Test on different Android versions

### Android Integration
- Use intents for app interaction
- Respect Android's permission model
- Handle storage access framework properly
- Work with content providers when needed
- Implement proper broadcast receivers

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

1. **"Permission denied"**
   - Check file permissions: `ls -la`
   - Verify storage access: `termux-setup-storage`
   - Check SELinux context if applicable

2. **"Command not found"**
   - Install missing package: `pkg install <package>`
   - Check PATH: `echo $PATH`
   - Verify shebang path

3. **"No such file or directory"**
   - Check PREFIX paths
   - Verify symbolic links
   - Check case sensitivity

4. **"Cannot allocate memory"**
   - Check available memory: `free -h`
   - Kill unnecessary processes
   - Increase swap if possible

## Workflow Automation

### Task Automation Rules
1. Use Termux:Boot for startup tasks
2. Implement proper logging
3. Handle network connectivity changes
4. Respect Doze mode and battery optimization
5. Use Termux:Widget for quick actions

### CI/CD in Termux
- Use local Git hooks
- Implement testing pipelines
- Automate builds with make or npm scripts
- Deploy using rsync or scp
- Monitor with custom scripts

## Communication Protocols

### User Interaction
- Always explain Termux-specific considerations
- Provide alternative solutions for limitations
- Include installation commands for dependencies
- Warn about battery/performance impact
- Suggest optimization opportunities

### Code Generation
When generating code:
1. Include proper error handling
2. Add comprehensive comments
3. Provide usage examples
4. Include dependency checks
5. Add performance considerations

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

# Clear temporary files
find /data/data/com.termux/files/usr/tmp -type f -mtime +7 -delete

# Rotate logs
find ~/.gemini/logs -name "*.log" -mtime +30 -delete

# Check disk usage
df -h
du -sh ~/.gemini/*
```

## Advanced Features

### Custom Tool Integration
When integrating new tools:
1. Check Termux compatibility
2. Verify architecture support (arm64, etc.)
3. Test resource consumption
4. Document installation process
5. Create wrapper scripts if needed

### Extension Development
For new extensions:
1. Follow modular design
2. Implement proper error handling
3. Include comprehensive tests
4. Document all features
5. Provide migration guides
```

### Context File Imports

Create specialized context files:

```markdown
# contexts/android-integration.md

# Android Integration Context

## Available Termux:API Commands

### Device Information
- `termux-battery-status` - Battery information
- `termux-brightness` - Screen brightness control
- `termux-call-log` - Call history
- `termux-camera-info` - Camera information
- `termux-contact-list` - Access contacts
- `termux-infrared-frequencies` - IR capabilities
- `termux-location` - GPS location
- `termux-sensor` - Sensor data
- `termux-telephony-deviceinfo` - Device info
- `termux-wifi-connectioninfo` - WiFi status
- `termux-wifi-scaninfo` - WiFi networks

### System Interaction
- `termux-clipboard-get/set` - Clipboard access
- `termux-dialog` - UI dialogs
- `termux-download` - Download manager
- `termux-fingerprint` - Biometric auth
- `termux-keystore` - Android keystore
- `termux-media-player` - Media control
- `termux-media-scan` - Media scanner
- `termux-microphone-record` - Audio recording
- `termux-notification` - Notifications
- `termux-notification-remove` - Clear notifications
- `termux-open` - Open files/URLs
- `termux-open-url` - Open URLs
- `termux-share` - Share content
- `termux-sms-list` - SMS history
- `termux-sms-send` - Send SMS
- `termux-storage-get` - Storage access
- `termux-toast` - Toast messages
- `termux-torch` - Flashlight control
- `termux-tts-engines` - TTS engines
- `termux-tts-speak` - Text to speech
- `termux-usb` - USB device access
- `termux-vibrate` - Vibration control
- `termux-volume` - Volume control
- `termux-wallpaper` - Wallpaper control
- `termux-wake-lock` - Wake lock control
- `termux-wake-unlock` - Release wake lock

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
1. Always check permission before use
2. Provide graceful fallbacks
3. Explain why permission is needed
4. Don't request unnecessary permissions
5. Cache permission status

## Intent Examples

### Common Intent Patterns
```bash
# Open URL in browser
termux-open-url "https://example.com"

# Share text
echo "Hello" | termux-share -a send

# Open specific app
am start -n com.android.settings/.Settings

# Send broadcast
am broadcast -a android.intent.action.BOOT_COMPLETED

# Start service
am startservice -n com.example/.MyService
```

## Storage Access

### Storage Paths After Setup
```bash
# Run first:
termux-setup-storage

# Then access:
~/storage/shared/          # Main storage
~/storage/downloads/       # Downloads folder
~/storage/dcim/           # Camera folder
~/storage/pictures/       # Pictures
~/storage/music/          # Music
~/storage/movies/         # Videos
~/storage/external-1/     # SD card (if present)
```

### File Access Patterns
- Use `~/storage/shared/` for user-accessible files
- Keep app data in `~/.local/share/`
- Use `$PREFIX/tmp/` for temporary files
- Cache in `~/.cache/`
```

## Tool Integration & Security

### Advanced Tool Restrictions

excludeTools: ["run_shell_command"]

Create sophisticated tool restriction patterns:

```json
{
  "name": "termux-secure",
  "version": "1.0.0",
  "excludeTools": [
    "run_shell_command(rm -rf)",
    "run_shell_command(su)",
    "run_shell_command(sudo)",
    "run_shell_command(chmod 777)",
    "run_shell_command(kill -9)",
    "run_shell_command(dd if=)",
    "run_shell_command(mkfs)",
    "file_delete(/data/data/com.termux/files/home/.bashrc)",
    "file_delete(/data/data/com.termux/files/home/.profile)",
    "file_write(/system)",
    "file_write(/proc)",
    "file_write(/sys)"
  ],
  "includeTools": [
    "read_file",
    "read_many_files",
    "file_write",
    "run_shell_command",
    "google_web_search",
    "save_memory",
    "web_fetch"
  ],
  "toolRestrictions": {
    "run_shell_command": {
      "allowedCommands": [
        "ls", "pwd", "echo", "cat", "grep", "find",
        "node", "python", "git", "npm", "pip",
        "termux-*", "pkg", "apt"
      ],
      "blockedPatterns": [
        "*/etc/*",
        "*/system/*",
        "*password*",
        "*secret*",
        "*private*key*"
      ],
      "requireConfirmation": true,
      "logCommands": true
    },
    "file_write": {
      "allowedPaths": [
        "~/projects/*",
        "~/.gemini/*",
        "/data/data/com.termux/files/home/*"
      ],
      "maxFileSize": "10MB",
      "allowedExtensions": [
        ".js", ".py", ".sh", ".json", ".md",
        ".txt", ".toml", ".yaml", ".yml"
      ]
    }
  }
}
```

### Security Middleware Implementation

```javascript
// security-middleware.js
class SecurityMiddleware {
  constructor(config) {
    this.config = config;
    this.auditLog = [];
  }
  
  async validateToolCall(tool, params) {
    const restriction = this.config.toolRestrictions[tool];
    if (!restriction) return true;
    
    // Log the attempt
    this.auditLog.push({
      timestamp: new Date(),
      tool,
      params,
      user: process.env.USER
    });
    
    // Check allowed commands
    if (tool === 'run_shell_command' && restriction.allowedCommands) {
      const command = params.command.split(' ')[0];
      if (!restriction.allowedCommands.includes(command)) {
        throw new Error(`Command '${command}' is not allowed`);
      }
    }
    
    // Check blocked patterns
    if (restriction.blockedPatterns) {
      for (const pattern of restriction.blockedPatterns) {
        const regex = new RegExp(pattern.replace('*', '.*'));
        if (regex.test(JSON.stringify(params))) {
          throw new Error(`Blocked pattern detected: ${pattern}`);
        }
      }
    }
    
    // Check file paths
    if (tool === 'file_write' && restriction.allowedPaths) {
      const filePath = params.path;
      const allowed = restriction.allowedPaths.some(allowedPath => {
        const regex = new RegExp('^' + allowedPath.replace('*', '.*') + '$');
        return regex.test(filePath);
      });
      
      if (!allowed) {
        throw new Error(`Path '${filePath}' is not in allowed paths`);
      }
    }
    
    // Check file size
    if (tool === 'file_write' && restriction.maxFileSize) {
      const maxSize = this.parseSize(restriction.maxFileSize);
      const content = params.content || '';
      if (content.length > maxSize) {
        throw new Error(`File size exceeds maximum of ${restriction.maxFileSize}`);
      }
    }
    
    // Require confirmation if needed
    if (restriction.requireConfirmation) {
      return await this.requestConfirmation(tool, params);
    }
    
    return true;
  }
  
  parseSize(sizeStr) {
    const units = { KB: 1024, MB: 1024*1024, GB: 1024*1024*1024 };
    const match = sizeStr.match(/^(\d+)(KB|MB|GB)$/i);
    if (!match) return parseInt(sizeStr);
    return parseInt(match[1]) * units[match[2].toUpperCase()];
  }
  
  async requestConfirmation(tool, params) {
    // In real implementation, this would interact with user
    console.log(`Confirmation required for ${tool}:`, params);
    // Return true for auto-approval in this example
    return true;
  }
  
  getAuditLog() {
    return this.auditLog;
  }
}

module.exports = SecurityMiddleware;
```

## Part III: Production Implementation

## Building Production-Ready Extensions

### Complete Extension Package Structure

```bash
termux-ultimate-extension/
├── gemini-extension.json
├── package.json
├── README.md
├── LICENSE
├── CHANGELOG.md
├── .github/
│   └── workflows/
│       └── test.yml
├── commands/
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
├── contexts/
│   ├── GEMINI.md
│   ├── android-integration.md
│   ├── security-policies.md
│   └── performance-guidelines.md
├── mcp-servers/
│   ├── termux-advanced/
│   │   ├── index.js
│   │   ├── package.json
│   │   └── test/
│   ├── android-bridge/
│   │   ├── index.js
│   │   └── package.json
│   └── security-monitor/
│       ├── index.js
│       └── package.json
├── scripts/
│   ├── install.sh
│   ├── uninstall.sh
│   ├── update.sh
│   └── test.sh
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── docs/
    ├── installation.md
    ├── configuration.md
    ├── commands.md
    └── troubleshooting.md
```

### Production gemini-extension.json

```json
{
  "name": "termux-ultimate",
  "version": "3.0.0",
  "description": "Comprehensive Termux integration for Gemini CLI",
  "author": "Your Name",
  "license": "MIT",
  "homepage": "https://github.com/yourusername/termux-ultimate",
  "repository": {
    "type": "git",
    "url": "https://github.com/yourusername/termux-ultimate.git"
  },
  "bugs": {
    "url": "https://github.com/yourusername/termux-ultimate/issues"
  },
  "engines": {
    "node": ">=18.0.0",
    "gemini-cli": ">=1.0.0"
  },
  "mcpServers": {
    "termux-advanced": {
      "command": "node",
      "args": ["./mcp-servers/termux-advanced/index.js"],
      "env": {
        "NODE_ENV": "production",
        "LOG_LEVEL": "${LOG_LEVEL:-info}"
      },
      "timeout": 30000,
      "trust": false,
      "includeTools": [
        "battery_status",
        "termux_notification",
        "storage_info",
        "termux_tts",
        "clipboard_manager"
      ]
    },
    "android-bridge": {
      "command": "node",
      "args": ["./mcp-servers/android-bridge/index.js"],
      "timeout": 60000
    },
    "security-monitor": {
      "command": "node", 
      "args": ["./mcp-servers/security-monitor/index.js"],
      "trust": true
    }
  },
  "contextFileName": "contexts/GEMINI.md",
  "additionalContexts": [
    "contexts/android-integration.md",
    "contexts/security-policies.md",
    "contexts/performance-guidelines.md"
  ],
  "excludeTools": [
    "run_shell_command(su)",
    "run_shell_command(sudo)",
    "run_shell_command(rm -rf /)",
    "file_delete(/system)",
    "file_write(/proc)",
    "file_write(/sys)"
  ],
  "dependencies": {
    "extensions": ["base-gemini-ext"],
    "packages": [
      "@modelcontextprotocol/sdk@^0.5.0",
      "sqlite3@^5.1.6"
    ],
    "termuxPackages": [
      "nodejs-lts",
      "python",
      "git",
      "termux-api"
    ]
  },
  "hooks": {
    "onLoad": "./scripts/on-load.js",
    "onUnload": "./scripts/on-unload.js",
    "beforeCommand": "./scripts/before-command.js",
    "afterCommand": "./scripts/after-command.js"
  },
  "configuration": {
    "properties": {
      "termux-ultimate.enableAdvancedFeatures": {
        "type": "boolean",
        "default": false,
        "description": "Enable advanced experimental features"
      },
      "termux-ultimate.logLevel": {
        "type": "string",
        "enum": ["debug", "info", "warn", "error"],
        "default": "info",
        "description": "Logging level for extension"
      }
    }
  }
}
```

### Installation Script

```bash
#!/data/data/com.termux/files/usr/bin/bash
# install.sh - Production installation script

set -euo pipefail
IFS=$'\n\t'

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EXTENSION_NAME="termux-ultimate"
EXTENSION_VERSION="3.0.0"
INSTALL_DIR="$HOME/.gemini/extensions/$EXTENSION_NAME"
LOG_FILE="$HOME/.gemini/logs/install-$(date +%Y%m%d-%H%M%S).log"

# Logging functions
log() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

# Header
echo "=================================" | tee -a "$LOG_FILE"
echo "Termux Ultimate Extension Installer" | tee -a "$LOG_FILE"
echo "Version: $EXTENSION_VERSION" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        error "Node.js is not installed. Run: pkg install nodejs-lts"
    fi
    
    NODE_VERSION=$(node -v | cut -d'v' -f2)
    REQUIRED_VERSION="18.0.0"
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$NODE_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        error "Node.js version $NODE_VERSION is too old. Required: v$REQUIRED_VERSION+"
    fi
    
    # Check Gemini CLI
    if ! command -v gemini &> /dev/null; then
        error "Gemini CLI is not installed. Run: npm install -g @google/gemini-cli"
    fi
    
    # Check Termux packages
    REQUIRED_PACKAGES=("git" "python" "termux-api")
    for pkg in "${REQUIRED_PACKAGES[@]}"; do
        if ! pkg list-installed 2>/dev/null | grep -q "^$pkg/"; then
            warn "Package '$pkg' is not installed. Installing..."
            pkg install -y "$pkg" || error "Failed to install $pkg"
        fi
    done
    
    # Check Termux:API app
    if ! termux-api-start 2>/dev/null; then
        warn "Termux:API app might not be installed. Some features may not work."
    fi
    
    log "Prerequisites check completed"
}

# Backup existing installation
backup_existing() {
    if [ -d "$INSTALL_DIR" ]; then
        log "Backing up existing installation..."
        BACKUP_DIR="$INSTALL_DIR.backup.$(date +%Y%m%d-%H%M%S)"
        mv "$INSTALL_DIR" "$BACKUP_DIR"
        log "Backup created at: $BACKUP_DIR"
    fi
}

# Install extension
install_extension() {
    log "Installing extension..."
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    
    # Copy files
    cp -r ./* "$INSTALL_DIR/" 2>/dev/null || true
    
    # Install Node.# Advanced Gemini CLI Setup and Configuration

After the initial installation, there are several important configuration steps and features to set up for a more powerful Gemini CLI experience. Here's a comprehensive guide to the advanced setup options.

## Configuration Files and Settings

### The settings.json File

The main configuration file for Gemini CLI is `settings.json`, which controls how the tool behaves. Settings are applied with the following precedence order:

- **Project Settings**: `.gemini/settings.json` (highest priority, overrides user and system settings)
- **User Settings**: `~/.gemini/settings.json` (overrides system settings)
- **System Settings**: `/etc/gemini-cli/settings.json` (lowest priority, applies to all users)

### Key Configuration Options

Create or edit your `~/.gemini/settings.json` file to include these important settings:

```json
{
  "theme": "Default",
  "selectedAuthType": "oauth-personal",
  "autoAccept": false,
  "sandbox": true,
  "vimMode": false,
  "checkpointing": true,
  "includeDirectories": ["../lib", "../docs"],
  "chatCompression": true,
  "usageStatisticsEnabled": true
}
```

**Important settings explained:**
- `autoAccept`: Auto-approves safe, read-only tool calls
- `sandbox`: Isolates tool execution (set to `true`, `"docker"`, or `"podman"`)
- `vimMode`: Enables Vim-style editing for the input prompt
- `checkpointing`: Enables the `/restore` command to undo file changes
- `includeDirectories`: Defines a multi-directory workspace

## Authentication Methods

### Option 1: OAuth Login (Recommended)

This is the simplest method for individual developers:

1. Start Gemini CLI by typing `gemini`
2. Choose OAuth and follow the browser authentication flow
3. Sign in with your personal Google account

**Benefits:**
- **Free tier**: 60 requests/min and 1,000 requests/day
- Access to Gemini 2.5 Pro with 1M token context window
- No API key management required

### Option 2: API Key Authentication

For specific model control or paid tier access:

1. Get your API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Set the environment variable:
   ```bash
   export GEMINI_API_KEY="YOUR_API_KEY"
   ```
3. Launch Gemini CLI with `gemini`

### Option 3: Vertex AI

For enterprise teams and production workloads:

```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
export GOOGLE_GENAI_USE_VERTEXAI=true
gemini
```

## Context Files (GEMINI.md)

Context files allow you to provide project-specific instructions to tailor Gemini CLI's behavior. These files use a hierarchical loading system:

1. **Global Context**: `~/.gemini/GEMINI.md` (applies to all projects)
2. **Project Context**: Search from current directory up to project root
3. **Sub-directory Context**: Scans subdirectories for component-specific instructions

### Creating a GEMINI.md File

Create a `GEMINI.md` file in your project root or `~/.gemini/` directory:

```markdown
# Project Instructions

## General Guidelines
- Use TypeScript for all new code
- Follow ESLint configuration
- Write comprehensive tests for new features

## Coding Standards
- Use 2 spaces for indentation
- Prefer functional programming patterns
- Always use strict equality (=== and !==)

## Dependencies
- Avoid adding new dependencies unless necessary
- Document reasons for any new packages
```

Use `/memory show` to see the combined context being sent to the model.

## MCP Server Integration

Model Context Protocol (MCP) servers extend Gemini CLI with custom tools and integrations. Configure them in your `settings.json`:

```json
{
  "mcpServers": {
    "github": {
      "httpUrl": "https://api.githubcopilot.com/mcp/",
      "headers": {
        "Authorization": "YOUR_GITHUB_PAT"
      },
      "timeout": 5000
    },
    "context7": {
      "httpUrl": "https://mcp.context7.com/mcp"
    }
  }
}
```

### Popular MCP Servers

- **GitHub MCP**: Integrates with GitHub repositories and issues
- **Context7**: Provides up-to-date documentation for frameworks
- **Firebase**: Manages Firebase projects
- **Google Workspace**: Works with Docs, Sheets, Calendar, and Gmail

Use `/mcp` to list configured servers and their available tools.

## IDE Integration

### VS Code Setup

Connect Gemini CLI to VS Code for enhanced context awareness:

1. Run `/ide install` to set up the extension
2. Use `/ide enable` to connect to your editor
3. Benefits include:
   - Automatic workspace context (recent files, cursor position, selected text)
   - Native diffing capabilities
   - Direct code change approval in the editor



## Custom Commands

Create reusable shortcuts for frequently used prompts. Store them in:
- `~/.gemini/commands/` (global commands)
- `<project>/.gemini/commands/` (project-specific)

### Example Custom Command

Create `~/.gemini/commands/test/gen.toml`:

```toml
description = "Generate unit tests for selected code"
prompt = """
Generate comprehensive unit tests for the following code: {{args}}
Use the project's existing testing framework and follow established patterns.
"""
```

## Advanced Features

### Checkpointing and Restore

Enable checkpointing to save project snapshots before file modifications:

```bash
# Enable via flag
gemini --checkpointing

# Or in settings.json
"checkpointing": true
```

Restore previous states with `/restore` command.

### Extensions

Create extensions by placing them in:
- `<workspace>/.gemini/extensions/`
- `~/.gemini/extensions/`

Each extension directory needs a `gemini-extension.json` file configuring MCP servers, tools, and context files.

### File Ignoring

Create a `.geminiignore` file to exclude files and directories:

```
/backups/
*.log
secret-config.json
node_modules/
```

## Useful Command-Line Flags

Launch Gemini CLI with specific options:

```bash
# Use a specific model
gemini -m gemini-2.5-flash

# Non-interactive mode for scripts
gemini -p "Explain this architecture"

# Include multiple directories
gemini --include-directories ../lib,../docs

# Enable sandbox mode
gemini --sandbox

# Auto-approve all tool calls (use with caution)
gemini --yolo

# Enable debug output
gemini -d
```



## Essential Slash Commands

Key commands to use within Gemini CLI:

- `/settings` - Open settings editor
- `/memory refresh` - Reload GEMINI.md files
- `/tools` - List available tools
- `/compress` - Compress chat context to save tokens
- `/chat save <tag>` - Save current conversation
- `/chat resume <tag>` - Resume saved conversation
- `/stats` - Show token usage
- `/init` - Generate starter GEMINI.md for your project
- `/theme` - Change visual theme

## Tips for Optimal Setup

1. **Start with basic configuration**: Begin with OAuth authentication and default settings
2. **Customize gradually**: Add GEMINI.md files and MCP servers as needed
3. **Use checkpointing**: Enable it for projects where you want rollback capability
4. **Configure project-specific settings**: Use local `.gemini/settings.json` for project overrides
5. **Set up relevant MCP servers**: Add GitHub, Context7, or other servers based on your workflow

With these advanced configurations, Gemini CLI becomes a powerful, customized AI assistant tailored to your specific development needs and workflow preferences.


# Complete Walkthrough: Integrating Custom Toolsets into Gemini CLI for Coding

Gemini CLI is an open-source AI agent that provides lightweight access to Gemini directly in your terminal, offering a free tier with 60 requests/min and 1,000 requests/day with a personal Google account. The real power of Gemini CLI lies in its extensibility through the Model Context Protocol (MCP), which allows you to seamlessly integrate custom tools and services into your AI-powered development workflow.

## Understanding the MCP Architecture

The Model Context Protocol provides a standardized way to connect external services and capabilities to the AI agent. When you configure MCP servers in Gemini CLI, they expose tools that become automatically integrated into the AI's available capabilities. The beauty of this system is that once configured, these custom tools become seamlessly integrated into Gemini CLI's workflow - the AI will automatically select and use them when appropriate based on your prompts.

## Setting Up Your Development Environment

### **Initial Installation**

First, ensure you have Gemini CLI installed globally. You'll need Node.js version 20 or higher:

```bash
# Install Gemini CLI globally
npm install -g @google/gemini-cli

# Verify installation
gemini --version
```

### **Project Structure**

Create a dedicated project directory for your custom toolset:

```bash
mkdir my-coding-toolset
cd my-coding-toolset
npm init -y
```

## Creating Your Custom MCP Server

### **Basic Server Implementation**

Let's build a practical MCP server that provides coding assistance tools. Create a file called `coding-assistant-server.js`:

```javascript
const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;

const execAsync = promisify(exec);

class CodingAssistantServer {
  constructor() {
    this.server = new Server(
      {
        name: 'coding-assistant',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
          resources: {},
        },
      }
    );
    
    this.setupTools();
  }
  
  setupTools() {
    this.server.setRequestHandler('tools/list', async () => ({
      tools: [
        {
          name: 'analyze_complexity',
          description: 'Analyze code complexity and suggest improvements',
          inputSchema: {
            type: 'object',
            properties: {
              code: {
                type: 'string',
                description: 'Code to analyze',
              },
              language: {
                type: 'string',
                description: 'Programming language',
                enum: ['javascript', 'python', 'go', 'java'],
              },
            },
            required: ['code', 'language'],
          },
        },
        {
          name: 'generate_tests',
          description: 'Generate unit tests for given code',
          inputSchema: {
            type: 'object',
            properties: {
              code: {
                type: 'string',
                description: 'Code to test',
              },
              framework: {
                type: 'string',
                description: 'Testing framework',
                default: 'jest',
              },
            },
            required: ['code'],
          },
        },
        {
          name: 'refactor_code',
          description: 'Refactor code following best practices',
          inputSchema: {
            type: 'object',
            properties: {
              code: {
                type: 'string',
                description: 'Code to refactor',
              },
              target: {
                type: 'string',
                description: 'Refactoring target',
                enum: ['performance', 'readability', 'modularity'],
              },
            },
            required: ['code', 'target'],
          },
        },
      ],
    }));
    
    this.server.setRequestHandler('tools/call', async (request) => {
      const { name, arguments: args } = request.params;
      
      try {
        switch (name) {
          case 'analyze_complexity':
            return await this.analyzeComplexity(args);
          case 'generate_tests':
            return await this.generateTests(args);
          case 'refactor_code':
            return await this.refactorCode(args);
          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `Error: ${error.message}`,
            },
          ],
        };
      }
    });
  }
  
  async analyzeComplexity(args) {
    // Implement complexity analysis logic
    const { code, language } = args;
    // This is a simplified example - in production, you'd use proper AST analysis
    const lines = code.split('\n').length;
    const conditions = (code.match(/if|else|switch|case/g) || []).length;
    const loops = (code.match(/for|while|do/g) || []).length;
    
    return {
      content: [
        {
          type: 'text',
          text: `Code Complexity Analysis:
- Lines of code: ${lines}
- Conditional statements: ${conditions}
- Loops: ${loops}
- Cyclomatic complexity estimate: ${conditions + loops + 1}
${conditions + loops > 10 ? '⚠️ High complexity detected. Consider breaking into smaller functions.' : '✅ Complexity is manageable.'}`,
        },
      ],
    };
  }
  
  async generateTests(args) {
    const { code, framework } = args;
    // Generate test template based on the code structure
    return {
      content: [
        {
          type: 'text',
          text: `Generated ${framework} test template:

\`\`\`javascript
describe('Function Tests', () => {
  test('should handle valid input', () => {
    // Test implementation
    expect(result).toBeDefined();
  });
  
  test('should handle edge cases', () => {
    // Edge case testing
  });
  
  test('should handle errors gracefully', () => {
    // Error handling tests
  });
});
\`\`\``,
        },
      ],
    };
  }
  
  async refactorCode(args) {
    const { code, target } = args;
    // Implement refactoring suggestions based on target
    return {
      content: [
        {
          type: 'text',
          text: `Refactoring for ${target}:
- Extract repeated code into functions
- Use descriptive variable names
- Apply single responsibility principle
- Consider using design patterns where appropriate`,
        },
      ],
    };
  }
  
  async start() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Coding Assistant MCP Server started');
  }
}

// Start the server
const server = new CodingAssistantServer();
server.start().catch(console.error);
```

### **Package Dependencies**

Install the required MCP SDK:

```bash
npm install @modelcontextprotocol/sdk
```

## Configuring Gemini CLI

### **Settings Configuration**

The Gemini CLI's behavior is controlled by configuration files. Create a `.gemini/settings.json` file in your project root to tell Gemini CLI how to run your MCP server:

```bash
mkdir -p .gemini
```

Add the following configuration:

```json
{
  "mcpServers": {
    "coding-assistant": {
      "command": "node",
      "args": ["./coding-assistant-server.js"],
      "env": {
        "NODE_ENV": "production"
      },
      "timeout": 30000,
      "includeTools": [
        "analyze_complexity",
        "generate_tests", 
        "refactor_code"
      ]
    }
  }
}
```

For global configuration that applies to all projects, you can also add this to `~/.gemini/settings.json`.

### **Project Context Configuration**

Create a `GEMINI.md` file to provide natural language guidelines and context to the AI:

```markdown
# Coding Assistant Project Context

## Available Custom Tools
- **analyze_complexity**: Analyzes code complexity and suggests improvements
- **generate_tests**: Automatically generates unit test templates
- **refactor_code**: Provides refactoring suggestions based on best practices

## Coding Standards
- Follow clean code principles
- Prioritize readability over clever solutions
- Always include error handling
- Write comprehensive tests for all functions

## Tool Usage Guidelines
When asked about code quality or improvements, always:
1. First use analyze_complexity to assess the current state
2. Suggest refactoring using refactor_code if complexity is high
3. Generate tests using generate_tests for any new or modified code
```

## Building and Testing Your Integration

### **Compile and Run**

First, create a simple test to ensure your server works:

```bash
# Test the server directly
node coding-assistant-server.js
```

### **Integration with Gemini CLI**

Launch Gemini CLI and verify your tools are loaded:

```bash
gemini
```

Press `Ctrl+T` to see available tools. Your custom tools should appear in the list.

### **Testing Your Tools**

Test your integrated tools with practical prompts:

```bash
# Test complexity analysis
gemini -p "Use the coding assistant to analyze this code: function calculate(a,b) { if(a>0) { for(let i=0;i<10;i++) { if(b>i) return a+b; } } return 0; }"

# Test test generation
gemini -p "Generate unit tests for a user authentication function"

# Test refactoring suggestions
gemini -p "Refactor this code for better readability"
```

## Advanced Integration Features

### **Adding AI-Powered Capabilities**

For more sophisticated features, you can integrate AI capabilities directly into your tools. Here's an example of adding code review using the Gemini API:

```javascript
// Add to your server implementation
{
  name: 'ai_code_review',
  description: 'AI-powered code review with Gemini',
  inputSchema: {
    type: 'object',
    properties: {
      code: {
        type: 'string',
        description: 'Code to review',
      },
      focus: {
        type: 'string',
        description: 'Review focus area',
        enum: ['security', 'performance', 'maintainability'],
        default: 'maintainability',
      },
    },
    required: ['code'],
  },
}

// Implementation
async aiCodeReview(args) {
  const { code, focus } = args;
  // Use Gemini API for intelligent code review
  // This requires GEMINI_API_KEY environment variable
  const review = await callGeminiAPI(code, focus);
  return {
    content: [
      {
        type: 'text',
        text: `AI Code Review (${focus}):\n${review}`,
      },
    ],
  };
}
```

### **Resource Providers**

MCP servers can also provide resources that the AI can access:

```javascript
setupResources() {
  this.server.setRequestHandler('resources/list', async () => ({
    resources: [
      {
        uri: 'coding://templates/react-component',
        name: 'React Component Template',
        description: 'Boilerplate for React components',
        mimeType: 'text/plain',
      },
      {
        uri: 'coding://docs/best-practices',
        name: 'Coding Best Practices',
        description: 'Team coding standards',
        mimeType: 'text/markdown',
      },
    ],
  }));
  
  this.server.setRequestHandler('resources/read', async (request) => {
    const { uri } = request.params;
    // Return appropriate resource content based on URI
  });
}
```

## Workflow Automation

### **CI/CD Integration**

You can use your custom tools in automated workflows:

```bash
#!/bin/bash
# pre-commit hook
gemini -p "Use coding assistant to analyze complexity of staged files"
gemini -p "Generate tests for any new functions"
```

### **Batch Processing**

Automate analysis across your entire codebase:

```bash
gemini --all_files -p "Use the coding assistant to analyze complexity of all JavaScript files and create a report"
```

## Best Practices

### **Tool Design Guidelines**

When creating custom tools, follow these principles:

1. **Single Responsibility**: Each tool should do one thing well
2. **Clear Naming**: Use descriptive names that indicate the tool's purpose
3. **Comprehensive Input Validation**: Validate all inputs and provide helpful error messages
4. **Consistent Output Format**: Return structured, predictable responses
5. **Performance Consideration**: Implement timeouts and handle long-running operations gracefully

### **Security Considerations**

Always validate and sanitize inputs, especially when executing shell commands or accessing files. Consider implementing permission controls for sensitive operations.

### **Documentation**

Document your tools thoroughly in the GEMINI.md file so the AI understands when and how to use them effectively.

## Extending with Multiple MCP Servers

You can configure multiple MCP servers to work together. For example, combine your coding assistant with GitHub integration and database tools:

```json
{
  "mcpServers": {
    "coding-assistant": {
      "command": "node",
      "args": ["./coding-assistant-server.js"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "./databases/"],
      "trust": true
    }
  }
}
```

This configuration allows the AI to seamlessly coordinate between code analysis, version control, and data storage operations.

By following this walkthrough, you've successfully integrated a custom coding toolset into Gemini CLI. The AI will now automatically discover and use these tools when appropriate, enhancing your development workflow with intelligent, context-aware assistance. The system's flexibility allows you to continuously expand and refine your toolset as your needs evolve.

# Complete In-Depth Guide to Creating and Using Extensions with Gemini CLI on Termux

## Table of Contents

### Part I: Foundation & Architecture
1. [Deep Dive into Gemini CLI Architecture](#deep-dive-into-gemini-cli-architecture)
2. [Complete Termux Setup & Optimization](#complete-termux-setup--optimization)
3. [Authentication Deep Dive](#authentication-deep-dive)
4. [Extension System Internals](#extension-system-internals)

### Part II: Advanced Extension Development
5. [MCP Server Architecture & Implementation](#mcp-server-architecture--implementation)
6. [Complex Command Hierarchies](#complex-command-hierarchies)
7. [Context Management System](#context-management-system)
8. [Tool Integration & Security](#tool-integration--security)

### Part III: Production Implementation
9. [Building Production-Ready Extensions](#building-production-ready-extensions)
10. [Performance Optimization Strategies](#performance-optimization-strategies)
11. [Debugging & Troubleshooting](#debugging--troubleshooting)
12. [Real-World Case Studies](#real-world-case-studies)

---

## Part I: Foundation & Architecture

## Deep Dive into Gemini CLI Architecture

### Core Components

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

### ReAct Loop Implementation

The Gemini command line interface (CLI) is an open source AI agent that provides access to Gemini directly in your terminal. The Gemini CLI uses a reason and act (ReAct) loop with your built-in tools and local or remote MCP servers to complete complex use cases like fixing bugs, creating new

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

### Token Management & Context Window

That free license gets you access to Gemini 2.5 Pro and its massive 1 million token context window.

The 1M token context window requires sophisticated management:

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

## Complete Termux Setup & Optimization

### Advanced Installation Process

The installation process is identical to Linux. You'll need Termux or a similar terminal emulator. I prefer Termux, make sure to download it from F-Droid store or GitHub. Version on Google Play is discontinued.

#### Step 1: Termux Environment Preparation

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

# Set up storage access
termux-setup-storage

# Configure Node.js environment
npm config set prefix ~/.npm-global
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify Node.js version (should be 18+)
node --version
npm --version
```

#### Step 2: Gemini CLI Installation with Error Handling

```bash
# Install Gemini CLI with retry logic
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

install_gemini_cli

# Verify installation
gemini --version
```

### Authentication Deep Dive

This is the simplest method. Run Gemini CLI with the --debug flag: ... Choose Google Login. The CLI will display a login URL in the terminal. Copy this link into your browser, authenticate with your Google account, and return to Termux. Gemini CLI will now be successfully authenticated.

#### Method 1: Debug Authentication (Manual)

```bash
#!/bin/bash
# auth-debug.sh - Debug authentication helper

echo "Starting Gemini CLI authentication in debug mode..."
echo "="*50

# Run in debug mode and capture output
gemini --debug 2>&1 | tee auth.log &
GEMINI_PID=$!

# Wait for authentication URL
echo "Waiting for authentication URL..."
while ! grep -q "https://accounts.google.com" auth.log 2>/dev/null; do
  sleep 1
done

# Extract and display URL
AUTH_URL=$(grep -o "https://accounts.google.com[^\"]*" auth.log | head -1)
echo ""
echo "Authentication URL found!"
echo "="*50
echo "$AUTH_URL"
echo "="*50
echo ""
echo "1. Copy the above URL"
echo "2. Open it in your browser"
echo "3. Complete authentication"
echo "4. Return here and press Enter"
read -r

# Clean up
rm -f auth.log
```

#### Method 2: Termux:API Integration (Automated)

For a more seamless experience, you can use Termux:API to open the login URL directly in your browser. Termux-api allows you to send commands to the Android system and send command to open a browser. This means that Termux would trigger Google authentication automatically by opening a browser, just like it would behave on desktop system. For that you need to install: Install Termux:API app from F-Droid. This will open the browser automatically, mimicking desktop behavior. Once authenticated, return to Termux and continue using Gemini CLI.

```bash
#!/bin/bash
# auth-api.sh - Automated authentication with Termux:API

# Check if Termux:API is installed
if ! command -v termux-open-url &> /dev/null; then
  echo "Installing termux-api package..."
  pkg install termux-api -y
fi

# Create authentication wrapper
cat > ~/gemini-auth-wrapper.sh << 'EOF'
#!/bin/bash

# Intercept authentication URL and open automatically
gemini "$@" 2>&1 | while IFS= read -r line; do
  echo "$line"
  
  # Check for Google auth URL
  if [[ "$line" =~ https://accounts.google.com ]]; then
    URL=$(echo "$line" | grep -o 'https://[^"]*' | head -1)
    if [ -n "$URL" ]; then
      echo "Opening authentication URL in browser..."
      termux-open-url "$URL"
      
      # Vibrate to notify user
      termux-vibrate -d 500
      
      # Show notification
      termux-notification \
        --title "Gemini CLI Authentication" \
        --content "Please complete authentication in browser" \
        --action "termux-open-url $URL"
    fi
  fi
done
EOF

chmod +x ~/gemini-auth-wrapper.sh

# Create alias for easy use
echo 'alias gemini-auth="~/gemini-auth-wrapper.sh"' >> ~/.bashrc
source ~/.bashrc

echo "✓ Automated authentication setup complete"
echo "Run 'gemini-auth' to start Gemini CLI with automatic browser opening"
```

#### Method 3: API Key Authentication

Get Your Key: Get an API key from Google AI Studio. Set Your Key: Make the key available to the CLI with one of these methods. Method 1: Shell Environment Variable Set the GEMINI_API_KEY environment variable. To use it across terminal sessions, add this line to your shell's profile (e.g., ~/.bashrc, ~/.zshrc).

```bash
#!/bin/bash
# setup-api-key.sh - Secure API key setup

# Secure API key storage with encryption
setup_api_key() {
  echo "Setting up Gemini API key..."
  
  # Create secure directory
  mkdir -p ~/.gemini/secure
  chmod 700 ~/.gemini/secure
  
  # Prompt for API key
  echo -n "Enter your Gemini API key: "
  read -rs API_KEY
  echo
  
  # Validate key format
  if [[ ! "$API_KEY" =~ ^[A-Za-z0-9_-]{39}$ ]]; then
    echo "Invalid API key format"
    return 1
  fi
  
  # Store encrypted (using simple base64 for Termux compatibility)
  echo "$API_KEY" | base64 > ~/.gemini/secure/api_key.enc
  chmod 600 ~/.gemini/secure/api_key.enc
  
  # Create loader script
  cat > ~/.gemini/load_api_key.sh << 'EOF'
#!/bin/bash
if [ -f ~/.gemini/secure/api_key.enc ]; then
  export GEMINI_API_KEY=$(base64 -d < ~/.gemini/secure/api_key.enc)
fi
EOF
  
  chmod +x ~/.gemini/load_api_key.sh
  
  # Add to bashrc
  echo 'source ~/.gemini/load_api_key.sh' >> ~/.bashrc
  
  echo "✓ API key configured successfully"
  echo "Run 'source ~/.bashrc' to load the key"
}

setup_api_key
```

## Extension System Internals

### Extension Discovery and Loading Process

name: The name of the extension. This is used to uniquely identify the extension and for conflict resolution when extension commands have the same name as user or project commands. ... mcpServers: A map of MCP servers to configure. The key is the name of the server, and the value is the server configuration. These servers will be loaded on startup just like MCP servers configured in a settings.json file. If both an extension and a settings.json file configure an MCP server with the same name, the server defined in the settings.json file takes precedence.

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
      path.join(os.homedir(), '.gemini', 'extensions'),  // Global
      path.join(process.cwd(), '.gemini', 'extensions')   // Project
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
            
            // Check for conflicts
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
    // Project extensions take precedence over global
    for (const conflict of this.conflicts) {
      const projectPath = path.join(process.cwd(), '.gemini', 'extensions');
      
      if (conflict.new.startsWith(projectPath)) {
        // Keep project extension
        console.log(`Extension conflict resolved: ${conflict.name} (using project version)`);
      } else {
        // Revert to existing
        const existing = this.extensions.get(conflict.name);
        existing.path = conflict.existing;
      }
    }
  }
}
```

### Extension Configuration Schema

```typescript
interface ExtensionConfig {
  name: string;
  version: string;
  description?: string;
  author?: string;
  license?: string;
  
  // MCP Server configuration
  mcpServers?: {
    [serverName: string]: {
      command: string;
      args?: string[];
      env?: Record<string, string>;
      cwd?: string;
      timeout?: number;
      trust?: boolean;
      includeTools?: string[];
      excludeTools?: string[];
    };
  };
  
  // Context configuration
  contextFileName?: string;
  additionalContexts?: string[];
  
  // Tool restrictions
  excludeTools?: string[];
  includeTools?: string[];
  
  // Dependencies
  dependencies?: {
    extensions?: string[];
    packages?: string[];
    termuxPackages?: string[];
  };
  
  // Hooks
  hooks?: {
    onLoad?: string;
    onUnload?: string;
    beforeCommand?: string;
    afterCommand?: string;
  };
}
```

## Part II: Advanced Extension Development

## MCP Server Architecture & Implementation

### Understanding MCP Protocol

An MCP server is an application that exposes tools and resources to the Gemini CLI through the Model Context Protocol, allowing it to interact with external systems and data sources. MCP servers act as a bridge between the Gemini model and your local environment or other services like APIs.

```javascript
// Complete MCP Server implementation for Termux
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
        name: 'termux-advanced',
        version: '2.0.0',
      },
      {
        capabilities: {
          tools: {},
          resources: {},
        },
      }
    );
    
    this.setupTools();
    this.setupResources();
  }
  
  setupTools() {
    // Battery status tool
    this.server.setRequestHandler('tools/list', async () => ({
      tools: [
        {
          name: 'battery_status',
          description: 'Get Android battery status via Termux',
          inputSchema: {
            type: 'object',
            properties: {},
          },
        },
        {
          name: 'termux_notification',
          description: 'Send Android notification',
          inputSchema: {
            type: 'object',
            properties: {
              title: {
                type: 'string',
                description: 'Notification title',
              },
              content: {
                type: 'string',
                description: 'Notification content',
              },
              priority: {
                type: 'string',
                enum: ['min', 'low', 'default', 'high', 'max'],
                default: 'default',
              },
              vibrate: {
                type: 'boolean',
                default: false,
              },
            },
            required: ['title', 'content'],
          },
        },
        {
          name: 'storage_info',
          description: 'Get Android storage information',
          inputSchema: {
            type: 'object',
            properties: {
              path: {
                type: 'string',
                description: 'Storage path to check',
                default: '/storage/emulated/0',
              },
            },
          },
        },
        {
          name: 'termux_tts',
          description: 'Text-to-speech using Android TTS',
          inputSchema: {
            type: 'object',
            properties: {
              text: {
                type: 'string',
                description: 'Text to speak',
              },
              language: {
                type: 'string',
                description: 'Language code (e.g., en-US)',
                default: 'en-US',
              },
              rate: {
                type: 'number',
                description: 'Speech rate (0.5 - 2.0)',
                minimum: 0.5,
                maximum: 2.0,
                default: 1.0,
              },
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
              action: {
                type: 'string',
                enum: ['get', 'set'],
                description: 'Clipboard action',
              },
              text: {
                type: 'string',
                description: 'Text to set (required for set action)',
              },
            },
            required: ['action'],
          },
        },
      ],
    }));
    
    // Tool execution handler
    this.server.setRequestHandler('tools/call', async (request) => {
      const { name, arguments: args } = request.params;
      
      try {
        switch (name) {
          case 'battery_status':
            return await this.getBatteryStatus();
            
          case 'termux_notification':
            return await this.sendNotification(args);
            
          case 'storage_info':
            return await this.getStorageInfo(args.path);
            
          case 'termux_tts':
            return await this.textToSpeech(args);
            
          case 'clipboard_manager':
            return await this.manageClipboard(args);
            
          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `Error executing ${name}: ${error.message}`,
            },
          ],
        };
      }
    });
  }
  
  async getBatteryStatus() {
    try {
      const { stdout } = await execAsync('termux-battery-status');
      const battery = JSON.parse(stdout);
      
      return {
        content: [
          {
            type: 'text',
            text: `Battery Status:
- Level: ${battery.percentage}%
- Status: ${battery.status}
- Health: ${battery.health}
- Temperature: ${battery.temperature}°C
- Plugged: ${battery.plugged}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to get battery status: ${error.message}`);
    }
  }
  
  async sendNotification(args) {
    const { title, content, priority, vibrate } = args;
    
    let command = `termux-notification --title "${title}" --content "${content}"`;
    
    if (priority && priority !== 'default') {
      command += ` --priority ${priority}`;
    }
    
    if (vibrate) {
      command += ' --vibrate 200,100,200';
    }
    
    try {
      await execAsync(command);
      return {
        content: [
          {
            type: 'text',
            text: `Notification sent: ${title}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to send notification: ${error.message}`);
    }
  }
  
  async getStorageInfo(storagePath = '/storage/emulated/0') {
    try {
      const { stdout } = await execAsync(`df -h ${storagePath}`);
      const lines = stdout.trim().split('\n');
      const data = lines[1].split(/\s+/);
      
      return {
        content: [
          {
            type: 'text',
            text: `Storage Information for ${storagePath}:
- Total: ${data[1]}
- Used: ${data[2]}
- Available: ${data[3]}
- Usage: ${data[4]}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to get storage info: ${error.message}`);
    }
  }
  
  async textToSpeech(args) {
    const { text, language, rate } = args;
    
    let command = `termux-tts-speak "${text}"`;
    
    if (language) {
      command += ` -l ${language}`;
    }
    
    if (rate) {
      command += ` -r ${rate}`;
    }
    
    try {
      await execAsync(command);
      return {
        content: [
          {
            type: 'text',
            text: `Speaking: "${text}" in ${language || 'default language'}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to speak text: ${error.message}`);
    }
  }
  
  async manageClipboard(args) {
    const { action, text } = args;
    
    try {
      if (action === 'get') {
        const { stdout } = await execAsync('termux-clipboard-get');
        return {
          content: [
            {
              type: 'text',
              text: `Clipboard content: ${stdout}`,
            },
          ],
        };
      } else if (action === 'set') {
        if (!text) {
          throw new Error('Text is required for set action');
        }
        await execAsync(`termux-clipboard-set "${text}"`);
        return {
          content: [
            {
              type: 'text',
              text: `Clipboard set to: ${text}`,
            },
          ],
        };
      }
    } catch (error) {
      throw new Error(`Clipboard operation failed: ${error.message}`);
    }
  }
  
  setupResources() {
    // Resource discovery
    this.server.setRequestHandler('resources/list', async () => ({
      resources: [
        {
          uri: 'termux://system/info',
          name: 'System Information',
          description: 'Termux and Android system information',
          mimeType: 'application/json',
        },
        {
          uri: 'termux://contacts/list',
          name: 'Contact List',
          description: 'Android contacts (requires permission)',
          mimeType: 'application/json',
        },
      ],
    }));
    
    // Resource reading
    this.server.setRequestHandler('resources/read', async (request) => {
      const { uri } = request.params;
      
      if (uri === 'termux://system/info') {
        return await this.getSystemInfo();
      } else if (uri === 'termux://contacts/list') {
        return await this.getContactList();
      }
      
      throw new Error(`Unknown resource: ${uri}`);
    });
  }
  
  async getSystemInfo() {
    try {
      const [deviceInfo, termuxInfo] = await Promise.all([
        execAsync('termux-info'),
        execAsync('uname -a'),
      ]);
      
      return {
        contents: [
          {
            uri: 'termux://system/info',
            mimeType: 'application/json',
            text: JSON.stringify({
              device: deviceInfo.stdout,
              system: termuxInfo.stdout,
              termuxHome: process.env.HOME,
              prefix: process.env.PREFIX,
            }, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to get system info: ${error.message}`);
    }
  }
  
  async getContactList() {
    try {
      const { stdout } = await execAsync('termux-contact-list');
      const contacts = JSON.parse(stdout);
      
      return {
        contents: [
          {
            uri: 'termux://contacts/list',
            mimeType: 'application/json',
            text: JSON.stringify(contacts, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to get contacts: ${error.message}`);
    }
  }
  
  async start() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Termux MCP Server started');
  }
}

// Start the server
const server = new TermuxMCPServer();
server.start().catch(console.error);
```

### Configuring MCP Servers in Extensions

The Gemini CLI uses the mcpServers configuration in your settings.json file to locate and connect to MCP servers. This configuration supports multiple servers with different transport mechanisms. You can configure MCP servers at the global level in the ~/.gemini/settings.json file or in your project's root directory, create or open the .gemini/settings.json file. Within the file, add the mcpServers configuration block. Add an mcpServers object to your settings.json file:

```json
{
  "mcpServers": {
    "termux-advanced": {
      "command": "node",
      "args": ["./mcp-servers/termux-advanced.js"],
      "env": {
        "TERMUX_HOME": "/data/data/com.termux/files/home",
        "ANDROID_DATA": "/storage/emulated/0",
        "NODE_ENV": "production"
      },
      "cwd": "~/.gemini/extensions/termux-suite",
      "timeout": 30000,
      "trust": false,
      "includeTools": [
        "battery_status",
        "termux_notification",
        "storage_info",
        "termux_tts",
        "clipboard_manager"
      ]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "~/databases/"],
      "trust": true
    }
  }
}
```

## Complex Command Hierarchies

### Namespace Architecture

Sub-directories are used to create namespaced commands, with the path separator (/ or \) being converted to a colon (:). A file at <project>/.gemini/commands/test.toml becomes the command /test. A file at <project>/.gemini/commands/git/commit.toml becomes the namespaced command /git:commit.

Create a sophisticated command structure:

```bash
# Create complex command hierarchy
mkdir -p ~/.gemini/extensions/termux-suite/commands/{android,dev,security,network,data}
```

### Android Integration Commands

```toml
# commands/android/intent.toml
description = "Create and execute Android intents"
prompt = """
Create an Android intent to {{args}}.

Provide:
1. The complete termux-open or am start command
2. Explanation of intent components:
   - Action (e.g., android.intent.action.VIEW)
   - Data URI if applicable
   - Package/Component if targeting specific app
   - Flags and extras
3. Alternative methods using termux-open
4. Security considerations

Include examples for common scenarios:
- Opening URLs
- Launching apps
- Sharing content
- Starting activities
"""

# commands/android/permissions.toml
description = "Check and request Android permissions"
prompt = """
For the operation: {{args}}

Analyze and provide:
1. Required Android permissions
2. How to check current permissions: !{termux-info}
3. Commands to request permissions
4. Fallback strategies if permissions are denied
5. Security best practices

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
1. Command to access the requested sensor
2. Data parsing strategy
3. Real-time monitoring setup
4. Battery impact considerations
5. Example processing script
"""
```

### Development Workflow Commands

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
1. Project structure with proper directories
2. Package installation commands (pkg and language-specific)
3. Configuration files (.bashrc additions, env vars)
4. Git setup with proper .gitignore
5. Editor configuration (vim/neovim)
6. Testing framework setup
7. Build scripts adapted for Termux
8. Debugging setup

Consider Termux limitations:
- No systemd (use termux-services)
- Modified FHS (PREFIX=/data/data/com.termux/files/usr)
- Limited syscalls
- Android storage restrictions
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
1. Analyze error symptoms
2. Check common Termux-specific issues:
   - Shebang paths (#!/data/data/com.termux/files/usr/bin/bash)
   - Permission problems
   - Missing dependencies
   - Path issues
3. Provide diagnostic commands
4. Suggest fixes with explanations
5. Create test cases to verify fix
"""

# commands/dev/optimize.toml
description = "Optimize code for Termux constraints"
prompt = """
Optimize this code/process for Termux: {{args}}

Current resource usage:
!{free -h}
!{df -h /data}
!{top -n 1 -b | head -20}

Optimization strategies:
1. Memory optimization (limited RAM on mobile)
2. Storage optimization (internal vs SD card)
3. Battery efficiency
4. Network usage reduction
5. Process management
6. Caching strategies
7. Background task handling

Provide optimized version with benchmarks.
"""
```

### Security Commands

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
1. File permissions check
2. Exposed credentials scan
3. Network connections review
4. Running processes analysis
5. Package vulnerabilities
6. Configuration weaknesses
7. Encryption status

Provide remediation steps for any issues found.
"""

# commands/security/encrypt.toml
description = "Implement encryption for sensitive data"
prompt = """
Set up encryption for: {{args}}

Available tools:
!{pkg list-installed | grep -E "gpg|openssl|crypt"}

Implement:
1. Choose appropriate encryption method
2. Key generation commands
3. Encryption/decryption scripts
4. Secure key storage strategy
5. Integration with Termux-keyring if applicable
6. Backup and recovery procedures

Create working examples with proper error handling.
"""
```

### Network Commands

```toml
# commands/network/tunnel.toml
description = "Set up network tunnels and proxies"
prompt = """
Create network tunnel for: {{args}}

Network status:
!{ip addr show}
!{netstat -tuln | head -20}

Configure:
1. SSH tunnel setup (if applicable)
2. VPN configuration
3. Proxy settings
4. Port forwarding rules
5. DNS configuration
6. Firewall rules (iptables if available)
7. Connection persistence
8. Auto-reconnect scripts

Handle Termux networking limitations appropriately.
"""

# commands/network/api.toml
description = "Create API client/server in Termux"
prompt = """
Build API {{args}} for Termux.

Requirements analysis based on:
- Available ports
- Network interfaces
- Security constraints

Implement:
1. Server setup (if server)
2. Client configuration (if client)
3. Authentication mechanism
4. Rate limiting
5. Error handling
6. Logging system
7. Testing endpoints
8. Documentation

Use Termux-compatible libraries and consider mobile constraints.
"""
```

## Context Management System

### Hierarchical Context Loading

Hierarchical Loading: The CLI combines GEMINI.md files from multiple locations. More specific files override general ones. The loading order is: Global Context: ~/.gemini/GEMINI.md (for instructions that apply to all your projects). Project/Ancestor Context: The CLI searches from your current directory up to the project root for GEMINI.md files. Sub-directory Context: The CLI also scans subdirectories for GEMINI.md files, allowing for component-specific instructions.

Create a comprehensive context system:

```markdown
# ~/.gemini/extensions/termux-suite/GEMINI.md

# Termux Development Context

## System Environment
You are operating in a Termux environment on Android. This is a Linux environment with significant constraints and unique characteristics.

### System Paths
- Home: /data/data/com.termux/files/home
- Prefix: /data/data/com.termux/files/usr
- Temp: /data/data/com.termux/files/usr/tmp
- Android Storage: ~/storage/ (after termux-setup-storage)
  - Shared: ~/storage/shared (maps to /storage/emulated/0)
  - Downloads: ~/storage/downloads
  - DCIM: ~/storage/dcim
  - Pictures: ~/storage/pictures
  - Music: ~/storage/music

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
1. Check command availability before use
2. Validate all inputs
3. Use trap for cleanup
4. Provide meaningful error messages
5. Log errors to: ~/.gemini/logs/

### Testing Requirements
- Test all scripts in actual Termux environment
- Check compatibility with different Android versions
- Verify termux-api calls work correctly
- Test with limited permissions
- Validate storage access

## Security Policies

### Forbidden Operations
NEVER attempt or suggest:
- Rooting device
- Modifying system files outside Termux
- Accessing other app's private data
- Running commands with `su` or `sudo`
- Disabling Android security features

### Credential Management
- Use termux-keyring when available
- Never store plaintext passwords
- Use environment variables from encrypted sources
- Implement proper session management
- Regular credential rotation

### Network Security
- Always use HTTPS when possible
- Validate SSL certificates
- Implement rate limiting
- Use SSH keys instead of passwords
- Configure fail2ban equivalents

## Performance Optimization

### Resource Constraints
Mobile devices have limited:
- RAM (typically 2-8GB, shared with Android)
- CPU (thermal throttling is common)
- Battery (optimize for power efficiency)
- Storage (internal is faster but limited)

### Optimization Strategies
1. **Memory Management**
   - Use streaming instead of loading entire files
   - Implement aggressive garbage collection
   - Monitor memory usage with `free -h`
   
2. **CPU Optimization**
   - Use nice values for background tasks
   - Implement task queuing
   - Avoid CPU-intensive operations during peak hours
   
3. **Battery Optimization**
   - Use wake locks sparingly
   - Batch network requests
   - Implement exponential backoff
   - Use Termux:Boot for scheduled tasks

## Integration Guidelines

### Termux:API Integration
When using Termux:API, always:
1. Check if termux-api package is installed
2. Verify Termux:API app is installed
3. Handle permission requests gracefully
4. Provide fallbacks for missing permissions
5. Test on different Android versions

### Android Integration
- Use intents for app interaction
- Respect Android's permission model
- Handle storage access framework properly
- Work with content providers when needed
- Implement proper broadcast receivers

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

1. **"Permission denied"**
   - Check file permissions: `ls -la`
   - Verify storage access: `termux-setup-storage`
   - Check SELinux context if applicable

2. **"Command not found"**
   - Install missing package: `pkg install <package>`
   - Check PATH: `echo $PATH`
   - Verify shebang path

3. **"No such file or directory"**
   - Check PREFIX paths
   - Verify symbolic links
   - Check case sensitivity

4. **"Cannot allocate memory"**
   - Check available memory: `free -h`
   - Kill unnecessary processes
   - Increase swap if possible

## Workflow Automation

### Task Automation Rules
1. Use Termux:Boot for startup tasks
2. Implement proper logging
3. Handle network connectivity changes
4. Respect Doze mode and battery optimization
5. Use Termux:Widget for quick actions

### CI/CD in Termux
- Use local Git hooks
- Implement testing pipelines
- Automate builds with make or npm scripts
- Deploy using rsync or scp
- Monitor with custom scripts

## Communication Protocols

### User Interaction
- Always explain Termux-specific considerations
- Provide alternative solutions for limitations
- Include installation commands for dependencies
- Warn about battery/performance impact
- Suggest optimization opportunities

### Code Generation
When generating code:
1. Include proper error handling
2. Add comprehensive comments
3. Provide usage examples
4. Include dependency checks
5. Add performance considerations

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

# Clear temporary files
find /data/data/com.termux/files/usr/tmp -type f -mtime +7 -delete

# Rotate logs
find ~/.gemini/logs -name "*.log" -mtime +30 -delete

# Check disk usage
df -h
du -sh ~/.gemini/*
```

## Advanced Features

### Custom Tool Integration
When integrating new tools:
1. Check Termux compatibility
2. Verify architecture support (arm64, etc.)
3. Test resource consumption
4. Document installation process
5. Create wrapper scripts if needed

### Extension Development
For new extensions:
1. Follow modular design
2. Implement proper error handling
3. Include comprehensive tests
4. Document all features
5. Provide migration guides
```

### Context File Imports

Create specialized context files:

```markdown
# contexts/android-integration.md

# Android Integration Context

## Available Termux:API Commands

### Device Information
- `termux-battery-status` - Battery information
- `termux-brightness` - Screen brightness control
- `termux-call-log` - Call history
- `termux-camera-info` - Camera information
- `termux-contact-list` - Access contacts
- `termux-infrared-frequencies` - IR capabilities
- `termux-location` - GPS location
- `termux-sensor` - Sensor data
- `termux-telephony-deviceinfo` - Device info
- `termux-wifi-connectioninfo` - WiFi status
- `termux-wifi-scaninfo` - WiFi networks

### System Interaction
- `termux-clipboard-get/set` - Clipboard access
- `termux-dialog` - UI dialogs
- `termux-download` - Download manager
- `termux-fingerprint` - Biometric auth
- `termux-keystore` - Android keystore
- `termux-media-player` - Media control
- `termux-media-scan` - Media scanner
- `termux-microphone-record` - Audio recording
- `termux-notification` - Notifications
- `termux-notification-remove` - Clear notifications
- `termux-open` - Open files/URLs
- `termux-open-url` - Open URLs
- `termux-share` - Share content
- `termux-sms-list` - SMS history
- `termux-sms-send` - Send SMS
- `termux-storage-get` - Storage access
- `termux-toast` - Toast messages
- `termux-torch` - Flashlight control
- `termux-tts-engines` - TTS engines
- `termux-tts-speak` - Text to speech
- `termux-usb` - USB device access
- `termux-vibrate` - Vibration control
- `termux-volume` - Volume control
- `termux-wallpaper` - Wallpaper control
- `termux-wake-lock` - Wake lock control
- `termux-wake-unlock` - Release wake lock

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
1. Always check permission before use
2. Provide graceful fallbacks
3. Explain why permission is needed
4. Don't request unnecessary permissions
5. Cache permission status

## Intent Examples

### Common Intent Patterns
```bash
# Open URL in browser
termux-open-url "https://example.com"

# Share text
echo "Hello" | termux-share -a send

# Open specific app
am start -n com.android.settings/.Settings

# Send broadcast
am broadcast -a android.intent.action.BOOT_COMPLETED

# Start service
am startservice -n com.example/.MyService
```

## Storage Access

### Storage Paths After Setup
```bash
# Run first:
termux-setup-storage

# Then access:
~/storage/shared/          # Main storage
~/storage/downloads/       # Downloads folder
~/storage/dcim/           # Camera folder
~/storage/pictures/       # Pictures
~/storage/music/          # Music
~/storage/movies/         # Videos
~/storage/external-1/     # SD card (if present)
```

### File Access Patterns
- Use `~/storage/shared/` for user-accessible files
- Keep app data in `~/.local/share/`
- Use `$PREFIX/tmp/` for temporary files
- Cache in `~/.cache/`
```

## Tool Integration & Security

### Advanced Tool Restrictions

excludeTools: ["run_shell_command"]

Create sophisticated tool restriction patterns:

```json
{
  "name": "termux-secure",
  "version": "1.0.0",
  "excludeTools": [
    "run_shell_command(rm -rf)",
    "run_shell_command(su)",
    "run_shell_command(sudo)",
    "run_shell_command(chmod 777)",
    "run_shell_command(kill -9)",
    "run_shell_command(dd if=)",
    "run_shell_command(mkfs)",
    "file_delete(/data/data/com.termux/files/home/.bashrc)",
    "file_delete(/data/data/com.termux/files/home/.profile)",
    "file_write(/system)",
    "file_write(/proc)",
    "file_write(/sys)"
  ],
  "includeTools": [
    "read_file",
    "read_many_files",
    "file_write",
    "run_shell_command",
    "google_web_search",
    "save_memory",
    "web_fetch"
  ],
  "toolRestrictions": {
    "run_shell_command": {
      "allowedCommands": [
        "ls", "pwd", "echo", "cat", "grep", "find",
        "node", "python", "git", "npm", "pip",
        "termux-*", "pkg", "apt"
      ],
      "blockedPatterns": [
        "*/etc/*",
        "*/system/*",
        "*password*",
        "*secret*",
        "*private*key*"
      ],
      "requireConfirmation": true,
      "logCommands": true
    },
    "file_write": {
      "allowedPaths": [
        "~/projects/*",
        "~/.gemini/*",
        "/data/data/com.termux/files/home/*"
      ],
      "maxFileSize": "10MB",
      "allowedExtensions": [
        ".js", ".py", ".sh", ".json", ".md",
        ".txt", ".toml", ".yaml", ".yml"
      ]
    }
  }
}
```

### Security Middleware Implementation

```javascript
// security-middleware.js
class SecurityMiddleware {
  constructor(config) {
    this.config = config;
    this.auditLog = [];
  }
  
  async validateToolCall(tool, params) {
    const restriction = this.config.toolRestrictions[tool];
    if (!restriction) return true;
    
    // Log the attempt
    this.auditLog.push({
      timestamp: new Date(),
      tool,
      params,
      user: process.env.USER
    });
    
    // Check allowed commands
    if (tool === 'run_shell_command' && restriction.allowedCommands) {
      const command = params.command.split(' ')[0];
      if (!restriction.allowedCommands.includes(command)) {
        throw new Error(`Command '${command}' is not allowed`);
      }
    }
    
    // Check blocked patterns
    if (restriction.blockedPatterns) {
      for (const pattern of restriction.blockedPatterns) {
        const regex = new RegExp(pattern.replace('*', '.*'));
        if (regex.test(JSON.stringify(params))) {
          throw new Error(`Blocked pattern detected: ${pattern}`);
        }
      }
    }
    
    // Check file paths
    if (tool === 'file_write' && restriction.allowedPaths) {
      const filePath = params.path;
      const allowed = restriction.allowedPaths.some(allowedPath => {
        const regex = new RegExp('^' + allowedPath.replace('*', '.*') + '$');
        return regex.test(filePath);
      });
      
      if (!allowed) {
        throw new Error(`Path '${filePath}' is not in allowed paths`);
      }
    }
    
    // Check file size
    if (tool === 'file_write' && restriction.maxFileSize) {
      const maxSize = this.parseSize(restriction.maxFileSize);
      const content = params.content || '';
      if (content.length > maxSize) {
        throw new Error(`File size exceeds maximum of ${restriction.maxFileSize}`);
      }
    }
    
    // Require confirmation if needed
    if (restriction.requireConfirmation) {
      return await this.requestConfirmation(tool, params);
    }
    
    return true;
  }
  
  parseSize(sizeStr) {
    const units = { KB: 1024, MB: 1024*1024, GB: 1024*1024*1024 };
    const match = sizeStr.match(/^(\d+)(KB|MB|GB)$/i);
    if (!match) return parseInt(sizeStr);
    return parseInt(match[1]) * units[match[2].toUpperCase()];
  }
  
  async requestConfirmation(tool, params) {
    // In real implementation, this would interact with user
    console.log(`Confirmation required for ${tool}:`, params);
    // Return true for auto-approval in this example
    return true;
  }
  
  getAuditLog() {
    return this.auditLog;
  }
}

module.exports = SecurityMiddleware;
```

## Part III: Production Implementation

## Building Production-Ready Extensions

### Complete Extension Package Structure

```bash
termux-ultimate-extension/
├── gemini-extension.json
├── package.json
├── README.md
├── LICENSE
├── CHANGELOG.md
├── .github/
│   └── workflows/
│       └── test.yml
├── commands/
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
├── contexts/
│   ├── GEMINI.md
│   ├── android-integration.md
│   ├── security-policies.md
│   └── performance-guidelines.md
├── mcp-servers/
│   ├── termux-advanced/
│   │   ├── index.js
│   │   ├── package.json
│   │   └── test/
│   ├── android-bridge/
│   │   ├── index.js
│   │   └── package.json
│   └── security-monitor/
│       ├── index.js
│       └── package.json
├── scripts/
│   ├── install.sh
│   ├── uninstall.sh
│   ├── update.sh
│   └── test.sh
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── docs/
    ├── installation.md
    ├── configuration.md
    ├── commands.md
    └── troubleshooting.md
```

### Production gemini-extension.json

```json
{
  "name": "termux-ultimate",
  "version": "3.0.0",
  "description": "Comprehensive Termux integration for Gemini CLI",
  "author": "Your Name",
  "license": "MIT",
  "homepage": "https://github.com/yourusername/termux-ultimate",
  "repository": {
    "type": "git",
    "url": "https://github.com/yourusername/termux-ultimate.git"
  },
  "bugs": {
    "url": "https://github.com/yourusername/termux-ultimate/issues"
  },
  "engines": {
    "node": ">=18.0.0",
    "gemini-cli": ">=1.0.0"
  },
  "mcpServers": {
    "termux-advanced": {
      "command": "node",
      "args": ["./mcp-servers/termux-advanced/index.js"],
      "env": {
        "NODE_ENV": "production",
        "LOG_LEVEL": "${LOG_LEVEL:-info}"
      },
      "timeout": 30000,
      "trust": false,
      "includeTools": [
        "battery_status",
        "termux_notification",
        "storage_info",
        "termux_tts",
        "clipboard_manager"
      ]
    },
    "android-bridge": {
      "command": "node",
      "args": ["./mcp-servers/android-bridge/index.js"],
      "timeout": 60000
    },
    "security-monitor": {
      "command": "node", 
      "args": ["./mcp-servers/security-monitor/index.js"],
      "trust": true
    }
  },
  "contextFileName": "contexts/GEMINI.md",
  "additionalContexts": [
    "contexts/android-integration.md",
    "contexts/security-policies.md",
    "contexts/performance-guidelines.md"
  ],
  "excludeTools": [
    "run_shell_command(su)",
    "run_shell_command(sudo)",
    "run_shell_command(rm -rf /)",
    "file_delete(/system)",
    "file_write(/proc)",
    "file_write(/sys)"
  ],
  "dependencies": {
    "extensions": ["base-gemini-ext"],
    "packages": [
      "@modelcontextprotocol/sdk@^0.5.0",
      "sqlite3@^5.1.6"
    ],
    "termuxPackages": [
      "nodejs-lts",
      "python",
      "git",
      "termux-api"
    ]
  },
  "hooks": {
    "onLoad": "./scripts/on-load.js",
    "onUnload": "./scripts/on-unload.js",
    "beforeCommand": "./scripts/before-command.js",
    "afterCommand": "./scripts/after-command.js"
  },
  "configuration": {
    "properties": {
      "termux-ultimate.enableAdvancedFeatures": {
        "type": "boolean",
        "default": false,
        "description": "Enable advanced experimental features"
      },
      "termux-ultimate.logLevel": {
        "type": "string",
        "enum": ["debug", "info", "warn", "error"],
        "default": "info",
        "description": "Logging level for extension"
      }
    }
  }
}
```

### Installation Script

```bash
#!/data/data/com.termux/files/usr/bin/bash
# install.sh - Production installation script

set -euo pipefail
IFS=$'\n\t'

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EXTENSION_NAME="termux-ultimate"
EXTENSION_VERSION="3.0.0"
INSTALL_DIR="$HOME/.gemini/extensions/$EXTENSION_NAME"
LOG_FILE="$HOME/.gemini/logs/install-$(date +%Y%m%d-%H%M%S).log"

# Logging functions
log() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

# Header
echo "=================================" | tee -a "$LOG_FILE"
echo "Termux Ultimate Extension Installer" | tee -a "$LOG_FILE"
echo "Version: $EXTENSION_VERSION" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        error "Node.js is not installed. Run: pkg install nodejs-lts"
    fi
    
    NODE_VERSION=$(node -v | cut -d'v' -f2)
    REQUIRED_VERSION="18.0.0"
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$NODE_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        error "Node.js version $NODE_VERSION is too old. Required: v$REQUIRED_VERSION+"
    fi
    
    # Check Gemini CLI
    if ! command -v gemini &> /dev/null; then
        error "Gemini CLI is not installed. Run: npm install -g @google/gemini-cli"
    fi
    
    # Check Termux packages
    REQUIRED_PACKAGES=("git" "python" "termux-api")
    for pkg in "${REQUIRED_PACKAGES[@]}"; do
        if ! pkg list-installed 2>/dev/null | grep -q "^$pkg/"; then
            warn "Package '$pkg' is not installed. Installing..."
            pkg install -y "$pkg" || error "Failed to install $pkg"
        fi
    done
    
    # Check Termux:API app
    if ! termux-api-start 2>/dev/null; then
        warn "Termux:API app might not be installed. Some features may not work."
    fi
    
    log "Prerequisites check completed"
}

# Backup existing installation
backup_existing() {
    if [ -d "$INSTALL_DIR" ]; then
        log "Backing up existing installation..."
        BACKUP_DIR="$INSTALL_DIR.backup.$(date +%Y%m%d-%H%M%S)"
        mv "$INSTALL_DIR" "$BACKUP_DIR"
        log "Backup created at: $BACKUP_DIR"
    fi
}

# Install extension
install_extension() {
    log "Installing extension..."
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    
    # Copy files
    cp -r ./* "$INSTALL_DIR/" 2>/dev/null || true
    
    # Install Node.


Now I'll create a comprehensive walkthrough for creating and using extensions with gemini-cli on Termux.

# Complete Guide to Creating and Using Extensions with Gemini CLI on Termux

## Table of Contents
1. [Prerequisites & Installation](#prerequisites--installation)
2. [Understanding Gemini CLI Extensions](#understanding-gemini-cli-extensions)
3. [Extension Architecture](#extension-architecture)
4. [Creating Your First Extension](#creating-your-first-extension)
5. [Advanced Extension Features](#advanced-extension-features)
6. [MCP Server Integration](#mcp-server-integration)
7. [Custom Commands with TOML](#custom-commands-with-toml)
8. [GEMINI.md Context Files](#geminimd-context-files)
9. [Best Practices & Tips](#best-practices--tips)

## Prerequisites & Installation

### Installing Gemini CLI on Termux

First, ensure you have Termux installed from F-Droid store or GitHub (not Google Play as that version is discontinued):

```bash
# Update packages
pkg update && pkg upgrade

# Install Node.js (required for Gemini CLI)
pkg install nodejs

# Verify installation
node --version  # Should be v18 or higher
npm --version
```

Install the Gemini CLI globally:

```bash
npm install -g @google/gemini-cli
```

### Authentication Setup

For Termux, you have two authentication methods:

#### Method 1: Debug Flag with Manual URL
Run Gemini CLI with the --debug flag, choose Google Login, copy the displayed authentication URL into your browser, authenticate with your Google account, and return to Termux:

```bash
gemini --debug
```

#### Method 2: Using Termux:API
For a more seamless experience, install Termux:API app from F-Droid and the termux-api package, which allows Termux to open the browser automatically:

```bash
# Install Termux:API package
pkg install termux-api

# Now Gemini CLI will open browser automatically
gemini
```

## Understanding Gemini CLI Extensions

Gemini CLI supports extensions that can be used to configure and extend its functionality. Extensions are powerful modules that can:

- Add custom MCP servers
- Provide custom commands
- Configure tool restrictions
- Add project-specific context

### Extension Locations

Gemini CLI looks for extensions in two locations and loads all extensions from both. If an extension with the same name exists in both locations, the extension in the workspace directory takes precedence:

1. **Global Extensions**: `~/.gemini/extensions/`
2. **Project Extensions**: `<workspace>/.gemini/extensions/`

## Extension Architecture

### Basic Extension Structure

Individual extensions exist as a directory that contains a gemini-extension.json file:

```
my-extension/
├── gemini-extension.json
├── GEMINI.md (optional)
└── commands/
    ├── command1.toml
    └── command2.toml
```

### The gemini-extension.json Configuration

The gemini-extension.json file contains the configuration for the extension with the following structure:

```json
{
  "name": "my-extension",
  "version": "1.0.0",
  "mcpServers": {
    "my-server": {
      "command": "node my-server.js"
    }
  },
  "contextFileName": "GEMINI.md",
  "excludeTools": ["run_shell_command"]
}
```

### Configuration Properties Explained

The name is used to uniquely identify the extension and for conflict resolution.

mcpServers: A map of MCP servers to configure. These servers will be loaded on startup just like MCP servers configured in a settings.json file. If both an extension and a settings.json file configure an MCP server with the same name, the server defined in the settings.json file takes precedence.

contextFileName: The name of the file that contains the context for the extension. This will be used to load the context from the workspace. If this property is not used but a GEMINI.md file is present in your extension directory, then that file will be loaded.

excludeTools: An array of tool names to exclude from the model. You can also specify command-specific restrictions for tools that support it, like the run_shell_command tool. For example, "excludeTools": ["run_shell_command(rm -rf)"] will block the rm -rf command.

## Creating Your First Extension

Let's create a practical extension for Termux development:

### Step 1: Create Extension Directory

```bash
# For global extension
mkdir -p ~/.gemini/extensions/termux-dev
cd ~/.gemini/extensions/termux-dev

# Or for project-specific extension
mkdir -p ./.gemini/extensions/termux-dev
cd ./.gemini/extensions/termux-dev
```

### Step 2: Create gemini-extension.json

```bash
cat > gemini-extension.json << 'EOF'
{
  "name": "termux-dev",
  "version": "1.0.0",
  "contextFileName": "TERMUX_CONTEXT.md",
  "excludeTools": [
    "run_shell_command(rm -rf /)",
    "run_shell_command(pkg uninstall termux-*)"
  ],
  "mcpServers": {}
}
EOF
```

### Step 3: Create Context File

```bash
cat > TERMUX_CONTEXT.md << 'EOF'
# Termux Development Context

## Environment
- Running on Android via Termux
- Limited to Termux-compatible packages
- No systemd, use termux-services instead
- Storage paths: ~/storage/ for Android access

## Package Management
- Use `pkg` instead of `apt`
- Install development tools: `pkg install git vim nodejs python`
- For GUI apps, use termux-x11 or VNC

## Best Practices
- Always use shebang `#!/data/data/com.termux/files/usr/bin/bash`
- Test scripts with `bash -n script.sh` before execution
- Use termux-api for Android integration
EOF
```

### Step 4: Add Custom Commands

Extensions can provide custom commands by placing TOML files in a commands/ subdirectory within the extension directory. These commands follow the same format as user and project custom commands:

```bash
mkdir commands
```

Create a package installation command:

```bash
cat > commands/pkg-install.toml << 'EOF'
description = "Install a package using Termux pkg manager"
prompt = """
Install the following package in Termux using the pkg package manager:
{{args}}

Provide the correct pkg install command and explain what the package does.
"""
EOF
```

Create a Termux API helper command:

```bash
cat > commands/termux-api.toml << 'EOF'
description = "Help with Termux API integration"
prompt = """
Provide code to use the Termux API for: {{args}}

Include:
1. Required termux-api package installation
2. Permission requirements if needed
3. Complete working example
4. Error handling
"""
EOF
```

## Advanced Extension Features

### Hierarchical Command Structure

An extension named gcp with the following structure: .gemini/extensions/gcp/ ├── gemini-extension.json └── commands/ ├── deploy.toml └── gcs/ └── sync.toml

You can organize commands in subdirectories:

```bash
# Create nested command structure
mkdir -p commands/android
mkdir -p commands/network

# Android-specific command
cat > commands/android/intent.toml << 'EOF'
description = "Create Android intent using Termux"
prompt = """
Generate a termux-open or am start command to: {{args}}
Include explanation of intent components and flags used.
"""
EOF

# Network utility command
cat > commands/network/scan.toml << 'EOF'
description = "Network scanning utilities for Termux"
prompt = """
Create a network scanning solution for: {{args}}
Use Termux-compatible tools like nmap, netcat, or curl.
Explain security implications and required permissions.
"""
EOF
```

### Dynamic Tool Exclusions

You can create context-aware tool restrictions:

```json
{
  "name": "safe-mode",
  "excludeTools": [
    "run_shell_command(rm -rf)",
    "run_shell_command(chmod 777)",
    "run_shell_command(kill -9)",
    "file_delete(/data/data/com.termux/files/home/.bashrc)"
  ]
}
```

## MCP Server Integration

An MCP server is an application that exposes tools and resources to the Gemini CLI through the Model Context Protocol, allowing it to interact with external systems and data sources. MCP servers act as a bridge between the Gemini model and your local environment or other services like APIs.

### Adding MCP Server to Extension

Create an MCP server configuration in your extension:

```json
{
  "name": "termux-mcp",
  "version": "1.0.0",
  "mcpServers": {
    "termux-helper": {
      "command": "node",
      "args": ["./mcp-servers/termux-helper.js"],
      "env": {
        "TERMUX_HOME": "/data/data/com.termux/files/home"
      }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### Creating a Simple MCP Server

Create a basic MCP server for Termux-specific operations:

```javascript
// mcp-servers/termux-helper.js
const { Server } = require('@modelcontextprotocol/sdk');

const server = new Server({
  name: 'termux-helper',
  version: '1.0.0',
  description: 'Termux-specific utilities'
});

server.addTool({
  name: 'check_battery',
  description: 'Check Android battery status',
  parameters: {},
  handler: async () => {
    const { exec } = require('child_process');
    return new Promise((resolve) => {
      exec('termux-battery-status', (error, stdout) => {
        if (error) {
          resolve({ error: error.message });
        } else {
          resolve({ battery: JSON.parse(stdout) });
        }
      });
    });
  }
});

server.addTool({
  name: 'send_notification',
  description: 'Send Android notification',
  parameters: {
    title: { type: 'string', required: true },
    content: { type: 'string', required: true }
  },
  handler: async ({ title, content }) => {
    const { exec } = require('child_process');
    return new Promise((resolve) => {
      exec(`termux-notification --title "${title}" --content "${content}"`, 
        (error) => {
          resolve({ 
            success: !error,
            error: error?.message 
          });
        }
      );
    });
  }
});

server.start();
```

## Custom Commands with TOML

Your command definition files must be written in the TOML format and use the .toml file extension. prompt (String): The prompt that will be sent to the Gemini model when the command is executed. This can be a single-line or multi-line string. description (String): A brief, one-line description of what the command does. This text will be displayed next to your command in the /help menu. If you omit this field, a generic description will be generated from the filename.

### Advanced TOML Command Features

Custom commands support two powerful methods for handling arguments. The CLI automatically chooses the correct method based on the content of your command's prompt. If your prompt contains the special placeholder {{args}}, the CLI will replace that placeholder with the text the user typed after the command name. When used in the main body of the prompt, the arguments are injected exactly as the user typed them.

Create an advanced command with argument handling:

```toml
# commands/dev.toml
description = "Setup development environment for a specific language"
prompt = """
Set up a complete Termux development environment for {{args}}.

Include:
1. Installing necessary packages via pkg
2. Setting up language-specific tools and package managers
3. Creating a sample project structure
4. Configuring editors (vim/neovim)
5. Setting up version control
6. Creating useful aliases in .bashrc

Ensure all commands are Termux-compatible and explain each step.
"""
```

## GEMINI.md Context Files

The CLI combines GEMINI.md files from multiple locations. More specific files override general ones. The loading order is: Global Context: ~/.gemini/GEMINI.md (for instructions that apply to all your projects). Project/Ancestor Context: The CLI searches from your current directory up to the project root for GEMINI.md files. Sub-directory Context: The CLI also scans subdirectories for GEMINI.md files, allowing for component-specific instructions.

### Creating Comprehensive Context

```markdown
# Termux Development Guidelines

## System Constraints
- Android Linux kernel with limited syscalls
- No root access by default
- Prefix system: /data/data/com.termux/files/usr

## Code Style
- Use POSIX-compliant shell scripts
- Prefer Python 3.9+ for automation
- Node.js 18+ for web services

## Import Additional Context
@./security-rules.md
@./api-guidelines.md

## Testing Requirements
- Test all scripts in Termux environment
- Verify termux-api calls work correctly
- Check file permissions (no chmod 777)
```

You can organize GEMINI.md files by importing other Markdown files with the @file.md syntax. This only supports .md files.

## Best Practices & Tips

### 1. Extension Development Workflow

```bash
# Create development structure
mkdir -p ~/gemini-dev/{extensions,mcp-servers,contexts}

# Symlink for easy testing
ln -s ~/gemini-dev/extensions/my-ext ~/.gemini/extensions/my-ext

# Test extension loading
gemini /extensions
```

### 2. Debugging Extensions

Use the debug flag to see extension loading:

```bash
gemini --debug
# Check console for extension loading messages
```

### 3. Version Control

Create a `.gitignore` for your extension:

```gitignore
# .gemini/extensions/my-ext/.gitignore
*.log
.env
node_modules/
tmp/
```

### 4. Extension Packaging

Create a distribution script:

```bash
#!/bin/bash
# package-extension.sh
EXTENSION_NAME="termux-dev"
VERSION="1.0.0"

tar -czf "${EXTENSION_NAME}-${VERSION}.tar.gz" \
  --exclude=node_modules \
  --exclude=.git \
  "${EXTENSION_NAME}/"

echo "Extension packaged as ${EXTENSION_NAME}-${VERSION}.tar.gz"
```

### 5. Testing Commands

Use /memory show to see the final combined context being sent to the model:

```bash
# In Gemini CLI
/memory show
/help  # See all available commands including custom ones
/termux-api:battery  # Use namespaced command
```

### 6. Performance Optimization

For Termux's limited resources:

```json
{
  "name": "termux-optimized",
  "excludeTools": [
    "web_fetch(https://very-large-file.com)"
  ],
  "settings": {
    "maxFileSize": "1MB",
    "timeout": 30000
  }
}
```

## Complete Example: Building a Full Extension

Here's a complete, production-ready extension for Termux development:

```bash
# Create the extension
mkdir -p ~/.gemini/extensions/termux-suite
cd ~/.gemini/extensions/termux-suite

# Main configuration
cat > gemini-extension.json << 'EOF'
{
  "name": "termux-suite",
  "version": "2.0.0",
  "contextFileName": "TERMUX.md",
  "excludeTools": [
    "run_shell_command(su)",
    "run_shell_command(sudo)",
    "file_delete(/system)"
  ],
  "mcpServers": {
    "android-bridge": {
      "command": "node",
      "args": ["./servers/android-bridge.js"]
    }
  }
}
EOF

# Context file
cat > TERMUX.md << 'EOF'
# Termux Complete Development Environment

## Available Tools
- Package manager: pkg (wrapper for apt)
- Python: 3.11 with pip
- Node.js: v20 with npm
- Git with GitHub CLI
- Termux:API for Android integration

## Project Standards
- All scripts must be POSIX-compliant
- Use ~/storage/shared for Android-accessible files
- Respect Android permissions model

## Security
- Never store credentials in plain text
- Use termux-keyring for sensitive data
- Validate all user inputs

@./additional-rules.md
EOF

# Create command structure
mkdir -p commands/{setup,android,dev}

# Setup command
cat > commands/setup/init.toml << 'EOF'
description = "Initialize new Termux project"
prompt = """
Create a new {{args}} project in Termux with:
1. Proper directory structure
2. Git initialization
3. README.md template
4. Basic CI/CD setup using GitHub Actions
5. Termux-specific configuration files
"""
EOF

# Android integration command
cat > commands/android/storage.toml << 'EOF'
description = "Setup Android storage access"
prompt = """
Configure Termux to access Android storage for: {{args}}
Include termux-setup-storage if needed and explain permissions.
"""
EOF

# Development helper
cat > commands/dev/debug.toml << 'EOF'
description = "Debug assistance for Termux apps"
prompt = """
Help debug this issue in Termux: {{args}}
Check for common Termux-specific problems like:
- Missing dependencies
- Permission issues
- Path problems
- API limitations
"""
EOF

echo "Extension created successfully!"
```

## Troubleshooting Common Issues

### Extension Not Loading

```bash
# Check extension discovery
gemini /extensions

# Verify JSON syntax
python -m json.tool < gemini-extension.json

# Check file permissions
ls -la ~/.gemini/extensions/
```

### Command Conflicts

When Gemini CLI starts, it loads all the extensions and merges their configurations. If there are any conflicts, the workspace configuration takes precedence.

### MCP Server Connection Issues

```bash
# Test MCP server independently
node ./mcp-servers/test-server.js

# Check environment variables
echo $GITHUB_TOKEN

# Verify network connectivity
curl -I https://api.github.com
```

## Conclusion

Gemini CLI extensions provide a powerful way to customize and extend the AI assistant's capabilities for your specific needs. On Termux, this becomes especially valuable for mobile development, allowing you to create sophisticated AI-powered tools that work within Android's constraints while leveraging the full power of Gemini's language models.

Remember to:
- Start simple and gradually add complexity
- Test extensions thoroughly in the Termux environment
- Share useful extensions with the community
- Keep security in mind when granting permissions
- Document your extensions well for future reference

With these tools and knowledge, you can transform Gemini CLI into a personalized AI assistant perfectly suited for mobile development on Termux.
## Gemini CLI on Termux: A Comprehensive Guide to Creating and Using Extensions

Harness the full potential of Google's Gemini large language models directly from your Android device with `gemini-cli` on Termux. This guide provides a detailed walkthrough on how to create and use extensions, empowering you to customize and expand the capabilities of this powerful command-line tool.

### 1. Setting the Stage: Installing Gemini CLI on Termux

Before diving into extension development, ensure you have `gemini-cli` up and running in your Termux environment.

**Prerequisites:**

*   **Termux:** Install Termux from F-Droid to ensure you have the latest version.
*   **Node.js:** `gemini-cli` is a Node.js application.

**Installation Steps:**

1.  **Update and upgrade Termux packages:**
    ```bash
    pkg update && pkg upgrade
    ```

2.  **Install Node.js:**
    ```bash
    pkg install nodejs
    ```

3.  **Install `gemini-cli`:** The recommended way to install is globally using `npm` (Node Package Manager), which comes with Node.js.
    ```bash
    npm install -g @google/gemini-cli
    ```

4.  **Authenticate with your Google Account:** Run the `gemini` command for the first time to log in.
    ```bash
    gemini
    ```
    This will prompt you to log in with your Google account, which provides a generous free tier of 60 requests per minute and 1,000 requests per day.

### 2. Extending Gemini CLI: An Overview

There are two primary methods for extending the functionality of `gemini-cli`:

*   **Custom Tools via Model Context Protocol (MCP):** This is the more powerful method, allowing you to define custom functions (tools) in a separate server that `gemini-cli` can connect to and utilize. This is ideal for integrating with external APIs or executing complex logic.
*   **Custom Slash Commands:** For simpler, prompt-based extensions, you can create custom slash commands. These are essentially reusable prompt templates that can accept arguments, making it easy to automate repetitive tasks.

### 3. Method 1: Building Custom Tools with MCP and Python

The Model Context Protocol (MCP) allows `gemini-cli` to discover and use tools exposed by an MCP server. We will create a simple MCP server in Python using the `fastmcp` library.

**Prerequisites for MCP Development:**

*   **Python:** Install Python in Termux.
    ```bash
    pkg install python
    ```
*   **fastmcp:** Install the `fastmcp` library using `pip`.
    ```bash
    pip install fastmcp
    ```

**Step-by-Step Guide to Creating a Custom Tool:**

**Step 1: Create the MCP Server File**

1.  Create a new directory for your MCP server and navigate into it:
    ```bash
    mkdir ~/gemini-extensions/my-mcp-server
    cd ~/gemini-extensions/my-mcp-server
    ```

2.  Create a Python file for your server, for example, `server.py`:
    ```bash
    touch server.py
    ```

3.  Open `server.py` in a text editor (like `nano`) and add the following code. This example creates a simple tool that greets a user.

    ```python
    from fastmcp import FastMCP

    # Initialize the MCP server
    mcp = FastMCP(name="MyTermuxTools")

    @mcp.tool
    def greet(name: str) -> str:
        """Greets the specified person."""
        return f"Hello, {name}! Greetings from your custom Termux tool."

    if __name__ == "__main__":
        mcp.run()
    ```

**Step 2: Run the MCP Server**

Keep this terminal window open and run your MCP server:

```bash
python server.py
```

Your custom tool is now running and waiting for connections from `gemini-cli`.

**Step 3: Configure `gemini-cli` to Use the MCP Server**

1.  Open a new Termux session.
2.  You need to edit the `gemini-cli` settings file. Open `~/.gemini/settings.json` in a text editor. If it doesn't exist, you may need to run `gemini` once to generate it.
3.  Add the `mcpServers` configuration to the JSON file. This tells `gemini-cli` how to connect to your running server.

    ```json
    {
      "mcpServers": {
        "my-termux-tools": {
          "command": "python",
          "args": ["/data/data/com.termux/files/home/gemini-extensions/my-mcp-server/server.py"],
          "transport": "stdio"
        }
      }
    }
    ```
    **Note:** You must use the full, absolute path to your `server.py` file within the Termux environment.

**Step 4: Use Your Custom Tool**

Now, you can start `gemini-cli` and use your new tool in a prompt:

```bash
gemini
```

Once in the `gemini` prompt, you can ask it to use your tool:

> `> Greet the world with my custom tool`

Gemini will recognize that the `greet` tool can fulfill this request, ask for your permission to use it, and then execute it, displaying the output from your Python script.

You can also verify that your MCP server is connected by running the `/mcp` command within the `gemini-cli` prompt.

### 4. Method 2: Creating Custom Slash Commands

For simpler, prompt-based extensions, custom slash commands are an excellent choice. They are defined in `.toml` files.

**Step-by-Step Guide to Creating a Slash Command:**

**Step 1: Create the Command Directory**

`gemini-cli` looks for global custom commands in the `~/.gemini/commands/` directory. Create it if it doesn't exist:

```bash
mkdir -p ~/.gemini/commands
```

**Step 2: Create the Command File**

Create a `.toml` file within the `commands` directory. The name of the file will be the name of your command. For example, to create a `/hello` command, create `hello.toml`:

```bash
touch ~/.gemini/commands/hello.toml
```

**Step 3: Define the Command**

Open `hello.toml` and add the following content. This command will take a name as an argument and insert it into the prompt.

```toml
description = "A friendly greeting command."
prompt = "Say hello to {{args}} in a cheerful way."
```

The `{{args}}` placeholder will be replaced by any text that follows the slash command.

**Step 4: Use the Slash Command**

Start `gemini-cli` and invoke your new command:

```bash
gemini
```

At the prompt, type:

> `> /hello Termux`

Gemini will then execute the prompt defined in your `.toml` file, resulting in a cheerful greeting to "Termux".

### 5. Bundled Extensions

For more organized and shareable extensions, you can create a "bundled extension". This involves creating a directory that contains a `gemini-extension.json` file, which can define MCP servers and other context for the extension.

**Example Structure:**

```
~/.gemini/extensions/my-cool-extension/
├── gemini-extension.json
└── my-mcp-server.py
```

The `gemini-extension.json` file might look like this:

```json
{
  "name": "my-cool-extension",
  "version": "1.0.0",
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["my-mcp-server.py"]
    }
  }
}
```

By following these methods, you can significantly enhance the capabilities of `gemini-cli` on your Android device, tailoring it to your specific workflows and needs.
Of course. Let's go deeper into extending `gemini-cli` on Termux, transforming it from a simple command-line assistant into a powerful, customized automation platform. This in-depth guide will cover advanced techniques, practical examples, and the underlying concepts that make these extensions work.

### Core Concepts: How Extensions Work

Before building, it's crucial to understand the "why." `gemini-cli` is designed to be an open and extensible framework. It achieves this through two primary mechanisms:

*   **Model Context Protocol (MCP):** This is the industry standard for allowing Large Language Models (LLMs) to interact with external services and tools. Think of an MCP server as a specialized API that your `gemini-cli` can connect to. When you ask Gemini a question, it not only considers its internal knowledge but also looks at the list of tools available from all connected MCP servers. If it determines a tool can help answer your prompt, it will ask for permission to use it. This is the most powerful way to extend Gemini, enabling it to interact with databases, external APIs, or even your local device hardware.
*   **Configuration Files:** `gemini-cli` uses a hierarchy of configuration files to manage its behavior. This includes `settings.json` for core settings, `.toml` files for custom commands, and `GEMINI.md` for providing project-specific context. This file-based system makes extensions portable and easy to manage.

---

### Method 1: Advanced Custom Tools with MCP

This method offers the highest degree of customization. We'll build a practical tool that leverages the `termux-api` to give Gemini control over your Android device's functions.

#### **Prerequisites: Setting up Termux API**

The `termux-api` allows scripts to access Android features like battery status, clipboard, notifications, and more.

1.  **Install the `termux-api` app:** Get it from F-Droid.
2.  **Install the `termux-api` package in Termux:**
    ```bash
    pkg install termux-api
    ```
3.  **Install a Python wrapper for the API:** This simplifies using the API in our Python script. There are several options, but `termux-api` is a good one.
    ```bash
    pip install termux-api
    ```

#### **Step 1: Crafting a More Powerful MCP Server**

Let's create an MCP server with tools to check the battery status and read the device's clipboard.

1.  **Create your project directory:**
    ```bash
    mkdir -p ~/gemini-extensions/termux-tools
    cd ~/gemini-extensions/termux-tools
    ```

2.  **Create the server file (`server.py`):**
    ```python
    import termux_api
    from fastmcp import FastMCP

    # Initialize the MCP server with a descriptive name
    mcp = FastMCP(name="TermuxDeviceTools")

    @mcp.tool
    def get_battery_status() -> dict:
        """
        Retrieves the current battery status of the Android device.
        Returns a dictionary with details like percentage, temperature, and health.
        """
        try:
            # The termux_api.battery() function returns a tuple (result, error)
            status, err = termux_api.battery()
            if err:
                return {"error": str(err)}
            return status
        except Exception as e:
            return {"error": f"Failed to execute termux-api: {e}"}

    @mcp.tool
    def get_clipboard_content() -> str:
        """
        Fetches the current text content from the Android device's clipboard.
        """
        try:
            content, err = termux_api.clipboard_get()
            if err:
                return f"Error: {err}"
            return content
        except Exception as e:
            return f"Error: Failed to execute termux-api: {e}"

    if __name__ == "__main__":
        # This makes the script runnable
        mcp.run()
    ```

    **Deep Dive into the Code:**
    *   `from fastmcp import FastMCP`: We import the core class from the `fastmcp` library, which simplifies creating MCP servers.
    *   `@mcp.tool`: This decorator is the magic that registers the Python function as a "tool" that Gemini can use. `fastmcp` automatically inspects the function's name, its docstring, and its type hints (`-> dict`, `-> str`) to create a schema that tells the LLM what the tool does, what arguments it takes, and what it returns. A well-written docstring is critical for the model to understand when and how to use your tool.
    *   `termux_api.battery()`: This function from the Python wrapper calls the underlying `termux-api` command and returns the result.
    *   **Error Handling:** Wrapping the API calls in `try...except` blocks is crucial for robustness. If the `termux-api` command fails, the tool will return a descriptive error message to the model instead of crashing the server.

#### **Step 2: Configuring `settings.json`**

The `~/.gemini/settings.json` file is where you tell `gemini-cli` how to find and run your MCP server.

1.  **Open or create the settings file:**
    ```bash
    nano ~/.gemini/settings.json
    ```

2.  **Add the `mcpServers` configuration.** Be sure to use the full, absolute path to your server script.
    ```json
    {
      "mcpServers": {
        "termux-tools": {
          "command": "python",
          "args": ["/data/data/com.termux/files/home/gemini-extensions/termux-tools/server.py"],
          "transport": "stdio"
        }
      }
    }
    ```
    **Configuration Explained:**
    *   `"termux-tools"`: This is a unique name you give to your server configuration.
    *   `"command"`: The executable to run. In this case, `python`.
    *   `"args"`: A list of arguments to pass to the command. The first and only argument here is the path to our script.
    *   `"transport": "stdio"`: This tells `gemini-cli` to communicate with the server process using standard input/output streams. It's the simplest method for local servers.

#### **Step 3: Interacting with Your Custom Tools**

1.  **Start `gemini-cli`:**
    ```bash
    gemini
    ```
2.  **Verify the connection:** Use the built-in `/mcp` command to see if your server is connected.
    ```
    > /mcp
    ```
    You should see `termux-tools` in the list of connected servers.

3.  **Use your tools with natural language:**
    *   `> What is the battery percentage on this phone?`
    *   `> Read the contents of my clipboard`

Gemini will now analyze your prompt, match it to the descriptions of your custom tools, and execute them to get the answer.

---

### Method 2: Advanced Slash Commands

Custom slash commands are reusable, parameterizable prompts defined in `.toml` files. They are perfect for automating repetitive prompting tasks.

#### **Step 1: Understanding the `.toml` Structure**

Let's create a more advanced command. The name of the command is determined by its file path. A file at `~/.gemini/commands/git/commit.toml` becomes the namespaced command `/git:commit`.

#### **Step 2: Creating a Multi-Argument Command**

Let's create a command `/writer:email` that drafts an email.

1.  **Create the directory and file:**
    ```bash
    mkdir -p ~/.gemini/commands/writer
    nano ~/.gemini/commands/writer/email.toml
    ```

2.  **Define the command in `email.toml`:**
    ```toml
    description = "Drafts a professional email. Usage: /writer:email <to> <subject> <main_point>"
    prompt = """
    You are a professional communication assistant.
    Draft a clear and concise email based on the following details.

    To: {{args[0]}}
    Subject: {{args[1]}}

    Body:
    Compose a professional email that clearly communicates the following main point: {{args[2..]}}.
    Keep the tone appropriate for a professional setting.
    """
    ```
    **Advanced Syntax:**
    *   `description`: A helpful message that appears in the autocomplete menu.
    *   `prompt`: The multi-line prompt template.
    *   `{{args[0]}}`: Accesses the first argument after the command.
    *   `{{args[1]}}`: Accesses the second argument.
    *   `{{args[2..]}}`: Accesses the third argument and all subsequent arguments, joining them together. This is useful for the main body of the email.

#### **Step 3: Using the Advanced Slash Command**

Now, in `gemini-cli`, you can draft an email with a single line:

> `> /writer:email "team@example.com" "Project Update" The new feature is ready for testing and has been deployed to the staging server.`

This command will be expanded using the template, sending a perfectly formatted request to the model and saving you significant time.

By mastering these advanced extension methods, you can tailor `gemini-cli` on Termux to your exact needs, creating a powerful and personalized AI assistant that lives right in your pocket.
# Complete In-Depth Guide to Creating and Using Extensions with Gemini CLI on Termux

## Table of Contents

### Part I: Foundation & Architecture
1. [Deep Dive into Gemini CLI Architecture](#deep-dive-into-gemini-cli-architecture)
2. [Complete Termux Setup & Optimization](#complete-termux-setup--optimization)
3. [Authentication Deep Dive](#authentication-deep-dive)
4. [Extension System Internals](#extension-system-internals)

### Part II: Advanced Extension Development
5. [MCP Server Architecture & Implementation](#mcp-server-architecture--implementation)
6. [Complex Command Hierarchies](#complex-command-hierarchies)
7. [Context Management System](#context-management-system)
8. [Tool Integration & Security](#tool-integration--security)

### Part III: Production Implementation
9. [Building Production-Ready Extensions](#building-production-ready-extensions)
10. [Performance Optimization Strategies](#performance-optimization-strategies)
11. [Debugging & Troubleshooting](#debugging--troubleshooting)
12. [Real-World Case Studies](#real-world-case-studies)

---

## Part I: Foundation & Architecture

## Deep Dive into Gemini CLI Architecture

### Core Components

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

### ReAct Loop Implementation

The Gemini command line interface (CLI) is an open source AI agent that provides access to Gemini directly in your terminal. The Gemini CLI uses a reason and act (ReAct) loop with your built-in tools and local or remote MCP servers to complete complex use cases like fixing bugs, creating new

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

### Token Management & Context Window

That free license gets you access to Gemini 2.5 Pro and its massive 1 million token context window.

The 1M token context window requires sophisticated management:

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

## Complete Termux Setup & Optimization

### Advanced Installation Process

The installation process is identical to Linux. You'll need Termux or a similar terminal emulator. I prefer Termux, make sure to download it from F-Droid store or GitHub. Version on Google Play is discontinued.

#### Step 1: Termux Environment Preparation

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

# Set up storage access
termux-setup-storage

# Configure Node.js environment
npm config set prefix ~/.npm-global
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify Node.js version (should be 18+)
node --version
npm --version
```

#### Step 2: Gemini CLI Installation with Error Handling

```bash
# Install Gemini CLI with retry logic
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

install_gemini_cli

# Verify installation
gemini --version
```

### Authentication Deep Dive

This is the simplest method. Run Gemini CLI with the --debug flag: ... Choose Google Login. The CLI will display a login URL in the terminal. Copy this link into your browser, authenticate with your Google account, and return to Termux. Gemini CLI will now be successfully authenticated.

#### Method 1: Debug Authentication (Manual)

```bash
#!/bin/bash
# auth-debug.sh - Debug authentication helper

echo "Starting Gemini CLI authentication in debug mode..."
echo "="*50

# Run in debug mode and capture output
gemini --debug 2>&1 | tee auth.log &
GEMINI_PID=$!

# Wait for authentication URL
echo "Waiting for authentication URL..."
while ! grep -q "https://accounts.google.com" auth.log 2>/dev/null; do
  sleep 1
done

# Extract and display URL
AUTH_URL=$(grep -o "https://accounts.google.com[^\"]*" auth.log | head -1)
echo ""
echo "Authentication URL found!"
echo "="*50
echo "$AUTH_URL"
echo "="*50
echo ""
echo "1. Copy the above URL"
echo "2. Open it in your browser"
echo "3. Complete authentication"
echo "4. Return here and press Enter"
read -r

# Clean up
rm -f auth.log
```

#### Method 2: Termux:API Integration (Automated)

For a more seamless experience, you can use Termux:API to open the login URL directly in your browser. Termux-api allows you to send commands to the Android system and send command to open a browser. This means that Termux would trigger Google authentication automatically by opening a browser, just like it would behave on desktop system. For that you need to install: Install Termux:API app from F-Droid. This will open the browser automatically, mimicking desktop behavior. Once authenticated, return to Termux and continue using Gemini CLI.

```bash
#!/bin/bash
# auth-api.sh - Automated authentication with Termux:API

# Check if Termux:API is installed
if ! command -v termux-open-url &> /dev/null; then
  echo "Installing termux-api package..."
  pkg install termux-api -y
fi

# Create authentication wrapper
cat > ~/gemini-auth-wrapper.sh << 'EOF'
#!/bin/bash

# Intercept authentication URL and open automatically
gemini "$@" 2>&1 | while IFS= read -r line; do
  echo "$line"
  
  # Check for Google auth URL
  if [[ "$line" =~ https://accounts.google.com ]]; then
    URL=$(echo "$line" | grep -o 'https://[^"]*' | head -1)
    if [ -n "$URL" ]; then
      echo "Opening authentication URL in browser..."
      termux-open-url "$URL"
      
      # Vibrate to notify user
      termux-vibrate -d 500
      
      # Show notification
      termux-notification \
        --title "Gemini CLI Authentication" \
        --content "Please complete authentication in browser" \
        --action "termux-open-url $URL"
    fi
  fi
done
EOF

chmod +x ~/gemini-auth-wrapper.sh

# Create alias for easy use
echo 'alias gemini-auth="~/gemini-auth-wrapper.sh"' >> ~/.bashrc
source ~/.bashrc

echo "✓ Automated authentication setup complete"
echo "Run 'gemini-auth' to start Gemini CLI with automatic browser opening"
```

#### Method 3: API Key Authentication

Get Your Key: Get an API key from Google AI Studio. Set Your Key: Make the key available to the CLI with one of these methods. Method 1: Shell Environment Variable Set the GEMINI_API_KEY environment variable. To use it across terminal sessions, add this line to your shell's profile (e.g., ~/.bashrc, ~/.zshrc).

```bash
#!/bin/bash
# setup-api-key.sh - Secure API key setup

# Secure API key storage with encryption
setup_api_key() {
  echo "Setting up Gemini API key..."
  
  # Create secure directory
  mkdir -p ~/.gemini/secure
  chmod 700 ~/.gemini/secure
  
  # Prompt for API key
  echo -n "Enter your Gemini API key: "
  read -rs API_KEY
  echo
  
  # Validate key format
  if [[ ! "$API_KEY" =~ ^[A-Za-z0-9_-]{39}$ ]]; then
    echo "Invalid API key format"
    return 1
  fi
  
  # Store encrypted (using simple base64 for Termux compatibility)
  echo "$API_KEY" | base64 > ~/.gemini/secure/api_key.enc
  chmod 600 ~/.gemini/secure/api_key.enc
  
  # Create loader script
  cat > ~/.gemini/load_api_key.sh << 'EOF'
#!/bin/bash
if [ -f ~/.gemini/secure/api_key.enc ]; then
  export GEMINI_API_KEY=$(base64 -d < ~/.gemini/secure/api_key.enc)
fi
EOF
  
  chmod +x ~/.gemini/load_api_key.sh
  
  # Add to bashrc
  echo 'source ~/.gemini/load_api_key.sh' >> ~/.bashrc
  
  echo "✓ API key configured successfully"
  echo "Run 'source ~/.bashrc' to load the key"
}

setup_api_key
```

## Extension System Internals

### Extension Discovery and Loading Process

name: The name of the extension. This is used to uniquely identify the extension and for conflict resolution when extension commands have the same name as user or project commands. ... mcpServers: A map of MCP servers to configure. The key is the name of the server, and the value is the server configuration. These servers will be loaded on startup just like MCP servers configured in a settings.json file. If both an extension and a settings.json file configure an MCP server with the same name, the server defined in the settings.json file takes precedence.

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
      path.join(os.homedir(), '.gemini', 'extensions'),  // Global
      path.join(process.cwd(), '.gemini', 'extensions')   // Project
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
            
            // Check for conflicts
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
    // Project extensions take precedence over global
    for (const conflict of this.conflicts) {
      const projectPath = path.join(process.cwd(), '.gemini', 'extensions');
      
      if (conflict.new.startsWith(projectPath)) {
        // Keep project extension
        console.log(`Extension conflict resolved: ${conflict.name} (using project version)`);
      } else {
        // Revert to existing
        const existing = this.extensions.get(conflict.name);
        existing.path = conflict.existing;
      }
    }
  }
}
```

### Extension Configuration Schema

```typescript
interface ExtensionConfig {
  name: string;
  version: string;
  description?: string;
  author?: string;
  license?: string;
  
  // MCP Server configuration
  mcpServers?: {
    [serverName: string]: {
      command: string;
      args?: string[];
      env?: Record<string, string>;
      cwd?: string;
      timeout?: number;
      trust?: boolean;
      includeTools?: string[];
      excludeTools?: string[];
    };
  };
  
  // Context configuration
  contextFileName?: string;
  additionalContexts?: string[];
  
  // Tool restrictions
  excludeTools?: string[];
  includeTools?: string[];
  
  // Dependencies
  dependencies?: {
    extensions?: string[];
    packages?: string[];
    termuxPackages?: string[];
  };
  
  // Hooks
  hooks?: {
    onLoad?: string;
    onUnload?: string;
    beforeCommand?: string;
    afterCommand?: string;
  };
}
```

## Part II: Advanced Extension Development

## MCP Server Architecture & Implementation

### Understanding MCP Protocol

An MCP server is an application that exposes tools and resources to the Gemini CLI through the Model Context Protocol, allowing it to interact with external systems and data sources. MCP servers act as a bridge between the Gemini model and your local environment or other services like APIs.

```javascript
// Complete MCP Server implementation for Termux
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
        name: 'termux-advanced',
        version: '2.0.0',
      },
      {
        capabilities: {
          tools: {},
          resources: {},
        },
      }
    );
    
    this.setupTools();
    this.setupResources();
  }
  
  setupTools() {
    // Battery status tool
    this.server.setRequestHandler('tools/list', async () => ({
      tools: [
        {
          name: 'battery_status',
          description: 'Get Android battery status via Termux',
          inputSchema: {
            type: 'object',
            properties: {},
          },
        },
        {
          name: 'termux_notification',
          description: 'Send Android notification',
          inputSchema: {
            type: 'object',
            properties: {
              title: {
                type: 'string',
                description: 'Notification title',
              },
              content: {
                type: 'string',
                description: 'Notification content',
              },
              priority: {
                type: 'string',
                enum: ['min', 'low', 'default', 'high', 'max'],
                default: 'default',
              },
              vibrate: {
                type: 'boolean',
                default: false,
              },
            },
            required: ['title', 'content'],
          },
        },
        {
          name: 'storage_info',
          description: 'Get Android storage information',
          inputSchema: {
            type: 'object',
            properties: {
              path: {
                type: 'string',
                description: 'Storage path to check',
                default: '/storage/emulated/0',
              },
            },
          },
        },
        {
          name: 'termux_tts',
          description: 'Text-to-speech using Android TTS',
          inputSchema: {
            type: 'object',
            properties: {
              text: {
                type: 'string',
                description: 'Text to speak',
              },
              language: {
                type: 'string',
                description: 'Language code (e.g., en-US)',
                default: 'en-US',
              },
              rate: {
                type: 'number',
                description: 'Speech rate (0.5 - 2.0)',
                minimum: 0.5,
                maximum: 2.0,
                default: 1.0,
              },
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
              action: {
                type: 'string',
                enum: ['get', 'set'],
                description: 'Clipboard action',
              },
              text: {
                type: 'string',
                description: 'Text to set (required for set action)',
              },
            },
            required: ['action'],
          },
        },
      ],
    }));
    
    // Tool execution handler
    this.server.setRequestHandler('tools/call', async (request) => {
      const { name, arguments: args } = request.params;
      
      try {
        switch (name) {
          case 'battery_status':
            return await this.getBatteryStatus();
            
          case 'termux_notification':
            return await this.sendNotification(args);
            
          case 'storage_info':
            return await this.getStorageInfo(args.path);
            
          case 'termux_tts':
            return await this.textToSpeech(args);
            
          case 'clipboard_manager':
            return await this.manageClipboard(args);
            
          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `Error executing ${name}: ${error.message}`,
            },
          ],
        };
      }
    });
  }
  
  async getBatteryStatus() {
    try {
      const { stdout } = await execAsync('termux-battery-status');
      const battery = JSON.parse(stdout);
      
      return {
        content: [
          {
            type: 'text',
            text: `Battery Status:
- Level: ${battery.percentage}%
- Status: ${battery.status}
- Health: ${battery.health}
- Temperature: ${battery.temperature}°C
- Plugged: ${battery.plugged}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to get battery status: ${error.message}`);
    }
  }
  
  async sendNotification(args) {
    const { title, content, priority, vibrate } = args;
    
    let command = `termux-notification --title "${title}" --content "${content}"`;
    
    if (priority && priority !== 'default') {
      command += ` --priority ${priority}`;
    }
    
    if (vibrate) {
      command += ' --vibrate 200,100,200';
    }
    
    try {
      await execAsync(command);
      return {
        content: [
          {
            type: 'text',
            text: `Notification sent: ${title}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to send notification: ${error.message}`);
    }
  }
  
  async getStorageInfo(storagePath = '/storage/emulated/0') {
    try {
      const { stdout } = await execAsync(`df -h ${storagePath}`);
      const lines = stdout.trim().split('\n');
      const data = lines[1].split(/\s+/);
      
      return {
        content: [
          {
            type: 'text',
            text: `Storage Information for ${storagePath}:
- Total: ${data[1]}
- Used: ${data[2]}
- Available: ${data[3]}
- Usage: ${data[4]}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to get storage info: ${error.message}`);
    }
  }
  
  async textToSpeech(args) {
    const { text, language, rate } = args;
    
    let command = `termux-tts-speak "${text}"`;
    
    if (language) {
      command += ` -l ${language}`;
    }
    
    if (rate) {
      command += ` -r ${rate}`;
    }
    
    try {
      await execAsync(command);
      return {
        content: [
          {
            type: 'text',
            text: `Speaking: "${text}" in ${language || 'default language'}`,
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to speak text: ${error.message}`);
    }
  }
  
  async manageClipboard(args) {
    const { action, text } = args;
    
    try {
      if (action === 'get') {
        const { stdout } = await execAsync('termux-clipboard-get');
        return {
          content: [
            {
              type: 'text',
              text: `Clipboard content: ${stdout}`,
            },
          ],
        };
      } else if (action === 'set') {
        if (!text) {
          throw new Error('Text is required for set action');
        }
        await execAsync(`termux-clipboard-set "${text}"`);
        return {
          content: [
            {
              type: 'text',
              text: `Clipboard set to: ${text}`,
            },
          ],
        };
      }
    } catch (error) {
      throw new Error(`Clipboard operation failed: ${error.message}`);
    }
  }
  
  setupResources() {
    // Resource discovery
    this.server.setRequestHandler('resources/list', async () => ({
      resources: [
        {
          uri: 'termux://system/info',
          name: 'System Information',
          description: 'Termux and Android system information',
          mimeType: 'application/json',
        },
        {
          uri: 'termux://contacts/list',
          name: 'Contact List',
          description: 'Android contacts (requires permission)',
          mimeType: 'application/json',
        },
      ],
    }));
    
    // Resource reading
    this.server.setRequestHandler('resources/read', async (request) => {
      const { uri } = request.params;
      
      if (uri === 'termux://system/info') {
        return await this.getSystemInfo();
      } else if (uri === 'termux://contacts/list') {
        return await this.getContactList();
      }
      
      throw new Error(`Unknown resource: ${uri}`);
    });
  }
  
  async getSystemInfo() {
    try {
      const [deviceInfo, termuxInfo] = await Promise.all([
        execAsync('termux-info'),
        execAsync('uname -a'),
      ]);
      
      return {
        contents: [
          {
            uri: 'termux://system/info',
            mimeType: 'application/json',
            text: JSON.stringify({
              device: deviceInfo.stdout,
              system: termuxInfo.stdout,
              termuxHome: process.env.HOME,
              prefix: process.env.PREFIX,
            }, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to get system info: ${error.message}`);
    }
  }
  
  async getContactList() {
    try {
      const { stdout } = await execAsync('termux-contact-list');
      const contacts = JSON.parse(stdout);
      
      return {
        contents: [
          {
            uri: 'termux://contacts/list',
            mimeType: 'application/json',
            text: JSON.stringify(contacts, null, 2),
          },
        ],
      };
    } catch (error) {
      throw new Error(`Failed to get contacts: ${error.message}`);
    }
  }
  
  async start() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Termux MCP Server started');
  }
}

// Start the server
const server = new TermuxMCPServer();
server.start().catch(console.error);
```

### Configuring MCP Servers in Extensions

The Gemini CLI uses the mcpServers configuration in your settings.json file to locate and connect to MCP servers. This configuration supports multiple servers with different transport mechanisms. You can configure MCP servers at the global level in the ~/.gemini/settings.json file or in your project's root directory, create or open the .gemini/settings.json file. Within the file, add the mcpServers configuration block. Add an mcpServers object to your settings.json file:

```json
{
  "mcpServers": {
    "termux-advanced": {
      "command": "node",
      "args": ["./mcp-servers/termux-advanced.js"],
      "env": {
        "TERMUX_HOME": "/data/data/com.termux/files/home",
        "ANDROID_DATA": "/storage/emulated/0",
        "NODE_ENV": "production"
      },
      "cwd": "~/.gemini/extensions/termux-suite",
      "timeout": 30000,
      "trust": false,
      "includeTools": [
        "battery_status",
        "termux_notification",
        "storage_info",
        "termux_tts",
        "clipboard_manager"
      ]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "~/databases/"],
      "trust": true
    }
  }
}
```

## Complex Command Hierarchies

### Namespace Architecture

Sub-directories are used to create namespaced commands, with the path separator (/ or \) being converted to a colon (:). A file at <project>/.gemini/commands/test.toml becomes the command /test. A file at <project>/.gemini/commands/git/commit.toml becomes the namespaced command /git:commit.

Create a sophisticated command structure:

```bash
# Create complex command hierarchy
mkdir -p ~/.gemini/extensions/termux-suite/commands/{android,dev,security,network,data}
```

### Android Integration Commands

```toml
# commands/android/intent.toml
description = "Create and execute Android intents"
prompt = """
Create an Android intent to {{args}}.

Provide:
1. The complete termux-open or am start command
2. Explanation of intent components:
   - Action (e.g., android.intent.action.VIEW)
   - Data URI if applicable
   - Package/Component if targeting specific app
   - Flags and extras
3. Alternative methods using termux-open
4. Security considerations

Include examples for common scenarios:
- Opening URLs
- Launching apps
- Sharing content
- Starting activities
"""

# commands/android/permissions.toml
description = "Check and request Android permissions"
prompt = """
For the operation: {{args}}

Analyze and provide:
1. Required Android permissions
2. How to check current permissions: !{termux-info}
3. Commands to request permissions
4. Fallback strategies if permissions are denied
5. Security best practices

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
1. Command to access the requested sensor
2. Data parsing strategy
3. Real-time monitoring setup
4. Battery impact considerations
5. Example processing script
"""
```

### Development Workflow Commands

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
1. Project structure with proper directories
2. Package installation commands (pkg and language-specific)
3. Configuration files (.bashrc additions, env vars)
4. Git setup with proper .gitignore
5. Editor configuration (vim/neovim)
6. Testing framework setup
7. Build scripts adapted for Termux
8. Debugging setup

Consider Termux limitations:
- No systemd (use termux-services)
- Modified FHS (PREFIX=/data/data/com.termux/files/usr)
- Limited syscalls
- Android storage restrictions
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
1. Analyze error symptoms
2. Check common Termux-specific issues:
   - Shebang paths (#!/data/data/com.termux/files/usr/bin/bash)
   - Permission problems
   - Missing dependencies
   - Path issues
3. Provide diagnostic commands
4. Suggest fixes with explanations
5. Create test cases to verify fix
"""

# commands/dev/optimize.toml
description = "Optimize code for Termux constraints"
prompt = """
Optimize this code/process for Termux: {{args}}

Current resource usage:
!{free -h}
!{df -h /data}
!{top -n 1 -b | head -20}

Optimization strategies:
1. Memory optimization (limited RAM on mobile)
2. Storage optimization (internal vs SD card)
3. Battery efficiency
4. Network usage reduction
5. Process management
6. Caching strategies
7. Background task handling

Provide optimized version with benchmarks.
"""
```

### Security Commands

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
1. File permissions check
2. Exposed credentials scan
3. Network connections review
4. Running processes analysis
5. Package vulnerabilities
6. Configuration weaknesses
7. Encryption status

Provide remediation steps for any issues found.
"""

# commands/security/encrypt.toml
description = "Implement encryption for sensitive data"
prompt = """
Set up encryption for: {{args}}

Available tools:
!{pkg list-installed | grep -E "gpg|openssl|crypt"}

Implement:
1. Choose appropriate encryption method
2. Key generation commands
3. Encryption/decryption scripts
4. Secure key storage strategy
5. Integration with Termux-keyring if applicable
6. Backup and recovery procedures

Create working examples with proper error handling.
"""
```

### Network Commands

```toml
# commands/network/tunnel.toml
description = "Set up network tunnels and proxies"
prompt = """
Create network tunnel for: {{args}}

Network status:
!{ip addr show}
!{netstat -tuln | head -20}

Configure:
1. SSH tunnel setup (if applicable)
2. VPN configuration
3. Proxy settings
4. Port forwarding rules
5. DNS configuration
6. Firewall rules (iptables if available)
7. Connection persistence
8. Auto-reconnect scripts

Handle Termux networking limitations appropriately.
"""

# commands/network/api.toml
description = "Create API client/server in Termux"
prompt = """
Build API {{args}} for Termux.

Requirements analysis based on:
- Available ports
- Network interfaces
- Security constraints

Implement:
1. Server setup (if server)
2. Client configuration (if client)
3. Authentication mechanism
4. Rate limiting
5. Error handling
6. Logging system
7. Testing endpoints
8. Documentation

Use Termux-compatible libraries and consider mobile constraints.
"""
```

## Context Management System

### Hierarchical Context Loading

Hierarchical Loading: The CLI combines GEMINI.md files from multiple locations. More specific files override general ones. The loading order is: Global Context: ~/.gemini/GEMINI.md (for instructions that apply to all your projects). Project/Ancestor Context: The CLI searches from your current directory up to the project root for GEMINI.md files. Sub-directory Context: The CLI also scans subdirectories for GEMINI.md files, allowing for component-specific instructions.

Create a comprehensive context system:

```markdown
# ~/.gemini/extensions/termux-suite/GEMINI.md

# Termux Development Context

## System Environment
You are operating in a Termux environment on Android. This is a Linux environment with significant constraints and unique characteristics.

### System Paths
- Home: /data/data/com.termux/files/home
- Prefix: /data/data/com.termux/files/usr
- Temp: /data/data/com.termux/files/usr/tmp
- Android Storage: ~/storage/ (after termux-setup-storage)
  - Shared: ~/storage/shared (maps to /storage/emulated/0)
  - Downloads: ~/storage/downloads
  - DCIM: ~/storage/dcim
  - Pictures: ~/storage/pictures
  - Music: ~/storage/music

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
1. Check command availability before use
2. Validate all inputs
3. Use trap for cleanup
4. Provide meaningful error messages
5. Log errors to: ~/.gemini/logs/

### Testing Requirements
- Test all scripts in actual Termux environment
- Check compatibility with different Android versions
- Verify termux-api calls work correctly
- Test with limited permissions
- Validate storage access

## Security Policies

### Forbidden Operations
NEVER attempt or suggest:
- Rooting device
- Modifying system files outside Termux
- Accessing other app's private data
- Running commands with `su` or `sudo`
- Disabling Android security features

### Credential Management
- Use termux-keyring when available
- Never store plaintext passwords
- Use environment variables from encrypted sources
- Implement proper session management
- Regular credential rotation

### Network Security
- Always use HTTPS when possible
- Validate SSL certificates
- Implement rate limiting
- Use SSH keys instead of passwords
- Configure fail2ban equivalents

## Performance Optimization

### Resource Constraints
Mobile devices have limited:
- RAM (typically 2-8GB, shared with Android)
- CPU (thermal throttling is common)
- Battery (optimize for power efficiency)
- Storage (internal is faster but limited)

### Optimization Strategies
1. **Memory Management**
   - Use streaming instead of loading entire files
   - Implement aggressive garbage collection
   - Monitor memory usage with `free -h`
   
2. **CPU Optimization**
   - Use nice values for background tasks
   - Implement task queuing
   - Avoid CPU-intensive operations during peak hours
   
3. **Battery Optimization**
   - Use wake locks sparingly
   - Batch network requests
   - Implement exponential backoff
   - Use Termux:Boot for scheduled tasks

## Integration Guidelines

### Termux:API Integration
When using Termux:API, always:
1. Check if termux-api package is installed
2. Verify Termux:API app is installed
3. Handle permission requests gracefully
4. Provide fallbacks for missing permissions
5. Test on different Android versions

### Android Integration
- Use intents for app interaction
- Respect Android's permission model
- Handle storage access framework properly
- Work with content providers when needed
- Implement proper broadcast receivers

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

1. **"Permission denied"**
   - Check file permissions: `ls -la`
   - Verify storage access: `termux-setup-storage`
   - Check SELinux context if applicable

2. **"Command not found"**
   - Install missing package: `pkg install <package>`
   - Check PATH: `echo $PATH`
   - Verify shebang path

3. **"No such file or directory"**
   - Check PREFIX paths
   - Verify symbolic links
   - Check case sensitivity

4. **"Cannot allocate memory"**
   - Check available memory: `free -h`
   - Kill unnecessary processes
   - Increase swap if possible

## Workflow Automation

### Task Automation Rules
1. Use Termux:Boot for startup tasks
2. Implement proper logging
3. Handle network connectivity changes
4. Respect Doze mode and battery optimization
5. Use Termux:Widget for quick actions

### CI/CD in Termux
- Use local Git hooks
- Implement testing pipelines
- Automate builds with make or npm scripts
- Deploy using rsync or scp
- Monitor with custom scripts

## Communication Protocols

### User Interaction
- Always explain Termux-specific considerations
- Provide alternative solutions for limitations
- Include installation commands for dependencies
- Warn about battery/performance impact
- Suggest optimization opportunities

### Code Generation
When generating code:
1. Include proper error handling
2. Add comprehensive comments
3. Provide usage examples
4. Include dependency checks
5. Add performance considerations

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

# Clear temporary files
find /data/data/com.termux/files/usr/tmp -type f -mtime +7 -delete

# Rotate logs
find ~/.gemini/logs -name "*.log" -mtime +30 -delete

# Check disk usage
df -h
du -sh ~/.gemini/*
```

## Advanced Features

### Custom Tool Integration
When integrating new tools:
1. Check Termux compatibility
2. Verify architecture support (arm64, etc.)
3. Test resource consumption
4. Document installation process
5. Create wrapper scripts if needed

### Extension Development
For new extensions:
1. Follow modular design
2. Implement proper error handling
3. Include comprehensive tests
4. Document all features
5. Provide migration guides
```

### Context File Imports

Create specialized context files:

```markdown
# contexts/android-integration.md

# Android Integration Context

## Available Termux:API Commands

### Device Information
- `termux-battery-status` - Battery information
- `termux-brightness` - Screen brightness control
- `termux-call-log` - Call history
- `termux-camera-info` - Camera information
- `termux-contact-list` - Access contacts
- `termux-infrared-frequencies` - IR capabilities
- `termux-location` - GPS location
- `termux-sensor` - Sensor data
- `termux-telephony-deviceinfo` - Device info
- `termux-wifi-connectioninfo` - WiFi status
- `termux-wifi-scaninfo` - WiFi networks

### System Interaction
- `termux-clipboard-get/set` - Clipboard access
- `termux-dialog` - UI dialogs
- `termux-download` - Download manager
- `termux-fingerprint` - Biometric auth
- `termux-keystore` - Android keystore
- `termux-media-player` - Media control
- `termux-media-scan` - Media scanner
- `termux-microphone-record` - Audio recording
- `termux-notification` - Notifications
- `termux-notification-remove` - Clear notifications
- `termux-open` - Open files/URLs
- `termux-open-url` - Open URLs
- `termux-share` - Share content
- `termux-sms-list` - SMS history
- `termux-sms-send` - Send SMS
- `termux-storage-get` - Storage access
- `termux-toast` - Toast messages
- `termux-torch` - Flashlight control
- `termux-tts-engines` - TTS engines
- `termux-tts-speak` - Text to speech
- `termux-usb` - USB device access
- `termux-vibrate` - Vibration control
- `termux-volume` - Volume control
- `termux-wallpaper` - Wallpaper control
- `termux-wake-lock` - Wake lock control
- `termux-wake-unlock` - Release wake lock

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
1. Always check permission before use
2. Provide graceful fallbacks
3. Explain why permission is needed
4. Don't request unnecessary permissions
5. Cache permission status

## Intent Examples

### Common Intent Patterns
```bash
# Open URL in browser
termux-open-url "https://example.com"

# Share text
echo "Hello" | termux-share -a send

# Open specific app
am start -n com.android.settings/.Settings

# Send broadcast
am broadcast -a android.intent.action.BOOT_COMPLETED

# Start service
am startservice -n com.example/.MyService
```

## Storage Access

### Storage Paths After Setup
```bash
# Run first:
termux-setup-storage

# Then access:
~/storage/shared/          # Main storage
~/storage/downloads/       # Downloads folder
~/storage/dcim/           # Camera folder
~/storage/pictures/       # Pictures
~/storage/music/          # Music
~/storage/movies/         # Videos
~/storage/external-1/     # SD card (if present)
```

### File Access Patterns
- Use `~/storage/shared/` for user-accessible files
- Keep app data in `~/.local/share/`
- Use `$PREFIX/tmp/` for temporary files
- Cache in `~/.cache/`
```

## Tool Integration & Security

### Advanced Tool Restrictions

excludeTools: ["run_shell_command"]

Create sophisticated tool restriction patterns:

```json
{
  "name": "termux-secure",
  "version": "1.0.0",
  "excludeTools": [
    "run_shell_command(rm -rf)",
    "run_shell_command(su)",
    "run_shell_command(sudo)",
    "run_shell_command(chmod 777)",
    "run_shell_command(kill -9)",
    "run_shell_command(dd if=)",
    "run_shell_command(mkfs)",
    "file_delete(/data/data/com.termux/files/home/.bashrc)",
    "file_delete(/data/data/com.termux/files/home/.profile)",
    "file_write(/system)",
    "file_write(/proc)",
    "file_write(/sys)"
  ],
  "includeTools": [
    "read_file",
    "read_many_files",
    "file_write",
    "run_shell_command",
    "google_web_search",
    "save_memory",
    "web_fetch"
  ],
  "toolRestrictions": {
    "run_shell_command": {
      "allowedCommands": [
        "ls", "pwd", "echo", "cat", "grep", "find",
        "node", "python", "git", "npm", "pip",
        "termux-*", "pkg", "apt"
      ],
      "blockedPatterns": [
        "*/etc/*",
        "*/system/*",
        "*password*",
        "*secret*",
        "*private*key*"
      ],
      "requireConfirmation": true,
      "logCommands": true
    },
    "file_write": {
      "allowedPaths": [
        "~/projects/*",
        "~/.gemini/*",
        "/data/data/com.termux/files/home/*"
      ],
      "maxFileSize": "10MB",
      "allowedExtensions": [
        ".js", ".py", ".sh", ".json", ".md",
        ".txt", ".toml", ".yaml", ".yml"
      ]
    }
  }
}
```

### Security Middleware Implementation

```javascript
// security-middleware.js
class SecurityMiddleware {
  constructor(config) {
    this.config = config;
    this.auditLog = [];
  }
  
  async validateToolCall(tool, params) {
    const restriction = this.config.toolRestrictions[tool];
    if (!restriction) return true;
    
    // Log the attempt
    this.auditLog.push({
      timestamp: new Date(),
      tool,
      params,
      user: process.env.USER
    });
    
    // Check allowed commands
    if (tool === 'run_shell_command' && restriction.allowedCommands) {
      const command = params.command.split(' ')[0];
      if (!restriction.allowedCommands.includes(command)) {
        throw new Error(`Command '${command}' is not allowed`);
      }
    }
    
    // Check blocked patterns
    if (restriction.blockedPatterns) {
      for (const pattern of restriction.blockedPatterns) {
        const regex = new RegExp(pattern.replace('*', '.*'));
        if (regex.test(JSON.stringify(params))) {
          throw new Error(`Blocked pattern detected: ${pattern}`);
        }
      }
    }
    
    // Check file paths
    if (tool === 'file_write' && restriction.allowedPaths) {
      const filePath = params.path;
      const allowed = restriction.allowedPaths.some(allowedPath => {
        const regex = new RegExp('^' + allowedPath.replace('*', '.*') + '$');
        return regex.test(filePath);
      });
      
      if (!allowed) {
        throw new Error(`Path '${filePath}' is not in allowed paths`);
      }
    }
    
    // Check file size
    if (tool === 'file_write' && restriction.maxFileSize) {
      const maxSize = this.parseSize(restriction.maxFileSize);
      const content = params.content || '';
      if (content.length > maxSize) {
        throw new Error(`File size exceeds maximum of ${restriction.maxFileSize}`);
      }
    }
    
    // Require confirmation if needed
    if (restriction.requireConfirmation) {
      return await this.requestConfirmation(tool, params);
    }
    
    return true;
  }
  
  parseSize(sizeStr) {
    const units = { KB: 1024, MB: 1024*1024, GB: 1024*1024*1024 };
    const match = sizeStr.match(/^(\d+)(KB|MB|GB)$/i);
    if (!match) return parseInt(sizeStr);
    return parseInt(match[1]) * units[match[2].toUpperCase()];
  }
  
  async requestConfirmation(tool, params) {
    // In real implementation, this would interact with user
    console.log(`Confirmation required for ${tool}:`, params);
    // Return true for auto-approval in this example
    return true;
  }
  
  getAuditLog() {
    return this.auditLog;
  }
}

module.exports = SecurityMiddleware;
```

## Part III: Production Implementation

## Building Production-Ready Extensions

### Complete Extension Package Structure

```bash
termux-ultimate-extension/
├── gemini-extension.json
├── package.json
├── README.md
├── LICENSE
├── CHANGELOG.md
├── .github/
│   └── workflows/
│       └── test.yml
├── commands/
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
├── contexts/
│   ├── GEMINI.md
│   ├── android-integration.md
│   ├── security-policies.md
│   └── performance-guidelines.md
├── mcp-servers/
│   ├── termux-advanced/
│   │   ├── index.js
│   │   ├── package.json
│   │   └── test/
│   ├── android-bridge/
│   │   ├── index.js
│   │   └── package.json
│   └── security-monitor/
│       ├── index.js
│       └── package.json
├── scripts/
│   ├── install.sh
│   ├── uninstall.sh
│   ├── update.sh
│   └── test.sh
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── docs/
    ├── installation.md
    ├── configuration.md
    ├── commands.md
    └── troubleshooting.md
```

### Production gemini-extension.json

```json
{
  "name": "termux-ultimate",
  "version": "3.0.0",
  "description": "Comprehensive Termux integration for Gemini CLI",
  "author": "Your Name",
  "license": "MIT",
  "homepage": "https://github.com/yourusername/termux-ultimate",
  "repository": {
    "type": "git",
    "url": "https://github.com/yourusername/termux-ultimate.git"
  },
  "bugs": {
    "url": "https://github.com/yourusername/termux-ultimate/issues"
  },
  "engines": {
    "node": ">=18.0.0",
    "gemini-cli": ">=1.0.0"
  },
  "mcpServers": {
    "termux-advanced": {
      "command": "node",
      "args": ["./mcp-servers/termux-advanced/index.js"],
      "env": {
        "NODE_ENV": "production",
        "LOG_LEVEL": "${LOG_LEVEL:-info}"
      },
      "timeout": 30000,
      "trust": false,
      "includeTools": [
        "battery_status",
        "termux_notification",
        "storage_info",
        "termux_tts",
        "clipboard_manager"
      ]
    },
    "android-bridge": {
      "command": "node",
      "args": ["./mcp-servers/android-bridge/index.js"],
      "timeout": 60000
    },
    "security-monitor": {
      "command": "node", 
      "args": ["./mcp-servers/security-monitor/index.js"],
      "trust": true
    }
  },
  "contextFileName": "contexts/GEMINI.md",
  "additionalContexts": [
    "contexts/android-integration.md",
    "contexts/security-policies.md",
    "contexts/performance-guidelines.md"
  ],
  "excludeTools": [
    "run_shell_command(su)",
    "run_shell_command(sudo)",
    "run_shell_command(rm -rf /)",
    "file_delete(/system)",
    "file_write(/proc)",
    "file_write(/sys)"
  ],
  "dependencies": {
    "extensions": ["base-gemini-ext"],
    "packages": [
      "@modelcontextprotocol/sdk@^0.5.0",
      "sqlite3@^5.1.6"
    ],
    "termuxPackages": [
      "nodejs-lts",
      "python",
      "git",
      "termux-api"
    ]
  },
  "hooks": {
    "onLoad": "./scripts/on-load.js",
    "onUnload": "./scripts/on-unload.js",
    "beforeCommand": "./scripts/before-command.js",
    "afterCommand": "./scripts/after-command.js"
  },
  "configuration": {
    "properties": {
      "termux-ultimate.enableAdvancedFeatures": {
        "type": "boolean",
        "default": false,
        "description": "Enable advanced experimental features"
      },
      "termux-ultimate.logLevel": {
        "type": "string",
        "enum": ["debug", "info", "warn", "error"],
        "default": "info",
        "description": "Logging level for extension"
      }
    }
  }
}
```

### Installation Script

```bash
#!/data/data/com.termux/files/usr/bin/bash
# install.sh - Production installation script

set -euo pipefail
IFS=$'\n\t'

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
EXTENSION_NAME="termux-ultimate"
EXTENSION_VERSION="3.0.0"
INSTALL_DIR="$HOME/.gemini/extensions/$EXTENSION_NAME"
LOG_FILE="$HOME/.gemini/logs/install-$(date +%Y%m%d-%H%M%S).log"

# Logging functions
log() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

# Header
echo "=================================" | tee -a "$LOG_FILE"
echo "Termux Ultimate Extension Installer" | tee -a "$LOG_FILE"
echo "Version: $EXTENSION_VERSION" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        error "Node.js is not installed. Run: pkg install nodejs-lts"
    fi
    
    NODE_VERSION=$(node -v | cut -d'v' -f2)
    REQUIRED_VERSION="18.0.0"
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$NODE_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        error "Node.js version $NODE_VERSION is too old. Required: v$REQUIRED_VERSION+"
    fi
    
    # Check Gemini CLI
    if ! command -v gemini &> /dev/null; then
        error "Gemini CLI is not installed. Run: npm install -g @google/gemini-cli"
    fi
    
    # Check Termux packages
    REQUIRED_PACKAGES=("git" "python" "termux-api")
    for pkg in "${REQUIRED_PACKAGES[@]}"; do
        if ! pkg list-installed 2>/dev/null | grep -q "^$pkg/"; then
            warn "Package '$pkg' is not installed. Installing..."
            pkg install -y "$pkg" || error "Failed to install $pkg"
        fi
    done
    
    # Check Termux:API app
    if ! termux-api-start 2>/dev/null; then
        warn "Termux:API app might not be installed. Some features may not work."
    fi
    
    log "Prerequisites check completed"
}

# Backup existing installation
backup_existing() {
    if [ -d "$INSTALL_DIR" ]; then
        log "Backing up existing installation..."
        BACKUP_DIR="$INSTALL_DIR.backup.$(date +%Y%m%d-%H%M%S)"
        mv "$INSTALL_DIR" "$BACKUP_DIR"
        log "Backup created at: $BACKUP_DIR"
    fi
}

# Install extension
install_extension() {
    log "Installing extension..."
    
    # Create installation directory
    mkdir -p "$INSTALL_DIR"
    
    # Copy files
    cp -r ./* "$INSTALL_DIR/" 2>/dev/null || true
    
    # Install Node.
