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
            text: `Battery Status:\n- Level: ${battery.percentage}%\n- Status: ${battery.status}\n- Health: ${battery.health}\n- Temperature: ${battery.temperature}°C\n- Plugged: ${battery.plugged}`,
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
            text: `Storage Information for ${storagePath}:\n- Total: ${data[1]}\n- Used: ${data[2]}\n- Available: ${data[3]}\n- Usage: ${data[4]}`,
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
