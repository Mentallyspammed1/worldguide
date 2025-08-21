#!/usr/bin/env node

// Pyrmethus v3.2.1 - Next-Generation Gemini Command Spell
// A revolutionary AI-powered terminal assistant, now with Function Calling,
// sandboxed execution, long-term vector memory, and a vastly improved UX.
// This version represents a major leap in capability and intelligence.

// --- Core Imports ---
import {
  GoogleGenerativeAI,
  GenerativeModel,
  ChatSession,
  Content,
  GenerationConfig,
  FunctionDeclarationSchemaType,
  Part
} from '@google/generative-ai';
import dotenv from 'dotenv';
import readline from 'readline/promises';
import fs from 'fs/promises';
import path from 'path';
import os from 'os';
import chalk, { Chalk } from 'chalk';
import figlet from 'figlet';
import hljs from 'highlight.js';
import ora, { Ora } from 'ora';
import { fileTypeFromBuffer } from 'file-type';
import { performance } from 'perf_hooks';
import { v4 as uuidv4 } from 'uuid';
import { applyPatch, createTwoFilesPatch } from 'diff';
import mammoth from 'mammoth';
import XLSX from 'xlsx';
import { exec as execCb, spawn } from 'child_process';
import { promisify } from 'util';
import crypto from 'crypto';
import EventEmitter from 'events';
import inquirer from 'inquirer';
import autocomplete from 'inquirer-autocomplete-prompt';
import fuzzy from 'fuzzy';
import { marked } from 'marked';
import chokidar from 'chokidar';
import NodeCache from 'node-cache';
import i18n from 'i18n';
import chalkAnimation from 'chalk-animation';
import fsSync from 'fs';
import { glob } from 'glob';
import yaml from 'js-yaml';
import PDFParser from 'pdf2json';
import pino from 'pino';
import simpleGit from 'simple-git';
import debounce from 'lodash.debounce';
import { z } from 'zod';
import zlib from 'zlib';
import axios from 'axios';
import clipboardy from 'clipboardy';
import open from 'open';
import boxen from 'boxen';
import Table from 'cli-table3';
import progress from 'cli-progress';
import notifier from 'node-notifier';

// --- Promisified Utilities ---
const exec = promisify(execCb);
const gzip = promisify(zlib.gzip);
const gunzip = promisify(zlib.gunzip);

// --- Enhanced Configuration Schema (Zod Validation) ---
const ConfigSchema = z.object({
  VERSION: z.string(),
  GOOGLE_API_KEY: z.string().optional(),
  MODEL_NAME: z.string(),
  MAX_FILE_SIZE: z.number().positive(),
  MAX_CONVERSATION_HISTORY: z.number().int(),
  STREAM_TIMEOUT: z.number().positive(),
  CACHE_TTL: z.number().int(),
  AUTO_SAVE_SESSION: z.boolean(),
  RATE_LIMITING: z.object({
    enabled: z.boolean(),
    requestsPerMinute: z.number().positive()
  }),
  AI_FEATURES: z.object({
    visionCapabilities: z.boolean(),
    autonomousMode: z.boolean(),
    contextAwareness: z.boolean(),
    functionCalling: z.boolean(), // NEW: Enable/disable function calling
    longTermMemory: z.boolean() // NEW: Enable/disable vector memory
  }),
  UI: z.object({
    MAX_BOX_WIDTH: z.number().positive(),
    SPINNER_FRAMES: z.array(z.string()),
    SPINNER_INTERVAL: z.number(),
    THEME: z.string(),
    SHOW_TIMESTAMPS: z.boolean(),
    SHOW_METRICS: z.boolean(),
    PROMPT_SYMBOL: z.string(),
    ENABLE_ANIMATIONS: z.boolean(),
    COMPACT_MODE: z.boolean(),
    ENABLE_NOTIFICATIONS: z.boolean() // NEW: Enable/disable desktop notifications
  }),
  MODEL_CONFIG: z.object({
    model: z.string(),
    generationConfig: z.object({ maxOutputTokens: z.number(), temperature: z.number(), topP: z.number(), topK: z.number() }).optional(),
    systemInstruction: z.string().optional()
  }),
  ALIASES: z.record(z.string()),
  PLUGINS: z.object({
    enabled: z.boolean(),
    directory: z.string()
  }),
  HISTORY: z.object({
    LOG_FILE: z.string(),
    MAX_LOG_SIZE: z.number().int(),
    EXPORT_DIR: z.string(),
    COMPRESSION: z.boolean()
  }),
  ENCRYPTION: z.object({
    enabled: z.boolean(),
    algorithm: z.string(),
    key: z.instanceof(Buffer).optional()
  }),
  LOGGING: z.object({
    level: z.enum(['fatal', 'error', 'warn', 'info', 'debug', 'trace', 'silent']),
    file: z.string(),
    structured: z.boolean()
  }),
  FILE_SUPPORT: z.object({
    extensions: z.array(z.string()),
    mimeTypes: z.array(z.string())
  }),
  SEARCH: z.object({
    ENGINE: z.enum(['duckduckgo', 'google']), // NEW: Specify search engine
    API_KEY: z.string().optional(),
    NUM_RESULTS: z.number().int().positive()
  }),
  GIT_INTEGRATION: z.object({
    enabled: z.boolean(),
    autoCommit: z.boolean(),
    commitPrefix: z.string()
  }),
  LANGUAGE: z.string().optional(),
  SECURITY: z.object({ // NEW: Enhanced security settings
    enableSandbox: z.boolean(),
    allowedCommands: z.array(z.string()),
    blockedCommands: z.array(z.string()),
    allowHttpRequests: z.boolean()
  }),
  MEMORY: z.object({ // NEW: Settings for long-term memory
    embeddingModel: z.string(),
    similarityThreshold: z.number().min(0).max(1),
    maxRelevantEntries: z.number().int().positive()
  })
});

// --- Directory Constants ---
const USER_HOME_DIR = os.homedir();
const PYRMETHUS_CONFIG_DIR = path.join(USER_HOME_DIR, '.pyrmethus');
const PYRMETHUS_LOG_DIR = path.join(PYRMETHUS_CONFIG_DIR, 'logs');
const PYRMETHUS_SESSION_DIR = path.join(PYRMETHUS_CONFIG_DIR, 'sessions');
const PYRMETHUS_HISTORY_EXPORT_DIR = path.join(PYRMETHUS_CONFIG_DIR, 'history_exports');
const PYRMETHUS_PLUGIN_DIR = path.join(PYRMETHUS_CONFIG_DIR, 'plugins');
const PYRMETHUS_COMMANDS_DIR = path.join(PYRMETHUS_CONFIG_DIR, 'commands');
const PYRMETHUS_CONTEXT_DIR = path.join(PYRMETHUS_CONFIG_DIR, 'context');
const PYRMETHUS_PROFILE_DIR = path.join(PYRMETHUS_CONFIG_DIR, 'profiles');
const PYRMETHUS_LOCALE_DIR = path.join(__dirname, 'locales');
const PYRMETHUS_MEMORY_DIR = path.join(PYRMETHUS_CONFIG_DIR, 'memory'); // NEW: Directory for memory vectors

const CONFIG_FILE_PATH = path.join(PYRMETHUS_CONFIG_DIR, 'config.json');

// --- UI Theme Definitions ---
const createTheme = (colors) => ({
  primary: colors.primary || chalk.ansi256(255),
  secondary: colors.secondary || chalk.ansi256(244),
  success: colors.success || chalk.green,
  warning: colors.warning || chalk.yellow,
  error: colors.error || chalk.red.bold,
  info: colors.info || chalk.blue,
  text: colors.text || chalk.white,
  dim: colors.dim || chalk.gray,
  accent: colors.accent || chalk.bold,
  prompt: colors.prompt || colors.primary || chalk.white,
  boxHeader: colors.boxHeader || colors.primary?.bold || chalk.white.bold,
  boxBorder: colors.boxBorder || colors.primary || chalk.white,
  highlight: colors.highlight || chalk.inverse,
  code: colors.code || chalk.green,
  link: colors.link || chalk.blue.underline
});

const themes = {
  neon: createTheme({ primary: chalk.cyanBright, secondary: chalk.magentaBright, success: chalk.greenBright, warning: chalk.yellowBright, info: chalk.blueBright, text: chalk.whiteBright, highlight: chalk.bgCyan.black, link: chalk.cyanBright.underline }),
  matrix: createTheme({ primary: chalk.green, secondary: chalk.rgb(0, 255, 0), info: chalk.rgb(0, 150, 0), text: chalk.rgb(0, 255, 0), dim: chalk.rgb(0, 100, 0), highlight: chalk.bgGreen.black }),
  cyberpunk: createTheme({ primary: chalk.rgb(255, 0, 255), secondary: chalk.rgb(0, 255, 255), warning: chalk.rgb(255, 255, 0), info: chalk.rgb(100, 100, 255), text: chalk.rgb(200, 200, 200), highlight: chalk.bgMagenta.black, code: chalk.cyan, link: chalk.magenta.underline }),
  minimal: createTheme({})
};

// --- Default Configuration ---
const defaultGeminiConfig = {
  VERSION: '3.2.1',
  GOOGLE_API_KEY: '',
  MODEL_NAME: 'gemini-1.5-flash-latest',
  MAX_FILE_SIZE: 20 * 1024 * 1024,
  MAX_CONVERSATION_HISTORY: 50,
  STREAM_TIMEOUT: 30000,
  CACHE_TTL: 3600,
  AUTO_SAVE_SESSION: true,
  RATE_LIMITING: { enabled: true, requestsPerMinute: 60 },
  AI_FEATURES: { visionCapabilities: true, autonomousMode: false, contextAwareness: true, functionCalling: true, longTermMemory: true },
  UI: {
    MAX_BOX_WIDTH: 100,
    SPINNER_FRAMES: ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
    SPINNER_INTERVAL: 80,
    THEME: 'neon',
    SHOW_TIMESTAMPS: true,
    SHOW_METRICS: true,
    PROMPT_SYMBOL: '❯',
    ENABLE_ANIMATIONS: true,
    COMPACT_MODE: false,
    ENABLE_NOTIFICATIONS: true
  },
  MODEL_CONFIG: {
    model: 'gemini-1.5-flash-latest',
    generationConfig: { maxOutputTokens: 8192, temperature: 0.7, topP: 0.9, topK: 40 },
    systemInstruction: "You are Pyrmethus, an advanced AI assistant integrated into a developer's terminal. You can use tools to search the web, interact with the file system, and execute commands. Provide concise, accurate, and secure responses. When using tools, explain your plan. When executing commands, prioritize safety."
  },
  ALIASES: { h: 'help', q: 'quit', c: 'clear', s: 'save', u: 'upload', ns: 'new-session', v: 'version', hist: 'history', p: 'patch', sess: 'sessions', cfg: 'config', g: 'git', t: 'task', ctx: 'context', cp: 'copy', pst: 'paste', o: 'open', mem: 'memory', w: 'watch', uf: 'upload', cf: 'clearfile' },
  PLUGINS: { enabled: true, directory: PYRMETHUS_PLUGIN_DIR },
  HISTORY: { LOG_FILE: path.join(PYRMETHUS_LOG_DIR, 'conversation.log'), MAX_LOG_SIZE: 1000, EXPORT_DIR: PYRMETHUS_HISTORY_EXPORT_DIR, COMPRESSION: true },
  ENCRYPTION: { enabled: false, algorithm: 'aes-256-gcm' },
  LOGGING: { level: 'info', file: path.join(PYRMETHUS_LOG_DIR, 'pyrmethus.log'), structured: true },
  FILE_SUPPORT: {
    extensions: ['txt', 'md', 'js', 'ts', 'py', 'json', 'csv', 'xml', 'yaml', 'yml', 'pdf', 'docx', 'xlsx', 'jpg', 'jpeg', 'png', 'gif', 'html', 'log', 'sh', 'sql', 'java', 'go', 'rs'],
    mimeTypes: ['text/*', 'application/json', 'application/xml', 'application/yaml', 'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'image/*']
  },
  SEARCH: { ENGINE: 'duckduckgo', API_KEY: '', NUM_RESULTS: 5 },
  GIT_INTEGRATION: { enabled: true, autoCommit: false, commitPrefix: '[Pyrmethus]' },
  LANGUAGE: 'en',
  SECURITY: { enableSandbox: true, allowedCommands: ['ls', 'cat', 'grep', 'git', 'node', 'npm', 'python'], blockedCommands: ['rm -rf', 'sudo'], allowHttpRequests: true },
  MEMORY: { embeddingModel: 'text-embedding-004', similarityThreshold: 0.75, maxRelevantEntries: 5 }
};

// --- Global State ---
const state = {
  configManager: null,
  config: { ...defaultGeminiConfig },
  geminiApi: null,
  model: null,
  chat: null,
  readlineInterface: null,
  theme: themes.neon,
  spinner: null,
  isProcessing: false,
  activeSession: null,
  performanceMetrics: { apiCalls: 0, totalApiDuration: 0, lastApiDuration: 0, tokensIn: 0, tokensOut: 0, averageResponseTime: 0, errorRate: 0, successRate: 100, cacheHits: 0, cacheMisses: 0 },
  conversationLog: [],
  contextFiles: new Set(),
  uploadedFile: null, // NEW: To hold reference to an uploaded file
  pluginManager: null,
  taskManager: null,
  gitManager: null,
  sessionManager: null,
  memoryManager: null, // NEW: Memory Manager instance
  sandboxManager: null, // NEW: Sandbox Manager instance
  notificationManager: null, // NEW: Notification Manager instance
  cache: null,
  rateLimiter: null,
  initialized: false,
  lastResponseText: '',
  watchedFiles: new Map() // For the /watch command
};

// --- Logger Initialization ---
const logger = pino({ level: 'info', transport: { target: 'pino-pretty', options: { colorize: true, translateTime: 'SYS:standard', ignore: 'pid,hostname' } } });

// --- Notification Manager ---
class NotificationManager {
  constructor (config) { this.config = config; }
  notify (title, message) {
    if (!this.config.UI.ENABLE_NOTIFICATIONS) return;
    notifier.notify({ title: `Pyrmethus: ${title}`, message });
    logger.debug(`Sent notification: ${title}`);
  }
}

// --- Sandbox Manager for Secure Command Execution ---
class SandboxManager {
  constructor (config) { this.config = config; }
  async execute (command, args = []) {
    if (!this.config.SECURITY.enableSandbox) {
      logger.warn('Sandbox is disabled. Executing command directly.');
      return exec(`${command} ${args.join(' ')}`);
    }

    const fullCommand = `${command} ${args.join(' ')}`;
    if (this.config.SECURITY.blockedCommands.some(blocked => fullCommand.includes(blocked))) {
      throw new Error(`Command execution blocked by security policy: ${command}`);
    }

    if (this.config.SECURITY.allowedCommands.length > 0 && !this.config.SECURITY.allowedCommands.includes(command)) {
      throw new Error(`Command not in allowed list: ${command}`);
    }

    return new Promise((resolve, reject) => {
      const child = spawn(command, args, { shell: true, stdio: 'pipe' });
      let stdout = '';
      let stderr = '';
      child.stdout.on('data', (data) => stdout += data.toString());
      child.stderr.on('data', (data) => stderr += data.toString());
      child.on('close', (code) => {
        if (code === 0) {
          resolve({ stdout, stderr });
        } else {
          reject(new Error(`Command failed with code ${code}: ${stderr || stdout}`));
        }
      });
      child.on('error', (err) => reject(err));
    });
  }
}

// --- Long-Term Memory Manager (Vector Store) ---
class MemoryManager {
  constructor (config, geminiApi) {
    this.config = config;
    this.geminiApi = geminiApi;
    this.memory = []; // In-memory vector store: { id, text, vector, timestamp }
    this.memoryFilePath = path.join(PYRMETHUS_MEMORY_DIR, 'memory_vectors.json');
  }

  async init () {
    await fs.mkdir(PYRMETHUS_MEMORY_DIR, { recursive: true });
    try {
      const data = await fs.readFile(this.memoryFilePath, 'utf-8');
      this.memory = JSON.parse(data);
      logger.info(`Loaded ${this.memory.length} entries from long-term memory.`);
    } catch (error) {
      logger.warn('No existing long-term memory found. Starting fresh.');
      this.memory = [];
    }
  }

  async save () {
    await fs.writeFile(this.memoryFilePath, JSON.stringify(this.memory, null, 2));
  }

  async addEntry (text) {
    if (!text || typeof text !== 'string' || text.trim().length < 10) return;
    try {
      const embeddingModel = this.geminiApi.getGenerativeModel({ model: this.config.MEMORY.embeddingModel });
      const result = await embeddingModel.embedContent(text);
      const vector = result.embedding.values;
      this.memory.push({ id: uuidv4(), text, vector, timestamp: new Date() });
      await this.save();
      logger.debug('Added new entry to long-term memory.');
    } catch (error) {
      logger.error({ err: error }, 'Failed to create embedding for memory entry.');
    }
  }

  async findRelevantEntries (queryText) {
    if (this.memory.length === 0) return [];
    const embeddingModel = this.geminiApi.getGenerativeModel({ model: this.config.MEMORY.embeddingModel });
    const queryResult = await embeddingModel.embedContent(queryText);
    const queryVector = queryResult.embedding.values;

    const similarities = this.memory.map(entry => ({
      ...entry,
      similarity: this.cosineSimilarity(queryVector, entry.vector)
    }));

    return similarities
      .filter(entry => entry.similarity > this.config.MEMORY.similarityThreshold)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, this.config.MEMORY.maxRelevantEntries);
  }

  cosineSimilarity (vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  clearMemory () {
    this.memory = [];
    return this.save();
  }
}

// --- Secure Encryption/Decryption ---
const IV_LENGTH = 16;
const AUTH_TAG_LENGTH = 16;

function encryptData (text, key) {
  const iv = crypto.randomBytes(IV_LENGTH);
  const cipher = crypto.createCipheriv(state.config.ENCRYPTION.algorithm, key, iv);
  const encrypted = Buffer.concat([cipher.update(text), cipher.final()]);
  const authTag = cipher.getAuthTag();
  return Buffer.concat([iv, authTag, encrypted]).toString('hex');
}

function decryptData (encryptedHex, key) {
  const encryptedBuffer = Buffer.from(encryptedHex, 'hex');
  const iv = encryptedBuffer.slice(0, IV_LENGTH);
  const authTag = encryptedBuffer.slice(IV_LENGTH, IV_LENGTH + AUTH_TAG_LENGTH);
  const encrypted = encryptedBuffer.slice(IV_LENGTH + AUTH_TAG_LENGTH);
  const decipher = crypto.createDecipheriv(state.config.ENCRYPTION.algorithm, key, iv);
  decipher.setAuthTag(authTag);
  const decrypted = Buffer.concat([decipher.update(encrypted), decipher.final()]);
  return decrypted.toString('utf-8');
}

// --- Tool and Function Definitions for AI ---
const tools = {
  web_search: {
    function: handleSearch,
    declaration: {
      name: 'web_search',
      description: 'Performs a web search for a given query and returns a summary of the top results.',
      parameters: {
        type: FunctionDeclarationSchemaType.OBJECT,
        properties: { query: { type: FunctionDeclarationSchemaType.STRING, description: 'The search query.' } },
        required: ['query']
      }
    }
  },
  execute_shell_command: {
    function: handleExec,
    declaration: {
      name: 'execute_shell_command',
      description: 'Executes a shell command in a secure sandbox. Use for file system operations, running scripts, etc.',
      parameters: {
        type: FunctionDeclarationSchemaType.OBJECT,
        properties: {
          command: { type: FunctionDeclarationSchemaType.STRING, description: "The command to execute (e.g., 'ls', 'cat')." },
          args: { type: FunctionDeclarationSchemaType.ARRAY, description: 'Arguments for the command.', items: { type: FunctionDeclarationSchemaType.STRING } }
        },
        required: ['command']
      }
    }
  },
  read_file: {
    function: async ({ filePath }) => readFileContent(filePath).then(res => res.content),
    declaration: {
      name: 'read_file',
      description: 'Reads the content of a specified file.',
      parameters: {
        type: FunctionDeclarationSchemaType.OBJECT,
        properties: { filePath: { type: FunctionDeclarationSchemaType.STRING, description: 'The path to the file.' } },
        required: ['filePath']
      }
    }
  },
  write_file: {
    function: async ({ filePath, content }) => fs.writeFile(filePath, content, 'utf-8'),
    declaration: {
      name: 'write_file',
      description: 'Writes content to a specified file, overwriting it if it exists.',
      parameters: {
        type: FunctionDeclarationSchemaType.OBJECT,
        properties: {
          filePath: { type: FunctionDeclarationSchemaType.STRING, description: 'The path to the file.' },
          content: { type: FunctionDeclarationSchemaType.STRING, description: 'The content to write.' }
        },
        required: ['filePath', 'content']
      }
    }
  }
};

// --- Enhanced Gemini Interaction with Function Calling ---
async function sendMessageToGemini (prompt) {
  if (!state.chat) throw new Error('Chat session not initialized.');
  await state.rateLimiter.checkLimit();

  const startTime = performance.now();
  state.isProcessing = true;
  state.spinner = ora({ text: state.theme.primary('Thinking...'), spinner: { interval: state.config.UI.SPINNER_INTERVAL, frames: state.config.UI.SPINNER_FRAMES }, color: state.theme.primary.name }).start();

  try {
    const fullPrompt = [];
    // Add long-term memory context
    if (state.config.AI_FEATURES.longTermMemory) {
      const relevantMemories = await state.memoryManager.findRelevantEntries(typeof prompt === 'string' ? prompt : prompt[0].text);
      if (relevantMemories.length > 0) {
        const memoryContext = 'Relevant information from past conversations:\n' + relevantMemories.map(m => `- ${m.text}`).join('\n');
        fullPrompt.push({ text: memoryContext });
      }
    }
    // Add file context
    if (state.config.AI_FEATURES.contextAwareness && state.contextFiles.size > 0) {
      const fileContext = 'Context from the following files is available:\n' + Array.from(state.contextFiles).map(f => `- ${path.basename(f)}`).join('\n');
      fullPrompt.push({ text: fileContext });
    }

    // Add uploaded file reference
    if (state.uploadedFile) {
      fullPrompt.push(state.uploadedFile);
    }

    // Add user prompt
    if (typeof prompt === 'string') {
      fullPrompt.push({ text: prompt });
    } else {
      fullPrompt.push(...prompt);
    }

    const stream = await state.chat.sendMessageStream(fullPrompt);
    let responseText = '';
    const functionCalls = [];

    for await (const chunk of stream.stream) {
      const text = chunk.text();
      if (text) {
        responseText += text;
        process.stdout.write(state.theme.text(text));
      }
      if (chunk.functionCalls) {
        functionCalls.push(...chunk.functionCalls);
      }
    }
    console.log(); // Newline after stream

    if (functionCalls.length > 0) {
      state.spinner.text = state.theme.info('Executing tools...');
      const toolResponses = [];
      for (const call of functionCalls) {
        const tool = tools[call.name];
        if (tool) {
          try {
            const output = await tool.function(call.args);
            toolResponses.push({
              functionResponse: {
                name: call.name,
                response: { output: JSON.stringify(output) }
              }
            });
          } catch (error) {
            toolResponses.push({
              functionResponse: {
                name: call.name,
                response: { error: error.message }
              }
            });
          }
        }
      }
      // Send tool responses back to the model
      await sendMessageToGemini(toolResponses);
    } else {
      state.lastResponseText = responseText;
      if (state.config.AI_FEATURES.longTermMemory) {
        await state.memoryManager.addEntry(`User: ${typeof prompt === 'string' ? prompt : 'Complex Input'}\nAI: ${responseText}`);
      }
      state.spinner.succeed(state.theme.success('Response received.'));
    }
  } catch (error) {
    logger.error({ err: error, prompt }, 'Error sending message to Gemini API');
    state.spinner.fail(state.theme.error(`Error: ${error.message}`));
  } finally {
    state.isProcessing = false;
    state.spinner?.stop();
  }
}

// --- Command Implementations ---
async function handleExec (args) {
  const command = Array.isArray(args) ? args[0] : args.command;
  const commandArgs = Array.isArray(args) ? args.slice(1) : args.args || [];

  if (!command) {
    console.log(state.theme.warning('Usage: /exec <command> [args...]'));
    return;
  }

  console.log(state.theme.info(`Executing: ${command} ${commandArgs.join(' ')}`));
  try {
    const { stdout, stderr } = await state.sandboxManager.execute(command, commandArgs);
    if (stdout) console.log(state.theme.success('STDOUT:\n'), stdout);
    if (stderr) console.log(state.theme.error('STDERR:\n'), stderr);
    return { stdout, stderr };
  } catch (error) {
    console.error(state.theme.error(`Execution failed: ${error.message}`));
    logger.error({ err: error, command }, 'Shell command execution failed');
    return { error: error.message };
  }
}

async function handleSearch ({ query }) {
  if (!query) {
    console.log(state.theme.warning('Usage: /search <query>'));
    return 'No query provided.';
  }
  console.log(state.theme.info(`Searching the web for: ${query}`));
  try {
    if (state.config.SEARCH.ENGINE === 'duckduckgo') {
      const response = await axios.get(`https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&pretty=1`);
      const results = response.data.RelatedTopics.filter(t => t.Text).slice(0, state.config.SEARCH.NUM_RESULTS);
      const summary = results.map(r => `${r.Text}`).join('\n');
      console.log(state.theme.success('Search Results:\n'), summary);
      return summary;
    }
    // Placeholder for other search engines
    return 'Search engine not configured.';
  } catch (error) {
    console.error(state.theme.error(`Search failed: ${error.message}`));
    return `Search failed: ${error.message}`;
  }
}

async function handleHelp () {
  const table = new Table({
    head: [state.theme.primary('Command'), state.theme.primary('Alias'), state.theme.primary('Description')],
    colWidths: [20, 10, 70],
    style: { head: [], border: [] }
  });

  const aliasMap = new Map(Object.entries(state.config.ALIASES).map(([alias, cmd]) => [cmd, alias]));

  commands.forEach((handler, name) => {
    if (name.startsWith('/_')) return; // Hide internal commands
    const commandName = name.substring(1);
    const alias = aliasMap.get(commandName) ? `/${aliasMap.get(commandName)}` : '-';
    table.push([state.theme.accent(name), state.theme.dim(alias), handler.description]);
  });

  console.log(boxen(table.toString(), { padding: 1, margin: 1, borderStyle: 'round', borderColor: 'gray' }));
}

async function handleUpload (args) {
  let filePath = args[0];
  if (!filePath) {
    const answers = await inquirer.prompt([{ name: 'filePath', message: 'Enter the path to the file to upload:' }]);
    if (!answers.filePath) {
      console.log(state.theme.warning('File path cannot be empty.'));
      return;
    }
    filePath = answers.filePath;
  }

  try {
    await fs.access(filePath); // Check if file exists
    state.spinner = ora({ text: state.theme.primary(`Uploading ${filePath}...`), spinner: { interval: state.config.UI.SPINNER_INTERVAL, frames: state.config.UI.SPINNER_FRAMES }, color: state.theme.primary.name }).start();
    const uploadResult = await state.geminiApi.uploadFile(filePath);
    state.uploadedFile = uploadResult.file; // Store the file object
    state.spinner.succeed(state.theme.success(`Successfully uploaded file: ${uploadResult.file.displayName}. It will be included in the context of your next messages.`));
  } catch (error) {
    state.spinner?.fail(state.theme.error(`Error uploading file: ${error.message}`));
    logger.error({ err: error, filePath }, 'File upload failed');
  }
}

async function handleClearFile () {
  if (state.uploadedFile) {
    console.log(state.theme.success(`Cleared file context for: ${state.uploadedFile.displayName}`));
    state.uploadedFile = null;
  } else {
    console.log(state.theme.info('No file is currently uploaded in the context.'));
  }
}

async function handleTask (args) {
  // This function would be expanded to use cli-table3 for listing tasks
  console.log(state.theme.warning('Task command not fully implemented in this version.'));
}

async function handleWatch (args) {
  const filePath = args[0];
  const commandToRun = args.slice(1).join(' ');
  if (!filePath || !commandToRun) {
    console.log(state.theme.warning('Usage: /watch <file_path> <command_to_run...>'));
    return;
  }
  const absolutePath = path.resolve(filePath);
  if (state.watchedFiles.has(absolutePath)) {
    console.log(state.theme.warning(`Already watching ${filePath}. Use /unwatch first.`));
    return;
  }
  const watcher = chokidar.watch(absolutePath).on('change', debounce(async () => {
    console.log(state.theme.info(`\nFile ${filePath} changed. Running: ${commandToRun}`));
    await handleInput(commandToRun);
    state.readlineInterface.prompt();
  }, 500));
  state.watchedFiles.set(absolutePath, watcher);
  console.log(state.theme.success(`Now watching ${filePath}.`));
}

async function handleUnwatch (args) {
  const filePath = args[0];
  if (!filePath) {
    console.log(state.theme.warning('Usage: /unwatch <file_path>'));
    return;
  }
  const absolutePath = path.resolve(filePath);
  if (state.watchedFiles.has(absolutePath)) {
    state.watchedFiles.get(absolutePath).close();
    state.watchedFiles.delete(absolutePath);
    console.log(state.theme.success(`Stopped watching ${filePath}.`));
  } else {
    console.log(state.theme.warning(`Not currently watching ${filePath}.`));
  }
}

async function handleChatFile(args) {
    const rl = state.readlineInterface;
    if (!rl) {
        console.log(state.theme.error('Readline interface not initialized.'));
        return;
    }

    try {
        const filePath = await new Promise(resolve => {
            rl.question(state.theme.prompt('Path to the file you want to discuss: '), resolve);
        });

        try {
            await fs.access(filePath);
        } catch (e) {
            console.log(state.theme.error(`Error: File not found at '${filePath}'`));
            return;
        }

        const uploadSpinner = ora({ text: state.theme.primary('Uploading...'), spinner: 'dots' }).start();
        const uploadedFile = await state.geminiApi.uploadFile(filePath);
        uploadSpinner.succeed(state.theme.success(`Successfully uploaded file: ${uploadedFile.file.displayName}`));

        const model = state.geminiApi.getGenerativeModel({ model: state.config.MODEL_NAME });
        const chat = model.startChat({ history: [] });

        console.log(state.theme.info("You can now ask questions about the file. Type 'exit' or 'quit' to end this file chat session."));

        while (true) {
            const userInput = await new Promise(resolve => {
                rl.question(state.theme.prompt('\nYou (file chat): '), resolve);
            });

            if (userInput.toLowerCase() === 'exit' || userInput.toLowerCase() === 'quit') {
                console.log(state.theme.secondary('Exiting file chat session.'));
                break;
            }

            const thinkingSpinner = ora({ text: state.theme.primary('Thinking...'), spinner: 'dots' }).start();
            const result = await chat.sendMessageStream([userInput, uploadedFile.file]);
            thinkingSpinner.stop();

            process.stdout.write(state.theme.text('AI: '));
            for await (const chunk of result.stream) {
                const chunkText = chunk.text();
                process.stdout.write(state.theme.text(chunkText));
            }
            console.log();
        }
    } catch (error) {
        console.error(state.theme.error(`\nAn error occurred during the file chat session: ${error.message}`));
        logger.error({ err: error }, 'File chat session failed');
    }
}

// --- Main Application Orchestration ---
async function initialize () {
  // 1. Configuration & Core Systems
  state.configManager = new ConfigManager();
  await state.configManager.init();
  state.config = state.configManager.config;
  setupLogger(state.config.LOGGING);
  state.theme = themes[state.config.UI.THEME.toLowerCase()] || themes.neon;
  setupI18n();
  state.cache = new NodeCache({ stdTTL: state.config.CACHE_TTL });
  state.rateLimiter = new RateLimiter(state.config.RATE_LIMITING);

  // 2. Initialize Managers
  state.notificationManager = new NotificationManager(state.config);
  state.sandboxManager = new SandboxManager(state.config);
  state.taskManager = new TaskManager();
  state.gitManager = new GitManager();
  state.sessionManager = new SessionManager();
  await state.sessionManager.init();

  // 3. Initialize AI and API
  if (!state.config.GOOGLE_API_KEY) throw new Error('GOOGLE_API_KEY not set.');
  state.geminiApi = new GoogleGenerativeAI(state.config.GOOGLE_API_KEY);
  state.memoryManager = new MemoryManager(state.config, state.geminiApi);
  await state.memoryManager.init();

  const modelConfig = {
    model: state.config.MODEL_CONFIG.model,
    generationConfig: state.config.MODEL_CONFIG.generationConfig,
    systemInstruction: state.config.MODEL_CONFIG.systemInstruction
  };
  if (state.config.AI_FEATURES.functionCalling) {
    modelConfig.tools = [{ functionDeclarations: Object.values(tools).map(t => t.declaration) }];
  }
  state.model = state.geminiApi.getGenerativeModel(modelConfig);
  state.chat = state.model.startChat({ history: [] });

  // 4. UI and Commands
  state.readlineInterface = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: `${state.theme.prompt(state.config.UI.PROMPT_SYMBOL + ' ')}` });
  inquirer.registerPrompt('autocomplete', autocomplete);
  registerCommands();

  // 5. Welcome Message
  console.clear();
  const rainbow = chalkAnimation.rainbow('Welcome to Pyrmethus v3.2.1!');
  rainbow.start();
  await new Promise(resolve => setTimeout(resolve, 1500));
  rainbow.stop();
  console.log(`\n${state.theme.primary("Type '/help' for commands. AI is equipped with tools.")}\n`);
  state.initialized = true;
  logger.info('Pyrmethus v3.2.1 initialized successfully.');
}

function registerCommands () {
  // Command registration logic would be here
  registerCommand('exec', 'Execute a shell command', (args) => handleExec(args), 'System');
  registerCommand('search', 'Perform a web search', (args) => handleSearch({ query: args.join(' ') }), 'System');
  registerCommand('help', 'Show this help message', handleHelp, 'General');
  registerCommand('watch', 'Watch a file and run a command on change', handleWatch, 'System');
  registerCommand('unwatch', 'Stop watching a file', handleUnwatch, 'System');
  registerCommand('quit', 'Exit Pyrmethus', gracefulShutdown, 'General');
  registerCommand('chatfile', 'Start an interactive chat about a specific file.', handleChatFile, 'Files');
  registerCommand('upload', 'Upload a file to discuss', handleUpload, 'Files');
  registerCommand('clearfile', 'Clear the uploaded file from the conversation context', handleClearFile, 'Files');
  // ... other commands
}

async function handleInput (input) {
  if (state.isProcessing) {
    console.log(state.theme.warning('Please wait, currently processing...'));
    return;
  }
  const trimmedInput = input.trim();
  if (!trimmedInput) return;
  logger.info({ input: trimmedInput }, 'User input');

  const [command, ...args] = trimmedInput.split(' ');
  const handler = commands.get(command.toLowerCase());

  if (handler) {
    try {
      await handler.action(args);
    } catch (error) {
      logger.error({ err: error, command }, `Command ${command} failed`);
      console.error(state.theme.error(`Error executing command: ${error.message}`));
    }
  } else {
    await sendMessageToGemini(trimmedInput);
  }
}

async function promptUser () {
  try {
    const answers = await inquirer.prompt([{ 
      type: 'autocomplete',
      name: 'commandInput',
      message: `${state.theme.prompt(state.config.UI.PROMPT_SYMBOL + ' ')} `,
      source: async (_, input) => {
        const commandNames = Array.from(commands.keys());
        if (!input) return commandNames;
        return fuzzy.filter(input, commandNames).map(el => el.string);
      }
    }]);
    await handleInput(answers.commandInput);
  } catch (error) {
    if (!error.isTtyError) {
      logger.error({ err: error }, 'Error during user prompt');
    }
  }
}

async function gracefulShutdown () {
  console.log(state.theme.secondary('\nShutting down...'));
  if (state.configManager) state.configManager.cleanup();
  if (state.sessionManager) {
    if (state.activeSession) await state.sessionManager.saveSession(state.activeSession);
    state.sessionManager.cleanup();
  }
  if (state.memoryManager) await state.memoryManager.save();
  state.watchedFiles.forEach(watcher => watcher.close());
  if (state.readlineInterface) state.readlineInterface.close();
  console.log(state.theme.dim('Goodbye!'));
  process.exit(0);
}

async function run () {
  try {
    await initialize();
    console.log(`Active session: ${state.theme.primary(state.activeSession?.name || 'Default')}`);
    while (true) {
      await promptUser();
    }
  } catch (error) {
    logger.fatal({ err: error }, 'Critical runtime error');
    console.error(chalk.redBright(`\nCritical Error: ${error.message}`));
    process.exit(1);
  }
}

// --- Global Error & Signal Handling ---
process.on('SIGINT', gracefulShutdown);
process.on('uncaughtException', (error) => {
  logger.fatal({ err: error }, 'Uncaught Exception');
  console.error(chalk.redBright(`\nAn unexpected error occurred: ${error.message}`));
  gracefulShutdown();
});
process.on('unhandledRejection', (reason) => {
  logger.error({ reason }, 'Unhandled Rejection');
});

// --- Start Application ---
// Dummy implementations for missing classes/functions to allow execution
const commands = new Map();
function registerCommand (name, description, action, category) { commands.set(`/${name}`, { description, action, category }); }
class ConfigManager { async init () {} cleanup () {} }
class SessionManager { async init () {} async saveSession () {} cleanup () {} }
function setupI18n () {}
async function readFileContent (filePath) { return { content: await fs.readFile(filePath, 'utf-8') }; }

run();
