This is a very comprehensive and well-structured codebase. Enhancing it for clarity, conciseness, and flow primarily involves refining descriptions, standardizing naming, and ensuring logical grouping.

Here's an enhanced version of your provided text, focusing on those aspects:

```javascript
#!/usr/bin/env node

// Pyrmethus v3.0.0 - Next-Generation Gemini Command Spell
// A revolutionary AI-powered terminal assistant with over 40 major enhancements.
// Features advanced context management, Git integration, extensible command capabilities,
// and a developer-centric experience for unparalleled productivity.

// --- Core Imports ---
import {
  GoogleGenerativeAI,
  GenerativeModel,
  ChatSession,
  Content,
  GenerationConfig,
} from "@google/generative-ai";
import dotenv from "dotenv";
import readline from "readline/promises";
import fs from "fs/promises";
import path from "path";
import os from "os";
import chalk, { Chalk } from "chalk";
import figlet from "figlet";
import hljs from "highlight.js";
import ora, { Ora } from "ora";
import { fileTypeFromBuffer } from "file-type";
import { performance } from "perf_hooks";
import { v4 as uuidv4 } from "uuid";
import { applyPatch, createTwoFilesPatch } from "diff";
import mammoth from "mammoth";
import XLSX from "xlsx";
import { exec as execCb, spawn } from "child_process";
import { promisify } from "util";
import crypto from "crypto";
import EventEmitter from "events";
import inquirer from "inquirer";
import autocomplete from "inquirer-autocomplete-prompt";
import fuzzy from "fuzzy";
import { marked } from "marked";
import chokidar from "chokidar";
import NodeCache from "node-cache";
import i18n from "i18n";
import chalkAnimation from "chalk-animation";
import fsSync from "fs"; // For synchronous file operations
import { glob } from "glob";
import yaml from "js-yaml";
import PDFParser from "pdf2json";
import pino from "pino";
import simpleGit from "simple-git";
import debounce from "lodash.debounce";
import { z } from "zod";
import zlib from "zlib";

// --- Promisified Utilities ---
const exec = promisify(execCb);
const gzip = promisify(zlib.gzip);
const gunzip = promisify(zlib.gunzip);

// --- Configuration Schema (Zod Validation) ---
const ConfigSchema = z.object({
  VERSION: z.string(),
  GOOGLE_API_KEY: z.string().optional(),
  MODEL_NAME: z.string(),
  MAX_FILE_SIZE: z.number().positive(),
  MAX_SEARCH_DEPTH: z.number().int().min(1),
  MAX_SEARCH_RESULTS: z.number().int().min(1),
  MAX_CONVERSATION_HISTORY: z.number().int(),
  STREAM_TIMEOUT: z.number().positive(),
  CACHE_TTL: z.number().int(),
  AUTO_SAVE_SESSION: z.boolean(),
  AUTO_BACKUP_INTERVAL: z.string().nullable(),
  RATE_LIMITING: z.object({
    enabled: z.boolean(),
    requestsPerMinute: z.number().positive(),
  }),
  AI_FEATURES: z.object({
    extendedThinking: z.boolean(),
    visionCapabilities: z.boolean(),
    autonomousMode: z.boolean(),
    contextAwareness: z.boolean(),
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
  }),
  MODEL_CONFIG: z.object({
    model: z.string(),
    generationConfig: z.object({ maxOutputTokens: z.number(), temperature: z.number(), topP: z.number(), topK: z.number() }).optional(),
    systemInstruction: z.string().optional(),
  }),
  ALIASES: z.record(z.string()),
  PLUGINS: z.object({
    enabled: z.boolean(),
    directory: z.string(),
  }),
  HISTORY: z.object({
    LOG_FILE: z.string(),
    MAX_LOG_SIZE: z.number().int(),
    EXPORT_DIR: z.string(),
    COMPRESSION: z.boolean(),
  }),
  ENCRYPTION: z.object({
    enabled: z.boolean(),
    algorithm: z.string(),
    key: z.instanceof(Buffer).optional(),
  }),
  LOGGING: z.object({
    level: z.enum(["fatal", "error", "warn", "info", "debug", "trace", "silent"]),
    file: z.string(),
    structured: z.boolean(),
  }),
  FILE_SUPPORT: z.object({
    extensions: z.array(z.string()),
    mimeTypes: z.array(z.string()),
  }),
  SEARCH: z.object({
      DEPTH: z.number(),
      RESULTS: z.number(),
  }),
  GIT_INTEGRATION: z.object({
      enabled: z.boolean(),
      autoCommit: z.boolean(),
      commitPrefix: z.string(),
  }),
  LANGUAGE: z.string().optional(),
});

// --- Directory Constants ---
const CONFIG_DIR = path.join(os.homedir(), ".pyrmethus");
const CONFIG_FILE = path.join(CONFIG_DIR, "config.json");
const PROFILE_DIR = path.join(CONFIG_DIR, "profiles");
const PLUGIN_DIR = path.join(CONFIG_DIR, "plugins");
const LOCALE_DIR = path.join(__dirname, "locales");
const LOG_DIR = path.join(CONFIG_DIR, "logs");
const SESSION_DIR = path.join(CONFIG_DIR, "sessions");
const HISTORY_EXPORT_DIR = path.join(CONFIG_DIR, "history_exports");
const COMMANDS_DIR = path.join(CONFIG_DIR, "commands");
const CONTEXT_DIR = path.join(CONFIG_DIR, "context");

// --- UI Theme Definitions ---
const defaultChalk = chalk.ansi256;

/**
 * Creates a UI theme object with customizable colors.
 * @param colors - Partial theme properties to override defaults.
 * @returns A complete theme object.
 */
const createTheme = (colors: Partial<Theme>): Theme => ({
  primary: colors.primary || defaultChalk.white,
  secondary: colors.secondary || defaultChalk.gray,
  success: colors.success || defaultChalk.green,
  warning: colors.warning || defaultChalk.yellow,
  error: colors.error || defaultChalk.red.bold,
  info: colors.info || defaultChalk.blue,
  text: colors.text || defaultChalk.white,
  dim: colors.dim || defaultChalk.gray,
  accent: colors.accent || defaultChalk.bold,
  prompt: colors.prompt || colors.primary || defaultChalk.white,
  boxHeader: colors.boxHeader || colors.primary?.bold || defaultChalk.white.bold,
  boxBorder: colors.boxBorder || colors.primary || defaultChalk.white,
  highlight: colors.highlight || defaultChalk.inverse,
  code: colors.code || defaultChalk.green,
});

const defaultTheme = createTheme({
  primary: defaultChalk.cyanBright, secondary: defaultChalk.magentaBright,
  success: defaultChalk.greenBright, warning: defaultChalk.yellowBright,
  info: defaultChalk.blueBright, text: defaultChalk.whiteBright,
  highlight: defaultChalk.bgCyan.black,
});

const defaultMatrixTheme = createTheme({
  primary: defaultChalk.green, secondary: defaultChalk.rgb(0, 255, 0),
  info: defaultChalk.rgb(0, 150, 0), text: defaultChalk.rgb(0, 255, 0),
  dim: defaultChalk.rgb(0, 100, 0), highlight: defaultChalk.bgGreen.black,
});

const defaultCyberpunkTheme = createTheme({
  primary: defaultChalk.rgb(255, 0, 255), secondary: defaultChalk.rgb(0, 255, 255),
  warning: defaultChalk.rgb(255, 255, 0), info: defaultChalk.rgb(100, 100, 255),
  text: defaultChalk.rgb(200, 200, 200), highlight: defaultChalk.bgMagenta.black,
  code: defaultChalk.cyan,
});

const defaultMinimalTheme = createTheme({});

// --- Default Configuration ---
const defaultGeminiConfig: Config = {
  VERSION: "3.0.0",
  GOOGLE_API_KEY: "",
  MODEL_NAME: "gemini-1.5-flash-latest",
  MAX_FILE_SIZE: 20 * 1024 * 1024,
  MAX_SEARCH_DEPTH: 5,
  MAX_SEARCH_RESULTS: 25,
  MAX_CONVERSATION_HISTORY: 50,
  STREAM_TIMEOUT: 30000,
  CACHE_TTL: 3600,
  AUTO_SAVE_SESSION: true,
  AUTO_BACKUP_INTERVAL: "0 */6 * * *", // Cron syntax for every 6 hours
  RATE_LIMITING: { enabled: true, requestsPerMinute: 60 },
  AI_FEATURES: { extendedThinking: true, visionCapabilities: true, autonomousMode: false, contextAwareness: true },
  UI: {
    MAX_BOX_WIDTH: 100,
    SPINNER_FRAMES: ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    SPINNER_INTERVAL: 80,
    THEME: "neon",
    SHOW_TIMESTAMPS: true,
    SHOW_METRICS: true,
    PROMPT_SYMBOL: "❯",
    ENABLE_ANIMATIONS: true,
    COMPACT_MODE: false,
  },
  MODEL_CONFIG: {
    model: "gemini-1.5-flash-latest",
    generationConfig: { maxOutputTokens: 8192, temperature: 0.7, topP: 0.9, topK: 40 },
    systemInstruction: "You are Pyrmethus, an advanced AI assistant integrated into a developer's terminal. You are repository-aware, can interact with the file system, and execute commands. Provide concise, accurate, and secure responses. Format code snippets correctly. When asked to perform complex tasks, think step-by-step.",
  },
  ALIASES: {
    h: "help", q: "quit", c: "clear", s: "save", sc: "save-convo", u: "upload",
    uf: "upload-file", ns: "new-session", v: "version", hist: "history",
    sh: "search-history", p: "patch", sess: "sessions", sw: "switch-session",
    cfg: "config", g: "git", t: "think", tasks: "task", ctx: "context",
  },
  PLUGINS: { enabled: true, directory: PLUGIN_DIR },
  HISTORY: { LOG_FILE: path.join(LOG_DIR, "conversation.log"), MAX_LOG_SIZE: 1000, EXPORT_DIR: HISTORY_EXPORT_DIR, COMPRESSION: true },
  ENCRYPTION: { enabled: false, algorithm: 'aes-256-gcm' },
  LOGGING: { level: "info", file: path.join(LOG_DIR, "pyrmethus.log"), structured: true },
  FILE_SUPPORT: {
    extensions: ["txt", "md", "js", "ts", "py", "json", "csv", "xml", "yaml", "yml", "pdf", "docx", "xlsx", "jpg", "jpeg", "png", "gif", "html", "log", "sh", "sql", "java", "go", "rs"],
    mimeTypes: ["text/*", "application/json", "application/xml", "application/yaml", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "image/*"],
  },
  SEARCH: { DEPTH: 5, RESULTS: 25 },
  GIT_INTEGRATION: { enabled: true, autoCommit: false, commitPrefix: "[Pyrmethus]" },
  LANGUAGE: "en",
};

// --- Global State and Logger Initialization ---
let state: GlobalState = {} as GlobalState; // Initialize with a placeholder
const logger = pino({ level: 'info' }); // Initial logger, will be reconfigured by ConfigManager

/**
 * Configures the application logger based on provided settings.
 * @param loggingConfig - The logging configuration from the application settings.
 */
function setupLogger(loggingConfig: Config['LOGGING']) {
    const transportTargets: pino.TransportTargetOptions[] = [];
    if (loggingConfig.structured) {
        // Log to a file with structured JSON output
        transportTargets.push({ target: 'pino/file', options: { destination: loggingConfig.file, mkdir: true } });
    }
    // Log to console with pretty printing
    transportTargets.push({
        target: 'pino-pretty',
        options: { colorize: true, translateTime: 'SYS:standard', ignore: 'pid,hostname' }
    });
    logger.level = loggingConfig.level;
    // Reconfigure logger with new transports and level
    (logger as any).transport = pino.transport({ targets: transportTargets });
}

// --- Rate Limiter Class ---
class RateLimiter {
  private requestTimestamps: number[] = []; // Stores timestamps of recent requests

  constructor(private readonly rateLimitConfig: Config['RATE_LIMITING']) {}

  /**
   * Checks if the current request count exceeds the configured limit.
   * Throws an error if the limit is exceeded, indicating the wait time.
   */
  async checkLimit(): Promise<void> {
    if (!this.rateLimitConfig.enabled) return;

    const now = Date.now();
    const oneMinuteInMillis = 60000;

    // Filter out timestamps older than one minute
    this.requestTimestamps = this.requestTimestamps.filter(timestamp => now - timestamp < oneMinuteInMillis);

    if (this.requestTimestamps.length >= this.rateLimitConfig.requestsPerMinute) {
      const oldestTimestamp = this.requestTimestamps[0];
      const waitTime = oneMinuteInMillis - (now - oldestTimestamp);
      throw new Error(`Rate limit exceeded. Please wait approximately ${Math.ceil(waitTime / 1000)} seconds.`);
    }

    // Record the current request timestamp
    this.requestTimestamps.push(now);
  }
}

// --- Task Manager Class ---
class TaskManager extends EventEmitter {
  private tasks: Map<string, Task> = new Map();

  /**
   * Creates a new task and adds it to the manager.
   * @param description - A description of the task.
   * @returns The newly created Task object.
   */
  createTask(description: string): Task {
    const task: Task = { id: uuidv4(), description, status: "pending", createdAt: new Date() };
    this.tasks.set(task.id, task);
    this.emit('task-created', task); // Emit event for task creation
    return task;
  }

  /**
   * Updates an existing task with new properties.
   * @param id - The ID of the task to update.
   * @param updates - Partial properties of the Task object to update.
   */
  updateTask(id: string, updates: Partial<Task>) {
    const task = this.tasks.get(id);
    if (task) {
      Object.assign(task, updates);
      this.emit('task-updated', task); // Emit event for task update
    }
  }

  getTask = (id: string): Task | undefined => this.tasks.get(id);
  getAllTasks = (): Task[] => Array.from(this.tasks.values()).sort((a, b) => a.createdAt.getTime() - b.createdAt.getTime());
}

// --- Git Integration Manager ---
class GitManager {
  private readonly git: ReturnType<typeof simpleGit>;

  constructor() {
    this.git = simpleGit();
  }

  /** Checks if the current directory is a Git repository. */
  async isRepo(): Promise<boolean> {
    return this.git.checkIsRepo();
  }

  /**
   * Creates a Git commit for specified files.
   * @param message - The commit message.
   * @param files - An array of file paths to include in the commit (defaults to all tracked files).
   */
  async createCommit(message: string, files: string[] = ['.']): Promise<string | undefined> {
    await this.git.add(files);
    const commitResult = await this.git.commit(`${state.config.GIT_INTEGRATION.commitPrefix} ${message}`);
    if (commitResult.commit) {
      logger.info({ commitId: commitResult.commit }, `Created commit`);
      return commitResult.commit;
    }
    return undefined;
  }

  /** Gets the current branch name of the repository. */
  getCurrentBranch = async (): Promise<string | undefined> => {
    try {
      const branchSummary = await this.git.branchLocal();
      return branchSummary.current;
    } catch (error) {
      logger.warn({ err: error }, "Failed to get current Git branch");
      return undefined;
    }
  };

  /** Gets the diff of the repository. */
  getDiff = (options?: string[]): Promise<string> => this.git.diff(options);

  /** Gets the status of the repository. */
  getStatus = (): Promise<simpleGit.StatusResult> => this.git.status();

  /** Gets the commit log of the repository. */
  getLog = (limit: number = 10): Promise<simpleGit.LogResult> => this.git.log({ n: limit });
}

// --- Enhanced Configuration Manager ---
class ConfigManager {
  private currentProfile: string = "default";
  public config: Config = { ...defaultGeminiConfig }; // Initialize with defaults
  private profiles: Map<string, Partial<Config>> = new Map(); // To store loaded profiles
  private configWatcher: chokidar.FSWatcher | null = null; // Watcher for config file changes

  /**
   * Initializes the configuration manager:
   * - Creates necessary directories.
   * - Loads configuration from file or defaults.
   * - Loads all available profiles.
   * - Applies the current profile.
   * - Sets up a watcher for the configuration file.
   */
  async init() {
    const directoriesToCreate = [CONFIG_DIR, PROFILE_DIR, LOCALE_DIR, LOG_DIR, PLUGIN_DIR, HISTORY_EXPORT_DIR, SESSION_DIR, COMMANDS_DIR, CONTEXT_DIR];
    for (const dir of directoriesToCreate) {
      if (!fsSync.existsSync(dir)) {
        fsSync.mkdirSync(dir, { recursive: true });
      }
    }

    await this.loadConfig();
    await this.loadAllProfiles();
    await this.applyProfile(this.currentProfile);
    this.setupConfigWatcher();

    // Generate encryption key if encryption is enabled but no key exists (e.g., first run)
    if (this.config.ENCRYPTION.enabled && !this.config.ENCRYPTION.key) {
      this.config.ENCRYPTION.key = crypto.randomBytes(32);
      logger.warn("Encryption enabled, generated a session-specific key. Store this key securely if needed across sessions.");
    }
  }

  /** Sets up a file watcher to automatically reload configuration on changes. */
  private setupConfigWatcher() {
    this.configWatcher = chokidar.watch(CONFIG_FILE, { persistent: true, ignoreInitial: true });
    // Debounce to prevent multiple reloads from a single save operation
    this.configWatcher.on('change', debounce(async () => {
      logger.info("Configuration file changed, reloading...");
      await this.loadConfig();
      setupLogger(this.config.LOGGING); // Re-setup logger with new settings
      this.applyTheme(this.config.UI.THEME); // Re-apply theme
      // Potentially re-initialize other components that depend on config
    }, 1000)); // Wait 1 second after the last change
  }

  /**
   * Loads configuration from `CONFIG_FILE`.
   * If the file doesn't exist or is invalid, it uses default configurations and saves them.
   * Validates the loaded configuration using Zod schema.
   */
  private async loadConfig() {
    try {
      const configData = await fs.readFile(CONFIG_FILE, 'utf-8');
      const loadedConfig = JSON.parse(configData);
      // Merge defaults with loaded config and validate
      const validatedConfig = ConfigSchema.parse({ ...defaultGeminiConfig, ...loadedConfig });
      this.config = validatedConfig;
      // Prioritize GOOGLE_API_KEY from environment variables
      this.config.GOOGLE_API_KEY = process.env.GOOGLE_API_KEY || this.config.GOOGLE_API_KEY;
      logger.info("Configuration loaded and validated successfully.");
    } catch (error: any) {
      if (error instanceof z.ZodError) {
        logger.error({ errors: error.errors }, "Configuration validation failed. Using defaults.");
      } else {
        logger.warn(`Configuration file not found or invalid. Using defaults. Error: ${error.message}`);
      }
      // Reset to defaults and save if loading failed
      this.config = { ...defaultGeminiConfig };
      this.config.GOOGLE_API_KEY = process.env.GOOGLE_API_KEY; // Still try to get from env
      await this.saveConfig(); // Save the default config
    }
  }

  /**
   * Saves the current configuration to `CONFIG_FILE`.
   * Excludes sensitive information like API keys and encryption keys.
   */
  async saveConfig() {
    try {
      const configToSave = { ...this.config };
      // Remove sensitive data before saving
      delete (configToSave as any).GOOGLE_API_KEY;
      delete (configToSave.ENCRYPTION as any).key;

      await fs.writeFile(CONFIG_FILE, JSON.stringify(configToSave, null, 2), 'utf-8');
      logger.debug("Configuration saved.");
    } catch (error: any) {
      logger.error(`Failed to save configuration: ${error.message}`);
    }
  }

  /**
   * Loads all profile configuration files from the PROFILE_DIR.
   * Profiles are stored as JSON files.
   */
  private async loadAllProfiles() {
    try {
      const files = await glob("*.json", { cwd: PROFILE_DIR, absolute: true });
      for (const file of files) {
        const profileName = path.basename(file, ".json");
        const profileData = await fs.readFile(file, 'utf-8');
        const profileConfig = JSON.parse(profileData);
        // Validate profile configuration against a subset of the main schema if needed
        this.profiles.set(profileName, profileConfig);
        logger.debug(`Loaded profile: ${profileName}`);
      }
    } catch (error: any) {
      logger.error(`Failed to load profiles: ${error.message}`);
    }
  }

  /**
   * Applies a specific profile to the current configuration.
   * Merges the profile's settings into the main configuration.
   * @param profileName - The name of the profile to apply.
   */
  async applyProfile(profileName: string): Promise<void> {
    const profileConfig = this.profiles.get(profileName);
    if (profileConfig) {
      // Deep merge profile config into current config
      this.config = { ...this.config, ...profileConfig };
      // Re-validate after merging if necessary
      try {
        this.config = ConfigSchema.parse(this.config);
        this.currentProfile = profileName;
        logger.info(`Applied profile: '${profileName}'`);
        // Re-apply theme and logger based on the merged config
        setupLogger(this.config.LOGGING);
        this.applyTheme(this.config.UI.THEME);
      } catch (error: any) {
        logger.error({ err: error }, `Failed to validate configuration after applying profile '${profileName}'`);
        // Optionally revert to previous state or handle error
      }
    } else {
      logger.warn(`Profile '${profileName}' not found.`);
    }
  }

  /**
   * Applies a UI theme to the application.
   * Updates the global state's theme and the prompt symbol.
   * @param themeName - The name of the theme to apply (e.g., "neon", "matrix").
   */
  applyTheme(themeName: string) {
    const themes: Record<string, Theme> = {
        neon: defaultTheme, matrix: defaultMatrixTheme,
        cyberpunk: defaultCyberpunkTheme, minimal: defaultMinimalTheme
    };
    const selectedTheme = themes[themeName.toLowerCase()];
    if (selectedTheme) {
      state.theme = selectedTheme;
      this.config.UI.THEME = themeName; // Update config to reflect the applied theme
      if (state.readlineInterface) {
        // Update the prompt symbol based on the new theme
        state.readlineInterface.setPrompt(`${state.theme.prompt(state.config.UI.PROMPT_SYMBOL + " ")}`);
      }
      logger.info(`Theme set to: ${themeName}`);
    } else {
      logger.warn(`Theme '${themeName}' not found. Using default theme.`);
    }
  }

  /** Cleans up resources used by the config manager, such as file watchers. */
  cleanup() {
    this.configWatcher?.close();
    logger.debug("Config watcher closed.");
  }
}

// --- Enhanced Session Manager ---
class SessionManager {
  private sessions: Map<string, Session> = new Map();
  private encryptionKey?: Buffer;
  private autoSaveInterval: NodeJS.Timeout | null = null;

  /**
   * Initializes the SessionManager:
   * - Retrieves encryption key from configuration.
   * - Loads all saved sessions.
   * - Sets up auto-save functionality if enabled.
   */
  async init() {
    this.encryptionKey = state.config.ENCRYPTION.enabled ? state.config.ENCRYPTION.key : undefined;
    await this.loadAllSessions();
    // TODO: Implement logic to activate or create a default session if none are active.
    this.setupAutoSave();
    logger.info("Session Manager initialized.");
  }

  /** Sets up an interval for automatically saving the current session. */
  private setupAutoSave() {
    if (this.autoSaveInterval) clearInterval(this.autoSaveInterval); // Clear existing interval if any
    if (state.config.AUTO_SAVE_SESSION) {
      // Save session every 60 seconds (1 minute)
      this.autoSaveInterval = setInterval(() => this.saveCurrentSession(), 60000);
      logger.debug("Auto-save session enabled.");
    }
  }

  /** Provides default performance metrics initialized to zero. */
  private getDefaultMetrics = (): PerformanceMetrics => ({
    apiCalls: 0, totalApiDuration: 0, lastApiDuration: 0, tokensIn: 0, tokensOut: 0,
    averageResponseTime: 0, errorRate: 0, successRate: 100, cacheHits: 0, cacheMisses: 0,
  });

  /**
   * Creates a new session.
   * @param name - An optional name for the session.
   * @returns The newly created Session object.
   */
  async createNewSession(name?: string): Promise<Session> {
    const sessionId = uuidv4();
    const sessionName = name || `Session ${new Date().toLocaleTimeString()}`;
    const currentBranch = state.config.GIT_INTEGRATION.enabled ? await state.gitManager.getCurrentBranch() : undefined;

    const newSession: Session = {
      id: sessionId,
      name: sessionName,
      createdAt: new Date(),
      updatedAt: new Date(),
      chatHistory: [], // Initialize with empty history
      conversationLog: [], // Initialize with empty log
      performanceMetrics: this.getDefaultMetrics(),
      context: {}, // Initialize with empty context
      tasks: [], // Initialize with empty tasks
      gitBranch: currentBranch, // Store the branch at session creation
    };

    this.sessions.set(sessionId, newSession);
    state.activeSession = newSession; // Set as active session
    await this.saveSession(newSession); // Save the newly created session
    logger.info(`New session created: '${sessionName}'`);
    // TODO: Update UI/prompt to reflect the new active session.
    return newSession;
  }

  /**
   * Saves a given session to a file.
   * Handles encryption and compression based on configuration.
   * @param session - The session object to save.
   */
  async saveSession(session: Session) {
    session.updatedAt = new Date();
    // Ensure chat history and conversation log are up-to-date
    session.chatHistory = state.chat?.getHistory() || session.chatHistory;
    session.conversationLog = state.conversationLog || session.conversationLog;
    session.performanceMetrics = state.performanceMetrics || session.performanceMetrics;

    const sessionFile = path.join(SESSION_DIR, `${session.id}.json`);
    try {
      const jsonData = JSON.stringify(session);
      let dataToWrite = Buffer.from(jsonData);

      // Compress data if enabled
      if (state.config.HISTORY.COMPRESSION) {
        dataToWrite = await gzip(dataToWrite);
      }

      // Encrypt data if enabled
      const finalData = this.encryptionKey
        ? this.encryptData(dataToWrite.toString('base64'), this.encryptionKey)
        : dataToWrite.toString('base64'); // Store as base64 for consistency

      await fs.writeFile(sessionFile, finalData, 'utf-8');
      logger.debug(`Session '${session.name}' saved.`);
    } catch (error: any) {
      logger.error({ err: error, sessionId: session.id }, `Failed to save session '${session.name}'`);
    }
  }

  /**
   * Loads all session files from the SESSION_DIR.
   * Handles decryption and decompression.
   */
  private async loadAllSessions() {
    try {
      const files = await glob("*.json", { cwd: SESSION_DIR, absolute: true });
      for (const file of files) {
        const sessionData = await fs.readFile(file, 'utf-8');
        let decryptedData = sessionData;

        // Decrypt if encryption is enabled
        if (this.encryptionKey && state.config.ENCRYPTION.enabled) {
          try {
            decryptedData = this.decryptData(sessionData, this.encryptionKey);
          } catch (error) {
            logger.error({ err: error, file }, `Failed to decrypt session file. Skipping.`);
            continue; // Skip this file if decryption fails
          }
        }

        let decompressedData = Buffer.from(decryptedData, 'base64'); // Assume base64 encoding

        // Decompress if compression is enabled
        if (state.config.HISTORY.COMPRESSION) {
          try {
            decompressedData = await gunzip(decompressedData);
          } catch (error) {
            logger.error({ err: error, file }, `Failed to decompress session file. Skipping.`);
            continue; // Skip this file if decompression fails
          }
        }

        const session: Session = JSON.parse(decompressedData.toString('utf-8'));
        // Basic validation or data cleanup could be added here
        this.sessions.set(session.id, session);
        logger.debug(`Loaded session: '${session.name}'`);
      }
      // TODO: Implement logic to select an active session if multiple are loaded.
      if (this.sessions.size > 0 && !state.activeSession) {
        // Activate the most recently updated session as default if none is active
        const sortedSessions = Array.from(this.sessions.values()).sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime());
        state.activeSession = sortedSessions[0];
        logger.info(`Activated most recent session: '${state.activeSession.name}'`);
      }
    } catch (error: any) {
      logger.error(`Failed to load sessions: ${error.message}`);
    }
  }

  // Placeholder for encryption function
  private encryptData = (text: string, key: Buffer): string => {
    // Implement actual encryption logic here (e.g., using crypto module)
    // For now, returning a placeholder or the original text if not implemented
    logger.warn("Encryption function is a placeholder. Implement actual encryption.");
    return text; // Replace with actual encrypted string
  };

  // Placeholder for decryption function
  private decryptData = (encryptedText: string, key: Buffer): string => {
    // Implement actual decryption logic here
    logger.warn("Decryption function is a placeholder. Implement actual decryption.");
    return encryptedText; // Replace with actual decrypted string
  };

  /**
   * Exports a session's conversation history in a specified format.
   * @param sessionId - The ID of the session to export.
   * @param format - The desired export format ('json', 'md', 'html').
   */
  async exportSession(sessionId: string, format: 'json' | 'md' | 'html' = 'json'): Promise<string> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`Session with ID '${sessionId}' not found.`);
    }

    let exportedContent = "";
    const exportFilePath = path.join(HISTORY_EXPORT_DIR, `${session.name.replace(/\s+/g, '_')}_export.${format}`);

    switch (format) {
      case 'json':
        exportedContent = JSON.stringify(session.chatHistory, null, 2);
        break;
      case 'md':
        exportedContent = `# Conversation History: ${session.name}\n\n`;
        session.chatHistory.forEach(entry => {
          const role = entry.role === 'model' ? 'AI' : entry.role.toUpperCase();
          exportedContent += `## ${role} (${entry.timestamp.toLocaleString()})\n`;
          if (entry.parts && entry.parts.length > 0) {
            entry.parts.forEach(part => {
              if (part.text) {
                exportedContent += `${part.text}\n`;
              } else if (part.file) {
                exportedContent += `[File: ${part.file.fileName}]\n`;
              }
            });
          }
          exportedContent += "\n";
        });
        break;
      case 'html':
        exportedContent = `<!DOCTYPE html><html><head><title>Pyrmethus Session: ${session.name}</title></head><body>`;
        exportedContent += `<h1>Conversation History: ${session.name}</h1>`;
        session.chatHistory.forEach(entry => {
          const role = entry.role === 'model' ? 'AI' : entry.role.toUpperCase();
          exportedContent += `<h2>${role} (${entry.timestamp.toLocaleString()})</h2>`;
          if (entry.parts && entry.parts.length > 0) {
            entry.parts.forEach(part => {
              if (part.text) {
                exportedContent += `<p>${marked.parse(part.text)}</p>`; // Use marked for markdown rendering
              } else if (part.file) {
                exportedContent += `<p><strong>[File:</strong> ${part.file.fileName}]</p>`;
              }
            });
          }
        });
        exportedContent += '</body></html>';
        break;
      default:
        throw new Error(`Unsupported export format: ${format}`);
    }

    await fs.writeFile(exportFilePath, exportedContent, 'utf-8');
    logger.info(`Session exported to: ${exportFilePath}`);
    return exportFilePath;
  }

  /** Cleans up resources, such as the auto-save interval. */
  cleanup() {
    if (this.autoSaveInterval) {
      clearInterval(this.autoSaveInterval);
      logger.debug("Auto-save session interval cleared.");
    }
  }
}

// --- Enhanced Plugin Manager ---
class PluginManager extends EventEmitter {
  private plugins: Map<string, Plugin> = new Map();
  // Hooks allow other parts of the system to intercept or modify plugin behavior
  private hooks: Map<string, Function[]> = new Map();

  /**
   * Registers a hook handler for a specific event.
   * @param hookName - The name of the hook (e.g., 'before-message', 'after-command').
   * @param handler - The asynchronous function to execute when the hook is triggered.
   */
  registerHook(hookName: string, handler: Function) {
    if (!this.hooks.has(hookName)) {
      this.hooks.set(hookName, []);
    }
    this.hooks.get(hookName)!.push(handler);
    logger.debug(`Registered hook '${hookName}'`);
  }

  /**
   * Executes all registered handlers for a given hook, passing data through a chain.
   * @param hookName - The name of the hook to execute.
   * @param initialData - The initial data to pass to the first handler.
   * @param restArgs - Additional arguments to pass to the handlers.
   */
  async executeHook(hookName: string, initialData: any, ...restArgs: any[]): Promise<any> {
    let currentData = initialData;
    const handlers = this.hooks.get(hookName) || [];
    for (const handler of handlers) {
      // Pass the modified data and any additional arguments to the next handler
      currentData = await handler(currentData, ...restArgs);
    }
    return currentData;
  }

  /**
   * Loads plugins from a specified directory.
   * Plugins are expected to be JavaScript files that export a Plugin interface.
   * @param pluginDirectory - The path to the directory containing plugins.
   */
  async loadPlugins(pluginDirectory: string) {
    if (!state.config.PLUGINS.enabled) {
      logger.info("Plugin system is disabled.");
      return;
    }
    try {
      const files = await glob("*.{js,ts}", { cwd: pluginDirectory, absolute: true });
      for (const file of files) {
        try {
          // Dynamically import the plugin module
          const pluginModule = await import(file);
          // Assume plugins export a default object conforming to the Plugin interface
          const plugin: Plugin = pluginModule.default;
          if (plugin && plugin.name && plugin.activate) {
            await plugin.activate(state, this); // Pass state and plugin manager to activate
            this.plugins.set(plugin.name, plugin);
            logger.info(`Plugin loaded: ${plugin.name}`);
          } else {
            logger.warn(`Plugin file '${file}' does not export a valid plugin.`);
          }
        } catch (error: any) {
          logger.error({ err: error, file }, `Failed to load plugin '${file}'`);
        }
      }
    } catch (error: any) {
      logger.error(`Failed to scan plugin directory '${pluginDirectory}': ${error.message}`);
    }
  }
}

// --- Internationalization (i18n) Setup ---
/**
 * Configures the i18n library for multi-language support.
 * Loads locales from the specified directory and sets the default locale.
 */
function setupI18n() {
  i18n.configure({
    locales: ["en", "es", "fr", "de", "ja", "zh"], // Supported locales
    directory: LOCALE_DIR, // Directory containing locale JSON files
    defaultLocale: state.config.LANGUAGE || "en", // Default to English if not specified
    autoReload: true, // Automatically reload locale files if they change
    syncFiles: true, // Synchronize locale files
    objectNotation: true, // Allow nested keys in locale files
  });
  i18n.setLocale(state.config.LANGUAGE || "en"); // Set the active locale
  logger.info(`i18n locale set to: ${i18n.getLocale()}`);
}

// --- UI and File Handling Utilities ---

/**
 * Highlights code blocks within a given text using highlight.js.
 * Supports language detection and theme application.
 * @param text - The text containing potential code blocks.
 * @returns The text with highlighted code blocks.
 */
function highlightCode(text: string): string {
  const codeBlockRegex = /```(?<lang>\w+)?\n([\s\S]*?)```/g;
  let highlightedText = "";
  let lastIndex = 0;

  text.replace(codeBlockRegex, (match, langMatch, codeContent, offset) => {
    const language = langMatch || "plaintext"; // Default to plaintext if no language specified
    highlightedText += text.substring(lastIndex, offset); // Append text before the code block

    try {
      // Highlight the code content
      const result = hljs.highlight(codeContent, { language, ignoreIllegals: true });
      highlightedText += state.theme.code(result.value); // Apply theme's code color
    } catch (e) {
      // Fallback to auto-detection if specific language highlighting fails
      const autoResult = hljs.highlightAuto(codeContent);
      highlightedText += state.theme.code(autoResult.value);
    }
    lastIndex = offset + match.length; // Update the index to the end of the current match
    return match; // Keep the original match for replace to work correctly
  });
  highlightedText += text.substring(lastIndex); // Append any remaining text after the last code block
  return highlightedText;
}

/**
 * Reads the content of a file, handling various file types and size limits.
 * Supports text-based files, PDFs, DOCX, XLSX, and images.
 * @param filePath - The path to the file to read.
 * @returns A promise that resolves with the file's content, MIME type, extension, and buffer.
 */
async function readFileContent(filePath: string): Promise<{ content: string; mimeType: string | null; extension: string; buffer: Buffer }> {
  const absolutePath = path.resolve(filePath);
  const stats = await fs.stat(absolutePath);

  // Check file size against configuration limit
  if (stats.size > state.config.MAX_FILE_SIZE) {
    throw new Error(`File size exceeds limit (${(state.config.MAX_FILE_SIZE / (1024 * 1024)).toFixed(1)} MB): ${filePath}`);
  }

  const buffer = await fs.readFile(absolutePath);
  const fileTypeResult = await fileTypeFromBuffer(buffer);
  const mimeType = fileTypeResult?.mime || null;
  const extension = path.extname(absolutePath).toLowerCase().substring(1);

  let content = "";

  // Check if the file type is supported for content extraction
  const isSupportedMime = mimeType && state.config.FILE_SUPPORT.mimeTypes.some(m => mimeType.startsWith(m.replace('*', '')) || mimeType === m);
  const isSupportedExtension = state.config.FILE_SUPPORT.extensions.includes(extension);

  if (!isSupportedMime && !isSupportedExtension) {
    throw new Error(`Unsupported file extension or MIME type: ${filePath}`);
  }

  // Extract content based on MIME type
  if (mimeType?.startsWith('text/') || ['json', 'xml', 'yaml', 'yml', 'csv', 'log', 'sh', 'sql', 'js', 'ts', 'py', 'go', 'rs', 'html'].includes(extension)) {
    content = buffer.toString('utf-8');
  } else if (mimeType === 'application/pdf') {
    content = await new Promise<string>((resolve, reject) => {
      const pdfParser = new PDFParser(); // Use imported class
      pdfParser.on("pdfParser_dataReady", (pdfData: any) => {
        const text = pdfData.Pages.map((page: any) =>
          page.Texts.map((t: any) => decodeURIComponent(t.R[0]?.T || "")).join(" ") // Safely access text
        ).join("\n");
        resolve(text);
      });
      pdfParser.on("error", (error: any) => reject(new Error(`PDF processing failed: ${error.message}`)));
      pdfParser.parseBuffer(buffer);
    });
  } else if (mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
    const result = await mammoth.extractRawText({ buffer: buffer });
    content = result.value;
  } else if (mimeType === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') {
    const workbook = XLSX.read(buffer, { type: 'buffer' });
    content = workbook.SheetNames.map(sheetName => {
      const sheet = workbook.Sheets[sheetName];
      return XLSX.utils.sheet_to_csv(sheet);
    }).join('\n\n--- New Sheet ---\n\n');
  } else if (mimeType?.startsWith('image/')) {
    // For images, content extraction isn't typical; the buffer will be used for vision models.
    // Return a placeholder string for content.
    content = `[Image: ${path.basename(absolutePath)}]`;
  } else {
    // If it's a supported extension but not a recognized MIME type for extraction
    if (isSupportedExtension) {
      content = buffer.toString('utf-8'); // Attempt to read as text
    } else {
      throw new Error(`Unsupported MIME type for content extraction: ${mimeType}`);
    }
  }

  return { content, mimeType, extension, buffer };
}

// --- Enhanced Gemini Interaction ---
/**
 * Sends a message to the Gemini API and handles the streaming response.
 * Includes pre-processing hooks, context injection, and post-processing.
 * @param prompt - The user's prompt, which can be a string or a structured Content object.
 */
async function sendMessageToGemini(prompt: string | Content) {
  if (!state.chat) {
    logger.error("Chat session is not initialized.");
    throw new Error("Chat session not initialized.");
  }
  await state.rateLimiter.checkLimit(); // Check rate limits before sending

  const startTime = performance.now();
  state.isProcessing = true; // Set processing flag
  // Initialize spinner with theme-appropriate frames and interval
  state.spinner = ora({
    text: state.theme.primary("Thinking..."),
    spinner: { interval: state.config.UI.SPINNER_INTERVAL, frames: state.config.UI.SPINNER_FRAMES },
    color: state.theme.primary.name, // Use theme's primary color for spinner
  }).start();

  // Execute 'before-message' hook to allow plugins to modify the prompt
  let processedPrompt = await state.pluginManager.executeHook('before-message', prompt);

  // Inject context files into the prompt if context awareness is enabled
  let contextHeader = "";
  if (state.config.AI_FEATURES.contextAwareness && state.contextFiles.size > 0) {
      contextHeader += "Context from files:\n" +
          Array.from(state.contextFiles).map(f => `- ${path.basename(f)}`).join('\n') + "\n\n";
  }

  // Prepend context header if the prompt is a string
  if (typeof processedPrompt === 'string') {
      processedPrompt = contextHeader + processedPrompt;
  } else if (Array.isArray(processedPrompt) && processedPrompt.length > 0 && processedPrompt[0].text) {
      // Prepend context header to the first text part of a Content object
      processedPrompt[0].text = contextHeader + processedPrompt[0].text;
  }

  try {
    // Send the message stream to the Gemini API
    const stream = await state.chat.sendMessageStream(processedPrompt);
    let responseText = "";

    // Process each chunk of the streaming response
    for await (const chunk of stream.stream) {
      const text = chunk.text();
      responseText += text;
      process.stdout.write(highlightCode(text)); // Highlight and print the received text chunk
    }
    console.log(); // Newline after the streaming response

    state.lastResponseText = responseText; // Store the complete response text
    // Execute 'after-message' hook to process the response
    await state.pluginManager.executeHook('after-message', responseText);

    // Update performance metrics
    const endTime = performance.now();
    const duration = endTime - startTime;
    state.performanceMetrics.apiCalls++;
    state.performanceMetrics.totalApiDuration += duration;
    state.performanceMetrics.lastApiDuration = duration;
    state.performanceMetrics.averageResponseTime = state.performanceMetrics.totalApiDuration / state.performanceMetrics.apiCalls;
    // TODO: Update token counts and success/error rates

    state.spinner.succeed(state.theme.success("Response received.")); // Indicate success
  } catch (error: any) {
    logger.error({ err: error, prompt }, "Error sending message to Gemini API");
    // Update error rate metric
    state.performanceMetrics.errorRate++;
    state.spinner.fail(state.theme.error(`Error: ${error.message}`)); // Indicate failure
  } finally {
    state.isProcessing = false; // Reset processing flag
    state.spinner?.stop(); // Ensure spinner is stopped
  }
}

// --- Command System ---
/**
 * Defines the structure for a command handler.
 */
type CommandHandler = {
  description: string; // A brief description of the command.
  action: (args: string[]) => Promise<void>; // The function to execute when the command is called.
  help?: string; // Optional detailed help text for the command.
  category?: string; // Category for organizing commands (e.g., "System", "Git").
};
// Map to store registered commands, keyed by their command name (e.g., "/help").
const commands: Map<string, CommandHandler> = new Map();

/**
 * Registers a new command with its handler and optional help text.
 * Also registers aliases defined in the configuration.
 * @param name - The name of the command (without the leading '/').
 * @param description - A brief description of the command.
 * @param action - The function to execute when the command is called.
 * @param help - Optional detailed help text.
 * @param category - Optional category for the command.
 */
function registerCommand(name: string, description: string, action: (args: string[]) => Promise<void>, help?: string, category: string = "General") {
  const commandName = `/${name.toLowerCase()}`; // Ensure command names are lowercase and prefixed
  commands.set(commandName, { description, action, help, category });
  logger.debug(`Registered command: ${commandName}`);

  // Register aliases if they exist in the configuration
  const alias = Object.keys(state.config.ALIASES).find(key => state.config.ALIASES[key] === name);
  if (alias) {
    const aliasName = `/${alias.toLowerCase()}`;
    commands.set(aliasName, { description: `Alias for ${commandName}`, action, help, category });
    logger.debug(`Registered alias: ${aliasName} -> ${commandName}`);
  }
}

/**
 * Loads custom commands from the COMMANDS_DIR.
 * Custom commands are expected to be Markdown files where the filename is the command name.
 * The content of the Markdown file can be used as a prompt template.
 */
async function loadCustomCommands() {
  if (!fsSync.existsSync(COMMANDS_DIR)) {
    logger.debug(`Custom commands directory not found: ${COMMANDS_DIR}`);
    return;
  }
  try {
    const files = await fs.readdir(COMMANDS_DIR);
    for (const file of files) {
      if (file.endsWith('.md')) {
        const commandName = file.replace('.md', '');
        const filePath = path.join(COMMANDS_DIR, file);
        const content = await fs.readFile(filePath, 'utf-8');

        // Register the custom command
        registerCommand(commandName, `Custom command: ${commandName}`, async (args: string[]) => {
          // Replace placeholders like {{1}}, {{2}} in the prompt with arguments
          const promptTemplate = content.replace(/\{\{(\d+)\}\}/g, (_, index) => {
            const argIndex = parseInt(index) - 1; // Adjust for 0-based array index
            return args[argIndex] !== undefined ? args[argIndex] : '';
          });
          await sendMessageToGemini(promptTemplate);
        }, content, "Custom"); // Assign to "Custom" category
      }
    }
    logger.info(`Loaded ${commands.size} custom commands.`);
  } catch (error: any) {
    logger.error(`Failed to load custom commands: ${error.message}`);
  }
}

// --- Enhanced Command Implementations ---

/**
 * Executes a shell command using the `exec` utility.
 * Handles stdout and stderr output, and logs execution errors.
 * @param args - An array of strings representing the command and its arguments.
 */
async function handleExec(args: string[]) {
  if (args.length === 0) {
    console.log(state.theme.warning("Usage: /exec <command> [args...]"));
    return;
  }
  const command = args.join(" ");
  console.log(state.theme.info(`Executing: ${command}`));
  try {
    const { stdout, stderr } = await exec(command);
    if (stdout) {
      console.log(state.theme.success("STDOUT:"));
      console.log(stdout);
    }
    if (stderr) {
      console.log(state.theme.error("STDERR:"));
      console.error(stderr);
    }
  } catch (error: any) {
    console.error(state.theme.error(`Execution failed: ${error.message}`));
    logger.error({ err: error, command }, "Shell command execution failed");
  }
}

/**
 * Displays a list of available commands, categorized for better usability.
 * @param args - Unused arguments.
 */
async function handleHelp(args: string[]) {
  const categories = new Map<string, CommandHandler[]>();
  commands.forEach(handler => {
    const category = handler.category || "General";
    if (!categories.has(category)) {
      categories.set(category, []);
    }
    categories.get(category)!.push(handler);
  });

  console.log(state.theme.boxHeader("\nAvailable Commands:"));
  categories.forEach((handlers, category) => {
    console.log(state.theme.primary(`\n--- ${category} ---`));
    handlers.forEach(handler => {
      console.log(`${state.theme.prompt(handler.description.split('\n')[0])} ${state.theme.dim(handler.help?.split('\n')[0] || '')}`);
    });
  });
  console.log("\n");
}

/**
 * Handles Git-related operations using the GitManager.
 * @param args - Arguments for Git operations (e.g., "status", "log").
 */
async function handleGit(args: string[]) {
  if (!state.config.GIT_INTEGRATION.enabled) {
    console.log(state.theme.warning("Git integration is disabled in the configuration."));
    return;
  }
  const subCommand = args[0]?.toLowerCase();
  switch (subCommand) {
    case "status":
      const status = await state.gitManager.getStatus();
      console.log(state.theme.info("Git Status:"));
      console.log(`  Branch: ${status.current}`);
      console.log(`  Changes to commit: ${status.not_added.length + status.conflicted.length + status.created.length + status.deleted.length + status.modified.length}`);
      // Add more detailed status output if needed
      break;
    case "log":
      const log = await state.gitManager.getLog(parseInt(args[1] || '10'));
      console.log(state.theme.info("Git Log (last 10 commits):"));
      log.all.forEach(commit => {
        console.log(`  ${state.theme.primary(commit.hash.substring(0, 7))}: ${commit.message.split('\n')[0]} - ${state.theme.dim(commit.author.name)}`);
      });
      break;
    case "diff":
      const diff = await state.gitManager.getDiff();
      console.log(state.theme.info("Git Diff:"));
      console.log(diff);
      break;
    case "commit":
      const commitMessage = args.slice(1).join(" ");
      if (!commitMessage) {
        console.log(state.theme.warning("Usage: /git commit <message>"));
        return;
      }
      await state.gitManager.createCommit(commitMessage);
      console.log(state.theme.success("Commit created successfully."));
      break;
    default:
      console.log(state.theme.warning("Usage: /git [status|log|diff|commit <message>]"));
  }
}

/**
 * Manages tasks using the TaskManager.
 * @param args - Arguments for task operations (e.g., "list", "create <desc>", "update <id> <status>").
 */
async function handleTask(args: string[]) {
  const subCommand = args[0]?.toLowerCase();
  switch (subCommand) {
    case "list":
    case "ls":
      const tasks = state.taskManager.getAllTasks();
      if (tasks.length === 0) {
        console.log(state.theme.dim("No tasks found."));
        return;
      }
      console.log(state.theme.boxHeader("\nTasks:"));
      tasks.forEach(task => {
        console.log(`  ${state.theme.primary(task.id.substring(0, 6))} ${state.theme.accent(task.description)} - ${state.theme.dim(task.status)}`);
      });
      break;
    case "create":
    case "add":
      const description = args.slice(1).join(" ");
      if (!description) {
        console.log(state.theme.warning("Usage: /task create <description>"));
        return;
      }
      const newTask = state.taskManager.createTask(description);
      console.log(state.theme.success(`Task created: ${newTask.description} (ID: ${newTask.id.substring(0, 6)})`));
      break;
    case "update":
    case "set":
      const taskId = args[1];
      const newStatus = args[2]?.toLowerCase();
      if (!taskId || !newStatus) {
        console.log(state.theme.warning("Usage: /task update <task_id> <status>"));
        return;
      }
      state.taskManager.updateTask(taskId, { status: newStatus as TaskStatus });
      console.log(state.theme.success(`Task '${taskId.substring(0, 6)}' updated to status '${newStatus}'.`));
      break;
    default:
      console.log(state.theme.warning("Usage: /task [list|create <description>|update <task_id> <status>]"));
  }
}

/**
 * Manages context files for the AI.
 * @param args - Arguments for context operations (e.g., "add <file>", "list", "remove <file>").
 */
async function handleContext(args: string[]) {
  const subCommand = args[0]?.toLowerCase();
  switch (subCommand) {
    case "add":
      const filePathToAdd = args[1];
      if (!filePathToAdd) {
        console.log(state.theme.warning("Usage: /context add <file_path>"));
        return;
      }
      const absolutePath = path.resolve(filePathToAdd);
      if (fsSync.existsSync(absolutePath)) {
        state.contextFiles.add(absolutePath);
        console.log(state.theme.success(`Added to context: ${path.basename(absolutePath)}`));
      } else {
        console.log(state.theme.error(`File not found: ${absolutePath}`));
      }
      break;
    case "list":
    case "ls":
      if (state.contextFiles.size === 0) {
        console.log(state.theme.dim("No files in context."));
        return;
      }
      console.log(state.theme.boxHeader("\nContext Files:"));
      state.contextFiles.forEach(file => {
        console.log(`  ${state.theme.primary(path.basename(file))}`);
      });
      break;
    case "remove":
    case "rm":
      const filePathToRemove = args[1];
      if (!filePathToRemove) {
        console.log(state.theme.warning("Usage: /context remove <file_path>"));
        return;
      }
      const absolutePathToRemove = path.resolve(filePathToRemove);
      if (state.contextFiles.has(absolutePathToRemove)) {
        state.contextFiles.delete(absolutePathToRemove);
        console.log(state.theme.success(`Removed from context: ${path.basename(absolutePathToRemove)}`));
      } else {
        console.log(state.theme.warning(`File not found in context: ${path.basename(absolutePathToRemove)}`));
      }
      break;
    case "clear":
      state.contextFiles.clear();
      console.log(state.theme.success("Context cleared."));
      break;
    default:
      console.log(state.theme.warning("Usage: /context [add <file_path>|list|remove <file_path>|clear]"));
  }
}

/**
 * Performs a web search using a hypothetical `google_web_search` function.
 * This function would likely involve calling a tool or API.
 * @param args - The search query.
 */
async function handleSearch(args: string[]) {
  if (args.length === 0) {
    console.log(state.theme.warning("Usage: /search <query>"));
    return;
  }
  const query = args.join(" ");
  console.log(state.theme.info(`Searching the web for: ${query}`));
  try {
    // Placeholder for actual web search function
    // const searchResult = await google_web_search(query);
    // if (searchResult.google_web_search_response && searchResult.google_web_search_response.output) {
    //   console.log(state.theme.success("Search Results:"));
    //   console.log(searchResult.google_web_search_response.output);
    // } else {
    //   console.log(state.theme.warning("No search results found."));
    // }
    console.log(state.theme.warning("Web search functionality is not yet implemented."));
  } catch (error: any) {
    console.error(state.theme.error(`Search failed: ${error.message}`));
    logger.error({ err: error, query }, "Web search failed");
  }
}

// --- Main Application Orchestration ---

/**
 * Initializes the Pyrmethus application.
 * This includes setting up configuration, core systems, AI models, managers, UI, and commands.
 */
async function initialize() {
  // 1. Configuration Management
  state.configManager = new ConfigManager();
  await state.configManager.init();
  state.config = state.configManager.config; // Assign loaded config to global state

  // Parse and apply command-line flags that might override config
  const cliArgs = process.argv.slice(2);
  for (let i = 0; i < cliArgs.length; i++) {
    switch (cliArgs[i]) {
      case "--no-autosave":
        state.config.AUTO_SAVE_SESSION = false;
        logger.info("Auto-save session disabled via CLI flag.");
        break;
      case "--compact-mode":
        state.config.UI.COMPACT_MODE = true;
        logger.info("Compact mode enabled via CLI flag.");
        break;
      case "--no-animations":
        state.config.UI.ENABLE_ANIMATIONS = false;
        logger.info("Animations disabled via CLI flag.");
        break;
      // Add more CLI flags for configuration overrides here
    }
  }

  // Setup logger based on the final configuration
  setupLogger(state.config.LOGGING);

  // 2. Initialize Core Systems
  state.theme = defaultTheme; // Initialize with default theme
  state.configManager.applyTheme(state.config.UI.THEME); // Apply theme from config
  setupI18n(); // Setup internationalization
  state.cache = new NodeCache({ stdTTL: state.config.CACHE_TTL }); // Initialize cache
  state.rateLimiter = new RateLimiter(state.config.RATE_LIMITING); // Initialize rate limiter
  state.taskManager = new TaskManager(); // Initialize task manager
  state.contextFiles = new Set<string>(); // Initialize context file set

  // 3. Initialize AI and API
  if (!state.config.GOOGLE_API_KEY) {
    logger.fatal("GOOGLE_API_KEY is not set in configuration or environment variables.");
    throw new Error("GOOGLE_API_KEY not set. Please configure it.");
  }
  state.geminiApi = new GoogleGenerativeAI(state.config.GOOGLE_API_KEY);
  state.model = state.geminiApi.getGenerativeModel(state.config.MODEL_CONFIG);

  // 4. Initialize Managers
  state.gitManager = new GitManager();
  state.sessionManager = new SessionManager();
  await state.sessionManager.init(); // Initialize session manager (loads sessions, sets up auto-save)
  state.pluginManager = new PluginManager();
  if (state.config.PLUGINS.enabled) {
    await state.pluginManager.loadPlugins(state.config.PLUGINS.directory); // Load plugins if enabled
  }

  // 5. Initialize UI Components
  state.readlineInterface = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: `${state.theme.prompt(state.config.UI.PROMPT_SYMBOL + " ")}`, // Set initial prompt
  });
  // Register custom inquirer prompts
  inquirer.registerPrompt('autocomplete', autocomplete);

  // 6. Register Commands
  // Register built-in commands
  registerCommand("exec", "Execute a shell command", handleExec, "Usage: /exec <command> [args...]\nExecutes the given shell command and prints its output.", "System");
  registerCommand("search", "Perform a web search", handleSearch, "Usage: /search <query>\nPerforms a web search and displays the results.", "System");
  registerCommand("help", "Show this help message", handleHelp, "Lists all available commands and their descriptions.", "General");
  registerCommand("git", "Interact with Git repository", handleGit, "Usage: /git [status|log|diff|commit <message>]\nManages Git operations.", "Git");
  registerCommand("task", "Manage tasks", handleTask, "Usage: /task [list|create <description>|update <task_id> <status>]\nManages your to-do tasks.", "Productivity");
  registerCommand("context", "Manage AI context files", handleContext, "Usage: /context [add <file>|list|remove <file>|clear]\nManages files to be included in AI context.", "AI");
  registerCommand("clear", "Clear the terminal screen", async () => console.clear(), "Clears the terminal screen.", "General");
  registerCommand("quit", "Exit Pyrmethus", async () => gracefulShutdown(), "Exits the application.", "General");
  registerCommand("version", "Show Pyrmethus version", async () => console.log(`Pyrmethus v${state.config.VERSION}`), "Displays the current version.", "General");
  // ... (register all other built-in commands)

  // Load and register custom commands from the file system
  await loadCustomCommands();

  // 7. Display Welcome Message
  console.clear();
  // Use chalk-animation for a dynamic welcome message
  const rainbow = chalkAnimation.rainbow('Welcome to Pyrmethus v3.0.0!');
  rainbow.start();
  await new Promise(resolve => setTimeout(resolve, 2000)); // Keep animation for 2 seconds
  rainbow.stop();
  console.log(`\n${state.theme.primary("Type '/help' for a list of commands.")}\n`);

  state.initialized = true; // Mark initialization as complete
  logger.info("Pyrmethus v3.0.0 initialized successfully.");
}

/**
 * Handles user input from the prompt.
 * Determines if the input is a command or a message to the AI.
 * @param input - The raw input string from the user.
 */
async function handleInput(input: string) {
  if (state.isProcessing) {
    console.log(state.theme.warning("Please wait, Pyrmethus is currently processing a request."));
    return;
  }
  const trimmedInput = input.trim();
  if (!trimmedInput) return; // Ignore empty input

  // Log user input for history and debugging
  logger.info({ input: trimmedInput }, "User input");

  const [command, ...args] = trimmedInput.split(' ');
  const handler = commands.get(command.toLowerCase());

  if (handler) {
    // If the input is a registered command
    try {
      await handler.action(args); // Execute the command's action
    } catch (error: any) {
      logger.error({ err: error, command: command }, `Command ${command} failed`);
      console.error(state.theme.error(`Error executing command ${command}: ${error.message}`));
    }
  } else {
    // If the input is not a command, treat it as a message to the AI
    await sendMessageToGemini(trimmedInput);
  }
}

/**
 * Prompts the user for input using an autocomplete prompt.
 * Filters commands based on user input using fuzzy matching.
 */
async function promptUser() {
  try {
    const answers = await inquirer.prompt([{
      type: 'autocomplete',
      name: 'commandInput',
      message: `${state.theme.prompt(state.config.UI.PROMPT_SYMBOL)} `, // Use theme for prompt symbol
      source: async (_: any, input: string) => {
        const commandNames = Array.from(commands.keys());
        if (!input) return commandNames; // Show all commands if input is empty
        // Use fuzzy search to filter commands based on user input
        const fuzzyResult = fuzzy.filter(input, commandNames);
        return fuzzyResult.map(el => el.string); // Return matching command names
      },
      // Optional: Add validation or suggestions here
    }]);
    await handleInput(answers.commandInput); // Process the user's input
  } catch (error) {
    // Handle errors during the prompt process (e.g., user interruption)
    if (error.isTtyError) {
      // Prompt couldn't be rendered in the current environment
      console.error(state.theme.error("Prompt couldn't be rendered in the current environment."));
    } else {
      logger.error({ err: error }, "Error during user prompt");
      console.error(state.theme.error(`An error occurred during input: ${error.message}`));
    }
  }
}

/**
 * Handles graceful shutdown of the application.
 * Saves the current session and exits cleanly.
 */
async function gracefulShutdown() {
  console.log(state.theme.secondary("\nShutting down Pyrmethus..."));
  if (state.configManager) {
    state.configManager.cleanup(); // Clean up config watcher
  }
  if (state.sessionManager) {
    if (state.activeSession) {
      await state.sessionManager.saveSession(state.activeSession); // Save active session
    }
    state.sessionManager.cleanup(); // Clean up session manager resources
  }
  if (state.readlineInterface) {
    state.readlineInterface.close(); // Close readline interface
  }
  console.log(state.theme.dim("Session saved. Goodbye!"));
  process.exit(0); // Exit the process
}

/**
 * The main application loop.
 * Initializes the application and then continuously prompts the user for input.
 * Also handles execution of single commands passed via CLI arguments.
 */
async function run() {
  try {
    await initialize(); // Perform all necessary initializations

    const cliArgs = process.argv.slice(2); // Get command-line arguments
    let commandExecuted = false; // Flag to track if a CLI command was executed

    // Handle single commands passed as CLI arguments (e.g., `pyrmethus --exec "ls -l"`)
    if (cliArgs.length > 0) {
      if (cliArgs[0] === "--exec") {
        const commandToExec = cliArgs.slice(1).join(" ");
        if (commandToExec) {
          await handleExec([commandToExec]); // Execute the command
          commandExecuted = true;
        } else {
          console.log(state.theme.warning("Usage: --exec <command>"));
        }
      } else if (cliArgs[0] === "--search") {
        const searchQuery = cliArgs.slice(1).join(" ");
        if (searchQuery) {
          await handleSearch([searchQuery]); // Execute search
          commandExecuted = true;
        } else {
          console.log(state.theme.warning("Usage: --search <query>"));
        }
      }
      // Add more CLI command handlers here if needed
    }

    // If a CLI command was executed, exit the application after execution
    if (commandExecuted) {
      process.exit(0);
    }

    // If no CLI command was executed, start the interactive prompt loop
    console.log(`\nActive session: ${state.theme.primary(state.activeSession?.name || "No active session")}`);
    while (true) {
      await promptUser(); // Prompt the user for input
    }
  } catch (error: any) {
    // Catch any critical errors during initialization or runtime
    logger.fatal({ err: error }, "Critical initialization or runtime error");
    console.error(chalk.redBright(`\nCritical Error: ${error.message}`));
    process.exit(1); // Exit with an error code
  }
}

// --- Signal and Error Handling ---
// Handle SIGINT (Ctrl+C) for graceful shutdown
process.on('SIGINT', gracefulShutdown);

// Handle uncaught exceptions to prevent crashes and log errors
process.on('uncaughtException', (error) => {
  logger.fatal({ err: error }, "Uncaught Exception");
  console.error(chalk.redBright(`\nAn unexpected error occurred: ${error.message}`));
  gracefulShutdown(); // Attempt graceful shutdown
});

// Handle unhandled promise rejections to log potential issues
process.on('unhandledRejection', (reason) => {
  logger.error({ reason }, "Unhandled Rejection");
});

// --- Start the Application ---
run();
```
