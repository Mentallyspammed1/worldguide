#!/usr/bin/env node

import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from '@google/generative-ai';
import dotenv from 'dotenv';
import fs from 'fs/promises';
import path from 'path';
import readline from 'readline';
import chalk from 'chalk';
import mime from 'mime-types';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { highlight } from 'cli-highlight';
import ora from 'ora';
import { exec, spawn } from 'child_process';
import util from 'util';
import os from 'os';

const execPromise = util.promisify(exec);

// --- Application Constants ---
const APP_NAME = 'NeonCLI';
const APP_VERSION = '2.0.0';
const DEFAULT_CONFIG_FILE = './neon_config.json';
const DEFAULT_HISTORY_FILE = './gemini_chat_history.json';
const DEFAULT_MACROS_FILE = './neon_macros.json';
const DEFAULT_SESSIONS_DIR = './sessions';
const DEFAULT_MODEL = 'gemini-1.5-pro-latest';
const DEFAULT_TEMP = 0.8;
const DEFAULT_MAX_HISTORY = 50;
const DEFAULT_SAFETY = 'BLOCK_NONE';
const ROLE_USER = 'user';
const ROLE_MODEL = 'model';
const CMD_PREFIX = '/';
const MACRO_PREFIX = '!';
const VALID_MACRO_NAME_REGEX = /^[a-zA-Z0-9_-]+$/;
const TEMP_FILE_PREFIX = 'neoncli-edit-';
const TOKEN_WARNING_THRESHOLD = 7000;
const KNOWN_MODELS = [
    'gemini-1.0-pro', 'gemini-1.0-pro-latest', 'gemini-pro',
    'gemini-1.5-flash', 'gemini-1.5-flash-latest',
    'gemini-1.5-pro', 'gemini-1.5-pro-latest',
];

// --- Default System Prompt ---
const DEFAULT_SYSTEM_PROMPT_TEXT = `You are ${APP_NAME} (v${APP_VERSION}), an advanced AI assistant running in a command-line interface.
You are running on model: {{MODEL_NAME}}.
Today's date is {{CURRENT_DATE}}.

**Capabilities & Behavior:**
- Use Markdown formatting extensively for clarity (code blocks, lists, bolding, etc.).
- **Specify the language** in code blocks (e.g., \`\`\`python). Assume 'bash' for shell scripts if unsure.
- Be concise unless verbosity is explicitly requested or necessary for detail.
- Ask clarifying questions if a request is ambiguous.
- If you need to reference previous parts of the conversation, do so clearly.
- Inform the user if you cannot fulfill a request and explain why (e.g., lack of real-time data, ethical boundaries).

**Code Execution (If Enabled by User):**
You can request shell or Python code execution. The user MUST confirm each request.
Structure your request within a **single** JSON code block:
- **Shell:** \`\`\`json\n{ "action": "run_shell", "command": "your_shell_command", "reason": "Explain why you need to run this." }\n\`\`\`
- **Python:** \`\`\`json\n{ "action": "run_python", "code": "your_python_code", "reason": "Explain why you need to run this." }\n\`\`\`
- **Reasoning is mandatory.** Be specific about the goal.
- Keep code focused on the immediate task. Do not attempt complex multi-step operations in one request.
- If execution is disabled or the user denies permission, state that you cannot proceed with the execution step and offer alternatives if possible.
- You will receive feedback (stdout, stderr, exit code) after execution, or a cancellation message. Use this feedback to inform your next response.`;


// --- Neon Sigils (Chalk Theme) ---
const neon = {
    userPrompt: chalk.cyanBright.bold, aiResponse: chalk.whiteBright, aiCodeBlock: chalk.white,
    systemInfo: chalk.blueBright.bold, commandHelp: chalk.greenBright, filePath: chalk.magentaBright,
    warning: chalk.yellowBright.bold, error: chalk.redBright.bold.inverse, debug: chalk.gray.dim,
    promptMarker: chalk.cyanBright.bold("❯ "), aiMarker: chalk.greenBright.bold("AI "),
    pasteMarker: chalk.yellowBright.bold("Paste> "), sysMarker: chalk.blueBright.bold("[System] "),
    errorMarker: chalk.redBright.bold.inverse("[Error]"), warnMarker: chalk.yellowBright.bold("[Warning] "),
    shellMarker: chalk.blue.bold("[Shell] "), pythonMarker: chalk.blue.bold("[Python] "),
    macroMarker: chalk.magentaBright.bold("[Macro] "), shellCommand: chalk.yellow, pythonCode: chalk.yellow,
    shellOutput: chalk.white, pythonOutput: chalk.white, macroName: chalk.magenta,
    macroContent: chalk.whiteBright, spinnerColor: 'cyan', thinkingText: 'Synthesizing...',
    searchHighlight: chalk.black.bgYellowBright, configKey: chalk.blue, configValue: chalk.whiteBright,
    tokenCount: chalk.yellowBright, separator: () => console.log(chalk.gray('─'.repeat(process.stdout.columns || 70))),
    editedMarker: chalk.yellow.dim('(edited) '), configDesc: chalk.gray.italic,
    statusBusy: chalk.redBright.bold('[Thinking] '), statusIdle: chalk.greenBright.bold('[Ready] '),
    executionRequest: chalk.yellow,
    confirmPrompt: chalk.yellowBright.bold,
};

// --- Safety Map ---
const SAFETY_MAP = {
    BLOCK_NONE: HarmBlockThreshold.BLOCK_NONE,
    BLOCK_LOW_AND_ABOVE: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    BLOCK_MEDIUM_AND_ABOVE: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    BLOCK_ONLY_HIGH: HarmBlockThreshold.BLOCK_ONLY_HIGH
};

// --- Configuration Loading ---
dotenv.config();
const argv = yargs(hideBin(process.argv))
    .option('api-key', { alias: 'k', type: 'string', description: 'Google Generative AI API Key (or use GEMINI_API_KEY env var)' })
    .option('model', { alias: 'm', type: 'string', description: `Model name (Default: ${DEFAULT_MODEL})` })
    .option('temperature', { alias: 't', type: 'number', description: `Temperature (0.0-2.0, Default: ${DEFAULT_TEMP})` })
    .option('config-file', { alias: 'cfg', type: 'string', default: DEFAULT_CONFIG_FILE, description: 'Path to the JSON configuration file' })
    .option('history-file', { alias: 'h', type: 'string', default: DEFAULT_HISTORY_FILE, description: 'Path to the chat history JSON file' })
    .option('macros-file', { type: 'string', default: DEFAULT_MACROS_FILE, description: 'Path to the macros JSON file' })
    .option('sessions-dir', { type: 'string', default: DEFAULT_SESSIONS_DIR, description: 'Directory to store chat sessions' })
    .option('safety', { alias: 's', type: 'string', choices: Object.keys(SAFETY_MAP), default: DEFAULT_SAFETY, description: `Safety threshold` })
    .option('max-history', { type: 'number', description: `Max history turns (pairs) to keep (Default: ${DEFAULT_MAX_HISTORY})` })
    .option('highlight', { type: 'boolean', default: true, description: 'Enable syntax highlighting for AI responses' })
    .option('allow-shell', { type: 'boolean', default: false, description: 'Allow AI to request shell command execution (requires confirmation)' })
    .option('shell', { type: 'string', default: process.platform === 'win32' ? 'powershell.exe' : '/bin/sh', description: 'Shell executable path for execution' })
    .option('allow-python', { type: 'boolean', default: false, description: 'Allow AI to request Python code execution (requires confirmation)' })
    .option('python-path', { type: 'string', default: process.env.TERMUX_VERSION ? 'python' : 'python3', description: 'Python executable path for execution' })
    .option('sandbox', { type: 'boolean', default: false, description: 'Attempt to sandbox shell/Python execution (experimental)' })
    .option('debug', { type: 'boolean', default: false, description: 'Enable detailed debug logging' })
    .version(APP_VERSION)
    .help().alias('help', 'H').alias('version', 'v')
    .argv;


// --- Configuration Manager ---
class ConfigManager {
    constructor(filePath) {
        this.filePath = path.resolve(filePath);
        this.config = {};
        this.initialLoadComplete = false;
    }

    getDefaults() {
        return {
            apiKey: null,
            modelName: DEFAULT_MODEL,
            temperature: DEFAULT_TEMP,
            safety: DEFAULT_SAFETY,
            maxHistory: DEFAULT_MAX_HISTORY,
            highlight: true,
            debug: false,
            allowShell: false,
            allowPython: false,
            systemPrompt: DEFAULT_SYSTEM_PROMPT_TEXT,
            useSystemPrompt: true,
        };
    }

    async load() {
        logDebug(`Loading config from: ${neon.filePath(this.filePath)}`);
        const defaults = this.getDefaults();
        this.config = { ...defaults };

        try {
            if (await checkFileExists(this.filePath)) {
                const content = await fs.readFile(this.filePath, 'utf8');
                if (content.trim()) {
                    const loaded = JSON.parse(content);
                    Object.keys(loaded).forEach(key => {
                        if (defaults.hasOwnProperty(key)) {
                            this.config[key] = loaded[key];
                        } else {
                            logWarning(`Unknown config key "${key}" found in ${this.filePath} and ignored.`);
                        }
                    });
                    if (!this.config.systemPrompt) this.config.systemPrompt = DEFAULT_SYSTEM_PROMPT_TEXT;
                    logSystem(`Loaded config from ${neon.filePath(this.filePath)}`);
                } else {
                    logSystem(`Config file ${neon.filePath(this.filePath)} is empty. Initializing with defaults.`);
                    await this.save();
                }
            } else {
                logSystem(`Config file ${neon.filePath(this.filePath)} not found. Creating with defaults.`);
                await this.save();
            }
        } catch (error) {
            logError(`Config load/parse failed: ${this.filePath}. Using defaults.`, error);
            this.config = { ...defaults };
        }

        this.config.apiKey = argv.apiKey || process.env.GEMINI_API_KEY || this.config.apiKey;

        const cliOverrides = ['model', 'temperature', 'safety', 'maxHistory', 'highlight', 'debug', 'allowShell', 'allowPython'];
        cliOverrides.forEach(key => {
            if (argv[key] !== undefined && argv[key] !== yargs.defaults[key]) {
                 const configKey = key === 'model' ? 'modelName' : key;
                 this.config[configKey] = argv[key];
                 logDebug(`CLI override: ${configKey} = ${argv[key]}`);
            }
        });

        this.config.temperature = clamp(this.config.temperature, 0.0, 2.0);
        this.config.maxHistory = Math.max(1, Math.floor(this.config.maxHistory || DEFAULT_MAX_HISTORY));
        if (!SAFETY_MAP[this.config.safety]) {
            logWarning(`Invalid safety level "${this.config.safety}". Falling back to ${DEFAULT_SAFETY}.`);
            this.config.safety = DEFAULT_SAFETY;
        }
         if (!KNOWN_MODELS.includes(this.config.modelName)) {
            logWarning(`Model "${this.config.modelName}" is not in the known list. Ensure compatibility.`);
         }
        ['highlight', 'debug', 'allowShell', 'allowPython', 'useSystemPrompt'].forEach(key => {
            this.config[key] = ['true', 'on', '1', 'yes'].includes(String(this.config[key]).toLowerCase());
        });

        this.initialLoadComplete = true;
        logDebug("Effective configuration loaded:", this.getAll());
    }

    async save() {
        if (!this.initialLoadComplete) return;
        logDebug(`Saving config to: ${neon.filePath(this.filePath)}`);
        const tempConfigFile = `${this.filePath}.${process.pid}.tmp`;
        try {
            await ensureDirectoryExists(this.filePath);
            const configToSave = { ...this.config };
            delete configToSave.apiKey;
            await fs.writeFile(tempConfigFile, JSON.stringify(configToSave, null, 2), 'utf8');
            await fs.rename(tempConfigFile, this.filePath);
             logDebug(`Config saved successfully.`);
        } catch (error) {
            logError(`Configuration save failed: ${this.filePath}`, error);
             try { await fs.unlink(tempConfigFile); } catch {}
        }
    }

    get(key) {
        const internalKey = key === 'model' ? 'modelName' : key;
        return this.config[internalKey];
     }

     async set(key, value) {
        const defaults = this.getDefaults();
        const internalKey = key === 'model' ? 'modelName' : key;

        if (!defaults.hasOwnProperty(internalKey)) {
            logWarning(`Attempted to set unknown config key: ${internalKey}`);
            return false;
        }

        let parsedValue = value;
        const defaultValueType = typeof defaults[internalKey];

        try {
             if (defaultValueType === 'boolean') {
                parsedValue = ['true', 'on', '1', 'yes'].includes(String(value).toLowerCase());
             } else if (defaultValueType === 'number') {
                parsedValue = parseFloat(value);
                if (isNaN(parsedValue)) throw new Error(`Invalid number format: "${value}"`);
                if (internalKey === 'temperature') parsedValue = clamp(parsedValue, 0.0, 2.0);
                if (internalKey === 'maxHistory') parsedValue = Math.max(1, Math.floor(parsedValue));
             } else if (internalKey === 'safety') {
                parsedValue = String(value).toUpperCase();
                if (!SAFETY_MAP[parsedValue]) throw new Error(`Invalid safety level: "${value}". Valid: ${Object.keys(SAFETY_MAP).join(', ')}`);
             } else if (internalKey === 'modelName') {
                 parsedValue = String(value);
                 if (!KNOWN_MODELS.includes(parsedValue)) {
                     logWarning(`Model "${parsedValue}" not in known list. Setting anyway.`);
                 }
             } else {
                parsedValue = String(value);
             }
        } catch (error) {
             logWarning(`Invalid value for ${internalKey}: ${error.message}`);
             return false;
        }

        const oldValue = this.config[internalKey];
        if (oldValue !== parsedValue) {
            this.config[internalKey] = parsedValue;
            logSystem(`Set ${neon.configKey(internalKey)} = ${neon.configValue(parsedValue)} (was: ${neon.configValue(oldValue)})`);
            await this.save();
            await applyConfigChange(internalKey, parsedValue, oldValue);
            return true;
        } else {
            logDebug(`Config set skipped: ${internalKey} value unchanged (${parsedValue})`);
            return true;
        }
    }

    getAll() {
        const { apiKey, ...rest } = this.config;
        return rest;
    }
}


// --- Global State ---
const configManager = new ConfigManager(argv.configFile);
const HISTORY_FILE = path.resolve(argv.historyFile);
const MACROS_FILE = path.resolve(argv.macrosFile);
const SESSIONS_DIR = path.resolve(argv.sessionsDir);
const SHELL_PATH = argv.shell;
const PYTHON_PATH = argv.pythonPath;
const IS_SANDBOXED = argv.sandbox;

let API_KEY, MODEL_NAME, MAX_HISTORY_PAIRS, IS_DEBUG_MODE, IS_HIGHLIGHTING_ACTIVE, IS_SHELL_ALLOWED,
    IS_PYTHON_ALLOWED, CURRENT_SYSTEM_PROMPT_TEMPLATE, USE_SYSTEM_PROMPT, generationConfig = {}, safetySettings = [];
let chatHistory = [], currentChatSession, genAI, aiModelInstance, isPastingMode = false,
    pasteBufferContent = [], lastTextResponse = null, saveFilePath = null, readlineInterface = null,
    isAiThinking = false, spinner = null, isWaitingForShellConfirmation = false, pendingShellCommand = null,
    isWaitingForPythonConfirmation = false, pendingPythonCode = null, macros = {}, isProcessingMacro = false,
    termuxToastAvailable = false, lastUserTextInput = null, commandQueue = [], tokenCache = null,
    isProcessingQueue = false;

const ALL_HARM_CATEGORIES = Object.values(HarmCategory);
const EDITOR = process.env.EDITOR || (process.platform === 'win32' ? 'notepad' : (process.env.TERMUX_VERSION ? 'nano' : 'vi'));


// --- Utility Functions ---
const logDebug = (msg, data) => IS_DEBUG_MODE && console.log(neon.debug(`[Debug] ${msg}`), data !== undefined ? util.inspect(data, { depth: 2, colors: true }) : '');
const logError = (msg, error) => {
    if (spinner?.isSpinning) spinner.fail(chalk.redBright('Error'));
    console.error(`\n${neon.errorMarker} ${neon.error(msg)}`);
    if (error) console.error(neon.error(`  > ${error.message || String(error)}`));
    sendTermuxToast(`Error: ${msg}`, 'error');
    safePromptRefresh();
};
const logWarning = (msg) => {
    if (spinner?.isSpinning) spinner.warn(chalk.yellowBright('Warning'));
    console.log(`\n${neon.warnMarker} ${neon.warning(msg)}`);
    sendTermuxToast(`Warning: ${msg}`, 'warning');
    safePromptRefresh();
};
const logSystem = (msg) => console.log(`${neon.sysMarker} ${neon.systemInfo(msg)}`);
const clearConsole = () => process.stdout.write(process.platform === 'win32' ? '\x1B[2J\x1B[0f' : '\x1Bc');
const checkFileExists = async (filePath) => { try { await fs.access(filePath); return true; } catch { return false; } }
const ensureDirectoryExists = async (filePath) => {
    const dir = path.dirname(filePath);
    try { await fs.mkdir(dir, { recursive: true }); } catch (error) { if (error.code !== 'EEXIST') throw error; }
};
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const safePromptRefresh = () => {
    if (!readlineInterface || readlineInterface.closed) return;
    try {
        const status = isAiThinking ? neon.statusBusy : neon.statusIdle;
        const modelInfo = MODEL_NAME ? `(${MODEL_NAME})` : '';
        const pasteModeInfo = isPastingMode ? neon.pasteMarker : '';
        const waitingInfo = isWaitingForShellConfirmation || isWaitingForPythonConfirmation ? chalk.yellowBright('[Confirm?] ') : '';
        const promptText = `${status}${waitingInfo}${neon.promptMarker}${modelInfo}${pasteModeInfo} `;

        readline.cursorTo(process.stdout, 0);
        readline.clearLine(process.stdout, 0);
        readlineInterface.setPrompt(promptText);
        readlineInterface.prompt(true);
    } catch (e) {
        logDebug("Error during safePromptRefresh (ignoring):", e);
    }
};

async function checkTermuxToast() {
    if (process.env.TERMUX_VERSION) {
        try {
            await execPromise('termux-toast --help', { timeout: 1000 });
            termuxToastAvailable = true;
            logDebug("Termux detected, toast notifications enabled.");
        } catch {
            termuxToastAvailable = false;
            logDebug("Termux detected, but termux-toast command failed or not found.");
        }
    }
}

function sendTermuxToast(message, level = 'info') {
    if (!termuxToastAvailable) return;
    const shortMessage = message.length > 100 ? message.substring(0, 97) + '...' : message;
    const color = level === 'error' ? '#FF0000' : level === 'warning' ? '#FFFF00' : '#FFFFFF';
    const command = `termux-toast -b ${color} "${shortMessage.replace(/"/g, '\\"')}"`;
    exec(command, (err) => {
        if (err) logDebug(`Failed to send Termux toast: ${err.message}`);
    });
}

function estimateTokenCountLocal(text) {
    if (!text) return 0;
    return Math.ceil((text.match(/\s+|./g) || []).length / 1.5);
}

async function openInEditor(content) {
    const tempFile = path.join(os.tmpdir(), `${TEMP_FILE_PREFIX}${Date.now()}.md`);
    let editorClosed = false;
    let originalInputState = null;

    try {
        logSystem(`Opening content in editor (${EDITOR}). Save and close editor when done.`);
        await fs.writeFile(tempFile, content, 'utf8');

        if (readlineInterface) {
            readlineInterface.pause();
            if (process.stdin.isTTY) {
                originalInputState = process.stdin.isRaw;
                process.stdin.setRawMode(true);
            }
        }
        await new Promise(resolve => setTimeout(resolve, 100));

        await new Promise((resolve, reject) => {
            const editorProcess = spawn(EDITOR, [tempFile], {
                stdio: 'inherit',
                shell: true,
                detached: false
            });

            editorProcess.on('error', (err) => {
                 editorClosed = true;
                 reject(new Error(`Failed to start editor '${EDITOR}': ${err.message}`));
            });

            editorProcess.on('close', (code) => {
                 editorClosed = true;
                 logDebug(`Editor process closed with code: ${code}`);
                 if (code !== 0) {
                     logWarning(`Editor exited with non-zero status code: ${code}. Reading file anyway.`);
                 }
                 resolve();
            });
        });

        if (readlineInterface) {
             if (process.stdin.isTTY && originalInputState !== null) {
                 process.stdin.setRawMode(originalInputState);
             }
            readlineInterface.resume();
        }
        await new Promise(resolve => setTimeout(resolve, 50));

        const updatedContent = await fs.readFile(tempFile, 'utf8');
        safePromptRefresh();
        return updatedContent;

    } catch (error) {
        logError(`Editor interaction failed:`, error);
        if (readlineInterface) {
            if (process.stdin.isTTY && originalInputState !== null && !editorClosed) {
                process.stdin.setRawMode(originalInputState);
            }
            if(!editorClosed) readlineInterface.resume();
        }
        safePromptRefresh();
        return null;
    } finally {
        try { await fs.unlink(tempFile); } catch { logWarning(`Failed to delete temp file: ${tempFile}`); }
    }
}


// --- Execution Functions ---
function sanitizeEnv() {
    const safeEnv = { ...process.env };
    Object.keys(safeEnv).forEach(key => {
        if (/^(AWS_|AZURE_|GOOGLE_|GITHUB_|CI_|RUNNER_|PASS|SECRET|TOKEN|API_KEY)/i.test(key)) {
            delete safeEnv[key];
        }
    });
    if (IS_SANDBOXED) {
        const commonPaths = process.platform === 'win32'
            ? ['C:\\Windows\\System32', 'C:\\Windows', 'C:\\Windows\\System32\\WindowsPowerShell\\v1.0']
            : ['/usr/bin', '/bin', '/usr/local/bin'];
        const currentDir = process.cwd();
        const allowedPaths = [...new Set([...commonPaths, currentDir])];
        safeEnv.PATH = allowedPaths.join(path.delimiter);
        safeEnv.SANDBOXED = 'true';
        logDebug("Using sandboxed environment variables. PATH:", safeEnv.PATH);
    }
    return safeEnv;
}

async function executeShellCommand(command, saveToPath) {
    if (!IS_SHELL_ALLOWED) {
        logWarning('Shell execution is disabled.');
        return { error: 'Shell execution disabled', stdout: '', stderr: '', code: -1 };
    }
    logSystem(`Executing shell command via ${neon.filePath(SHELL_PATH)}: ${neon.shellCommand(command)}`);
    const spinnerExec = ora({ text: `Running shell...`, color: neon.spinnerColor }).start();
    let result = { stdout: '', stderr: '', code: 1, error: 'Execution failed' };

    try {
        const quotedCommand = process.platform === 'win32' ? command : command.replace(/(["$`\\])/g, '\\$1');
        const execCmd = process.platform === 'win32'
            ? `${SHELL_PATH} -NoProfile -NonInteractive -Command "${command}"`
            : `${SHELL_PATH} -c "${quotedCommand}"`;
        logDebug(`Executing prepared command: ${execCmd}`);

        const { stdout, stderr } = await execPromise(execCmd, {
            env: sanitizeEnv(),
            timeout: 30000,
            maxBuffer: 10 * 1024 * 1024,
            shell: false
        });

        spinnerExec.succeed('Shell command finished.');
        result = { stdout: stdout.trim(), stderr: stderr.trim(), code: 0, error: null };
        logDebug("Shell stdout:", result.stdout);
        if (result.stderr) logDebug("Shell stderr:", result.stderr);

    } catch (error) {
         spinnerExec.fail('Shell command failed.');
        result = {
             stdout: error.stdout?.trim() || '',
             stderr: error.stderr?.trim() || error.message || 'Unknown execution error',
             code: error.code || 1,
             error: 'Shell command execution failed'
        };
        logError(`Shell command failed (Code: ${result.code})`, result.stderr);
    } finally {
        if (saveToPath) {
            try {
                const resolvedSavePath = path.resolve(saveToPath);
                await ensureDirectoryExists(resolvedSavePath);
                const outputToSave = `Exit Code: ${result.code}\n${result.error ? `Error: ${result.error}\n` : ''}Stderr:\n${result.stderr}\n\nStdout:\n${result.stdout}`;
                await fs.writeFile(resolvedSavePath, outputToSave, 'utf8');
                logSystem(`Shell output (Code: ${result.code}) saved to ${neon.filePath(resolvedSavePath)}`);
            } catch (saveError) {
                logError(`Failed to save shell output to ${saveToPath}`, saveError);
            }
        }
         safePromptRefresh();
    }
    return result;
}

async function executePythonCode(code, saveToPath) {
    if (!IS_PYTHON_ALLOWED) {
        logWarning('Python execution is disabled.');
        return { error: 'Python execution disabled', stdout: '', stderr: '', code: -1 };
    }
    logSystem(`Executing Python code via ${neon.filePath(PYTHON_PATH)}...`);
    const spinnerExec = ora({ text: `Running python code...`, color: neon.spinnerColor }).start();
    const tempFilePath = path.join(os.tmpdir(), `neoncli_python_${Date.now()}.py`);
    let result = { stdout: '', stderr: '', code: 1, error: 'Execution setup failed' };

    try {
        await fs.writeFile(tempFilePath, code, 'utf8');
        logDebug(`Python code written to temp file: ${tempFilePath}`);

        result = await new Promise((resolve) => {
            let stdout = '', stderr = '';
            const proc = spawn(PYTHON_PATH, [tempFilePath], {
                env: sanitizeEnv(),
                timeout: 60000,
                stdio: ['ignore', 'pipe', 'pipe']
            });

            proc.stdout.on('data', (data) => stdout += data.toString());
            proc.stderr.on('data', (data) => stderr += data.toString());

            proc.on('close', (code) => {
                 if (code === 0) {
                     spinnerExec.succeed('Python code finished.');
                 } else {
                     spinnerExec.fail(`Python code failed (Code: ${code}).`);
                     logError(`Python execution failed with exit code ${code}.`);
                 }
                 logDebug("Python stdout:", stdout.trim());
                 if (stderr.trim()) logDebug("Python stderr:", stderr.trim());
                 resolve({ stdout: stdout.trim(), stderr: stderr.trim(), code, error: code ? 'Python execution failed' : null });
            });

            proc.on('error', (err) => {
                spinnerExec.fail('Python process failed to start.');
                logError('Failed to spawn Python process:', err);
                resolve({ stdout: '', stderr: err.message, code: 1, error: 'Python process spawn failed' });
            });
        });

    } catch (fileError) {
        spinnerExec.fail('Python execution setup failed.');
        logError('Error writing Python temp file:', fileError);
        result.stderr = fileError.message;
    } finally {
        await fs.unlink(tempFilePath).catch(e => logDebug("Failed to delete python temp file:", e));

         if (saveToPath) {
            try {
                const resolvedSavePath = path.resolve(saveToPath);
                await ensureDirectoryExists(resolvedSavePath);
                const outputToSave = `Exit Code: ${result.code}\n${result.error ? `Error: ${result.error}\n` : ''}Stderr:\n${result.stderr}\n\nStdout:\n${result.stdout}`;
                await fs.writeFile(resolvedSavePath, outputToSave, 'utf8');
                logSystem(`Python output (Code: ${result.code}) saved to ${neon.filePath(resolvedSavePath)}`);
            } catch (saveError) {
                logError(`Failed to save Python output to ${saveToPath}`, saveError);
            }
        }
        safePromptRefresh();
    }
    return result;
}


// --- History Management ---
async function loadChatHistory() {
    chatHistory = [];
    tokenCache = null;
    if (!await checkFileExists(HISTORY_FILE)) {
        logSystem(`History file (${neon.filePath(HISTORY_FILE)}) not found. Starting fresh.`);
        return;
    }
    try {
        const data = await fs.readFile(HISTORY_FILE, 'utf8');
        if (!data.trim()) {
             logSystem(`History file empty. Starting fresh.`);
             return;
        }
        const loadedHistory = JSON.parse(data);
        if (!Array.isArray(loadedHistory)) throw new Error("History is not an array.");

        const validEntries = loadedHistory.filter(entry => {
            const isValid = isValidHistoryEntry(entry);
            if (!isValid && IS_DEBUG_MODE) logDebug(`Invalid history entry skipped: ${JSON.stringify(entry).substring(0, 100)}...`);
            return isValid;
        });

        if (loadedHistory.length !== validEntries.length) {
             logWarning(`Loaded ${validEntries.length} valid history entries (${loadedHistory.length - validEntries.length} invalid entries skipped).`);
        }
        chatHistory = validEntries;

        trimHistory(false);
        logSystem(`Loaded ${Math.ceil(chatHistory.length / 2)} turns from ${neon.filePath(HISTORY_FILE)}`);

    } catch (error) {
        logError(`History load failed: ${HISTORY_FILE}. Starting fresh.`, error);
        chatHistory = [];
        try {
             const backupPath = `${HISTORY_FILE}.${Date.now()}.bak`;
             await fs.copyFile(HISTORY_FILE, backupPath);
             logWarning(`Backed up corrupted history file to ${backupPath}`);
             await fs.writeFile(HISTORY_FILE, '[]', 'utf8');
        } catch (backupError) { logError(`Failed to back up or reset corrupted history file:`, backupError); }
    }
}

async function saveChatHistory() {
    if (!chatHistory) {
        logDebug("Skipping history save: Not initialized.");
        return;
    }
     if (chatHistory.length === 0) {
         logDebug("Skipping history save: Empty.");
         return;
     }
    logDebug(`Saving ${Math.ceil(chatHistory.length / 2)} turns to ${neon.filePath(HISTORY_FILE)}`);
    const tempHistoryFile = `${HISTORY_FILE}.${process.pid}.tmp`;
    try {
        await ensureDirectoryExists(HISTORY_FILE);
        await fs.writeFile(tempHistoryFile, JSON.stringify(chatHistory, null, 2), 'utf8');
        await fs.rename(tempHistoryFile, HISTORY_FILE);
        logDebug("History saved successfully.");
    } catch (error) {
        logError(`History save failed: ${HISTORY_FILE}`, error);
        try { await fs.unlink(tempHistoryFile); } catch {}
    }
}

function trimHistory(logTrim = true) {
    if (!chatHistory) return false;
    const maxEntries = MAX_HISTORY_PAIRS * 2;
    if (chatHistory.length > maxEntries) {
        const removedCount = chatHistory.length - maxEntries;
        chatHistory = chatHistory.slice(-maxEntries);
        if (logTrim) logSystem(`History trimmed to last ${MAX_HISTORY_PAIRS} turns. Removed ${removedCount} entries.`);
        tokenCache = null;
        return true;
    }
    return false;
}

function isValidHistoryEntry(entry) {
    return entry && typeof entry === 'object' &&
           (entry.role === ROLE_USER || entry.role === ROLE_MODEL) &&
           Array.isArray(entry.parts) && entry.parts.length > 0 &&
           entry.parts.every(part =>
               (typeof part.text === 'string') ||
               (typeof part.inlineData === 'object' && part.inlineData !== null &&
                typeof part.inlineData.mimeType === 'string' && typeof part.inlineData.data === 'string')
           );
}

// --- Macro Management ---
async function loadMacros() {
    macros = {};
    if (!await checkFileExists(MACROS_FILE)) {
        logSystem(`Macros file (${neon.filePath(MACROS_FILE)}) not found. No macros loaded.`);
        return;
    }
    try {
        const data = await fs.readFile(MACROS_FILE, 'utf8');
        if (!data.trim()) {
             logSystem(`Macros file empty.`);
             return;
        }
        const loadedMacros = JSON.parse(data);
        if (typeof loadedMacros !== 'object' || loadedMacros === null) throw new Error("Macros file is not a JSON object.");

        Object.entries(loadedMacros).forEach(([name, content]) => {
            if (VALID_MACRO_NAME_REGEX.test(name) && typeof content === 'string') {
                macros[name] = content;
            } else {
                logWarning(`Invalid macro skipped: Name "${name}" or content type invalid.`);
            }
        });
        logSystem(`Loaded ${Object.keys(macros).length} macros from ${neon.filePath(MACROS_FILE)}`);
    } catch (error) {
        logError(`Macros load failed: ${MACROS_FILE}.`, error);
        macros = {};
    }
}

async function saveMacros() {
    logDebug(`Saving macros to: ${neon.filePath(MACROS_FILE)}`);
    const tempMacroFile = `${MACROS_FILE}.${process.pid}.tmp`;
    try {
        await ensureDirectoryExists(MACROS_FILE);
        await fs.writeFile(tempMacroFile, JSON.stringify(macros, null, 2), 'utf8');
        await fs.rename(tempMacroFile, MACROS_FILE);
        logDebug("Macros saved successfully.");
    } catch (error) {
        logError(`Macros save failed: ${MACROS_FILE}`, error);
        try { await fs.unlink(tempMacroFile); } catch {}
    }
}


// --- File Processing ---
async function convertFileToGenerativePart(filePath) {
    const resolvedPath = path.resolve(filePath);
    logSystem(`Processing file: ${neon.filePath(resolvedPath)}`);
    try {
        const stats = await fs.stat(resolvedPath);
        if (!stats.isFile()) throw new Error('Path is not a file.');
        if (stats.size > 50 * 1024 * 1024) {
            logWarning(`File size (${(stats.size / 1024 / 1024).toFixed(1)}MB) is large. Processing may be slow or fail.`);
        }

        const buffer = await fs.readFile(resolvedPath);
        const mimeType = mime.lookup(resolvedPath);

        if (!mimeType) throw new Error('Could not determine MIME type.');
        if (!mimeType.startsWith('image/') && !mimeType.startsWith('video/') && !mimeType.startsWith('audio/') && !mimeType.startsWith('text/')) {
             logWarning(`MIME type "${mimeType}" might not be directly supported by the model for inline data. Sending as text instead.`);
             if (stats.size < 1024 * 1024 && mimeType.startsWith('text/')) {
                return { text: `File Content (${path.basename(filePath)}):\n\n${buffer.toString('utf8')}` };
             } else {
                 throw new Error(`Unsupported or large non-text file type: ${mimeType}`);
             }
        }

        logSystem(`Detected MIME type: ${mimeType}`);
        return {
            inlineData: {
                data: buffer.toString('base64'),
                mimeType
            }
        };
    } catch (error) {
        logError(`Failed to process file "${filePath}":`, error);
        return null;
    }
}


// --- Core AI Interaction ---
async function confirmExecution(type, input, reason) {
    if (!readlineInterface) return false;

    const question = `\n${neon.sysMarker} ${neon.executionRequest(`AI requests to run ${type}:`)}`
        + `\n${type === 'Shell' ? neon.shellCommand(input) : neon.pythonCode(input)}`
        + `\n${neon.sysMarker} ${neon.executionRequest(`Reason: ${reason || '(No reason provided)'}`)}`
        + `\n${neon.sysMarker} ${neon.confirmPrompt(`Allow execution? (yes/no): `)}`;

    return new Promise((resolve) => {
        readlineInterface.question(question, (answer) => {
            const confirmation = answer.trim().toLowerCase();
            if (confirmation === 'yes' || confirmation === 'y') {
                logSystem(`${type} execution confirmed by user.`);
                resolve(true);
            } else {
                logSystem(`${type} execution denied by user.`);
                resolve(false);
            }
            safePromptRefresh();
        });
    });
}

async function detectAndHandleExecutionRequest(responseText) {
    const jsonMatch = responseText.match(/```json\s*(\{[\s\S]+?\})\s*```/);
    if (!jsonMatch) return { handled: false };

    let request;
    try { request = JSON.parse(jsonMatch[1]); } catch { return { handled: false }; }

    const { action, command, code, reason } = request;
    let execType = null, execInput = null, isAllowed = false;

    if (action === 'run_shell' && typeof command === 'string' && command.trim()) {
        execType = 'Shell'; execInput = command.trim(); isAllowed = IS_SHELL_ALLOWED;
    } else if (action === 'run_python' && typeof code === 'string' && code.trim()) {
        execType = 'Python'; execInput = code.trim(); isAllowed = IS_PYTHON_ALLOWED;
    } else {
        return { handled: false };
    }

    logSystem(`AI proposed ${execType} execution.`);
    if (!isAllowed) {
        logWarning(`${execType} execution requested but globally disabled.`);
        const feedback = `User feedback: Cannot execute the requested ${execType.toLowerCase()} because execution is disabled in the application settings.`;
        queueTask({ handler: sendMessageToAI, parts: [{ text: feedback }], isFeedback: true });
        return { handled: true, executed: false };
    }

    isWaitingForShellConfirmation = (execType === 'Shell');
    isWaitingForPythonConfirmation = (execType === 'Python');
    pendingShellCommand = isWaitingForShellConfirmation ? execInput : null;
    pendingPythonCode = isWaitingForPythonConfirmation ? execInput : null;
    safePromptRefresh();

    const confirmed = await confirmExecution(execType, execInput, reason);

    isWaitingForShellConfirmation = false;
    isWaitingForPythonConfirmation = false;
    pendingShellCommand = null;
    pendingPythonCode = null;

    if (confirmed) {
        const result = (execType === 'Shell') ? await executeShellCommand(execInput) : await executePythonCode(execInput);
        const feedback = formatExecutionFeedback(execType.toLowerCase(), execInput, result);
        queueTask({ handler: sendMessageToAI, parts: [{ text: feedback }], isFeedback: true });
        return { handled: true, executed: true };
    } else {
        const feedback = `User feedback: User denied the request to execute the ${execType.toLowerCase()}.`;
        queueTask({ handler: sendMessageToAI, parts: [{ text: feedback }], isFeedback: true });
        return { handled: true, executed: false };
    }
}

function formatExecutionFeedback(type, input, result) {
    let feedback = `User feedback: Execution of ${type} completed.\n`;
    feedback += `Exit Code: ${result.code}\n`;
    if (result.error) feedback += `Error: ${result.error}\n`;
    if (result.stderr) feedback += `Stderr:\n${result.stderr.substring(0, 500)}${result.stderr.length > 500 ? '...' : ''}\n`;
    if (result.stdout) feedback += `Stdout:\n${result.stdout.substring(0, 1000)}${result.stdout.length > 1000 ? '...' : ''}\n`;
    else if (!result.stderr && result.code === 0) feedback += `Stdout: (empty)\n`;
    return feedback;
}

function applyHighlightingPrint(text, isEdited = false) {
    if (!text) return;
    const prefix = isEdited ? neon.editedMarker : '';
    if (IS_HIGHLIGHTING_ACTIVE) {
        try {
            const highlighted = highlight(text, { language: 'markdown', ignoreIllegals: true, theme: { keyword: chalk.blueBright, built_in: chalk.cyan, string: chalk.green, comment: chalk.gray, function: chalk.yellow, title: chalk.magentaBright, section: chalk.magentaBright.bold, code: chalk.white, number: chalk.yellowBright } });
            console.log(prefix + highlighted);
        } catch (e) {
            logDebug("Highlighting failed, printing raw:", e);
            console.log(prefix + neon.aiResponse(text));
        }
    } else {
        console.log(prefix + neon.aiResponse(text));
    }
}

async function sendMessageToAI(parts, isFeedback = false, isEditOrRegen = false) {
    if (!parts || parts.length === 0 || parts.every(p => !p.text && !p.inlineData)) {
        logWarning("Attempted to send empty message.");
        return;
    }
    if (isWaitingForShellConfirmation || isWaitingForPythonConfirmation) {
        logWarning('Cannot send message while awaiting confirmation.');
        return;
    }
    if (!aiModelInstance) {
        logError('AI model not initialized. Cannot send message.');
        return;
    }

    isAiThinking = true;
    let firstChunkReceived = false;
    if (!isFeedback) {
        lastUserTextInput = parts.map(p => p.text || `[File: ${p.inlineData?.mimeType}]`).join('\n');
        spinner = ora({ text: neon.thinkingText, color: neon.spinnerColor, spinner: 'dots' }).start();
    }
    safePromptRefresh();

    let accumulatedResponseText = '';
    let usageMetadata = null;
    let finalContent = null;
    let finishReason = null;
    let promptFeedback = null;
    let streamError = null;
    let streamedContent = false;

    try {
        const historyForAPI = [...chatHistory];
        if (!isFeedback) {
            historyForAPI.push({ role: ROLE_USER, parts });
            const maxEntries = MAX_HISTORY_PAIRS * 2;
            if (historyForAPI.length > maxEntries) {
                historyForAPI.splice(0, historyForAPI.length - maxEntries);
                logDebug(`History snapshot trimmed to ${historyForAPI.length} entries for API call.`);
            }
        }

        currentChatSession = aiModelInstance.startChat({
            history: historyForAPI,
            generationConfig,
            safetySettings,
        });

        logDebug("Sending message to AI...");
        const stream = await currentChatSession.sendMessageStream(parts);

        if (!isFeedback) {
            process.stdout.write('\n' + neon.aiMarker);
        }

        for await (const chunk of stream) {
            const chunkText = chunk.text();
            if (chunkText) {
                if (!firstChunkReceived && !isFeedback && spinner?.isSpinning) {
                     spinner.stop();
                     firstChunkReceived = true;
                }
                accumulatedResponseText += chunkText;
                streamedContent = true;
                if (!isFeedback) {
                    process.stdout.write(neon.aiResponse(chunkText));
                }
            }
        }

         if (streamedContent && !isFeedback) {
             process.stdout.write('\n');
         } else if (!streamedContent && !isFeedback && spinner?.isSpinning) {
             spinner.stop();
         }

        const finalResponse = await stream.response;
        logDebug("Final response object received:", finalResponse);

        usageMetadata = finalResponse.usageMetadata;
        promptFeedback = finalResponse.promptFeedback;
        finalContent = finalResponse.candidates?.[0]?.content;
        finishReason = finalResponse.candidates?.[0]?.finishReason;

        if (finalContent?.parts?.map(p => p.text).join('') && !accumulatedResponseText) {
            accumulatedResponseText = finalContent.parts.map(p => p.text).join('');
            logDebug("Using text from final response object as stream was empty.");
             if (!isFeedback) {
                 process.stdout.write('\n' + neon.aiMarker);
                 applyHighlightingPrint(accumulatedResponseText, isEditOrRegen);
                 process.stdout.write('\n');
             }
        }

        if (promptFeedback?.blockReason) {
            logWarning(`Request or Response Blocked: ${promptFeedback.blockReason}`);
            accumulatedResponseText = `[Blocked by Safety Filter: ${promptFeedback.blockReason}]`;
             if (!isFeedback && !streamedContent) applyHighlightingPrint(accumulatedResponseText);
        } else if (finishReason === 'STOP' || finishReason === 'MAX_TOKENS' || (finishReason === 'OTHER' && accumulatedResponseText)) {
            if (finishReason === 'MAX_TOKENS') logWarning("Response may be truncated: Max tokens reached.");
            if (finishReason === 'OTHER') logWarning("Response finished with reason 'OTHER'. Processing content.");

            const execResult = await detectAndHandleExecutionRequest(accumulatedResponseText);

            if (execResult.handled) {
                 logDebug("Execution request handled by detectAndHandleExecutionRequest.");
                 lastTextResponse = accumulatedResponseText;
             } else {
                 if (!isFeedback && finalContent && finalContent.role === ROLE_MODEL) {
                     chatHistory.push({ role: ROLE_USER, parts });
                     chatHistory.push(finalContent);
                     trimHistory();
                     await saveChatHistory();
                     lastTextResponse = accumulatedResponseText;
                     tokenCache = null;

                     if (saveFilePath && !isFeedback) {
                         try {
                             await ensureDirectoryExists(saveFilePath);
                             await fs.writeFile(saveFilePath, accumulatedResponseText, 'utf8');
                             logSystem(`Response saved to ${neon.filePath(saveFilePath)}`);
                             saveFilePath = null;
                         } catch (saveError) { logError(`Failed to save response to ${saveFilePath}`, saveError); saveFilePath = null; }
                     }
                      if (!isFeedback && !streamedContent) {
                          applyHighlightingPrint(accumulatedResponseText, isEditOrRegen);
                      }

                 } else if (isFeedback) {
                      logDebug("System feedback sent, not added to history or processed further.");
                 } else if (!finalContent || finalContent.role !== ROLE_MODEL) {
                      logWarning("Received response metadata but no valid model content to save.");
                 }
             }

        } else if (finishReason === 'SAFETY') {
             logWarning(`Response stopped due to safety settings.`);
             accumulatedResponseText = `[Response Blocked by Safety Filter]`;
              if (!isFeedback && !streamedContent) applyHighlightingPrint(accumulatedResponseText);
        } else {
             logWarning(`Response generation stopped unexpectedly or yielded no content. Reason: ${finishReason || 'Unknown'}`);
              if (accumulatedResponseText && !isFeedback && !streamedContent) {
                 applyHighlightingPrint(accumulatedResponseText + `\n[Response Incomplete or Error: ${finishReason || 'Unknown'}]`);
              } else if (!isFeedback) {
                 logError(`AI interaction failed to produce content. Finish Reason: ${finishReason || 'Unknown'}`);
              }
        }

    } catch (error) {
        streamError = error;
        if (spinner?.isSpinning) spinner.fail('AI Error');
        logError('AI communication or processing error:', error);
        if (error.message?.includes('API key not valid')) logError('Hint: Check API key validity/permissions.');
        else if (error.status === 429 || error.message?.includes('429')) logWarning('API rate limit likely hit. Try again later.');
        else if (error.message?.includes('FETCH_ERROR') || error.message?.includes('ECONNREFUSED')) logWarning('Network error connecting to AI service.');
        else if (error.message?.includes('Invalid JSON payload')) logError('Hint: Possible issue with history format or input data.');
    } finally {
        isAiThinking = false;
        if (spinner?.isSpinning) spinner.stop();

        if (!streamError && usageMetadata && !isFeedback && finishReason !== 'SAFETY' && !promptFeedback?.blockReason) {
            const { promptTokenCount = 0, candidatesTokenCount = 0, totalTokenCount = 0 } = usageMetadata;
            logSystem(neon.tokenCount(`Tokens: ${promptTokenCount} (prompt) + ${candidatesTokenCount} (response) = ${totalTokenCount} (total)`));
             if (totalTokenCount > TOKEN_WARNING_THRESHOLD) {
                 logWarning(`High token count (${totalTokenCount}). Consider '/clear' or '/context'.`);
             }
        }

        setTimeout(processQueue, 0);
        safePromptRefresh();
    }
}


// --- Command Handlers ---
const commandHandlers = {
    help: () => {
        console.log(neon.separator());
        logSystem(`${APP_NAME} v${APP_VERSION} - Command Help`);
        console.log(neon.separator());
        const cmds = {
            'Chat & History': [
                 ['/edit', 'Edit the last user message in your editor.'],
                 ['/regen', 'Regenerate the last AI response.'],
                 ['/paste', 'Enter multi-line paste mode. End with /endpaste.'],
                 ['/endpaste', 'Submit content entered in paste mode.'],
                 ['/clear', 'Clear the current chat history.'],
                 ['/history [num]', 'Show last [num] chat turns (default: 10).'],
                 ['/search <query>', 'Search chat history for <query>.'],
                 ['/tokens', 'Estimate/show token count for current history (uses API).'],
                 ['/context <num>', `Set max history pairs (current: ${MAX_HISTORY_PAIRS}).`]
            ],
             'Files & Saving': [
                 ['/file <path> [prompt]', 'Load file content (text/image/video/audio) into chat.'],
                 ['/save <filepath>', 'Set filepath to save the next AI response.']
            ],
             'Model & Generation': [
                 ['/model [name]', `Show or set the AI model (current: ${MODEL_NAME}).`],
                 ['/model list', 'List known compatible models.'],
                 ['/model reload', 'Re-initialize the connection to the current model.'],
                 ['/temp <value>', `Set temperature (0.0-2.0, current: ${configManager.get('temperature')}).`],
                 ['/safety [level]', `Show or set safety level (current: ${configManager.get('safety')}). Levels: ${Object.keys(SAFETY_MAP).join(', ')}`],
                 ['/system view|edit|set <text>|reset|toggle', 'Manage the system prompt.']
            ],
             'Code Execution': [
                 ['/shell on|off', `Toggle AI shell request ability (current: ${IS_SHELL_ALLOWED ? 'ON' : 'OFF'}). Requires --allow-shell flag.`],
                 ['/shell run <cmd>', 'Manually execute a shell command.'],
                 ['/shell save <file> <cmd>', 'Manually execute and save output.'],
                 ['/python on|off', `Toggle AI Python request ability (current: ${IS_PYTHON_ALLOWED ? 'ON' : 'OFF'}). Requires --allow-python flag.`],
                 ['/python run <code>', 'Manually execute Python code.'],
                 ['/python save <file> <code>', 'Manually execute Python and save output.']
            ],
             'Macros': [
                 ['/macro define <name> <text>', 'Define macro `!<name>`. Supports $1, $*, $#, $0.'],
                 ['/macro undef <name>', 'Delete macro `!<name>`.'],
                 ['/macro list', 'List defined macros.']
            ],
             'Session Management': [
                 ['/session save <name>', 'Save current chat history as session.'],
                 ['/session load <name>', 'Load chat history from session.'],
                 ['/session list', 'List saved sessions.']
            ],
             'Configuration & Control': [
                 ['/config list', 'Show current configuration (excluding API key).'],
                 ['/config set <key> <value>', 'Set a configuration option (persisted).'],
                 ['/highlight on|off', `Toggle syntax highlighting (current: ${IS_HIGHLIGHTING_ACTIVE ? 'ON' : 'OFF'}).`],
                 ['/debug on|off', `Toggle debug logging (current: ${IS_DEBUG_MODE ? 'ON' : 'OFF'}).`],
                 ['/exit | /quit | /bye', 'Exit the application cleanly.']
            ]
        };
         Object.entries(cmds).forEach(([category, commandList]) => {
             logSystem(`\n${chalk.underline(category)}:`);
             commandList.forEach(([cmd, desc]) => {
                 console.log(`  ${neon.commandHelp(cmd.padEnd(30))} ${desc}`);
             });
         });
         console.log(neon.separator());
    },
    exit: async () => await gracefulExit(),
    quit: async () => await gracefulExit(),
    bye: async () => await gracefulExit(),
    clear: async () => {
         if (chatHistory.length === 0) return logSystem("History is already empty.");
         chatHistory = [];
         tokenCache = null;
         lastTextResponse = null;
         lastUserTextInput = null;
         if (aiModelInstance) {
              currentChatSession = aiModelInstance.startChat({ history: [], generationConfig, safetySettings });
              logSystem('Chat history cleared and session state reset.');
         } else {
              logSystem('Chat history cleared.');
         }
         await saveChatHistory();
    },
    history: (args) => {
         if (chatHistory.length === 0) return logSystem("Chat history is empty.");
         const maxTurns = Math.ceil(chatHistory.length / 2);
         const numTurnsToShow = Math.min(parseInt(args) || 10, maxTurns);
         const numEntriesToShow = numTurnsToShow * 2;
         const startIndex = Math.max(0, chatHistory.length - numEntriesToShow);

         logSystem(`--- Displaying Last ${numTurnsToShow} Chat Turns ---`);
         chatHistory.slice(startIndex).forEach((entry, index) => {
             const absoluteIndex = startIndex + index;
             const turnNumber = Math.floor(absoluteIndex / 2) + 1;
             const marker = entry.role === ROLE_USER ? neon.userPrompt(`User [${turnNumber}]:`) : neon.aiResponse(`AI [${turnNumber}]:  `);
             const preview = entry.parts?.map(p => p.text || `[${p.inlineData?.mimeType || 'Data Part'}]`).join('\n').replace(/\n+/g, ' ').substring(0, 120) || '[Empty Entry]';
             console.log(`${marker} ${preview}${preview.length === 120 ? '...' : ''}`);
         });
         console.log(neon.separator());
    },
    file: async (args) => {
         const argParts = args.trim().split(' ');
         const filePath = argParts[0];
         const promptText = argParts.slice(1).join(' ') || `Process this file: ${path.basename(filePath)}`;
         if (!filePath) return logWarning("Usage: /file <filepath> [optional prompt]");

         const filePart = await convertFileToGenerativePart(filePath);
         if (filePart) {
             const partsToSend = [filePart];
             if (promptText) partsToSend.push({ text: promptText });
             queueTask({ handler: sendMessageToAI, parts: partsToSend });
         } else {
             logError(`Could not process file: ${filePath}`);
         }
    },
    paste: () => {
         if (isPastingMode) return logWarning("Already in paste mode. /endpaste to finish.");
         isPastingMode = true;
         pasteBufferContent = [];
         logSystem('Entered paste mode. Type /endpaste on a new line to submit, Ctrl+C to cancel paste mode.');
         safePromptRefresh();
    },
    endpaste: async () => {
         if (!isPastingMode) return logWarning("Not in paste mode. /paste to start.");
         isPastingMode = false;
         const content = pasteBufferContent.join('\n');
         pasteBufferContent = [];
         if (content.trim()) {
             logSystem(`Submitting ${content.split('\n').length} lines...`);
             queueTask({ handler: sendMessageToAI, parts: [{ text: content }] });
         } else {
             logSystem("Paste mode ended. No content submitted.");
             safePromptRefresh();
         }
    },
    edit: async () => {
         if (isAiThinking) return logWarning("Cannot edit while AI is thinking.");
         if (isWaitingForShellConfirmation || isWaitingForPythonConfirmation) return logWarning("Cannot edit while awaiting confirmation.");

         let lastUserIndex = -1;
         for(let i = chatHistory.length - 1; i >= 0; i--) {
             if (chatHistory[i].role === ROLE_USER) {
                 lastUserIndex = i;
                 break;
             }
         }

         if (lastUserIndex === -1) return logWarning('No user message found in history to edit.');

         const lastUserEntry = chatHistory[lastUserIndex];
         const textToEdit = lastUserEntry.parts?.map(p => p.text || `[Unsupported part: ${p.inlineData?.mimeType}]`).join('\n') || '';

         const editedContent = await openInEditor(textToEdit);

         if (editedContent !== null && editedContent.trim() !== textToEdit.trim()) {
             const historyBeforeEdit = chatHistory.slice(0, lastUserIndex);
             chatHistory = historyBeforeEdit;
             tokenCache = null;
             logSystem('Applying edited message and regenerating response...');
             queueTask({ handler: sendMessageToAI, parts: [{ text: editedContent }], isEditOrRegen: true });
             await saveChatHistory();
         } else if (editedContent === textToEdit || (editedContent !== null && editedContent.trim() === textToEdit.trim())) {
             logSystem("Edit cancelled or no changes made.");
             safePromptRefresh();
         } else {
             logWarning("Failed to get edited content.");
             safePromptRefresh();
         }
    },
    regen: async () => {
          if (isAiThinking) return logWarning("Cannot regenerate while AI is thinking.");
          if (isWaitingForShellConfirmation || isWaitingForPythonConfirmation) return logWarning("Cannot regenerate while awaiting confirmation.");

          let lastUserIndex = -1;
          for(let i = chatHistory.length - 1; i >= 0; i--) {
              if (chatHistory[i].role === ROLE_USER) {
                  lastUserIndex = i;
                  break;
              }
          }

          if (lastUserIndex === -1) return logWarning('No previous user message found to regenerate response for.');

          const lastUserParts = chatHistory[lastUserIndex].parts;
          if (!lastUserParts || lastUserParts.length === 0) return logWarning("Last user message seems empty or invalid.");

          if (lastUserIndex < chatHistory.length - 1 && chatHistory[lastUserIndex + 1].role === ROLE_MODEL) {
              const historyBeforeRegen = chatHistory.slice(0, lastUserIndex + 1);
              chatHistory = historyBeforeRegen;
              tokenCache = null;
              logSystem('Removed last AI response. Regenerating...');
              await saveChatHistory();
          } else {
              logSystem('No previous AI response to remove. Regenerating based on last user input...');
          }

          queueTask({ handler: sendMessageToAI, parts: lastUserParts, isEditOrRegen: true });
    },
    save: (args) => {
          const filePath = args.trim();
          if (!filePath) {
              saveFilePath = null; logSystem("Save path cleared. Next response will not be saved.");
          } else {
              saveFilePath = path.resolve(filePath);
              logSystem(`Next AI response will be saved to: ${neon.filePath(saveFilePath)}`);
          }
    },
    temp: async (args) => await configManager.set('temperature', args.trim()),
    model: async (args) => {
         const action = args.trim().toLowerCase();
         if (!action) return logSystem(`Current model: ${neon.configValue(MODEL_NAME)}`);
         if (action === 'list') {
             logSystem("Known models (availability depends on API key/region):");
             KNOWN_MODELS.forEach(m => console.log(`  - ${neon.configValue(m)}`));
             return;
         }
         if (action === 'reload') {
             logSystem(`Reloading model connection for ${neon.configValue(MODEL_NAME)}...`);
             queueTask({ handler: async () => {
                 await initializeModelInstance(false);
                 logSystem(`Model ${neon.configValue(MODEL_NAME)} reloaded.`);
             }});
             return;
         }
         logSystem(`Attempting to set model to: ${action}`);
         queueTask({ handler: async () => await configManager.set('modelName', action) });
    },
    safety: async (args) => await configManager.set('safety', args.trim().toUpperCase()),
    debug: async (args) => {
         const input = args.trim().toLowerCase();
         let newState;
         if (input === 'on' || input === 'true' || input === '1' || input === 'yes') newState = true;
         else if (input === 'off' || input === 'false' || input === '0' || input === 'no') newState = false;
         else newState = !IS_DEBUG_MODE;
         queueTask({ handler: async () => await configManager.set('debug', newState) });
    },
    highlight: async (args) => {
         const input = args.trim().toLowerCase();
         let newState;
         if (input === 'on' || input === 'true' || input === '1' || input === 'yes') newState = true;
         else if (input === 'off' || input === 'false' || input === '0' || input === 'no') newState = false;
         else newState = !IS_HIGHLIGHTING_ACTIVE;
         queueTask({ handler: async () => await configManager.set('highlight', newState) });
    },
    search: (query) => {
         if (!query || query.trim().length < 2) return logWarning("Provide search query (at least 2 characters).");
         const lowerQuery = query.toLowerCase();
         let matchCount = 0;
         logSystem(`--- Searching History for "${query}" ---`);
         const regex = new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');

         chatHistory.forEach((entry, index) => {
             const textContent = entry.parts?.map(p => p.text || '').join('\n') || '';
             if (textContent.toLowerCase().includes(lowerQuery)) {
                 matchCount++;
                 const turnNumber = Math.floor(index / 2) + 1;
                 const marker = entry.role === ROLE_USER ? neon.userPrompt(`User [${turnNumber}]:`) : neon.aiResponse(`AI [${turnNumber}]:  `);
                 const preview = textContent.replace(/\n+/g, ' ').substring(0, 150);
                 try {
                     const highlightedPreview = preview.replace(regex, (match) => neon.searchHighlight(match));
                     console.log(`${marker} ${highlightedPreview}${preview.length === 150 ? '...' : ''}`);
                 } catch (e) {
                      console.log(`${marker} ${preview}${preview.length === 150 ? '...' : ''}`);
                 }
             }
         });
         logSystem(matchCount > 0 ? `Found ${matchCount} matching entries.` : `No matches found for "${query}".`);
         console.log(neon.separator());
    },
    shell: async (args) => {
         const [action, ...rest] = args.trim().split(' ');
         const commandOrFile = rest[0];
         const command = action === 'save' ? rest.slice(1).join(' ') : rest.join(' ');

         if (action === 'on' || action === 'off') {
             if (!argv.allowShell) return logWarning("Shell execution not enabled via --allow-shell flag. Cannot toggle.");
             const newState = action === 'on';
             queueTask({ handler: async () => await configManager.set('allowShell', newState) });
         } else if (action === 'run' && command) {
             if (!IS_SHELL_ALLOWED) return logWarning("Shell execution is disabled. Use '/shell on' first (if --allow-shell is active).");
             queueTask({ handler: async () => await executeShellCommand(command) });
         } else if (action === 'save' && commandOrFile && command) {
             if (!IS_SHELL_ALLOWED) return logWarning("Shell execution is disabled. Use '/shell on' first (if --allow-shell is active).");
             queueTask({ handler: async () => await executeShellCommand(command, commandOrFile) });
         } else if (!action) {
             logSystem(`Shell execution capability: ${argv.allowShell ? (IS_SHELL_ALLOWED ? 'ENABLED' : 'DISABLED (use /shell on)') : 'DISABLED (by --allow-shell flag)'}.`);
             logSystem(`Use '/shell on|off|run <cmd>|save <file> <cmd>'.`);
         } else {
             logWarning(`Invalid /shell usage. See /help.`);
         }
    },
    python: async (args) => {
         const [action, ...rest] = args.trim().split(' ');
         const fileOrCode = rest[0];
         const code = action === 'save' ? rest.slice(1).join(' ') : rest.join(' ');

         if (action === 'on' || action === 'off') {
             if (!argv.allowPython) return logWarning("Python execution not enabled via --allow-python flag. Cannot toggle.");
             const newState = action === 'on';
             queueTask({ handler: async () => await configManager.set('allowPython', newState) });
         } else if (action === 'run' && code) {
             if (!IS_PYTHON_ALLOWED) return logWarning("Python execution is disabled. Use '/python on' first (if --allow-python is active).");
             queueTask({ handler: async () => await executePythonCode(code) });
         } else if (action === 'save' && fileOrCode && code) {
             if (!IS_PYTHON_ALLOWED) return logWarning("Python execution is disabled. Use '/python on' first (if --allow-python is active).");
             queueTask({ handler: async () => await executePythonCode(code, fileOrCode) });
         } else if (!action) {
             logSystem(`Python execution capability: ${argv.allowPython ? (IS_PYTHON_ALLOWED ? 'ENABLED' : 'DISABLED (use /python on)') : 'DISABLED (by --allow-python flag)'}.`);
             logSystem(`Use '/python on|off|run <code>|save <file> <code>'.`);
         } else {
             logWarning(`Invalid /python usage. See /help.`);
         }
    },
    macro: async (args) => {
         const [action, name, ...contentParts] = args.trim().split(' ');
         const content = contentParts.join(' ');
         if (action === 'define' && name && content) {
             if (!VALID_MACRO_NAME_REGEX.test(name)) return logWarning(`Invalid macro name: "${name}". Use only letters, numbers, underscore, hyphen.`);
             macros[name] = content;
             logSystem(`Macro ${neon.macroName(`!${name}`)} defined.`);
             await saveMacros();
         } else if (action === 'undef' && name) {
             if (macros[name]) {
                 delete macros[name];
                 logSystem(`Macro ${neon.macroName(`!${name}`)} undefined.`);
                 await saveMacros();
             } else logWarning(`Macro !${name} not found.`);
         } else if (action === 'list') {
             if (Object.keys(macros).length === 0) return logSystem("No macros defined.");
             logSystem("Defined macros:");
             Object.entries(macros).forEach(([n, c]) => console.log(`  ${neon.macroName(`!${n}`)}: ${neon.macroContent(c.substring(0, 80))}${c.length > 80 ? '...' : ''}`));
         } else logWarning("Usage: /macro define <name> <content> | undef <name> | list");
    },
    session: async (args) => {
          const [action, ...nameParts] = args.trim().split(' ');
          const name = nameParts.join(' ').trim();

          if (!action || !name && (action === 'save' || action === 'load')) {
              return logWarning(`Usage: /session save <name> | load <name> | list`);
          }

          try {
              await ensureDirectoryExists(SESSIONS_DIR);
              if (action === 'save') {
                  if (chatHistory.length === 0) return logWarning("Cannot save an empty chat history as a session.");
                  const sessionFile = path.join(SESSIONS_DIR, `${name}.json`);
                  queueTask({ handler: async () => {
                      await fs.writeFile(sessionFile, JSON.stringify(chatHistory, null, 2));
                      logSystem(`Session saved: ${neon.filePath(name)}`);
                  }});
              } else if (action === 'load') {
                  const sessionFile = path.join(SESSIONS_DIR, `${name}.json`);
                  if (!await checkFileExists(sessionFile)) return logError(`Session file not found: ${neon.filePath(sessionFile)}`);
                  queueTask({ handler: async () => {
                      const data = await fs.readFile(sessionFile, 'utf8');
                      const loadedHistory = JSON.parse(data);
                      if (!Array.isArray(loadedHistory)) throw new Error("Session file is invalid (not an array).");

                      const validEntries = loadedHistory.filter(isValidHistoryEntry);
                      if (validEntries.length !== loadedHistory.length) {
                          logWarning(`Loaded ${validEntries.length} valid session entries (${loadedHistory.length - validEntries.length} invalid skipped).`);
                      }
                      if (validEntries.length === 0) return logWarning("Session loaded successfully, but contained no valid entries. History cleared.");

                      chatHistory = validEntries;
                      tokenCache = null;
                      lastTextResponse = null;
                      lastUserTextInput = null;
                      trimHistory(false);

                      if (aiModelInstance) {
                          currentChatSession = aiModelInstance.startChat({ history: chatHistory, generationConfig, safetySettings });
                      }
                      logSystem(`Session loaded: ${neon.filePath(name)} (${Math.ceil(chatHistory.length / 2)} turns). History replaced.`);
                      await saveChatHistory();
                  }});
              } else if (action === 'list') {
                  const files = (await fs.readdir(SESSIONS_DIR)).filter(f => f.endsWith('.json'));
                  if (files.length === 0) logSystem("No saved sessions found.");
                  else { logSystem("Saved sessions:"); files.forEach(f => console.log(`  - ${neon.filePath(f.replace('.json', ''))}`)); }
              } else {
                  logWarning(`Unknown /session action: ${action}. Use save, load, or list.`);
              }
          } catch (error) { logError(`Session operation failed for "${name}":`, error); }
    },
    config: async (args) => {
         const [action, key, ...valueParts] = args.trim().split(' ');
         const value = valueParts.join(' ');
         if (action === 'list' || !action) {
             logSystem("Current Configuration (excluding API key):"); console.log(neon.separator());
             const configDesc = { modelName: 'AI model', temperature: 'Randomness (0-2)', safety: 'Safety filter level', maxHistory: 'History pairs retained', highlight: 'Syntax highlighting', debug: 'Debug logging', allowShell: 'Allow AI shell exec (config)', allowPython: 'Allow AI python exec (config)', systemPrompt: 'AI instructions', useSystemPrompt: 'Send system prompt' };
             Object.entries(configManager.getAll()).forEach(([k, v]) => {
                 const desc = configDesc[k] ? ` - ${neon.configDesc(configDesc[k])}` : '';
                 const displayValue = k === 'systemPrompt' ? '[Use /system view]' : neon.configValue(v);
                 console.log(`  ${neon.configKey(k.padEnd(15))}: ${displayValue}${desc}`);
             });
              logSystem(`Files: Cfg=${neon.filePath(configManager.filePath)}, Hist=${neon.filePath(HISTORY_FILE)}, Mac=${neon.filePath(MACROS_FILE)}, Sess=${neon.filePath(SESSIONS_DIR)}`);
              logSystem(`Runtime: Shell Allowed=${IS_SHELL_ALLOWED}, Python Allowed=${IS_PYTHON_ALLOWED} (influenced by flags & config)`);
             console.log(neon.separator());
         } else if (action === 'set' && key && valueParts.length > 0) {
             queueTask({ handler: async () => await configManager.set(key, value) });
         } else {
             logWarning("Usage: /config list | /config set <key> <value>");
         }
    },
    system: async (args) => {
         const [action, ...rest] = args.trim().split(' ');
         const content = rest.join(' ');
         if (action === 'view') {
             logSystem("--- Current System Prompt ---");
             console.log(configManager.get('systemPrompt') || chalk.italic('(Empty - Using Default Internally)'));
             console.log(neon.separator());
             logSystem(`System prompt usage currently: ${USE_SYSTEM_PROMPT ? 'ENABLED' : 'DISABLED'}. Use '/system toggle'.`);
         } else if (action === 'edit') {
             const currentPrompt = configManager.get('systemPrompt') || DEFAULT_SYSTEM_PROMPT_TEXT;
             const edited = await openInEditor(currentPrompt);
             if (edited !== null && edited !== currentPrompt) {
                 queueTask({ handler: async () => await configManager.set('systemPrompt', edited) });
             } else if (edited === currentPrompt) {
                  logSystem("Edit cancelled or no changes made.");
                  safePromptRefresh();
             } else {
                  logWarning("Failed to get edited system prompt.");
                  safePromptRefresh();
             }
         } else if (action === 'set' && content) {
             queueTask({ handler: async () => await configManager.set('systemPrompt', content) });
         } else if (action === 'reset') {
             queueTask({ handler: async () => await configManager.set('systemPrompt', DEFAULT_SYSTEM_PROMPT_TEXT) });
         } else if (action === 'toggle') {
             queueTask({ handler: async () => await configManager.set('useSystemPrompt', !USE_SYSTEM_PROMPT) });
         } else {
             logWarning("Usage: /system view|edit|set <text>|reset|toggle");
         }
    },
    tokens: async () => {
         if (isAiThinking) return logWarning("Cannot count tokens while AI is thinking.");
         if (tokenCache !== null) return logSystem(`Cached history token count (API): ${neon.tokenCount(tokenCache)}`);
         if (!aiModelInstance) return logError("AI model not initialized. Cannot count tokens.");
         if (chatHistory.length === 0) { tokenCache = 0; return logSystem("History empty (0 tokens)."); }

         const spinnerTokens = ora({ text: `Counting tokens via API...`, color: neon.spinnerColor }).start();
         try {
             const historyForCount = chatHistory.map(entry => ({ role: entry.role, parts: entry.parts.map(p => p.text ? { text: p.text } : p) }));
             logDebug("History for token count:", historyForCount);
             const { totalTokens } = await aiModelInstance.countTokens(historyForCount);
             spinnerTokens.succeed(`Current History Token Count (API): ${neon.tokenCount(totalTokens)}`);
             tokenCache = totalTokens;
             if (totalTokens > TOKEN_WARNING_THRESHOLD) logWarning(`High token count. Consider '/clear', '/context', or saving/loading sessions.`);
         } catch (error) {
             spinnerTokens.fail('API token counting failed.');
             logError('Token counting error:', error);
             tokenCache = null;
             const localEstimate = chatHistory.reduce((sum, entry) => sum + (entry.parts?.reduce((partSum, part) => partSum + estimateTokenCountLocal(part.text), 0) || 0), 0);
             logSystem(`Local rough estimate: ~${localEstimate} tokens.`);
         } finally { safePromptRefresh(); }
    },
    context: async (args) => {
         const value = args.trim();
         if (!value) return logSystem(`Current max history pairs: ${MAX_HISTORY_PAIRS} (Total ${MAX_HISTORY_PAIRS * 2} entries)`);
         const num = parseInt(value);
         if (isNaN(num) || num < 1) return logWarning("Provide a positive integer for the number of history *pairs* (user + AI) to keep.");
         queueTask({ handler: async () => await configManager.set('maxHistory', num) });
    },
};


// --- Initialization ---
async function applyConfigChange(key, newValue, oldValue) {
    logDebug(`Applying config change: ${key} = ${newValue} (was: ${oldValue})`);
    let needsModelReload = false;
    let needsChatReset = false;

    switch (key) {
        case 'debug': IS_DEBUG_MODE = newValue; break;
        case 'highlight': IS_HIGHLIGHTING_ACTIVE = newValue; break;
        case 'temperature':
            generationConfig.temperature = newValue;
            needsChatReset = true;
            break;
        case 'maxHistory':
            MAX_HISTORY_PAIRS = newValue;
            if(trimHistory()) {
                 needsChatReset = true;
                 await saveChatHistory();
            }
            break;
        case 'allowShell':
            IS_SHELL_ALLOWED = argv.allowShell && newValue;
            logSystem(`Shell execution config set to ${newValue}. Runtime status: ${IS_SHELL_ALLOWED ? 'ENABLED' : 'DISABLED'}`);
            break;
        case 'allowPython':
            IS_PYTHON_ALLOWED = argv.allowPython && newValue;
            logSystem(`Python execution config set to ${newValue}. Runtime status: ${IS_PYTHON_ALLOWED ? 'ENABLED' : 'DISABLED'}`);
            break;
        case 'modelName':
            MODEL_NAME = newValue;
            needsModelReload = true;
            break;
        case 'safety':
            safetySettings = ALL_HARM_CATEGORIES.map(c => ({ category: c, threshold: SAFETY_MAP[newValue] || SAFETY_MAP[DEFAULT_SAFETY] }));
            needsModelReload = true;
            logDebug("Updated safety settings:", safetySettings);
            break;
        case 'systemPrompt':
            CURRENT_SYSTEM_PROMPT_TEMPLATE = newValue;
            if (USE_SYSTEM_PROMPT) needsModelReload = true;
            break;
        case 'useSystemPrompt':
            USE_SYSTEM_PROMPT = newValue;
            needsModelReload = true;
            break;
    }

    if (needsModelReload && genAI) {
        logSystem("Config change requires reloading AI model connection...");
        await initializeModelInstance(false);
    } else if (needsChatReset && aiModelInstance && !needsModelReload) {
        logSystem("Config change requires resetting chat session state...");
         currentChatSession = aiModelInstance.startChat({ history: chatHistory, generationConfig, safetySettings });
         tokenCache = null;
    }

    safePromptRefresh();
}

async function initializeModelInstance(showBanner = true) {
    if (!API_KEY) {
        const errorMsg = 'API key is missing. Set GEMINI_API_KEY env var, use --api-key flag, or set via /config set apiKey <key> (key not saved to config file).';
        if (!genAI) {
            throw new Error(errorMsg);
        } else {
            logError(errorMsg);
            aiModelInstance = null;
            currentChatSession = null;
            return;
        }
    }

    logSystem(`Initializing Google Generative AI with model: ${neon.configValue(MODEL_NAME)}...`);
    const initSpinner = ora({ text: `Connecting to ${MODEL_NAME}...`, color: neon.spinnerColor }).start();

    try {
        if (!genAI) {
            genAI = new GoogleGenerativeAI(API_KEY);
        }

        const systemInstructionContent = USE_SYSTEM_PROMPT ? getInterpolatedSystemPrompt() : undefined;
        logDebug(systemInstructionContent ? "Using system instruction." : "System instruction disabled.");

        aiModelInstance = genAI.getGenerativeModel({
            model: MODEL_NAME,
            safetySettings: safetySettings,
            generationConfig: generationConfig,
            systemInstruction: systemInstructionContent ? { role: "system", parts: [{ text: systemInstructionContent }] } : undefined,
        });

        currentChatSession = aiModelInstance.startChat({
             history: chatHistory,
        });

        initSpinner.succeed(`Initialized ${neon.configValue(MODEL_NAME)}.`);
        if (showBanner) {
            logSystem(`Safety: ${neon.configValue(configManager.get('safety'))}. History: ${neon.configValue(MAX_HISTORY_PAIRS)} turns.`);
            logSystem(`Type '/help' for commands, '!' for macros.`);
        }
        tokenCache = null;

    } catch (error) {
        initSpinner.fail(`Failed to initialize AI model (${MODEL_NAME})`);
        logError(`Initialization Error:`, error);
        aiModelInstance = null;
        currentChatSession = null;
        if (error.message?.includes('API key not valid')) logError('Hint: Check API key validity/permissions.');
        else if (error.message?.includes('Could not find model')) logError(`Hint: Model "${MODEL_NAME}" might be invalid or unavailable. Try '/model list' or check Google AI documentation.`);
        else if (error.status === 403 || error.message?.includes('PERMISSION_DENIED')) logError('Hint: API key lacks permissions or the Generative Language API is not enabled for your project.');
        else if (error.message?.includes('Quota') || error.status === 429) logError('Hint: API quota exceeded. Check your usage limits.');

         if (!readlineInterface) process.exit(1);
    } finally {
        safePromptRefresh();
    }
}

function getInterpolatedSystemPrompt() {
    let prompt = CURRENT_SYSTEM_PROMPT_TEMPLATE || DEFAULT_SYSTEM_PROMPT_TEXT;
    try {
        prompt = prompt.replace(/\{\{APP_NAME\}\}/g, APP_NAME);
        prompt = prompt.replace(/\{\{APP_VERSION\}\}/g, APP_VERSION);
        prompt = prompt.replace(/\{\{MODEL_NAME\}\}/g, MODEL_NAME || 'N/A');
        prompt = prompt.replace(/\{\{CURRENT_DATE\}\}/g, new Date().toLocaleDateString());
    } catch (e) {
        logError("Failed to interpolate system prompt variables", e);
        return DEFAULT_SYSTEM_PROMPT_TEXT;
    }
    return prompt;
}

let isExiting = false;

async function gracefulExit(code = 0) {
    logSystem('\nShutting down gracefully...');
    isExiting = true;

    if (readlineInterface) {
        readlineInterface.close();
        readlineInterface = null;
    }

    if (spinner?.isSpinning) spinner.stop();

    isWaitingForShellConfirmation = false;
    isWaitingForPythonConfirmation = false;

    await new Promise(resolve => setTimeout(resolve, 200));

    logSystem('Saving final state...');
    await Promise.allSettled([
        saveChatHistory(),
        saveMacros(),
        configManager.save()
    ]).then(results => {
        results.forEach((result, i) => {
            if (result.status === 'rejected') {
                const task = ['History', 'Macros', 'Config'][i];
                console.error(neon.error(`Failed to save ${task} on exit: ${result.reason?.message || result.reason}`));
            }
        });
    });

    logSystem('Goodbye!');
    await new Promise(resolve => setTimeout(resolve, 100));
    process.exit(code);
}


// --- Input Parsing ---
function parseCommand(line) {
    if (!line.startsWith(CMD_PREFIX)) return null;
    const match = line.match(/^\/(\w+)(?:\s+(.*))?$/s);
    if (!match) return null;
    return { command: match[1].toLowerCase(), args: match[2]?.trim() || '' };
}

function expandMacro(line) {
    if (!line.startsWith(MACRO_PREFIX) || isProcessingMacro) return null;
    const match = line.match(/^!([a-zA-Z0-9_-]+)(?:\s+(.*))?$/s);
    if (!match) return null;

    const name = match[1];
    const rawArgs = match[2?.trim()] || '';

    if (macros[name]) {
        isProcessingMacro = true;
        logDebug(`Expanding macro: !${name} with args: "${rawArgs}"`);
        let expanded = macros[name];
        const argList = rawArgs.split(/\s+/).filter(a => a);

        try {
            expanded = expanded.replace(/\$0/g, name);
            expanded = expanded.replace(/\$\*/g, rawArgs);
            expanded = expanded.replace(/\$#/g, argList.length.toString());
            argList.forEach((arg, index) => {
                const placeholder = new RegExp(`\\$${index + 1}`, 'g');
                expanded = expanded.replace(placeholder, arg);
            });

            logMacroExpansion(name, rawArgs, expanded);
            isProcessingMacro = false;
            return expanded.trim();
        } catch (e) {
             logError(`Error expanding macro !${name}:`, e);
             isProcessingMacro = false;
             return line;
        }
    } else {
         logWarning(`Macro !${name} not found.`);
         return null;
    }
}

function logMacroExpansion(name, args, result) {
    console.log(
        `${neon.macroMarker} Expanded ${neon.macroName(`!${name}`)}` +
        (args ? ` with [${neon.macroContent(args)}]` : '') +
        `\n${neon.macroMarker} > ${neon.macroContent(result)}`
    );
}


// --- Main Loop & Input Handling ---
function queueTask(task) {
    if (isExiting) {
        logDebug("Ignoring task queue attempt during exit.");
        return;
    }
    commandQueue.push(task);
    logDebug(`Task queued (${commandQueue.length} total). Type: ${task.handler?.name || 'unknown'}`);
    processQueue();
}

async function processQueue() {
    if (isProcessingQueue || commandQueue.length === 0 || isExiting) {
        return;
    }
    if (isAiThinking || isWaitingForShellConfirmation || isWaitingForPythonConfirmation) {
        logDebug(`Queue processing deferred (AI Thinking: ${isAiThinking}, Awaiting Confirm: ${isWaitingForShellConfirmation || isWaitingForPythonConfirmation})`);
        return;
    }

    isProcessingQueue = true;
    const task = commandQueue.shift();
    logDebug(`Processing task (${commandQueue.length} remaining): ${task.handler?.name || 'unknown'}`);

    try {
        if (task.handler === sendMessageToAI) {
            await sendMessageToAI(task.parts, task.isFeedback, task.isEditOrRegen);
        } else if (task.handler) {
            await task.handler(task.args);
            safePromptRefresh();
            isProcessingQueue = false;
            processQueue();
        } else {
             logWarning("Dequeued task has no handler.");
             isProcessingQueue = false;
             processQueue();
        }
    } catch (error) {
        logError(`Error processing queued task (${task.handler?.name || 'unknown'}):`, error);
        isProcessingQueue = false;
        safePromptRefresh();
        processQueue();
    }
}

async function handleLineInput(line) {
    if (isExiting) return;

    const trimmedLine = line.trim();

    if (isPastingMode) {
        if (trimmedLine.toLowerCase() === '/endpaste') {
            queueTask({ handler: commandHandlers.endpaste, args: '' });
        } else {
            pasteBufferContent.push(line);
            safePromptRefresh();
        }
        return;
    }

    if (isWaitingForShellConfirmation || isWaitingForPythonConfirmation) {
        logWarning("Please answer the confirmation prompt (yes/no) first.");
        safePromptRefresh();
        return;
    }

    if (!trimmedLine) {
        safePromptRefresh();
        return;
    }

    let lineToProcess = trimmedLine;
    const expanded = expandMacro(lineToProcess);
    if (expanded !== null) {
        lineToProcess = expanded;
        if (!lineToProcess) {
            safePromptRefresh();
            return;
        }
    }

    const cmd = parseCommand(lineToProcess);

    if (cmd && commandHandlers[cmd.command]) {
        queueTask({ handler: commandHandlers[cmd.command], args: cmd.args });
    } else if (lineToProcess) {
        queueTask({ handler: sendMessageToAI, parts: [{ text: lineToProcess }] });
    } else {
        safePromptRefresh();
    }
}


async function main() {
    process.on('SIGINT', () => { if (!isExiting) gracefulExit(0); });
    process.on('SIGTERM', () => { if (!isExiting) gracefulExit(0); });
    process.on('unhandledRejection', (reason, promise) => {
        console.error(neon.error('\n[FATAL] Unhandled Rejection:'));
        console.error(neon.error(reason?.stack || reason));
        if (!isExiting) gracefulExit(1);
    });
    process.on('uncaughtException', (error) => {
        console.error(neon.error('\n[FATAL] Uncaught Exception:'));
        console.error(neon.error(error.stack || error));
        if (!isExiting) gracefulExit(1);
    });


    console.log(chalk.cyanBright.bold(`\n${APP_NAME} v${APP_VERSION} - Neon Powered CLI Assistant`));
    console.log(neon.separator());

    await configManager.load();

    API_KEY = configManager.get('apiKey');
    MODEL_NAME = configManager.get('modelName');
    MAX_HISTORY_PAIRS = configManager.get('maxHistory');
    IS_DEBUG_MODE = configManager.get('debug');
    IS_HIGHLIGHTING_ACTIVE = configManager.get('highlight');
    IS_SHELL_ALLOWED = argv.allowShell && configManager.get('allowShell');
    IS_PYTHON_ALLOWED = argv.allowPython && configManager.get('allowPython');
    CURRENT_SYSTEM_PROMPT_TEMPLATE = configManager.get('systemPrompt');
    USE_SYSTEM_PROMPT = configManager.get('useSystemPrompt');
    generationConfig.temperature = configManager.get('temperature');
    safetySettings = ALL_HARM_CATEGORIES.map(c => ({ category: c, threshold: SAFETY_MAP[configManager.get('safety')] }));

    await Promise.allSettled([
        checkTermuxToast(),
        loadChatHistory(),
        loadMacros(),
        ensureDirectoryExists(SESSIONS_DIR)
    ]).then(results => results.forEach((r, i) => {
        if (r.status === 'rejected') {
            const taskName = ['Termux Toast Check', 'History Load', 'Macros Load', 'Sessions Dir Check'][i];
            logWarning(`Subsystem init failed for ${taskName}:`, r.reason);
        }
    }));

    try {
        await initializeModelInstance(true);
        if (!aiModelInstance) throw new Error("AI Model instance is null after initialization attempt.");
    } catch (initError) {
         console.error(chalk.redBright.bold.inverse('\n[FATAL STARTUP ERROR]'));
         console.error(chalk.redBright("Could not initialize the AI model. Please check your API key, model name, and network connection."));
         process.exit(1);
    }

    readlineInterface = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        prompt: '',
        terminal: true,
        completer: (line) => {
            const completions = [];
            const currentInput = line.toLowerCase();

            if (line.startsWith(CMD_PREFIX)) {
                 const cmdPart = line.substring(1).toLowerCase();
                 const potentialCommands = Object.keys(commandHandlers).filter(c => c.startsWith(cmdPart));
                 potentialCommands.forEach(c => completions.push(`${CMD_PREFIX}${c}`));
            } else if (line.startsWith(MACRO_PREFIX)) {
                 const macroPart = line.substring(1).toLowerCase();
                 const potentialMacros = Object.keys(macros).filter(m => m.startsWith(macroPart));
                 potentialMacros.forEach(m => completions.push(`${MACRO_PREFIX}${m}`));
            }
             const hits = completions.filter(c => c.toLowerCase().startsWith(currentInput));
             return [hits.length ? hits : completions, line];
        },
        historySize: 1000,
        removeHistoryDuplicates: true,
    });

    readlineInterface.on('line', (line) => {
        try {
            handleLineInput(line);
        } catch (e) {
            logError("Error handling input line:", e);
            safePromptRefresh();
        }
    });

    process.stdout.on('resize', () => {
          safePromptRefresh();
     });

    readlineInterface.on('close', () => {
        if (!isExiting) {
            gracefulExit(0);
        }
    });

    safePromptRefresh();
}

main().catch(error => {
    console.error(chalk.redBright.bold.inverse('\n[FATAL STARTUP ERROR]'));
    console.error(chalk.redBright(error.stack || error));
    process.exit(1);
});
