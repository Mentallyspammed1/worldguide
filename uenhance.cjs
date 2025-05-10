#!/usr/bin/env node

// interactive_enhancer_v3.js
// Pyrmethus, the Termux Coding Wizard, presents: The Advanced Interactive File Enhancement Spell!

const fs = require('fs').promises;
const path = require('path');
const readline = require('readline');
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require('@google/generative-ai');
const nc = require('nanocolors');
const winston = require('winston');

// ~ Arcane Constants & Configuration ~
const SCRIPT_CONFIG = {
    DEFAULT_MODEL_NAME: "gemini-2.5-pro-exp-03-25", // Stable model for reliability
    MAX_API_CALLS_PER_MINUTE: 59,
    LOG_FILE_NAME: "enhancement_log_v3.txt",
    MAX_FILE_SIZE_BYTES: 2 * 1024 * 1024, // 2MB limit
    BACKUP_SUFFIX: '.bak', // For backup files
    VALIDATE_OUTPUT: true, // Validate enhanced content
    SUPPORTED_EXTENSIONS: ['js', 'py', 'java', 'c', 'cpp', 'go', 'rs', 'sh', 'rb', 'php', 'md', 'txt', 'rst', 'tex', 'json', 'yaml', 'xml', 'ini', 'toml', 'conf'],
    SAFETY_SETTINGS: [
        { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
        { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
        { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
        { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
    ],
};

// ~ Enhanced Logger Configuration ~
const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: winston.format.combine(
        winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss,SSS' }),
        winston.format.json() // Structured logging for better parsing
    ),
    transports: [
        new winston.transports.File({ filename: SCRIPT_CONFIG.LOG_FILE_NAME, options: { flags: 'a' }, encoding: 'utf8' }),
        new winston.transports.Console({
            format: winston.format.combine(
                winston.format.colorize(),
                winston.format.printf(({ timestamp, level, message }) => `${timestamp} [${level.toUpperCase()}] ${message}`)
            )
        })
    ]
});

// ~ Utility Functions ~
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const printMessage = (message, type = 'info', colorFn = nc.cyan) => {
    const prefix = type === 'error' ? '❌ Arcane Anomaly: ' : '✨ Pyrmethus whispers: ';
    const formattedMessage = `${nc.bold(colorFn(prefix))}${colorFn(message)}`;
    logger[type](message);
    console.log(formattedMessage);
};

const askQuestion = (rlInterface, query, defaultValue = null, colorFn = nc.blue) => {
    const prompt = defaultValue ? `${query} ${nc.dim(`(default: ${defaultValue})`)} ` : `${query} `;
    return new Promise(resolve => rlInterface.question(colorFn(prompt), answer => resolve(answer.trim() || defaultValue || '')));
};

const checkFileExists = async (filePath) => {
    try {
        const stats = await fs.stat(filePath);
        return stats.isFile();
    } catch (error) {
        if (error.code === 'ENOENT') return false;
        throw error;
    }
};

const backupFile = async (filePath) => {
    const backupPath = `${filePath}${SCRIPT_CONFIG.BACKUP_SUFFIX}`;
    try {
        await fs.copyFile(filePath, backupPath);
        logger.info(`Backup created: ${backupPath}`);
        printMessage(`Backup inscribed at ${nc.blue(backupPath)}`, 'info', nc.green);
    } catch (error) {
        logger.error(`Backup failed for ${filePath}: ${error.message}`);
        throw new Error(`Backup failed: ${error.message}`);
    }
};

const validateContent = (content, fileExtension) => {
    if (!content) return false;
    if (SCRIPT_CONFIG.SUPPORTED_EXTENSIONS.includes(fileExtension)) {
        // Basic validation for specific types
        if (['json', 'yaml', 'toml'].includes(fileExtension)) {
            try {
                if (fileExtension === 'json') JSON.parse(content);
                // Add YAML/TOML parsing if libraries are added
                return true;
            } catch {
                return false;
            }
        }
        return true; // Other types assumed valid if non-empty
    }
    return true; // Default to true for unknown types
};

// ~ Core Functions ~

/**
 * Configures the Gemini AI model with safety settings and model selection.
 * @returns {Object} Configured generative model
 * @throws {Error} If API key is missing or model configuration fails
 */
const configureGeminiOracle = async () => {
    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey) throw new Error("GOOGLE_API_KEY not set.");

    const modelName = process.env.GEMINI_MODEL_NAME || SCRIPT_CONFIG.DEFAULT_MODEL_NAME;
    printMessage(`Attuning to ${nc.yellow(modelName)} Oracle...`, 'info', nc.magenta);

    try {
        const genAI = new GoogleGenerativeAI(apiKey);
        const model = genAI.getGenerativeModel({
            model: modelName,
            safetySettings: SCRIPT_CONFIG.SAFETY_SETTINGS,
        });
        printMessage(`Gemini Oracle (${modelName}) attuned with BLOCK_NONE safety.`, 'info', nc.green);
        logger.info(`Configured model: ${modelName}, safety: ${JSON.stringify(SCRIPT_CONFIG.SAFETY_SETTINGS)}`);
        return model;
    } catch (error) {
        printMessage(`Failed to attune Oracle: ${error.message}`, 'error', nc.red);
        throw error;
    }
};

/**
 * Reads and validates the input file content.
 * @param {string} filePath - Path to the input file
 * @returns {string} File content
 * @throws {Error} If file reading fails or size exceeds limit
 */
const readScrollContent = async (filePath) => {
    printMessage(`Unfurling scroll: ${nc.blue(filePath)}...`);
    try {
        const content = await fs.readFile(filePath, 'utf-8');
        if (content.length > SCRIPT_CONFIG.MAX_FILE_SIZE_BYTES) {
            throw new Error(`Scroll too vast (${(content.length / 1024 / 1024).toFixed(2)}MB). Max: ${SCRIPT_CONFIG.MAX_FILE_SIZE_BYTES / 1024 / 1024}MB`);
        }
        logger.info(`Read ${filePath}: ${content.length} bytes`);
        return content;
    } catch (error) {
        printMessage(`Failed to read ${filePath}: ${error.message}`, 'error', nc.red);
        throw error;
    }
};

/**
 * Consults Gemini AI to enhance file content based on its type.
 * @param {string} fileIdentifier - File name for prompt
 * @param {string} fileExtension - File extension
 * @param {string} content - Original content
 * @param {Object} model - Configured Gemini model
 * @returns {Object} Enhanced content, explanation, and API call count
 */
const consultOracleForEnhancement = async (fileIdentifier, fileExtension, content, model) => {
    const typeHint = fileExtension ? ` and is a '.${fileExtension}' file` : '';
    const codeBlockHint = fileExtension || 'text';
    const prompt = `
You are Pyrmethus's familiar, an expert in content enhancement.
Enhance the following file content for '${fileIdentifier}'${typeHint}.

For code (e.g., js, py, java):
- Add/improve docstrings, comments, type hints.
- Optimize performance, readability, and maintainability.
- Fix bugs, adhere to best practices.
For text (e.g., md, txt):
- Improve clarity, grammar, structure.
- Enhance tone and formatting.
For configs (e.g., json, yaml):
- Ensure validity, add comments, improve organization.

Return FULL enhanced content in a single code block:
\`\`\`${codeBlockHint}
... enhanced content ...
\`\`\`
Follow with a BRIEF explanation of changes.
If no changes needed, return original content with explanation.
If content is invalid, explain why.

Original Content:
\`\`\`
${content}
\`\`\`
`;

    printMessage(`Consulting Oracle for ${nc.blue(fileIdentifier)}...`, 'info', nc.cyan);
    logger.info(`Prompt length for ${fileIdentifier}: ${prompt.length}`);

    try {
        const result = await model.generateContent(prompt);
        const responseText = result.response.text().trim();
        logger.info(`Response length for ${fileIdentifier}: ${responseText.length}`);

        const contentPattern = /```(?:[a-zA-Z0-9_.-]+)?\n([\s\S]*?)\n```/;
        const match = responseText.match(contentPattern);
        let enhancedContent = content;
        let explanation = "No explanation provided.";

        if (match && match[1]) {
            enhancedContent = match[1];
            const explanationStart = responseText.indexOf(match[0]) + match[0].length;
            explanation = responseText.substring(explanationStart).trim() || "No explanation provided.";
            if (!validateContent(enhancedContent, fileExtension)) {
                logger.warn(`Invalid enhanced content for ${fileIdentifier}`);
                return { enhancedContent: content, explanation: "Enhanced content invalid; reverting to original.", callsMade: 1 };
            }
        } else {
            explanation = responseText;
            printMessage(`No valid content block in response for ${fileIdentifier}.`, 'error', nc.yellow);
        }

        if (result.response.promptFeedback?.blockReason) {
            logger.warn(`Blocked prompt for ${fileIdentifier}: ${result.response.promptFeedback.blockReason}`);
            return { enhancedContent: content, explanation: `Blocked: ${result.response.promptFeedback.blockReason}`, callsMade: 1 };
        }

        return { enhancedContent, explanation, callsMade: 1 };
    } catch (error) {
        printMessage(`Oracle consultation failed for ${fileIdentifier}: ${error.message}`, 'error', nc.red);
        return { enhancedContent: content, explanation: `Error: ${error.message}`, callsMade: 1 };
    }
};

/**
 * Writes enhanced content to the output file.
 * @param {string} filePath - Output file path
 * @param {string} content - Content to write
 * @throws {Error} If writing fails
 */
const inscribeEnhancedScroll = async (filePath, content) => {
    printMessage(`Inscribing to ${nc.blue(filePath)}...`);
    try {
        await fs.writeFile(filePath, content, 'utf-8');
        logger.info(`Wrote to ${filePath}`);
        printMessage(`Inscribed ${nc.blue(filePath)} successfully!`, 'info', nc.green);
    } catch (error) {
        printMessage(`Failed to inscribe ${filePath}: ${error.message}`, 'error', nc.red);
        throw error;
    }
};

// ~ Main Function ~
/**
 * Orchestrates the interactive file enhancement process.
 */
const main = async () => {
    logger.info("Starting Interactive File Enhancement Spell v3");
    printMessage("Hark! The Advanced Enhancement Spell begins...", 'info', nc.magenta);

    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

    try {
        // Input file
        const inputFilePath = path.resolve(await askQuestion(rl, nc.bold("Enter input scroll path: "), null, nc.cyan));
        if (!inputFilePath || !(await checkFileExists(inputFilePath))) {
            throw new Error(`Invalid input scroll: ${inputFilePath}`);
        }

        const fileExtension = path.extname(inputFilePath).slice(1).toLowerCase();
        if (!SCRIPT_CONFIG.SUPPORTED_EXTENSIONS.includes(fileExtension)) {
            printMessage(`Warning: Extension '.${fileExtension}' may not be fully supported.`, 'info', nc.yellow);
        }

        // Output file
        const defaultOutputName = `${path.basename(inputFilePath, path.extname(inputFilePath))}_enhanced${path.extname(inputFilePath)}`;
        const outputFilePath = path.resolve(await askQuestion(rl, nc.bold("Enter output parchment path: "), defaultOutputName, nc.cyan));

        printMessage(`Source: ${nc.blue(inputFilePath)}`, 'info', nc.cyan);
        printMessage(`Destination: ${nc.blue(outputFilePath)}`, 'info', nc.cyan);

        // Handle overwrites
        if (inputFilePath === outputFilePath) {
            if ((await askQuestion(rl, nc.bold(nc.yellow(`Overwrite ${nc.blue(inputFilePath)}? (yes/NO): `)), 'NO', nc.yellow)).toLowerCase() !== 'yes') {
                printMessage("Spell aborted to preserve original scroll.", 'info', nc.yellow);
                return;
            }
            await backupFile(inputFilePath);
        } else if (await checkFileExists(outputFilePath)) {
            if ((await askQuestion(rl, nc.bold(nc.yellow(`Overwrite ${nc.blue(outputFilePath)}? (yes/NO): `)), 'NO', nc.yellow)).toLowerCase() !== 'yes') {
                printMessage("Spell aborted to preserve existing parchment.", 'info', nc.yellow);
                return;
            }
        }

        // Ensure output directory exists
        const outputDir = path.dirname(outputFilePath);
        if (!(await checkFileExists(outputDir))) {
            await fs.mkdir(outputDir, { recursive: true });
            printMessage(`Created sanctuary: ${nc.blue(outputDir)}`, 'info', nc.yellow);
        }

        // Configure and enhance
        const model = await configureGeminiOracle();
        const startTime = Date.now();
        const originalContent = await readScrollContent(inputFilePath);
        const { enhancedContent, explanation } = await consultOracleForEnhancement(
            path.basename(inputFilePath),
            fileExtension,
            originalContent,
            model
        );

        printMessage(`Oracle's insight:\n${nc.dim(explanation)}`, 'info', nc.green);

        // Write output
        if (enhancedContent !== originalContent) {
            await inscribeEnhancedScroll(outputFilePath, enhancedContent);
        } else {
            printMessage(`No enhancements made for ${nc.blue(inputFilePath)}.`, 'info', nc.yellow);
        }

        const elapsed = (Date.now() - startTime) / 1000;
        printMessage(`Ritual complete in ${nc.yellow(elapsed.toFixed(2))} seconds.`, 'info', nc.magenta);

    } catch (error) {
        printMessage(`Spell disrupted: ${error.message}`, 'error', nc.red);
        process.exitCode = 1;
    } finally {
        printMessage("Closing Oracle connection.", 'info', nc.gray);
        rl.close();
    }
};

// ~ Execute ~
if (require.main === module) {
    main();
}