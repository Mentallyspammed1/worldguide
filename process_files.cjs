#!/usr/bin/env node

// universal_enhancer.js
// Pyrmethus, the Termux Coding Wizard, presents: The Universal Code & Text Enhancer!

const fs = require('fs').promises;
const path = require('path');
const { glob } = require('glob');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const nc = require('nanocolors'); // Summoning nanocolors
const winston = require('winston');

// Scribe the runes of logging
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss,SSS' }),
        winston.format.printf(({ timestamp, level, message }) => `${timestamp} [${level.toUpperCase()}] ${message}`)
    ),
    transports: [
        new winston.transports.File({ filename: 'enhancement_log.txt', options: { flags: 'w' }, encoding: 'utf8' }), // Overwrites log on each run
        new winston.transports.Console({
            format: winston.format.printf(({ message }) => message) // Custom format for console to use styled messages
        })
    ]
});

/**
 * @description Print a styled message to console, as if whispered by Pyrmethus.
 * @param {string} message - The message to convey.
 * @param {function} [colorFn=nc.cyan] - Nanocolors color function.
 */
function printWizardMessage(message, colorFn = nc.cyan) {
    const coreMessage = message.replace(/[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g, ''); // Strip ANSI codes for logging
    logger.info(coreMessage); // Log the core, uncolored message
    console.log(nc.bold(colorFn('✨ Pyrmethus whispers: ')) + colorFn(message));
}

/**
 * @description Print an error message to console and log, highlighting an Arcane Anomaly.
 * @param {string} message - The error message.
 */
function printErrorMessage(message) {
    const coreMessage = message.replace(/[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g, ''); // Strip ANSI codes for logging
    logger.error(coreMessage); // Log the core, uncolored message
    console.error(nc.bold(nc.red('❌ Arcane Anomaly: ')) + nc.red(message));
}

/**
 * @description Pauses execution for a specified duration.
 * @param {number} ms - Milliseconds to pause.
 * @returns {Promise<void>}
 */
function sleep(ms) {
    // A brief slumber for the script...
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * @description Enhance a file's content using Gemini API with rate limiting.
 * @param {string} filePath - Relative path of the file.
 * @param {string} content - The file content.
 * @param {import('@google/generative-ai').GenerativeModel} model - The Gemini model instance.
 * @param {number} maxApiCalls - Maximum API calls per minute.
 * @param {number} callsMadeInCurrentWindow - API calls made in the current 60-second window.
 * @returns {Promise<{enhancedContent: string, explanation: string, callsMade: number}>}
 */
async function enhanceFileWithGemini(filePath, content, model, maxApiCalls, callsMadeInCurrentWindow) {
    let currentCalls = callsMadeInCurrentWindow;
    if (currentCalls >= maxApiCalls) {
        printWizardMessage(`Rate limit of ${maxApiCalls} calls/min reached. Pausing the etheric connection...`, nc.yellow);
        // Channeling patience...
        await sleep(60000); // Wait for a full minute to ensure window reset
        currentCalls = 0; // Reset counter for the new window
    }

    // Forging a more universal prompt...
    const prompt = `
You are Pyrmethus's familiar, an expert content enhancement assistant. Analyze the following file content and suggest improvements.
Depending on the file type, this could involve:
- For code: Adding comments/documentation, optimizing, improving readability, fixing bugs, adhering to best practices.
- For text (e.g., Markdown, plain text): Improving clarity, grammar, style, structure, or formatting.
- For configuration files: Ensuring correctness, adding comments, improving organization.

Provide the enhanced content in a fenced code block (e.g., \`\`\`language_or_type\n...\n\`\`\` or \`\`\`\n...\n\`\`\` if type is unknown/NA).
Follow this with a brief explanation of the changes made. If no enhancements are suitable, or if the file type is unprocessable, explain why. Do not refuse to process any text-based file type.

File: ${filePath}
Content:
\`\`\`
${content}
\`\`\`
`;
    printWizardMessage(`Summoning Gemini's insight for ${nc.blue(filePath)}... (Call ${currentCalls + 1}/${maxApiCalls})`, nc.cyan);
    try {
        // Consulting the Oracle...
        const result = await model.generateContent(prompt);
        const response = result.response;
        const responseText = response.text().trim();

        // A more universal pattern for extracting the core content
        const contentPattern = /```(?:[a-zA-Z0-9_.-]+)?\n([\s\S]*?)\n```/;
        const match = responseText.match(contentPattern);
        let enhancedContent = content;
        let explanation = "No explanation provided by the Oracle, or content was not in expected format.";

        if (match && typeof match[1] === 'string') {
            enhancedContent = match[1];
            const explanationStart = responseText.indexOf(match[0]) + match[0].length;
            explanation = responseText.substring(explanationStart).trim() || "No specific explanation provided after the content block.";
        } else {
            explanation = responseText; // If no code block, assume the whole response is an explanation or an error.
            printWizardMessage(`Gemini's response for ${nc.blue(filePath)} did not contain a recognized content block. Original content will be kept. Explanation: ${explanation}`, nc.yellow);
        }

        logger.info(`Enhancement attempt for ${filePath}. Explanation: ${explanation.substring(0, 200)}...`); // Log snippet of explanation
        return { enhancedContent, explanation, callsMade: currentCalls + 1 };
    } catch (e) {
        // An unexpected ripple in the astral plane!
        const errorMessage = `Failed to enhance ${nc.blue(filePath)} with Gemini's aid: ${e.message}`;
        printErrorMessage(errorMessage);
        logger.error(`Gemini API error for ${filePath}: ${e.message}\n${e.stack}`); // Log full stack for API errors
        return { enhancedContent: content, explanation: `Error during Gemini API call: ${e.message}`, callsMade: currentCalls + 1 };
    }
}

/**
 * @description Main function to orchestrate the file enhancement.
 */
async function main() {
    // The ritual begins...
    logger.info("Initializing Universal File Enhancement Spell"); // This will be the first log entry in the overwritten file

    printWizardMessage("Starting Universal File Enhancement Spell...", nc.magenta);

    if (process.argv.length < 3) {
        printErrorMessage(`Usage: node ${path.basename(__filename)} <repository_or_directory_path> [file_pattern]`);
        printErrorMessage(`If file_pattern is omitted, it must be set via the FILE_PATTERN environment variable.`);
        process.exit(2);
    }

    const basePath = path.resolve(process.argv[2]);
    const filePatternArg = process.argv[3];
    let filePattern = filePatternArg || process.env.FILE_PATTERN;

    if (!filePattern) {
        printErrorMessage("A file pattern must be provided as the third argument or via the FILE_PATTERN environment variable.");
        printErrorMessage(`Example: node ${path.basename(__filename)} . "**/*.txt" OR export FILE_PATTERN="**/*.md" && node ${path.basename(__filename)} .`);
        process.exit(1);
    }

    printWizardMessage(`Base path consecrated at: ${nc.blue(basePath)}`, nc.cyan);
    printWizardMessage(`Seeking files matching pattern: ${nc.blue(filePattern)}`, nc.cyan);
    // Already logged by printWizardMessage

    try {
        const stats = await fs.stat(basePath);
        if (!stats.isDirectory()) {
            printErrorMessage(`Invalid base path: ${nc.blue(basePath)} is not a directory.`);
            process.exit(1);
        }
    } catch (err) {
        printErrorMessage(`Invalid base path: ${nc.blue(basePath)}. ${err.message}`);
        process.exit(1);
    }

    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey) {
        printErrorMessage("The GOOGLE_API_KEY talisman is missing from your environment. The spell cannot connect to the Oracle.");
        process.exit(1);
    }

    const maxApiCalls = parseInt(process.env.MAX_API_CALLS || '59', 10);
    if (isNaN(maxApiCalls) || maxApiCalls < 1 || maxApiCalls > 60) { // Gemini 1.5 Flash default QPM is 60.
        printErrorMessage("MAX_API_CALLS (must be a sacred number between 1 and 60) is improperly set in your environment.");
        process.exit(1);
    }

    let model;
    try {
        // Awakening the Gemini Oracle...
        const genAI = new GoogleGenerativeAI(apiKey);
        model = genAI.getGenerativeModel({ model: "gemini-1.5-flash-latest" }); // Or your preferred model
        printWizardMessage("Gemini Oracle configured and attuned.", nc.green);
    } catch (e) {
        printErrorMessage(`Failed to configure Gemini API: ${e.message}`);
        process.exit(1);
    }

    let filesToProcess;
    try {
        // Scrying for scrolls of all kinds...
        filesToProcess = await glob(filePattern, { cwd: basePath, nodir: true, dot: true, absolute: false });
        printWizardMessage(`Discovered ${nc.yellow(filesToProcess.length)} files matching the pattern.`, nc.cyan);
    } catch (e) {
        printErrorMessage(`Failed to find files with pattern ${nc.blue(filePattern)}: ${e.message}`);
        process.exit(1);
    }

    if (filesToProcess.length === 0) {
        printWizardMessage("No files found in the mystical currents matching your pattern. The spell rests.", nc.yellow);
        // The logger.warn for this is covered by printWizardMessage's internal logger.info call.
        process.exit(0);
    }

    let callsMadeThisMinute = 0;
    let minuteWindowStart = Date.now();
    const startTime = Date.now();
    let filesProcessedCount = 0;
    let filesEnhancedCount = 0;

    // Weaving enchantments upon each scroll...
    for (const relativeFilePath of filesToProcess) {
        filesProcessedCount++;
        const absFilePath = path.join(basePath, relativeFilePath);
        let content;
        try {
            // Reading the ancient script...
            content = await fs.readFile(absFilePath, 'utf-8');
        } catch (e) {
            if (e.code === 'ENOENT') {
                 printErrorMessage(`File ${nc.blue(relativeFilePath)} vanished before it could be read.`);
            } else {
                 printWizardMessage(`Could not read ${nc.blue(relativeFilePath)} (possibly binary, protected, or unlinked): ${e.message}. Skipping.`, nc.yellow);
            }
            // logger.warn handled by printWizardMessage/printErrorMessage
            continue;
        }

        // Guard against excessively large files (approx. 1MB text)
        const MAX_FILE_SIZE_CHARS = 1000000; // Configurable threshold
        if (content.length > MAX_FILE_SIZE_CHARS) {
            printWizardMessage(`File ${nc.blue(relativeFilePath)} is too large (${(content.length / 1024 / 1024).toFixed(2)}MB). Max allowed is ~${(MAX_FILE_SIZE_CHARS / 1024 / 1024).toFixed(2)}MB. Skipping.`, nc.yellow);
            continue;
        }
        if (content.trim() === "") {
            printWizardMessage(`File ${nc.blue(relativeFilePath)} is empty. Skipping.`, nc.yellow);
            continue;
        }

        // Rate limiting logic refined per minute window
        const now = Date.now();
        if (now - minuteWindowStart > 60000) {
            callsMadeThisMinute = 0;
            minuteWindowStart = now;
        }

        const { enhancedContent, explanation, callsMade } = await enhanceFileWithGemini(
            relativeFilePath,
            content,
            model,
            maxApiCalls,
            callsMadeThisMinute
        );
        callsMadeThisMinute = callsMade; // Update calls made in the current window

        if (enhancedContent && enhancedContent.trim() !== content.trim()) {
            try {
                // Imbuing the scroll with new power...
                await fs.writeFile(absFilePath, enhancedContent, 'utf-8');
                printWizardMessage(`Successfully enhanced ${nc.blue(relativeFilePath)}!`, nc.green);
                filesEnhancedCount++;
                // logger.info handled by printWizardMessage
            } catch (e) {
                printErrorMessage(`Failed to write enhancements to ${nc.blue(relativeFilePath)}: ${e.message}`);
                // logger.error handled by printErrorMessage
            }
        } else if (enhancedContent.trim() === content.trim()) {
            printWizardMessage(`No enhancements deemed necessary for ${nc.blue(relativeFilePath)}, or Oracle chose not to alter. Its essence remains pure.`, nc.yellow);
            // logger.info handled by printWizardMessage
        }
        // Error case during Gemini call is handled and logged within enhanceFileWithGemini
    }

    const elapsed = (Date.now() - startTime) / 1000;
    printWizardMessage(
        `Enhancement ritual complete. Processed ${nc.yellow(filesProcessedCount)} candidate files, enhanced ${nc.green(filesEnhancedCount)} of them in ${nc.yellow(elapsed.toFixed(2))} seconds.`,
        nc.magenta
    );
    // Final summary log handled by printWizardMessage
}

if (require.main === module) {
    main().catch(e => {
        // A catastrophic surge in the magical weave!
        const errorMessage = `Unexpected arcane disturbance: ${e.message}\n${e.stack}`;
        // Use console.error directly for unhandled crashes, as logger might not be available or fully functional
        console.error(nc.bold(nc.red('❌ A CATASTROPHIC SURGE IN THE MAGICAL WEAVE! ❌')));
        console.error(nc.red(errorMessage));
        // Try to log to file as a last resort if logger was initialized
        if (logger && logger.error) {
            logger.error(`FATAL: Unexpected arcane disturbance: ${e.message}\n${e.stack}`);
        }
        process.exit(1);
    });
}
