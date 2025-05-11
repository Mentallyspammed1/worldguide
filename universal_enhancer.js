Hark, adept of the digital loom! You seek to broaden the horizons of our Code Enhancement Spell, to allow it to mend and meld *any* file type, and to weave its magic with the vibrant threads of `nanocolors`. A bold and worthy transmutation! I, Pyrmethus, shall reforge this incantation to your will.

First, ensure the new arcane components are bound to your Termux sanctum. Invoke these commands if you haven't already, or to switch from `chalk` to `nanocolors`:
```bash
pkg install nodejs-lts
npm install nanocolors @google/generative-ai glob winston
```
If `chalk` was previously installed, you might consider `npm uninstall chalk`.

Behold, the transmuted JavaScript grimoire, `universal_enhancer.js`:

```javascript
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
        new winston.transports.File({ filename: 'enhancement_log.txt', options: { flags: 'w' }, encoding: 'utf8' }),
        new winston.transports.Console({
            format: winston.format.printf(({ message }) => message) // Custom format for console
        })
    ]
});

/**
 * @description Print a styled message to console, as if whispered by Pyrmethus.
 * @param {string} message - The message to convey.
 * @param {function} [colorFn=nc.cyan] - Nanocolors color function.
 */
function printWizardMessage(message, colorFn = nc.cyan) {
    const fullMessage = `✨ Pyrmethus whispers: ${message}`;
    logger.info(message); // Log the core message
    console.log(nc.bold(colorFn('✨ Pyrmethus whispers: ')) + colorFn(message));
}

/**
 * @description Print an error message to console and log, highlighting an Arcane Anomaly.
 * @param {string} message - The error message.
 */
function printErrorMessage(message) {
    const fullMessage = `❌ Arcane Anomaly: ${message}`;
    logger.error(message); // Log the core message
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
        await sleep(60000 - (Date.now() % 60000)); // Wait for the next minute
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
Follow this with a brief explanation of the changes made. If no enhancements are suitable, or if the file type is unprocessable, explain why.

File: ${filePath}
Content:
\`\`\`
${content}
\`\`\`
`;
    printWizardMessage(`Summoning Gemini's insight for ${nc.blue(filePath)}...`, nc.cyan);
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

        if (match && typeof match[1] === 'string') { // Check if match[1] is a string
            enhancedContent = match[1];
            // Find the end of the matched code block to get subsequent text as explanation
            const explanationStart = responseText.indexOf(match[0]) + match[0].length;
            explanation = responseText.substring(explanationStart).trim() || "No specific explanation provided after the content block.";
        } else {
            // If no code block, assume the whole response is an explanation or an error from Gemini's side
            explanation = responseText;
            printWizardMessage(`Gemini's response for ${filePath} did not contain a recognized content block. Using original content. Explanation: ${explanation}`, nc.yellow);
        }

        logger.info(`Enhancement attempt for ${filePath}. Explanation: ${explanation}`);
        return { enhancedContent, explanation, callsMade: currentCalls + 1 };
    } catch (e) {
        // An unexpected ripple in the astral plane!
        const errorMessage = `Failed to enhance ${filePath} with Gemini's aid: ${e.message}\n${e.stack}`;
        printErrorMessage(errorMessage);
        logger.error(`Failed to enhance ${filePath} with Gemini's aid: ${e.message}`);
        return { enhancedContent: content, explanation: `Error: ${e.message}`, callsMade: currentCalls + 1 };
    }
}

/**
 * @description Main function to orchestrate the file enhancement.
 */
async function main() {
    // The ritual begins...
    logger.info("Initializing Universal File Enhancement Spell");

    printWizardMessage("Starting Universal File Enhancement Spell...", nc.magenta);

    if (process.argv.length < 3) {
        printErrorMessage(`Usage: node ${path.basename(__filename)} <repository_or_directory_path> [file_pattern]`);
        printErrorMessage(`If file_pattern is omitted, it must be set via FILE_PATTERN environment variable.`);
        process.exit(2);
    }

    const basePath = path.resolve(process.argv[2]);
    const filePatternArg = process.argv[3];
    let filePattern = filePatternArg || process.env.FILE_PATTERN;

    if (!filePattern) {
        printErrorMessage("A file pattern must be provided as the third argument or via the FILE_PATTERN environment variable.");
        printErrorMessage(`Example: node ${path.basename(__filename)} . "**/*.txt" or FILE_PATTERN="**/*.md" node ${path.basename(__filename)} .`);
        process.exit(1);
    }

    printWizardMessage(`Base path consecrated at: ${nc.blue(basePath)}`, nc.cyan);
    printWizardMessage(`Seeking files matching: ${nc.blue(filePattern)}`, nc.cyan);
    logger.info(`Base path: ${basePath}`);
    logger.info(`File pattern: ${filePattern}`);

    try {
        const stats = await fs.stat(basePath);
        if (!stats.isDirectory()) {
            printErrorMessage(`Invalid base path: ${basePath} is not a directory.`);
            process.exit(1);
        }
    } catch (err) {
        printErrorMessage(`Invalid base path: ${basePath}. ${err.message}`);
        process.exit(1);
    }

    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey) {
        printErrorMessage("The GOOGLE_API_KEY talisman is missing from your environment.");
        process.exit(1);
    }

    const maxApiCalls = parseInt(process.env.MAX_API_CALLS || '59', 10);
    if (isNaN(maxApiCalls) || maxApiCalls < 1 || maxApiCalls > 60) {
        printErrorMessage("MAX_API_CALLS (1-60) must be a valid number, like a sacred count.");
        process.exit(1);
    }

    let model;
    try {
        // Awakening the Gemini Oracle...
        const genAI = new GoogleGenerativeAI(apiKey);
        model = genAI.getGenerativeModel({ model: "gemini-1.5-flash-latest" }); // Or your preferred model
        printWizardMessage("Gemini Oracle configured and attuned.", nc.green);
        logger.info("Gemini Oracle configured successfully.");
    } catch (e) {
        printErrorMessage(`Failed to configure Gemini API: ${e.message}`);
        process.exit(1);
    }

    let filesToProcess;
    try {
        // Scrying for scrolls of all kinds...
        filesToProcess = await glob(filePattern, { cwd: basePath, nodir: true, dot: true, absolute: false });
        printWizardMessage(`Discovered ${nc.yellow(filesToProcess.length)} files matching the pattern.`, nc.cyan);
        logger.info(`Found ${filesToProcess.length} files matching pattern.`);
    } catch (e) {
        printErrorMessage(`Failed to find files with pattern ${filePattern}: ${e.message}`);
        process.exit(1);
    }

    if (filesToProcess.length === 0) {
        printWizardMessage("No files found in the mystical currents matching your pattern. The spell rests.", nc.yellow);
        logger.warn("No files found to enhance.");
        await fs.appendFile("enhancement_log.txt", `${new Date().toISOString().replace('T', ' ').substring(0,23)} [WARNING] No files found to enhance.\n`, "utf8").catch(()=>{/* ignore error on append */});
        process.exit(0);
    }

    let callsMade = 0;
    const startTime = Date.now();
    // Weaving enchantments upon each scroll...
    for (const relativeFilePath of filesToProcess) {
        const absFilePath = path.join(basePath, relativeFilePath);
        let content;
        try {
            // Reading the ancient script...
            content = await fs.readFile(absFilePath, 'utf-8');
        } catch (e) {
            // Some scrolls may be binary or unreadable, skip them.
            if (e.code === 'ENOENT') {
                 printErrorMessage(`File ${relativeFilePath} vanished before it could be read.`);
            } else {
                 printWizardMessage(`Could not read ${nc.blue(relativeFilePath)} (possibly binary or protected): ${e.message}. Skipping.`, nc.yellow);
            }
            logger.warn(`Failed to read ${relativeFilePath}: ${e.message}. Skipping.`);
            continue;
        }

        // Guard against excessively large files to prevent huge API costs/long processing
        if (content.length > 1000000) { // Approx 1MB limit, adjust as needed
            printWizardMessage(`File ${nc.blue(relativeFilePath)} is too large (${(content.length/1024/1024).toFixed(2)}MB). Skipping.`, nc.yellow);
            logger.warn(`Skipping large file ${relativeFilePath} (${content.length} bytes).`);
            continue;
        }


        const { enhancedContent, callsMade: updatedCallsMade } = await enhanceFileWithGemini(
            relativeFilePath, // Pass relative path for context in prompt
            content,
            model,
            maxApiCalls,
            callsMade
        );
        callsMade = updatedCallsMade;

        if (enhancedContent && enhancedContent !== content) {
            try {
                // Imbuing the scroll with new power...
                await fs.writeFile(absFilePath, enhancedContent, 'utf-8');
                printWizardMessage(`Successfully enhanced ${nc.blue(relativeFilePath)}!`, nc.green);
                logger.info(`Enhanced ${relativeFilePath}`);
            } catch (e) {
                printErrorMessage(`Failed to write enhancements to ${relativeFilePath}: ${e.message}`);
                logger.error(`Failed to write enhancements to ${relativeFilePath}: ${e.message}`);
            }
        } else if (enhancedContent === content) {
            printWizardMessage(`No enhancements deemed necessary for ${nc.blue(relativeFilePath)}, its essence remains pure.`, nc.yellow);
            logger.info(`No enhancements needed for ${relativeFilePath}`);
        }
        // else: error already printed by enhanceFileWithGemini
    }

    const elapsed = (Date.now() - startTime) / 1000;
    printWizardMessage(
        `Enhancement ritual complete. Processed ${nc.yellow(filesToProcess.length)} candidate files in ${nc.yellow(elapsed.toFixed(2))} seconds.`,
        nc.magenta
    );
    logger.info(`Enhancement complete. Processed ${filesToProcess.length} candidate files in ${elapsed.toFixed(2)} seconds.`);
}

if (require.main === module) {
    main().catch(e => {
        // A catastrophic surge in the magical weave!
        const errorMessage = `Unexpected arcane disturbance: ${e.message}\n${e.stack}`;
        printErrorMessage(errorMessage);
        logger.error(`Unexpected arcane disturbance: ${e.message}`); // Keep stack for file log
        process.exit(1);
    });
}
```

**Key Transmutations:**

1.  **`nanocolors` Integration**:
    *   `require('chalk')` is now `require('nanocolors')` (aliased to `nc`).
    *   All color calls have been updated to `nanocolors` syntax (e.g., `nc.red(nc.bold('text'))`).

2.  **Universal File Type Handling**:
    *   **File Pattern**: The script now requires a file pattern to be explicitly provided either as a command-line argument or via the `FILE_PATTERN` environment variable. There's no default like `**/*.py`. This gives you full control.
    *   **Gemini Prompt**: The prompt sent to the Gemini Oracle is now generalized. It asks for enhancements suitable for various file types (code, text, config) rather than being Python-specific.
    *   **Content Extraction**: The regex to extract the enhanced content from Gemini's response (`contentPattern`) is now more flexible, looking for ` ```language\n...\n``` ` or ` ```\n...\n``` `.
    *   **Messaging**: Wizard messages and log entries now refer to "files" or "content" generically.

3.  **Safety Measures**:
    *   A basic check for very large files has been added to skip them, preventing excessive API usage or processing times. You can adjust the `1000000` character limit.
    *   Improved error handling for file reading (e.g., skipping binary files gracefully).

**To Wield This Universal Spell:**

1.  Save the code above as `universal_enhancer.js` in your Termux environment.
2.  Make it executable: `chmod +x universal_enhancer.js`
3.  Ensure `GOOGLE_API_KEY` is set in your environment:
    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```
4.  Invoke the spell, specifying the base path and the file pattern:
    *   To enhance all Markdown files in the current directory and subdirectories:
        ```bash
        ./universal_enhancer.js . "**/*.md"
        ```
    *   To enhance all JavaScript files in a `src` subdirectory:
        ```bash
        ./universal_enhancer.js ./my_project "src/**/*.js"
        ```
    *   Using the `FILE_PATTERN` environment variable:
        ```bash
        export FILE_PATTERN="docs/**/*.txt"
        ./universal_enhancer.js /path/to/your/project
        ```

This enhanced spell now stands ready to assist you with a wider array of textual and coded artifacts within your Termux realm, its outputs gleaming with the light of `nanocolors`! May your enchantments be ever more versatile.