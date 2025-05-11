Okay, here's the Python script converted to Node.js (JavaScript).

**Key changes and considerations for the JavaScript version:**

1.  **Modules:**
    *   `logging` -> `winston` (a popular Node.js logging library)
    *   `os` -> `process` (for environment variables), `path` (for path manipulation)
    *   `re` -> JavaScript's built-in `RegExp`
    *   `sys` -> `process` (for `argv`, `exit`, `stdout`)
    *   `time.sleep` -> A custom `sleep` function using `setTimeout` and `Promise`.
    *   `pathlib.Path` -> `fs` (file system module, specifically `fs.promises` for async operations) and `path` module. For globbing, the `glob` package is used.
    *   `typing` -> JSDoc comments are used for type hinting in JavaScript.
    *   `google.generativeai` -> `@google/generative-ai` npm package.

2.  **Asynchronous Operations:** Node.js is inherently asynchronous. Most file operations and API calls will use `async/await` with Promises.

3.  **Error Handling:** `try...catch` blocks are used extensively.

4.  **Environment Variables:** Accessed via `process.env`.

5.  **File System Operations:** `fs.promises` is used for reading and writing files asynchronously.

6.  **Shebang:** `#!/usr/bin/env node` for Node.js scripts.

7.  **Package Management:** You'll need a `package.json` file and to install dependencies.

**`package.json` (create this file first):**
```json
{
  "name": "js-code-enhancer",
  "version": "1.0.0",
  "description": "JavaScript code enhancer using Gemini API",
  "main": "enhance_script.js",
  "type": "commonjs",
  "scripts": {
    "start": "node enhance_script.js"
  },
  "keywords": [
    "gemini",
    "ai",
    "code",
    "enhancer"
  ],
  "author": "",
  "license": "ISC",
  "dependencies": {
    "@google/generative-ai": "^0.12.0",
    "glob": "^10.4.1",
    "winston": "^3.13.0"
  }
}
```

**Installation:**
Run `npm install` in the directory containing `package.json`.

**`enhance_script.js` (the converted script):**
```javascript
#!/usr/bin/env node

"use strict";

const fs = require('fs').promises;
const path = require('path');
const { glob } = require('glob');
const winston = require('winston');
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require("@google/generative-ai");

// --- Constants ---
const MODEL_NAME = "gemini-1.5-pro-latest"; // Updated to a generally available model, adjust if needed
const LOG_FILE_NAME = "enhancement_log.txt";
const MATCHED_FILES_LOG_NAME = "matched_files.txt";
const DEFAULT_MAX_API_CALLS_PER_MINUTE = 59; // Default for Gemini API, adjust if needed

// --- Configure Logging ---
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        winston.format.printf(({ timestamp, level, message, module, funcName, lineno }) => {
            let context = '';
            if (module && funcName && lineno) {
                context = ` [${module}.${funcName}:${lineno}]`;
            } else if (module && funcName) {
                context = ` [${module}.${funcName}]`;
            } else if (module) {
                context = ` [${module}]`;
            }
            return `${timestamp} - ${level.toUpperCase()} -${context} - ${message}`;
        })
    ),
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: LOG_FILE_NAME, options: { flags: 'a' }, encoding: 'utf-8' })
    ]
});

// Helper for logging with context
const logWithContext = (level, message, context = {}) => {
    logger.log(level, message, context);
};


// --- API Configuration ---
/**
 * Configures and returns the Gemini GenerativeModel.
 * Reads the API key from the GOOGLE_API_KEY environment variable.
 * Sets safety configurations for the model.
 * Exits the script if configuration fails.
 * @returns {Promise<import("@google/generative-ai").GenerativeModel|null>}
 */
async function configureApi() {
    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey) {
        logWithContext('error', "GOOGLE_API_KEY environment variable not set. Please set it to your API key.", { module: 'main', funcName: 'configureApi' });
        process.exit(1);
    }

    try {
        const genAI = new GoogleGenerativeAI(apiKey);
        const safetySettings = [
            { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
            { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
            { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
            { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
        ];
        const model = genAI.getGenerativeModel({ model: MODEL_NAME, safetySettings });
        logWithContext('info', `Successfully configured Gemini API with model: ${MODEL_NAME}`, { module: 'main', funcName: 'configureApi' });
        return model;
    } catch (e) {
        logWithContext('error', `Error configuring Gemini API: ${e.message}`, { module: 'main', funcName: 'configureApi' });
        process.exit(1);
    }
    return null; // Should not be reached
}

// --- File Operations ---
/**
 * Gets a sorted list of files matching the pattern in the base directory.
 * Logs the matched files to MATCHED_FILES_LOG_NAME.
 * Exits if the base directory is invalid or an error occurs during file search.
 * @param {string} baseDirStr
 * @param {string} pattern
 * @returns {Promise<string[]>}
 */
async function getMatchingFiles(baseDirStr, pattern) {
    const baseDirPath = path.resolve(baseDirStr);
    const matchedFilesLogPath = path.resolve(MATCHED_FILES_LOG_NAME);
    const context = { module: 'main', funcName: 'getMatchingFiles' };

    try {
        const stats = await fs.stat(baseDirPath);
        if (!stats.isDirectory()) {
            const errorMsg = `Base directory '${baseDirPath}' does not exist or is not a directory.`;
            logWithContext('error', errorMsg, context);
            await fs.writeFile(matchedFilesLogPath, `Error: ${errorMsg}\n`, 'utf-8');
            process.exit(1);
        }
    } catch (err) {
        const errorMsg = `Base directory '${baseDirPath}' does not exist or is not a directory. Error: ${err.message}`;
        logWithContext('error', errorMsg, context);
        await fs.writeFile(matchedFilesLogPath, `Error: ${errorMsg}\n`, 'utf-8').catch(e => console.error("Failed to write to matched_files.log:", e)); // Best effort
        process.exit(1);
    }

    let files = [];
    try {
        // glob expects forward slashes, even on Windows, for patterns
        const globPattern = path.join(baseDirPath, pattern).replace(/\\/g, '/');
        files = await glob(globPattern, { nodir: true, absolute: true });
        files.sort();

        let logContent = `Matched files for pattern '${pattern}' in directory '${baseDirPath}':\n`;
        if (files.length > 0) {
            logContent += files.join("\n") + "\n";
        } else {
            logContent += "No files matched the pattern.\n";
        }
        await fs.writeFile(matchedFilesLogPath, logContent, 'utf-8');

        if (files.length > 0) {
            logWithContext('info', `Found ${files.length} files matching pattern '${pattern}' in '${baseDirPath}'. See ${MATCHED_FILES_LOG_NAME}.`, context);
        } else {
            logWithContext('info', `No files matched pattern '${pattern}' in '${baseDirPath}'. See ${MATCHED_FILES_LOG_NAME}.`, context);
        }
        return files;

    } catch (e) {
        const errorMsg = `Error during file search with pattern '${pattern}' in directory '${baseDirPath}': ${e.message}`;
        logWithContext('error', errorMsg, context);
        await fs.writeFile(matchedFilesLogPath, `Error finding files with pattern '${pattern}' in '${baseDirPath}': ${e.message}\n`, 'utf-8').catch(e => console.error("Failed to write to matched_files.log:", e));
        process.exit(1);
    }
    return []; // Should not be reached
}

/**
 * Reads content of a file. Returns null if an error occurs.
 * @param {string} filePathStr
 * @returns {Promise<string|null>}
 */
async function readFileContent(filePathStr) {
    const filePath = path.resolve(filePathStr);
    const context = { module: 'main', funcName: 'readFileContent' };
    try {
        const content = await fs.readFile(filePath, 'utf-8');
        logWithContext('info', `Successfully read ${filePath}`, context);
        return content;
    } catch (e) {
        logWithContext('error', `Failed to read ${filePath}: ${e.message}`, context);
        return null;
    }
}

/**
 * Writes content to a file. Returns true on success, false otherwise.
 * @param {string} filePathStr
 * @param {string} content
 * @returns {Promise<boolean>}
 */
async function writeFileContent(filePathStr, content) {
    const filePath = path.resolve(filePathStr);
    const context = { module: 'main', funcName: 'writeFileContent' };
    try {
        await fs.writeFile(filePath, content, 'utf-8');
        logWithContext('info', `Successfully wrote enhanced code to ${filePath}`, context);
        return true;
    } catch (e) {
        logWithContext('error', `Failed to write to ${filePath}: ${e.message}`, context);
        return false;
    }
}

// --- Code Enhancement ---
/**
 * Enhances Python code using the Gemini API.
 * Returns the enhanced code string if successful and changed, otherwise null.
 * @param {import("@google/generative-ai").GenerativeModel} model
 * @param {string} code
 * @param {string} filePath
 * @returns {Promise<string|null>}
 */
async function enhanceCode(model, code, filePath) {
    const context = { module: 'main', funcName: 'enhanceCode' };
    // The prompt is for Python code enhancement, so it remains largely the same.
    // If you were enhancing JS code, you'd change "Python" to "JavaScript" and the ```python block to ```javascript
    const prompt = `You are an expert Python code reviewer and enhancer.
Analyze the following Python code and provide an improved version.
Focus on these areas:
1.  **Code Readability & Documentation**: Improve clarity, add or refine docstrings (following Google style or NumPy style if appropriate), and comments where necessary.
2.  **Performance Optimization**: Suggest more efficient algorithms or data structures if applicable, reduce redundancy.
3.  **Error Handling**: Implement robust error handling using try-except blocks and input validation where sensible.
4.  **Pythonic Best Practices & PEP 8**: Ensure the code adheres to PEP 8 guidelines (naming, style, structure) and idiomatic Python.
5.  **Type Hinting**: Add or improve type hints for function parameters, return types, and variables for better static analysis and readability (PEP 484).
6.  **Modern Python Features**: Utilize modern Python features (e.g., f-strings, context managers, comprehensions) where they improve the code.

IMPORTANT:
-   Return ONLY the complete, enhanced Python code block.
-   Wrap the entire Python code block within \`\`\`python ... \`\`\`.
-   Include comments within the code to explain significant changes or rationale if not self-evident.
-   Do not include any conversational preamble or concluding remarks outside the code block.
-   If the code is already excellent and requires no changes, return the original code within the \`\`\`python ... \`\`\` block.

Original code from file: ${filePath}
\`\`\`python
${code}
\`\`\``;

    try {
        logWithContext('debug', `Sending code from ${filePath} to Gemini API for enhancement.`, { ...context, lineno: 'N/A' }); // lineno not easily available
        const result = await model.generateContent(prompt);
        const response = result.response;
        const responseText = response.text();


        if (!responseText) {
            logWithContext('warn', `No enhancement suggestions or empty response received for ${filePath}.`, context);
            if (response.promptFeedback && response.promptFeedback.blockReason) {
                logWithContext('warn', `Prompt was blocked for ${filePath}. Reason: ${response.promptFeedback.blockReason}. Details: ${JSON.stringify(response.promptFeedback.safetyRatings)}`, context);
            }
            return null;
        }

        // Extract Python code block using a more robust regex
        const codeMatch = responseText.match(/```python\s*([\s\S]*?)\s*```/);
        if (!codeMatch || !codeMatch[1]) {
            logWithContext('error', `No valid Python code block (\`\`\`python ... \`\`\`) found in the API response for ${filePath}.`, context);
            logWithContext('debug', `Full response for ${filePath}:\n${responseText}`, context);
            return null;
        }

        const enhancedCode = codeMatch[1].trim();

        if (enhancedCode === code.trim()) {
            logWithContext('info', `No functional changes suggested by the API for ${filePath}. Original code retained.`, context);
            return null; // Indicate no change needed
        }

        logWithContext('info', `Successfully generated enhancements for ${filePath}.`, context);
        return enhancedCode;

    } catch (e) {
        logWithContext('error', `API call failed for ${filePath}: ${e.message}`, context);
        // The JS library might not have e.response.prompt_feedback in the same way
        // but you can log the whole error object for details.
        if (e.message.includes("SAFETY")) { // A common indicator for content filter issues
             logWithContext('error', `API call for ${filePath} might have been blocked by safety filters. Full error: ${JSON.stringify(e)}`, context);
        }
        return null;
    }
}

/**
 * Simple sleep function.
 * @param {number} ms - Milliseconds to sleep.
 * @returns {Promise<void>}
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// --- Main Logic ---
/**
 * Main function to process and enhance files.
 * @param {string} baseDir
 * @param {string} filePattern
 * @returns {Promise<void>}
 */
async function main(baseDir, filePattern) {
    const mainContext = { module: 'main', funcName: 'main' };
    logWithContext('info', `Starting code enhancement process for directory '${baseDir}', pattern '${filePattern}'.`, mainContext);
    logWithContext('info', `Log file: ${path.resolve(LOG_FILE_NAME)}`, mainContext);
    logWithContext('info', `Matched files list: ${path.resolve(MATCHED_FILES_LOG_NAME)}`, mainContext);

    const maxApiCallsPerMinute = parseInt(process.env.MAX_API_CALLS || DEFAULT_MAX_API_CALLS_PER_MINUTE.toString(), 10);
    logWithContext('info', `Using MAX_API_CALLS_PER_MINUTE: ${maxApiCallsPerMinute}`, mainContext);

    const model = await configureApi();
    if (!model) {
        logWithContext('error', "Failed to configure API model. Exiting.", { ...mainContext, level: 'critical' }); // winston uses error for critical
        return;
    }

    const filesToProcess = await getMatchingFiles(baseDir, filePattern);

    if (filesToProcess.length === 0) {
        logWithContext('warn', "No files found matching the pattern. Nothing to enhance.", mainContext);
        return; // Exit if no files
    }

    const delayMilliseconds = (maxApiCallsPerMinute > 0 ? (60.0 / maxApiCallsPerMinute) * 1000 : 0) + 100; // + 0.1 sec buffer
    logWithContext('info', `Calculated API call delay: ${(delayMilliseconds / 1000).toFixed(2)} seconds per call.`, mainContext);

    let modifiedFilesCount = 0;
    let processedFilesCount = 0;

    for (const filePathStr of filesToProcess) {
        processedFilesCount++;
        logWithContext('info', `Processing file ${processedFilesCount}/${filesToProcess.length}: ${filePathStr}`, mainContext);

        const originalCode = await readFileContent(filePathStr);
        if (originalCode === null) {
            logWithContext('warn', `Skipping ${filePathStr} due to read error.`, mainContext);
            continue; // Skip to next file
        }

        if (!originalCode.trim()) {
            logWithContext('info', `Skipping ${filePathStr} as it is empty or contains only whitespace.`, mainContext);
            continue;
        }

        const enhancedCode = await enhanceCode(model, originalCode, filePathStr);

        if (enhancedCode) { // If enhancement was successful and code changed
            if (await writeFileContent(filePathStr, enhancedCode)) {
                modifiedFilesCount++;
            } else {
                logWithContext('error', `Failed to write changes to ${filePathStr}. Original file remains unchanged.`, mainContext);
            }
        } else {
            logWithContext('info', `No enhancements applied to ${filePathStr} (either no suggestions, error, or no change).`, mainContext);
        }

        if (processedFilesCount < filesToProcess.length) {
            if (delayMilliseconds > 100) { // Only sleep if a meaningful delay is set
                logWithContext('debug', `Waiting for ${(delayMilliseconds / 1000).toFixed(2)} seconds before next API call...`, mainContext);
                await sleep(delayMilliseconds);
            }
        }
    }

    logWithContext('info', `--- Enhancement Process Completed ---`, mainContext);
    logWithContext('info', `Total files matched: ${filesToProcess.length}`, mainContext);
    logWithContext('info', `Files processed: ${processedFilesCount}`, mainContext);
    logWithContext('info', `Files modified: ${modifiedFilesCount}`, mainContext);
    logWithContext('info', `Log file: ${path.resolve(LOG_FILE_NAME)}`, mainContext);
}

// --- Script Entry Point ---
if (require.main === module) {
    if (process.argv.length !== 4) {
        const scriptName = path.basename(process.argv[1]);
        logger.error(`Usage: node ${scriptName} <base_directory> <file_pattern>`);
        logger.error(`Example: node ${scriptName} ./my_project '**/*.py'`); // Keep example as .py if enhancing Python
        process.exit(1);
    }

    const cliBaseDir = process.argv[2];
    const cliFilePattern = process.argv[3];

    // Validate base_directory existence before starting main logic
    fs.stat(cliBaseDir)
        .then(stats => {
            if (!stats.isDirectory()) {
                logger.error(`Error: Base directory '${cliBaseDir}' exists but is not a directory.`);
                process.exit(1);
            }
            main(cliBaseDir, cliFilePattern).catch(err => {
                logger.error(`Unhandled error in main function: ${err.message}\n${err.stack}`);
                process.exit(1);
            });
        })
        .catch(err => {
            logger.error(`Error: Base directory '${cliBaseDir}' does not exist or is not accessible: ${err.message}`);
            process.exit(1);
        });
}
```

**To run the script:**

1.  Save the code above as `enhance_script.js`.
2.  Save the `package.json` content.
3.  Run `npm install`.
4.  Set your `GOOGLE_API_KEY` environment variable:
    *   Linux/macOS: `export GOOGLE_API_KEY="YOUR_API_KEY_HERE"`
    *   Windows (cmd): `set GOOGLE_API_KEY=YOUR_API_KEY_HERE`
    *   Windows (PowerShell): `$env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"`
5.  Execute the script:
    `node enhance_script.js ./your_project_directory '**/*.py'`
    (Assuming you are still enhancing Python files as per the original script's prompt. If you want to enhance JS files, you'd change the pattern to `'**/*.js'` and modify the prompt inside `enhanceCode` function accordingly.)

This JavaScript version aims to mirror the functionality and structure of your Python script as closely as possible while adhering to Node.js conventions. Remember that the Gemini Pro 1.5 model might have different rate limits or behaviors than the experimental one you used, so monitor its performance. I've also updated the `MODEL_NAME` to `gemini-1.5-pro-latest` which is a generally available and powerful model. Adjust if you have access to a specific experimental one.
