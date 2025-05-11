#!/usr/bin/env node

"use strict";

const fs = require('fs').promises;
const path = require('path');
const { glob } = require('glob');
const winston = require('winston');
const { exec } = require('child_process');
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require("@google/generative-ai");

// --- Constants ---
const MODEL_NAME = "gemini-1.5-pro-latest";
const LOG_FILE_NAME = "enhancement_log.txt";
const MATCHED_FILES_LOG_NAME = "matched_files.txt";
const DEFAULT_MAX_API_CALLS_PER_MINUTE = 59;
const SCRIPT_NAME = path.basename(process.argv[1] || 'enhance_code.js');
const BACKUP_DIR_NAME = 'code_enhance_backups';
const PRE_COMMIT_MSG = 'Pre-enhancement backup';
const POST_COMMIT_MSG_PREFIX = 'Enhanced via script:';

// --- Configure Logging ---
const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: winston.format.combine(
        winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        winston.format.printf(({ timestamp, level, message, module, funcName, lineno, filePath }) => {
            let contextString = '';
            if (module) {
                contextString = funcName ? `${module}.${funcName}` : module;
                if (lineno) {
                    contextString += `:${lineno}`;
                }
            }
            const fileInfo = filePath ? ` (file: ${path.basename(filePath)})` : '';
            return `${timestamp} - ${level.toUpperCase()}${contextString ? ` [${contextString}]` : ''} - ${message}${fileInfo}`;
        })
    ),
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({
            filename: LOG_FILE_NAME,
            options: { flags: 'a' },
            encoding: 'utf-8'
        })
    ]
});

const logWithContext = (level, message, context = {}) => {
    logger.log(level, message, context);
};

// --- Utility Functions ---
function exitWithError(message, context = {}, error) {
    logWithContext('error', error ? `${message}: ${error.message}` : message, context);
    if (error?.stack) {
        logWithContext('debug', `Stack trace: ${error.stack}`, context);
    }
    process.exit(1);
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// --- Backup & Git Functions ---
async function isGitRepo(cwd) {
    return new Promise((resolve) => {
        exec('git rev-parse --is-inside-work-tree', { cwd }, (error) => {
            resolve(!error);
        });
    });
}

async function createBackupDir(baseDir) {
    const timestamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14);
    const backupDir = path.join(baseDir, BACKUP_DIR_NAME, timestamp);
    
    try {
        await fs.mkdir(backupDir, { recursive: true });
        return backupDir;
    } catch (e) {
        exitWithError(`Failed to create backup directory: ${backupDir}`, {}, e);
    }
}

async function backupFile(originalPath, backupDir, baseDir) {
    try {
        const relativePath = path.relative(baseDir, originalPath);
        const backupPath = path.join(backupDir, relativePath);
        
        await fs.mkdir(path.dirname(backupPath), { recursive: true });
        await fs.copyFile(originalPath, backupPath);
        return true;
    } catch (e) {
        logWithContext('error', `Failed to create backup for ${path.basename(originalPath)}`, {
            module: SCRIPT_NAME,
            funcName: 'backupFile',
            filePath: originalPath
        });
        return false;
    }
}

async function gitCommitFile(filePath, message, cwd) {
    return new Promise((resolve, reject) => {
        exec(`git add "${filePath}" && git commit -m "${message}"`, { cwd }, (error) => {
            error ? reject(error) : resolve();
        });
    });
}

// --- API Configuration ---
async function configureApi() {
    const apiKey = process.env.GOOGLE_API_KEY;
    const context = { module: SCRIPT_NAME, funcName: 'configureApi' };

    if (!apiKey) {
        exitWithError("GOOGLE_API_KEY environment variable not set.", context);
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
        logWithContext('info', `Configured Gemini API with model: ${MODEL_NAME}`, context);
        return model;
    } catch (e) {
        exitWithError(`Error configuring Gemini API`, context, e);
    }
}

// --- File Operations ---
async function getMatchingFiles(baseDirStr, pattern) {
    const baseDirPath = path.resolve(baseDirStr);
    const matchedFilesLogPath = path.resolve(MATCHED_FILES_LOG_NAME);
    const context = { module: SCRIPT_NAME, funcName: 'getMatchingFiles' };

    try {
        await fs.stat(baseDirPath);
    } catch (err) {
        exitWithError(`Error accessing base directory '${baseDirPath}'`, context, err);
    }

    try {
        const globPattern = path.join(baseDirPath, pattern).replace(/\\/g, '/');
        logWithContext('debug', `Searching with glob: ${globPattern}`, context);
        
        const files = await glob(globPattern, { nodir: true, absolute: true });
        
        if (!Array.isArray(files)) {
            exitWithError(`Glob returned non-array type: ${typeof files}`, context);
        }
        
        files.sort();

        await fs.writeFile(matchedFilesLogPath, 
            `Matched files for '${pattern}' in '${baseDirPath}':\n${files.join("\n") || "No files matched"}\n`,
            'utf-8'
        );

        return files;
    } catch (e) {
        exitWithError(`File search failed for '${pattern}'`, context, e);
    }
}

async function readFileContent(filePathStr) {
    const filePath = path.resolve(filePathStr);
    const context = { module: SCRIPT_NAME, funcName: 'readFileContent', filePath };
    try {
        return await fs.readFile(filePath, 'utf-8');
    } catch (e) {
        logWithContext('error', `Read failed: ${e.message}`, context);
        return null;
    }
}

async function writeFileContent(filePathStr, content) {
    const filePath = path.resolve(filePathStr);
    const context = { module: SCRIPT_NAME, funcName: 'writeFileContent', filePath };
    try {
        await fs.writeFile(filePath, content, 'utf-8');
        logWithContext('info', `File written successfully`, context);
        return true;
    } catch (e) {
        logWithContext('error', `Write failed: ${e.message}`, context);
        return false;
    }
}

// --- Code Enhancement ---
async function enhanceCode(model, code, filePath) {
    const context = { module: SCRIPT_NAME, funcName: 'enhanceCode', filePath };
    // IMPORTANT: The prompt is for Python code enhancement.
    // If enhancing JavaScript or other languages, update "Python" to the target language
    // and the ```python block to ```<language_name>.
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

Original code from file: ${path.basename(filePath)}
\`\`\`python
${code}
\`\`\``;

    try {
        const result = await model.generateContent(prompt);
        const response = result.response;
        
        if (response.promptFeedback?.blockReason) {
            logWithContext('warn', `Prompt blocked: ${response.promptFeedback.blockReason}`, context);
            return null;
        }

        const responseText = response.text();
        const codeMatch = responseText.match(/```python\s*([\s\S]*?)\s*```/); // Adjust 'python' if enhancing other languages
        if (!codeMatch?.[1]) {
            logWithContext('error', `No valid code block found in API response. Ensure the prompt asks for a specific language block like \`\`\`python ... \`\`\`.`, context);
            logWithContext('debug', `Full API response for ${filePath}:\n${responseText}`, context);
            return null;
        }

        const enhancedCode = codeMatch[1].trim();
        return enhancedCode === code.trim() ? null : enhancedCode;
    } catch (e) {
        logWithContext('error', `API call failed: ${e.message}`, context);
        return null;
    }
}

// --- Main Logic ---
async function main(baseDir, filePattern) {
    const mainContext = { module: SCRIPT_NAME, funcName: 'main' };
    const baseDirResolved = path.resolve(baseDir);

    // Setup backups
    const backupDir = await createBackupDir(baseDirResolved);
    logWithContext('info', `Backup directory: ${backupDir}`, mainContext);

    // Git initialization
    let isGitRepository = false;
    try {
        isGitRepository = await isGitRepo(baseDirResolved);
        if (isGitRepository) {
            await new Promise((resolve, reject) => {
                exec(`git commit --allow-empty -m "${PRE_COMMIT_MSG}"`, 
                    { cwd: baseDirResolved }, 
                    (error) => error ? reject(error) : resolve()
                );
            });
            logWithContext('info', 'Created pre-enhancement Git commit', mainContext);
        } else {
            logWithContext('info', 'Not a Git repository, or Git not found. Proceeding with file backups only.', mainContext);
        }
    } catch (e) {
        logWithContext('warn', `Git pre-commit failed: ${e.message}. Proceeding with file backups only.`, mainContext);
        isGitRepository = false;
    }

    // Rest of main logic
    const model = await configureApi();
    const filesToProcess = await getMatchingFiles(baseDir, filePattern);
    
    if (filesToProcess.length === 0) {
        logWithContext('info', "No files to process", mainContext);
        return;
    }

    const maxApiCalls = parseInt(process.env.MAX_API_CALLS, 10) || DEFAULT_MAX_API_CALLS_PER_MINUTE;
    const delayMs = (60.0 / maxApiCalls) * 1000 + 100; // Add a small buffer
    
    let modifiedCount = 0, processedCount = 0, failedCount = 0;

    for (const filePathStr of filesToProcess) {
        processedCount++;
        const fileContext = { ...mainContext, filePath: filePathStr };
        logWithContext('info', `Processing file ${processedCount}/${filesToProcess.length}: ${path.basename(filePathStr)}`, fileContext);

        // Backup file
        if (!(await backupFile(filePathStr, backupDir, baseDirResolved))) {
            logWithContext('warn', `Skipping file ${path.basename(filePathStr)} due to backup failure.`, fileContext);
            failedCount++;
            continue;
        }

        const originalCode = await readFileContent(filePathStr);
        if (!originalCode) {
            failedCount++;
            continue;
        }

        try {
            const enhancedCode = await enhanceCode(model, originalCode, filePathStr);
            if (!enhancedCode) { // No valid enhancement or API decided no changes needed
                logWithContext('info', `No valid enhancements or no changes suggested for ${path.basename(filePathStr)}.`, fileContext);
                continue;
            }

            if (await writeFileContent(filePathStr, enhancedCode)) {
                modifiedCount++;
                logWithContext('info', `Successfully enhanced and saved ${path.basename(filePathStr)}.`, fileContext);
                // Git commit
                if (isGitRepository) {
                    try {
                        await gitCommitFile(
                            filePathStr,
                            `${POST_COMMIT_MSG_PREFIX} ${path.basename(filePathStr)}`,
                            baseDirResolved
                        );
                        logWithContext('info', `Git commit successful for ${path.basename(filePathStr)}`, fileContext);
                    } catch (e) {
                        logWithContext('error', `Git commit failed for ${path.basename(filePathStr)}: ${e.message}`, fileContext);
                        // Continue processing other files even if a commit fails
                    }
                }
            } else {
                logWithContext('error', `Failed to write enhanced code for ${path.basename(filePathStr)}. Original file preserved in backup.`, fileContext);
                failedCount++;
            }
        } catch (error) {
            logWithContext('error', `Processing error for ${path.basename(filePathStr)}: ${error.message}`, fileContext);
            logWithContext('debug', `Stack trace for ${path.basename(filePathStr)}: ${error.stack}`, fileContext);
            failedCount++;
        }

        if (processedCount < filesToProcess.length) {
            await sleep(delayMs);
        }
    }

    logWithContext('info', `--- Enhancement Summary ---`, mainContext);
    logWithContext('info', `Total files matched: ${filesToProcess.length}`, mainContext);
    logWithContext('info', `Files processed: ${processedCount}`, mainContext);
    logWithContext('info', `Files successfully modified: ${modifiedCount}`, mainContext);
    logWithContext('info', `Files failed or skipped: ${failedCount}`, mainContext);
    logWithContext('info', `Backups created in: ${backupDir}`, mainContext);
    logWithContext('info', `Detailed logs in: ${LOG_FILE_NAME}`, mainContext);
}

// --- Script Entry ---
async function runScript() {
    if (process.argv.length !== 4) {
        logger.error(`Usage: node ${SCRIPT_NAME} <base_directory> <file_pattern>`);
        logger.error(`Example: node ${SCRIPT_NAME} ./my_project_path '**/*.py'`);
        logger.error(`Note: The example prompt in 'enhanceCode' is for Python. Update it and the pattern for other languages.`);
        process.exit(1);
    }

    const [,, baseDir, pattern] = process.argv;
    
    try {
        // Basic validation for base directory
        const stats = await fs.stat(baseDir);
        if (!stats.isDirectory()) {
            exitWithError(`Base directory '${baseDir}' is not a valid directory.`);
        }
        await main(baseDir, pattern);
    } catch (err) {
        // This catch handles errors from fs.stat or if main itself throws an unhandled error early on.
        exitWithError(`Initialization error or invalid base directory: ${baseDir}`, {}, err);
    }
}

if (require.main === module) {
    runScript().catch(err => {
        // Fallback for truly unhandled exceptions from runScript's top level or initial setup.
        logWithContext('error', `Critical unhandled error in script execution: ${err.message}`, { module: SCRIPT_NAME, funcName: 'globalCatch' });
        logWithContext('debug', `Stack: ${err.stack}`, { module: SCRIPT_NAME, funcName: 'globalCatch' });
        process.exit(1);
    });
}
```

**Key Features of the Enhanced Script:**

This script is packed with features to provide a safe, reliable, and efficient code enhancement workflow:

1.  **Robust Automated Backups:**
    *   **Timestamped Directories:** Automatically creates a unique, timestamped directory (e.g., `code_enhance_backups/YYYYMMDDHHMMSS`) for each backup session, preventing accidental overwrites.
    *   **Preserved File Structure:** Faithfully replicates the original directory structure within the backup folder, making it easy to locate and restore files if needed.
    *   **Pre-Modification Safeguard:** Backups are performed for each file *before* it is modified by the enhancement process. If a backup fails, the file is skipped.

2.  **Seamless Git Integration (Optional but Recommended):**
    *   **Pre-Enhancement Snapshot:** If the target directory is a Git repository, an initial empty commit with a message (e.g., "Pre-enhancement backup") is made to mark the state before any changes by this script.
    *   **Granular Commits:** Each successfully enhanced file is individually added and committed with a descriptive message (e.g., "Enhanced via script: filename.ext"), providing a detailed history of changes.
    *   **Graceful Fallback:** If Git operations fail or the directory isn't a Git repository (or Git is not installed/found), the script defaults to using file-based backups, ensuring originals are always protected.

3.  **Enhanced Safety and Reliability:**
    *   **Conditional Modification:** Files are only updated if the AI model returns valid and distinct enhancements (i.e., the enhanced code is different from the original).
    *   **Integrity Preservation:** Original files are meticulously preserved in the backup directory. If any step in the enhancement or writing process for a specific file encounters an error, the original file on disk remains unchanged (post-backup).
    *   **Comprehensive Logging:** Detailed logs capture all operations, warnings, and errors, facilitating easy troubleshooting and monitoring. Logs are output to both the console and a persistent file (`enhancement_log.txt`). A list of matched files is saved to `matched_files.txt`.

**Usage Instructions:**

To use this script:

1.  **Set Your Google API Key:**
    The script requires your Google Generative AI API key. Set it as an environment variable in your terminal session:
    ```bash
    export GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY_HERE"
    ```
    Replace `"YOUR_ACTUAL_API_KEY_HERE"` with your valid API key.

2.  **Run the Script:**
    Execute the script from your terminal using Node.js, providing the base directory of your project and a glob pattern to match the files you want to enhance.
    ```bash
    node enhance_code.js ./your_project_directory '**/*.py'
    ```
    *   `./your_project_directory`: Replace with the path to the root directory containing the code files you wish to enhance.
    *   `'**/*.py'`: This is an example glob pattern that matches all Python files (`.py`) in all subdirectories. Adjust this pattern to match your target files (e.g., `'**/*.js'` for JavaScript, `'src/**/*.java'` for Java files within a `src` directory). **Important:** The default prompt within the `enhanceCode` function is tailored for Python. You will need to modify this prompt if you are enhancing code in other languages.

**Expected Outcome:**
The script will process the matched files one by one:
*   It will create a backup of each original file in a timestamped subdirectory within `your_project_directory/code_enhance_backups/`.
*   If your project directory is a Git repository, it will attempt to create an initial commit and then commit each successfully enhanced file.
*   It will log its progress, including any errors or warnings, to the console and to `enhancement_log.txt`. A list of all files matched by your pattern will be saved in `matched_files.txt`.
*   Original files in your working directory will only be overwritten if the AI suggests valid improvements and the file writing process is successful. Otherwise, the original (or its backed-up version) is preserved.

---

The enhanced script detailed above, with its backup and Git integration, provides a comprehensive solution for automated code improvement. It builds upon a more fundamental version that focuses primarily on the core AI-driven enhancement logic without these additional safety features.

For context, or if a simpler script without automated backups or Git integration is preferred as a starting point, this foundational version is provided below. Note that this version uses a different Gemini model (`gemini-2.5-pro-exp-03-25` as specified in its constants) and lacks the sophisticated error handling, backup, and Git features of the primary script presented earlier.

```javascript
#!/usr/bin/env node

"use strict";

const fs = require('fs').promises;
const path = require('path');
const { glob } = require('glob'); // Assumes glob v7+ which exports { glob } as an async function
const winston = require('winston');
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require("@google/generative-ai");

// --- Constants ---
const MODEL_NAME = "gemini-2.5-pro-exp-03-25"; // Corrected from original with extra newlines
const LOG_FILE_NAME = "enhancement_log.txt";
const MATCHED_FILES_LOG_NAME = "matched_files.txt";
const DEFAULT_MAX_API_CALLS_PER_MINUTE = 59; // Default for Gemini API (check official docs for current limits)
const SCRIPT_NAME = path.basename(process.argv[1] || 'enhance_code.js');

// --- Configure Logging ---
const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info', // Allow configuring log level via env
    format: winston.format.combine(
        winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        winston.format.printf(({ timestamp, level, message, module, funcName, lineno, filePath }) => {
            let contextString = '';
            if (module) {
                contextString = funcName ? `${module}.${funcName}` : module;
                if (lineno) {
                    contextString += `:${lineno}`;
                }
            }
            const fileInfo = filePath ? ` (file: ${path.basename(filePath)})` : '';
            return `${timestamp} - ${level.toUpperCase()}${contextString ? ` [${contextString}]` : ''} - ${message}${fileInfo}`;
        })
    ),
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({
            filename: LOG_FILE_NAME,
            options: { flags: 'a' }, // options for fs.createWriteStream
            encoding: 'utf-8'
        })
    ]
});

// Helper for logging with context
const logWithContext = (level, message, context = {}) => {
    logger.log(level, message, context);
};

// --- Utility Functions ---
/**
 * Exits the script with an error message.
 * @param {string} message - The error message.
 * @param {object} [context] - Optional logging context.
 * @param {Error} [error] - Optional error object for stack trace.
 */
function exitWithError(message, context = {}, error) {
    logWithContext('error', error ? `${message}: ${error.message}` : message, context);
    if (error && error.stack) {
        logWithContext('debug', `Stack trace: ${error.stack}`, context);
    }
    process.exit(1);
}

/**
 * Simple sleep function.
 * @param {number} ms - Milliseconds to sleep.
 * @returns {Promise<void>}
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// --- API Configuration ---
/**
 * Configures and returns the Gemini GenerativeModel.
 * Reads the API key from the GOOGLE_API_KEY environment variable.
 * Sets safety configurations for the model.
 * Exits the script if configuration fails.
 * @returns {Promise<import("@google/generative-ai").GenerativeModel>}
 */
async function configureApi() {
    const apiKey = process.env.GOOGLE_API_KEY;
    const context = { module: SCRIPT_NAME, funcName: 'configureApi' };

    if (!apiKey) {
        // exitWithError will terminate the script.
        exitWithError("GOOGLE_API_KEY environment variable not set. Please set it to your API key.", context);
        return null; // Should be unreachable, but satisfies linters/compilers
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
        logWithContext('info', `Successfully configured Gemini API with model: ${MODEL_NAME}`, context);
        return model;
    } catch (e) {
        exitWithError(`Error configuring Gemini API`, context, e);
        return null; // Should be unreachable
    }
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
    const context = { module: SCRIPT_NAME, funcName: 'getMatchingFiles' };

    try {
        const stats = await fs.stat(baseDirPath);
        if (!stats.isDirectory()) {
            const errorMsg = `Base directory '${baseDirPath}' does not exist or is not a directory.`;
            // Best effort to log to file before exiting
            await fs.writeFile(matchedFilesLogPath, `Error: ${errorMsg}\n`, 'utf-8')
                .catch(e => logWithContext('warn', `Failed to write error to ${MATCHED_FILES_LOG_NAME}: ${e.message}`, context));
            exitWithError(errorMsg, context);
            return []; // Should be unreachable
        }
    } catch (err) {
        const errorMsg = `Error accessing base directory '${baseDirPath}'`;
        await fs.writeFile(matchedFilesLogPath, `Error: ${errorMsg}: ${err.message}\n`, 'utf-8')
            .catch(e => logWithContext('warn', `Failed to write error to ${MATCHED_FILES_LOG_NAME}: ${e.message}`, context));
        exitWithError(errorMsg, context, err);
        return []; // Should be unreachable
    }

    let files;
    try {
        // glob expects forward slashes, even on Windows, for patterns
        const globPattern = path.join(baseDirPath, pattern).replace(/\\/g, '/');
        logWithContext('debug', `Searching for files with glob pattern: ${globPattern}`, context);
        
        files = await glob(globPattern, { nodir: true, absolute: true });

        if (!Array.isArray(files)) {
            // This case indicates a serious issue with the glob library or its usage/version.
            const errorMsg = `File search (glob) did not return an array. Received type: ${typeof files}. This may indicate an issue with the 'glob' library version or environment. Expected an array of file paths.`;
            await fs.writeFile(matchedFilesLogPath, `Critical Error: ${errorMsg}\nGlob pattern: ${globPattern}\n`, 'utf-8')
                .catch(e => logWithContext('warn', `Failed to write critical glob error to ${MATCHED_FILES_LOG_NAME}: ${e.message}`, context));
            exitWithError(errorMsg, context);
            return []; // Should be unreachable
        }
        
        files.sort(); // Sort alphabetically for consistent processing order

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
        const errorMsg = `Error during file search with pattern '${pattern}' in directory '${baseDirPath}'`;
        await fs.writeFile(matchedFilesLogPath, `Error: ${errorMsg}: ${e.message}\n`, 'utf-8')
            .catch(writeErr => logWithContext('warn', `Failed to write search error to ${MATCHED_FILES_LOG_NAME}: ${writeErr.message}`, context));
        exitWithError(errorMsg, context, e);
        return []; // Should be unreachable
    }
}

/**
 * Reads content of a file. Returns null if an error occurs.
 * @param {string} filePathStr
 * @returns {Promise<string|null>}
 */
async function readFileContent(filePathStr) {
    const filePath = path.resolve(filePathStr);
    const context = { module: SCRIPT_NAME, funcName: 'readFileContent', filePath };
    try {
        const content = await fs.readFile(filePath, 'utf-8');
        logWithContext('debug', `Successfully read file.`, context);
        return content;
    } catch (e) {
        logWithContext('error', `Failed to read file: ${e.message}`, context);
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
    const context = { module: SCRIPT_NAME, funcName: 'writeFileContent', filePath };
    try {
        await fs.writeFile(filePath, content, 'utf-8');
        logWithContext('info', `Successfully wrote enhanced code to file.`, context);
        return true;
    } catch (e) {
        logWithContext('error', `Failed to write to file: ${e.message}`, context);
        return false;
    }
}

// --- Code Enhancement ---
/**
 * Enhances code using the Gemini API.
 * This function is currently hardcoded to enhance Python code.
 * To enhance other languages, the prompt needs to be adjusted accordingly.
 * @param {import("@google/generative-ai").GenerativeModel} model
 * @param {string} code
 * @param {string} filePath
 * @returns {Promise<string|null>}
 */
async function enhanceCode(model, code, filePath) {
    const context = { module: SCRIPT_NAME, funcName: 'enhanceCode', filePath };
    // IMPORTANT: The prompt is for Python code enhancement.
    // If enhancing JavaScript or other languages, update "Python" to the target language
    // and the ```python block to ```<language_name>.
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

Original code from file: ${path.basename(filePath)}
\`\`\`python
${code}
\`\`\``;

    try {
        logWithContext('debug', `Sending code to Gemini API for enhancement.`, context);
        const result = await model.generateContent(prompt);
        const response = result.response;
        
        if (response.promptFeedback && response.promptFeedback.blockReason) {
            logWithContext('warn', `Prompt was blocked. Reason: ${response.promptFeedback.blockReason}. Details: ${JSON.stringify(response.promptFeedback.safetyRatings)}`, context);
            return null;
        }

        const responseText = response.text();
        if (!responseText) {
            logWithContext('warn', `No enhancement suggestions or empty response received.`, context);
            return null;
        }

        // Extract Python code block
        const codeMatch = responseText.match(/```python\s*([\s\S]*?)\s*```/);
        if (!codeMatch || !codeMatch[1]) {
            logWithContext('error', `No valid Python code block (\`\`\`python ... \`\`\`) found in the API response.`, context);
            logWithContext('debug', `Full response for ${filePath}:\n${responseText}`, context);
            return null;
        }

        const enhancedCode = codeMatch[1].trim();

        if (enhancedCode === code.trim()) {
            logWithContext('info', `No functional changes suggested by the API. Original code retained.`, context);
            return null; 
        }

        logWithContext('info', `Successfully generated enhancements.`, context);
        return enhancedCode;

    } catch (e) {
        logWithContext('error', `API call failed: ${e.message}`, context);
        if (e.message && e.message.toLowerCase().includes("safety")) {
             logWithContext('warn', `API call might have been blocked by safety filters. Full error: ${e.toString()}`, context);
        } else if (e.response && e.response.promptFeedback) { 
             logWithContext('warn', `API call failed with prompt feedback: ${JSON.stringify(e.response.promptFeedback)}`, context);
        } else {
            logWithContext('debug', `Full API error details: ${JSON.stringify(e, Object.getOwnPropertyNames(e))}`, context);
        }
        return null;
    }
}

// --- Main Logic ---
/**
 * Main function to process and enhance files.
 * @param {string} baseDir
 * @param {string} filePattern
 * @returns {Promise<void>}
 */
async function main(baseDir, filePattern) {
    const mainContext = { module: SCRIPT_NAME, funcName: 'main' };
    logWithContext('info', `Starting code enhancement process for directory '${baseDir}', pattern '${filePattern}'.`, mainContext);
    logWithContext('info', `Log file: ${path.resolve(LOG_FILE_NAME)}`, mainContext);
    logWithContext('info', `Matched files list: ${path.resolve(MATCHED_FILES_LOG_NAME)}`, mainContext);

    let parsedMaxApiCalls = parseInt(process.env.MAX_API_CALLS, 10);
    if (isNaN(parsedMaxApiCalls) || parsedMaxApiCalls <= 0) {
        logWithContext('warn', `Invalid or no MAX_API_CALLS environment variable found (value: "${process.env.MAX_API_CALLS}"). Using default: ${DEFAULT_MAX_API_CALLS_PER_MINUTE} calls/minute.`, mainContext);
        parsedMaxApiCalls = DEFAULT_MAX_API_CALLS_PER_MINUTE;
    }
    
    const maxApiCallsPerMinute = Math.max(1, parsedMaxApiCalls); // Ensure at least 1 to prevent division by zero/negative delay
    logWithContext('info', `API call rate limit: ${maxApiCallsPerMinute} calls/minute.`, mainContext);

    const model = await configureApi(); // Exits on failure

    const filesToProcess = await getMatchingFiles(baseDir, filePattern); // Exits on failure

    if (filesToProcess.length === 0) {
        logWithContext('info', "No files found matching the pattern. Nothing to enhance.", mainContext);
        return;
    }

    const delayMilliseconds = (60.0 / maxApiCallsPerMinute) * 1000 + 100; // +100ms buffer
    logWithContext('info', `Calculated API call delay: ${(delayMilliseconds / 1000).toFixed(2)} seconds per call.`, mainContext);

    let modifiedFilesCount = 0;
    let processedFilesCount = 0;
    let failedFilesCount = 0;

    for (const filePathStr of filesToProcess) {
        processedFilesCount++;
        const fileContext = { ...mainContext, filePath: filePathStr }; // Add filePath to context for per-file logs
        logWithContext('info', `Processing file ${processedFilesCount}/${filesToProcess.length}: ${path.basename(filePathStr)}`, fileContext);

        const originalCode = await readFileContent(filePathStr);
        if (originalCode === null) {
            logWithContext('warn', `Skipping due to read error.`, fileContext);
            failedFilesCount++;
            continue;
        }

        if (!originalCode.trim()) {
            logWithContext('info', `Skipping as it is empty or contains only whitespace.`, fileContext);
            continue;
        }

        try {
            const enhancedCode = await enhanceCode(model, originalCode, filePathStr);

            if (enhancedCode) {
                if (await writeFileContent(filePathStr, enhancedCode)) {
                    modifiedFilesCount++;
                } else {
                    logWithContext('error', `Failed to write changes. Original file remains unchanged.`, fileContext);
                    failedFilesCount++;
                }
            } else {
                logWithContext('info', `No enhancements applied or generated.`, fileContext);
            }
        } catch (error) { // Catch unexpected errors during enhanceCode or writeFileContent
            logWithContext('error', `An unexpected error occurred during processing: ${error.message}`, fileContext);
            logWithContext('debug', `Error stack: ${error.stack}`, fileContext);
            failedFilesCount++;
        }

        if (processedFilesCount < filesToProcess.length && delayMilliseconds > 100) {
            logWithContext('debug', `Waiting for ${(delayMilliseconds / 1000).toFixed(2)} seconds before next API call...`, mainContext);
            await sleep(delayMilliseconds);
        }
    }

    logWithContext('info', `--- Enhancement Process Completed ---`, mainContext);
    logWithContext('info', `Total files matched: ${filesToProcess.length}`, mainContext);
    logWithContext('info', `Files processed: ${processedFilesCount}`, mainContext);
    logWithContext('info', `Files modified: ${modifiedFilesCount}`, mainContext);
    if (failedFilesCount > 0) {
        logWithContext('warn', `Files failed during processing (read/write/enhancement error): ${failedFilesCount}`, mainContext);
    }
    logWithContext('info', `Log file: ${path.resolve(LOG_FILE_NAME)}`, mainContext);
    logWithContext('info', `Matched files list: ${path.resolve(MATCHED_FILES_LOG_NAME)}`, mainContext);
}

// --- Script Entry Point ---
async function runScript() {
    if (process.argv.length !== 4) {
        logger.error(`Usage: node ${SCRIPT_NAME} <base_directory> <file_pattern>`);
        logger.error(`Example: node ${SCRIPT_NAME} ./my_project '**/*.py'`);
        logger.error(`Note: The example uses '*.py' because the current 'enhanceCode' prompt is for Python.`);
        logger.error(`Change the pattern and the prompt in 'enhanceCode' if targeting other languages.`);
        process.exit(1);
    }

    const cliBaseDir = process.argv[2];
    const cliFilePattern = process.argv[3];

    try {
        const stats = await fs.stat(cliBaseDir);
        if (!stats.isDirectory()) {
            // exitWithError will use logger and terminate.
            exitWithError(`Error: Base directory '${cliBaseDir}' exists but is not a directory.`);
        }
    } catch (err) {
        exitWithError(`Error: Base directory '${cliBaseDir}' does not exist or is not accessible`, {}, err);
    }

    // main function contains its own comprehensive try-catch for its operations.
    // Errors in configureApi or getMatchingFiles will call exitWithError.
    await main(cliBaseDir, cliFilePattern);
}

if (require.main === module) {
    runScript().catch(err => {
        // This catch is a fallback for truly unhandled exceptions from runScript's top level await or initial setup.
        // Most errors should be handled and logged by exitWithError or within main.
        logWithContext('error', `Critical unhandled error in script execution: ${err.message}`, { module: SCRIPT_NAME, funcName: 'globalCatch' });
        logWithContext('debug', `Stack: ${err.stack}`, { module: SCRIPT_NAME, funcName: 'globalCatch' });
        process.exit(1);
    });
}
