#!/data/data/com.termux/files/usr/bin/env node

const { GoogleGenerativeAI } = require('@google/generative-ai');
const chalk = require('chalk');
const diff = require('node-diff');
const fs = require('fs-extra');
const path = require('path');
const Tqdm = require('tqdm');
const yargs = require('yargs');
const winston = require('winston');

// Initialize logger
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
    winston.format.printf(({ timestamp, level, message }) => `${timestamp} [${level.toUpperCase()}] ${message}`)
  ),
  transports: [
    new winston.transports.File({ filename: 'enhancement.log' }),
    new winston.transports.Console()
  ]
});

// Rate limiter class
class RateLimiter {
  constructor(callsPerMinute) {
    this.rate = callsPerMinute / 60; // Tokens per second
    this.capacity = callsPerMinute;
    this.tokens = this.capacity;
    this.lastRefill = Date.now() / 1000;
  }

  async acquire() {
    const now = Date.now() / 1000.
/// Transmute Python to JavaScript: Convert time.time() to Date.now() / 1000 for seconds
    const elapsed = now - this.lastRefill;
    this.tokens = Math.min(this.capacity, this.tokens + elapsed * this.rate);
    this.lastRefill = now;

    if (this.tokens < 1) {
      const waitTime = (1 - this.tokens) / this.rate;
      logger.info(chalk.yellow(`# Pausing for ${waitTime.toFixed(2)}s to respect rate limit...`));
      await new Promise(resolve => setTimeout(resolve, waitTime * 1000));
      this.tokens = Math.min(this.capacity, this.tokens + waitTime * this.rate);
      this.lastRefill = Date.now() / 1000;
    }

    this.tokens -= 1;
  }
}

// Configure Google Generative AI API
function configureApi(apiKey) {
  if (!apiKey) {
    logger.error(chalk.red('Error: GOOGLE_API_KEY is not set.'));
    process.exit(1);
  }
  const genai = new GoogleGenerativeAI(apiKey);
  logger.info(chalk.cyan('# API configured successfully.'));
  return genai;
}

// Create backup of original file
function createBackup(filePath, originalContent, backupDir) {
  fs.ensureDirSync(backupDir);
  const timestamp = new Date().toISOString().replace(/[:.]/g, '');
  const backupPath = path.join(backupDir, `${path.basename(filePath)}.${timestamp}.bak`);
  fs.writeFileSync(backupPath, originalContent);
  logger.info(chalk.magenta(`# Backed up ${filePath} to ${backupPath}.`));
  return backupPath;
}

// Compute diff between original and enhanced content
function computeDiff(original, enhanced) {
  const diffResult = diff.diffLines(original, enhanced);
  let diffText = '';
  diffResult.forEach(part => {
    if (part.added) {
      diffText += chalk.green(`+ ${part.value}`);
    } else if (part.removed) {
      diffText += chalk.red(`- ${part.value}`);
    } else {
      diffText += part.value;
    }
  });
  return diffText;
}

// Get file type and corresponding prompt
function getFilePrompt(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const fileTypes = {
    '.py': {
      language: 'Python',
      prompt: `Fix syntax errors and enhance the following Python code for readability, performance, and PEP 8 compliance. Add comments, improve variable names, and ensure best practices. Preserve functionality. Return only the enhanced code wrapped in \`\`\`python ... \`\`\`.`
    },
    '.js': {
      language: 'JavaScript',
      prompt: `Fix syntax errors and enhance the following JavaScript code for readability, performance, and ESLint compliance. Use modern ES6+ syntax, add comments, and improve structure. Preserve functionality. Return only the enhanced code wrapped in \`\`\`javascript ... \`\`\`.`
    },
    '.html': {
      language: 'HTML',
      prompt: `Fix syntax errors and enhance the following HTML code for semantic structure, accessibility (ARIA attributes), and modern standards (HTML5). Add comments and improve formatting. Preserve functionality. Return only the enhanced code wrapped in \`\`\`html ... \`\`\`.`
    },
    '.css': {
      language: 'CSS',
      prompt: `Fix syntax errors and enhance the following CSS code for readability, performance, and modern standards (e.g., CSS3, BEM naming). Add comments and optimize selectors. Preserve functionality. Return only the enhanced code wrapped in \`\`\`css ... \`\`\`.`
    },
    '.md': {
      language: 'Markdown',
      prompt: `Enhance the following Markdown content for clarity, structure, and consistency. Improve headings, lists, links, and formatting. Add missing metadata if applicable. Return only the enhanced content wrapped in \`\`\`markdown ... \`\`\`.`
    },
    'default': {
      language: 'Text',
      prompt: `Fix any errors and enhance the following text content for clarity, structure, and best practices specific to its format. Add comments or metadata if appropriate. Return only the enhanced content wrapped in \`\`\`text ... \`\`\`.`
    }
  };
  return fileTypes[ext] || fileTypes.default;
}

// Enhance a single file
async function enhanceFile(filePath, modelName, rateLimiter, genai, dryRun = false) {
  try {
    logger.info(chalk.blue(`# Summoning enhancements for ${filePath}...`));
    if (!fs.existsSync(filePath)) {
      logger.error(chalk.red(`Error: File ${filePath} does not exist.`));
      return false;
    }

    // Read original file
    const originalContent = fs.readFileSync(filePath, 'utf-8');

    // Get file-specific prompt
    const { language, prompt: promptTemplate } = getFilePrompt(filePath);
    logger.info(chalk.blue(`# Detected file type: ${language}`));
    const prompt = `${promptTemplate}\n\n\`\`\`${language.toLowerCase()}\n${originalContent}\n\`\`\``;

    // Rate limiting
    await rateLimiter.acquire();

    // Call the AI model
    const model = genai.getGenerativeModel({ model: modelName });
    const result = await model.generateContent(prompt);

    if (!result || !result.response || !result.response.text) {
      logger.error(chalk.red(`Error: No valid response from AI for ${filePath}.`));
      return false;
    }

    // Extract enhanced content
    let enhancedContent = result.response.text().trim();
    const codeBlockRegex = new RegExp(`\\\`\`\`${language.toLowerCase()}[\\s\\n]*(.*?)[\\s\\n]*\\\`\`\`$`, 's');
    const match = enhancedContent.match(codeBlockRegex);
    if (match && match[1]) {
      enhancedContent = match[1].trim();
    } else {
      logger.warn(chalk.yellow(`# Response not properly formatted; assuming raw content.`));
    }

    // Compute and log diff
    const diffText = computeDiff(originalContent, enhancedContent);
    logger.info(chalk.cyan(`# Diff for ${filePath}:\n${diffText}`));

    if (dryRun) {
      logger.info(chalk.yellow(`# Dry run: Changes not applied to ${filePath}.`));
      return true;
    }

    // Create backup
    const backupPath = createBackup(filePath, originalContent, '.enhancement_backups');

    // Write enhanced content
    fs.writeFileSync(filePath, enhancedContent);

    logger.info(chalk.green(`Successfully enhanced ${filePath}.`));
    logger.info(chalk.yellow(`# Original length: ${originalContent.length} chars, Enhanced length: ${enhancedContent.length} chars`));
    return true;

  } catch (e) {
    logger.error(chalk.red(`Error enhancing ${filePath}: ${e.message}`));
    return false;
  }
}

// Main function
async function main() {
  // Parse arguments
  const argv = yargs
    .scriptName('enhance_file')
    .usage('$0 <file_path> [options]')
    .positional('file_path', { describe: 'Path to the file to enhance', type: 'string' })
    .option('model', { describe: 'AI model to use', type: 'string', default: 'gemini-1.5-pro' })
    .option('max-calls', { describe: 'Max API calls per minute (1-60)', type: 'number', default: 59 })
    .option('dry-run', { describe: 'Preview changes without applying them', type: 'boolean', default: false })
    .help()
    .argv;

  logger.info(chalk.magenta('# Pyrmethus File Enhancer Initialized'));

  // Validate max-calls
  if (argv['max-calls'] < 1 || argv['max-calls'] > 60) {
    logger.error(chalk.red(`Error: max-calls must be between 1 and 60, got ${argv['max-calls']}.`));
    process.exit(1);
  }

  // Configure API
  const genai = configureApi(process.env.GOOGLE_API_KEY);

  // Initialize rate limiter
  const rateLimiter = new RateLimiter(argv['max-calls']);

  // Enhance file with progress feedback
  const tqdm = new Tqdm({ total: 1, desc: 'Enhancing', barFormat: '{l_bar}{bar}| {n}/{total} [{elapsed}]' });
  const success = await enhanceFile(argv.file_path, argv.model, rateLimiter, genai, argv['dry-run']);
  tqdm.update(1);
  tqdm.close();

  if (success) {
    logger.info(chalk.green('# Enhancement spell completed successfully.'));
    process.exit(0);
  } else {
    logger.error(chalk.red('# Enhancement spell failed.'));
    process.exit(1);
  }
}

// Run main
main().catch(e => {
  logger.error(chalk.red(`Fatal error: ${e.message}`));
  process.exit(1);
});
