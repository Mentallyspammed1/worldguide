// twin-range-bot/scripts/reviewCode.js
const { GoogleGenerativeAI } = require(' @google/generative-ai');
const axios = require('axios');
const fs = require('fs');

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: 'gemini-2.5-pro' });

async function reviewCode() {
  const prNumber = process.env.GITHUB_EVENT_NUMBER;
  const repo = process.env.GITHUB_REPOSITORY;
  const files = await axios.get(`https://api.github.com/repos/${repo}/pulls/${prNumber}/files`, {
    headers: { Authorization: `token ${process.env.GITHUB_TOKEN}` },
  });
  for (const file of files.data) {
    const content = fs.readFileSync(file.filename, 'utf-8');
    const prompt = `Review this TypeScript code from ${file.filename} for issues and suggestions:\n${content}`;
    const result = await model.generateContent(prompt);
    const review = await result.response.text();
    await axios.post(
      `https://api.github.com/repos/${repo}/pulls/${prNumber}/comments`,
      { body: `Gemini Review for ${file.filename}:\n${review}`, path: file.filename, position: 1 },
      { headers: { Authorization: `token ${process.env.GITHUB_TOKEN}` } }
    );
  }
}

reviewCode().catch(console.error);

