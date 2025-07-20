// twin-range-bot/src/services/geminiService.ts
import { GoogleGenerativeAI } from '@google/generative-ai';
import { logger } from '../core/logger';
import { BotConfig, KlineData } from '../../types';

export class GeminiService {
  private genAI: GoogleGenerativeAI | null;
  private model: any;

  constructor(apiKey: string | null, config: BotConfig) {
    this.genAI = apiKey ? new GoogleGenerativeAI(apiKey) : null;
    this.model = this.genAI ? this.genAI.getGenerativeModel({ model: 'gemini-2.5-pro' }) : null;
    logger.info('GeminiService initialized', { hasApiKey: !!apiKey, symbols: config.symbols });
  }

  async analyzeMarketData(symbol: string, klines: KlineData[]): Promise<string> {
    if (!this.model) {
      logger.warn('Gemini API not available for market analysis', { symbol });
      return 'No API key provided';
    }
    const prompt = `Analyze the following kline data for ${symbol} and provide a concise trading signal (Buy, Sell, Hold) with a brief rationale:\n${JSON.stringify(klines.slice(-5), null, 2)}`;
    try {
      const result = await this.model.generateContent(prompt);
      const response = await result.response;
      const signal = response.text();
      logger.info('Gemini market analysis', { symbol, signal });
      return signal;
    } catch (err: any) {
      logger.error('Gemini market analysis failed', { symbol, error: err.message });
      return 'Analysis failed';
    }
  }

  async reviewCode(fileContent: string, fileName: string): Promise<string> {
    if (!this.model) {
      logger.warn('Gemini API not available for code review', { fileName });
      return 'No API key provided';
    }
    const prompt = `Review the following TypeScript code from ${fileName} for issues, best practices, and optimization suggestions:\n${fileContent}`;
    try {
      const result = await this.model.generateContent(prompt);
      const response = await result.response;
      const review = response.text();
      logger.info('Gemini code review completed', { fileName });
      return review;
    } catch (err: any) {
      logger.error('Gemini code review failed', { fileName, error: err.message });
      return 'Review failed';
    }
  }
}