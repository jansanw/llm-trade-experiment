import json
import logging
import aiohttp
import pandas as pd
from typing import Dict, Optional
from .base import LLMProvider

class DeepSeekProvider(LLMProvider):
    """DeepSeek implementation of the LLM provider."""
    
    def __init__(self, api_key: str):
        """Initialize provider with API key."""
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)

    def _generate_prompt(self, hourly_df: pd.DataFrame, min15_df: pd.DataFrame, min5_df: pd.DataFrame, min1_df: pd.DataFrame, additional_context: dict = None) -> str:
        """Generate analysis prompt from market data."""
        # Get summary stats for each timeframe
        summaries = []
        for name, df in [
            ("hourly", hourly_df),
            ("15-minute", min15_df),
            ("5-minute", min5_df),
            ("1-minute", min1_df)
        ]:
            if df.empty:
                continue
                
            current = df.iloc[-1]
            recent = df.tail(10)
            
            price_change = ((current['close'] - recent.iloc[0]['close']) / recent.iloc[0]['close']) * 100
            volatility = recent['close'].std()
            volume_trend = "Increasing" if recent['volume'].is_monotonic_increasing else "Decreasing"
            
            summary = f"{name.capitalize()} candles:\n"
            summary += f"Current: Open={current['open']:.2f}, High={current['high']:.2f}, Low={current['low']:.2f}, Close={current['close']:.2f}\n"
            summary += "Recent trends:\n"
            summary += f"- Price change last 10 periods: {price_change:.2f}%\n"
            summary += f"- Volatility (10-period std): {volatility:.2f}\n"
            summary += f"- Volume trend: {volume_trend}\n"
            summaries.append(summary)
            
        prompt = "You are a professional futures trader. Analyze the following market data and provide a trading decision.\n\n"
        prompt += "Market Data:\n"
        prompt += "\n".join(f"{i+1}. Last 100 {summary}" for i, summary in enumerate(summaries))
        
        if additional_context:
            prompt += f"\nAdditional Context:\n{additional_context}\n"
            
        prompt += "\nBased on this data, should we go long or short? Provide:\n"
        prompt += "1. Position (-1.0 for full short to 1.0 for full long)\n"
        prompt += "2. Confidence level (0.0 to 1.0)\n"
        prompt += "3. Brief explanation of your reasoning\n\n"
        prompt += "Format your response as a JSON object with keys: position, confidence, reasoning"
        
        return prompt

    async def get_trading_decision(self, hourly_df: pd.DataFrame, min15_df: pd.DataFrame, min5_df: pd.DataFrame, min1_df: pd.DataFrame, additional_context: Optional[Dict] = None) -> Dict:
        """Get trading decision from DeepSeek."""
        print("Starting trading decision request") # Debug print
        self.logger.info("Starting trading decision request")

        # Generate prompt
        print("Generating prompt...") # Debug print
        prompt = self._generate_prompt(hourly_df, min15_df, min5_df, min1_df, additional_context)
        print("Prompt generated") # Debug print
        self.logger.info("\n=== Generated Prompt ===")
        self.logger.info(prompt)
        self.logger.info("=" * 80)

        # Make API request
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "system", "content": "You are a professional futures trader. Always respond with a valid JSON object containing position (-1.0 to 1.0), confidence (0.0 to 1.0), and reasoning."},
                {"role": "user", "content": prompt}
            ]
        }

        print(f"Making API request to {url}") # Debug print
        try:
            async with aiohttp.ClientSession() as session:
                print("Session created") # Debug print
                async with session.post(url, headers=headers, json=payload) as response:
                    print(f"Got response with status {response.status}") # Debug print
                    response_text = await response.text()
                    print(f"Response text: {response_text[:200]}...") # Debug print first 200 chars
                    self.logger.info("\n=== Raw Response ===")
                    self.logger.info(response_text)
                    self.logger.info("=" * 80)

                    if response.status != 200:
                        print(f"Request failed with status {response.status}") # Debug print
                        return {
                            "position": 0.0,
                            "confidence": 0.0,
                            "reasoning": f"API request failed: {response_text}"
                        }

                    # Try to parse response
                    try:
                        print("Parsing response as JSON") # Debug print
                        data = json.loads(response_text)
                        content = data["choices"][0]["message"]["content"].strip()
                        print(f"Got content: {content[:200]}...") # Debug print first 200 chars
                        self.logger.info("\n=== Model Response ===")
                        self.logger.info(content)
                        self.logger.info("=" * 80)

                        # Clean up content - remove markdown code blocks if present
                        content = content.replace("```json", "").replace("```", "").strip()
                        # Find the first { and last }
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        if start >= 0 and end > start:
                            content = content[start:end]
                        
                        print(f"Cleaned content: {content}") # Debug print

                        # Try to parse content as JSON
                        try:
                            print("Parsing content as JSON") # Debug print
                            result = json.loads(content)
                            print(f"Final result: {result}") # Debug print
                            return result
                        except json.JSONDecodeError:
                            print("Failed to parse content as JSON") # Debug print
                            return {
                                "position": 0.0,
                                "confidence": 0.0,
                                "reasoning": f"Failed to parse model response as JSON: {content}"
                            }

                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        print(f"Failed to process API response: {e}") # Debug print
                        return {
                            "position": 0.0,
                            "confidence": 0.0,
                            "reasoning": f"Failed to process API response: {str(e)}"
                        }

        except Exception as e:
            print(f"Request failed with error: {e}") # Debug print
            return {
                "position": 0.0,
                "confidence": 0.0,
                "reasoning": f"Request failed: {str(e)}"
            }

    async def test_api_connection(self):
        """Test the API connection with a simple request."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"API test failed: {error_text}")
                        return False
                    return True
        except Exception as e:
            self.logger.error(f"API test failed: {str(e)}")
            return False 