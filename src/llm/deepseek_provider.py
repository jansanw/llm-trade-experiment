import json
import logging
import aiohttp
import pandas as pd
from typing import Dict, Optional
from .base import LLMProvider

# Configure logging to ignore DEBUG from other libraries
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('peewee').setLevel(logging.WARNING)

class DeepSeekProvider(LLMProvider):
    """DeepSeek implementation of the LLM provider."""
    
    def __init__(self, api_key: str, dry_run: bool = False):
        """Initialize provider with API key.
        
        Args:
            api_key: DeepSeek API key
            dry_run: If True, only log the prompt without making API calls
        """
        self.api_key = api_key
        self.dry_run = dry_run
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
        # Generate prompt
        prompt = self._generate_prompt(hourly_df, min15_df, min5_df, min1_df, additional_context)
        self.logger.info("\n=== Generated Prompt ===\n%s\n%s", prompt, "=" * 80)

        if self.dry_run:
            return {
                "position": 0.0,
                "confidence": 0.0,
                "reasoning": "Dry run mode - no API call made"
            }

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

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    response_text = await response.text()
                    self.logger.info("\n=== Raw Response ===\n%s\n%s", response_text, "=" * 80)

                    if response.status != 200:
                        print(f"API request failed: {response_text}")
                        return {
                            "position": 0.0,
                            "confidence": 0.0,
                            "reasoning": f"API request failed: {response_text}"
                        }

                    # Try to parse response
                    try:
                        data = json.loads(response_text)
                        content = data["choices"][0]["message"]["content"].strip()

                        # Clean up content - remove markdown code blocks if present
                        content = content.replace("```json", "").replace("```", "").strip()
                        # Find the first { and last }
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        if start >= 0 and end > start:
                            content = content[start:end]

                        # Try to parse content as JSON
                        try:
                            result = json.loads(content)
                            print(f"Trading decision: {json.dumps(result, indent=2)}")
                            return result
                        except json.JSONDecodeError:
                            print(f"Failed to parse model response as JSON: {content}")
                            return {
                                "position": 0.0,
                                "confidence": 0.0,
                                "reasoning": f"Failed to parse model response as JSON: {content}"
                            }

                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        print(f"Failed to process API response: {e}")
                        return {
                            "position": 0.0,
                            "confidence": 0.0,
                            "reasoning": f"Failed to process API response: {str(e)}"
                        }

        except Exception as e:
            print(f"Request failed: {e}")
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