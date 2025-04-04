import json
import logging
import aiohttp
import pandas as pd
import platform
import asyncio
from typing import Dict, Optional
from .base import LLMProvider
from src.prompts.generators import BasePromptGenerator, PromptV0, PromptFVG

# Configure Windows-specific event loop policy
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging to ignore DEBUG from other libraries
logging.getLogger('polygon').setLevel(logging.WARNING)
logging.getLogger('peewee').setLevel(logging.WARNING)

class DeepSeekProvider(LLMProvider):
    """DeepSeek implementation of the LLM provider."""
    
    def __init__(self, api_key: str, dry_run: bool = False, prompt_generator: Optional[BasePromptGenerator] = None):
        """Initialize provider with API key.
        
        Args:
            api_key: DeepSeek API key
            dry_run: If True, only log the prompt without making API calls
            prompt_generator: Optional prompt generator to use. Defaults to PromptFVG
        """
        self.api_key = api_key
        self.dry_run = dry_run
        self.prompt_generator = prompt_generator or PromptFVG()
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

    def _format_reasoning(self, reasoning_dict: dict) -> str:
        """Format the reasoning dictionary into a readable string."""
        if not isinstance(reasoning_dict, dict):
            return str(reasoning_dict)
            
        sections = []
        for key, value in reasoning_dict.items():
            # Convert key from snake_case to Title Case
            title = key.replace('_', ' ').title()
            sections.append(f"{title}: {value}")
            
        return "\n".join(sections)

    async def get_trading_decision(self, hourly_df: pd.DataFrame, min15_df: pd.DataFrame, min5_df: pd.DataFrame, min1_df: pd.DataFrame, additional_context: Optional[Dict] = None) -> Dict:
        """Get a trading decision from the model."""
        # Generate prompt using the configured generator
        prompt = self.prompt_generator.generate(
            hourly_df=hourly_df,
            min15_df=min15_df,
            min5_df=min5_df,
            min1_df=min1_df,
            additional_context=additional_context
        )
        
        self.logger.info("Generated prompt:")
        self.logger.info(prompt)
        
        if self.dry_run:
            self.logger.info("Dry run mode - skipping API call")
            return {
                "current_position": {
                    "position": 0.0,
                    "confidence": 0.0,
                    "take_profit": None,
                    "stop_loss": None
                },
                "daily_bias": {
                    "direction": 0.0,
                    "confidence": 0.0,
                    "key_levels": []
                },
                "reasoning": "Dry run mode - no API call made"
            }
        
        # Prepare API request
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "deepseek-reasoner",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional futures trader. You will analyze market data and provide trading decisions in JSON format with position, confidence, and reasoning fields."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    self.logger.info(f"API response status: {response.status}")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"API error response: {error_text}")
                        raise ValueError(f"API request failed with status {response.status}: {error_text}")
                    
                    raw_response = await response.text()
                    self.logger.info(f"Raw API response: {raw_response}")
                    
                    try:
                        response_json = json.loads(raw_response)
                        content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # Clean up content (remove markdown code blocks if present)
                        content = content.replace("```json", "").replace("```", "").strip()
                        
                        # Find the JSON object in the content
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        if start == -1 or end == 0:
                            raise ValueError("No JSON object found in response")
                        
                        json_str = content[start:end]
                        self.logger.info(f"Cleaned content for parsing: {json_str}")
                        
                        # Parse the response
                        decision = json.loads(json_str)
                        
                        # Validate decision format first
                        required_sections = {
                            "daily_bias": ["direction", "confidence", "key_levels"],
                            "current_position": ["position", "confidence", "take_profit", "stop_loss"],
                            "reasoning": ["daily_context", "levels_analysis", "trend_alignment", "volume_analysis", "risk_reward"]
                        }
                        
                        for section, required_keys in required_sections.items():
                            if section not in decision:
                                raise ValueError(f"Missing required section: {section}")
                            if not all(key in decision[section] for key in required_keys):
                                raise ValueError(f"Missing required keys in {section}: {required_keys}")
                        
                        # Validate value ranges
                        daily_bias = decision["daily_bias"]
                        current_pos = decision["current_position"]
                        
                        if not -1.0 <= float(daily_bias["direction"]) <= 1.0:
                            raise ValueError(f"Daily bias direction out of range: {daily_bias['direction']}")
                        if not 0.0 <= float(daily_bias["confidence"]) <= 1.0:
                            raise ValueError(f"Daily bias confidence out of range: {daily_bias['confidence']}")
                            
                        if not -1.0 <= float(current_pos["position"]) <= 1.0:
                            raise ValueError(f"Current position value out of range: {current_pos['position']}")
                        if not 0.0 <= float(current_pos["confidence"]) <= 1.0:
                            raise ValueError(f"Current position confidence out of range: {current_pos['confidence']}")
                        
                        # Validate take-profit and stop-loss are numeric
                        try:
                            current_pos["take_profit"] = float(current_pos["take_profit"])
                            current_pos["stop_loss"] = float(current_pos["stop_loss"])
                        except (ValueError, TypeError):
                            raise ValueError("take_profit and stop_loss must be numeric values")
                        
                        # Validate take-profit and stop-loss make sense for the position
                        current_price = float(min1_df.iloc[-1]['close'])
                        if current_pos["position"] > 0:  # Long position
                            if current_pos["take_profit"] <= current_price:
                                raise ValueError("take_profit must be above current price for long positions")
                            if current_pos["stop_loss"] >= current_price:
                                raise ValueError("stop_loss must be below current price for long positions")
                        elif current_pos["position"] < 0:  # Short position
                            if current_pos["take_profit"] >= current_price:
                                raise ValueError("take_profit must be below current price for short positions")
                            if current_pos["stop_loss"] <= current_price:
                                raise ValueError("stop_loss must be above current price for short positions")
                        
                        # Format the reasoning after validation
                        reasoning_dict = decision['reasoning']
                        formatted_reasoning = []
                        for key, value in reasoning_dict.items():
                            # Convert key from snake_case to Title Case
                            title = key.replace('_', ' ').title()
                            formatted_reasoning.append(f"{title}: {value}")
                        
                        # Return the decision in the format expected by the trading bot
                        final_decision = {
                            "current_position": {
                                "position": float(current_pos["position"]),
                                "confidence": float(current_pos["confidence"]),
                                "take_profit": float(current_pos["take_profit"]),
                                "stop_loss": float(current_pos["stop_loss"])
                            },
                            "daily_bias": {
                                "direction": float(daily_bias["direction"]),
                                "confidence": float(daily_bias["confidence"]),
                                "key_levels": daily_bias["key_levels"]
                            },
                            "reasoning": "\n".join(formatted_reasoning)
                        }
                        
                        self.logger.info(f"Final decision: {final_decision}")
                        return final_decision
                        
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse JSON response: {e}")
                        self.logger.error(f"Raw response that failed to parse: {raw_response}")
                        raise
                        
        except Exception as e:
            self.logger.error(f"Error getting trading decision: {str(e)}")
            return {
                "current_position": {
                    "position": 0.0,
                    "confidence": 0.0,
                    "take_profit": None,
                    "stop_loss": None
                },
                "daily_bias": {
                    "direction": 0.0,
                    "confidence": 0.0,
                    "key_levels": []
                },
                "reasoning": f"Error getting trading decision: {str(e)}"
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