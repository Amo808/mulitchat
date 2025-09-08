import asyncio
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any
import aiohttp
from .base_provider import BaseAdapter, Message, GenerationParams, ChatResponse, ModelInfo, ModelProvider, ModelType, ProviderConfig

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)


class GeminiAdapter(BaseAdapter):
    """Google Gemini AI Provider Adapter with streaming support"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = config.base_url or "https://generativelanguage.googleapis.com"
        self.base_url = self.base_url.rstrip("/")
        self.session = None
        
        # Initialize tokenizer for accurate token counting
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                # Use cl100k_base encoding (GPT-4 tokenizer) as approximation for Gemini
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self.logger.info("Tiktoken tokenizer initialized for accurate token counting")
            except Exception as e:
                self.logger.warning(f"Failed to initialize tiktoken: {e}")
                self.tokenizer = None
        else:
            self.logger.warning("Tiktoken not available, using character-based token estimation")
        
        # Debug: Check if API key is available
        if self.api_key:
            self.logger.info(f"Gemini API key loaded successfully (length: {len(self.api_key)})")
        else:
            self.logger.error("Gemini API key not found in environment variables!")

    @property
    def name(self) -> str:
        return "Google Gemini"

    @property
    def supported_models(self) -> List[ModelInfo]:
        return [
            # Gemini 2.5 Pro - самая мощная модель с мышлением
            ModelInfo(
                id="gemini-2.5-pro",
                name="gemini-2.5-pro",
                display_name="Gemini 2.5 Pro (Most Powerful Thinking)",
                provider=ModelProvider.GEMINI,
                context_length=2000000,  # 2M context window
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                type=ModelType.CHAT,
                max_output_tokens=8192,
                recommended_max_tokens=4096,
                pricing={
                    "input_tokens_small": 1.25,   # <= 200k tokens
                    "input_tokens_large": 2.50,   # > 200k tokens
                    "output_tokens_small": 10.00, # <= 200k tokens (including thinking)
                    "output_tokens_large": 15.00, # > 200k tokens (including thinking)
                    "context_caching_small": 0.31, # <= 200k tokens
                    "context_caching_large": 0.625, # > 200k tokens
                    "context_storage": 4.50       # per 1M tokens per hour
                },
                description="Most powerful thinking model with advanced reasoning and multimodal understanding"
            ),
            # Gemini 2.5 Flash - лучшее соотношение цена/качество
            ModelInfo(
                id="gemini-2.5-flash",
                name="gemini-2.5-flash",
                display_name="Gemini 2.5 Flash (Best Value + Thinking)",
                provider=ModelProvider.GEMINI,
                context_length=1000000,  # 1M context window
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                type=ModelType.CHAT,
                max_output_tokens=8192,
                recommended_max_tokens=4096,
                pricing={
                    "input_tokens": 0.30,    # text/image/video, audio is 1.00
                    "output_tokens": 2.50,   # including thinking tokens
                    "context_caching": 0.075, # text/image/video, audio is 0.25
                    "context_storage": 1.00,  # per 1M tokens per hour
                    "audio_input": 1.00,     # audio input special pricing
                    "audio_caching": 0.25,   # audio caching special pricing
                    "live_api_input_text": 0.50,
                    "live_api_input_audio": 3.00,
                    "live_api_output_text": 2.00,
                    "live_api_output_audio": 12.00
                },
                description="Best price-performance with adaptive thinking and comprehensive capabilities"
            ),
            # Gemini 2.5 Flash Lite - самая экономичная
            ModelInfo(
                id="gemini-2.5-flash-lite",
                name="gemini-2.5-flash-lite",
                display_name="Gemini 2.5 Flash Lite (Fastest & Cheapest)",
                provider=ModelProvider.GEMINI,
                context_length=1000000,
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                type=ModelType.CHAT,
                max_output_tokens=8192,
                recommended_max_tokens=4096,
                pricing={
                    "input_tokens": 0.10,    # text/image/video
                    "input_tokens_audio": 0.30, # audio input
                    "output_tokens": 0.40,   # including thinking tokens
                    "context_caching": 0.025, # text/image/video
                    "context_caching_audio": 0.125, # audio caching
                    "context_storage": 1.00  # per 1M tokens per hour
                },
                description="Most cost-effective model with high throughput and low latency"
            ),
            # Gemini 2.0 Flash - новое поколение
            ModelInfo(
                id="gemini-2.0-flash",
                name="gemini-2.0-flash",
                display_name="Gemini 2.0 Flash (Next-Gen)",
                provider=ModelProvider.GEMINI,
                context_length=1000000,
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                type=ModelType.CHAT,
                max_output_tokens=8192,
                recommended_max_tokens=4096,
                pricing={
                    "input_tokens": 0.10,    # text/image/video
                    "input_tokens_audio": 0.70, # audio input
                    "output_tokens": 0.40,   # all outputs
                    "context_caching": 0.025, # text/image/video per 1M tokens
                    "context_caching_audio": 0.175, # audio caching per 1M tokens
                    "context_storage": 1.00, # per 1M tokens per hour
                    "image_generation": 0.039, # per image (1290 tokens)
                    "live_api_input_text": 0.35,
                    "live_api_input_audio": 2.10,
                    "live_api_output_text": 1.50,
                    "live_api_output_audio": 8.50
                },
                description="Next-generation features with speed and real-time streaming"
            ),
            # Gemini 1.5 Pro - устаревшая но мощная
            ModelInfo(
                id="gemini-1.5-pro",
                name="gemini-1.5-pro",
                display_name="Gemini 1.5 Pro (Legacy)",
                provider=ModelProvider.GEMINI,
                context_length=2000000,
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                type=ModelType.CHAT,
                max_output_tokens=8192,
                recommended_max_tokens=4096,
                pricing={
                    "input_tokens_small": 1.25,  # <= 128k tokens
                    "input_tokens_large": 2.50,  # > 128k tokens
                    "output_tokens_small": 5.00, # <= 128k tokens
                    "output_tokens_large": 10.00, # > 128k tokens
                    "context_caching_small": 0.3125, # <= 128k tokens
                    "context_caching_large": 0.625,  # > 128k tokens
                    "context_storage": 4.50,     # per 1M tokens per hour
                    "grounding_search": 35.0     # per 1k grounding requests
                },
                description="Complex reasoning tasks requiring greater intelligence (legacy)"
            ),
            # Gemini 1.5 Flash - устаревшая но быстрая
            ModelInfo(
                id="gemini-1.5-flash",
                name="gemini-1.5-flash",
                display_name="Gemini 1.5 Flash (Legacy)",
                provider=ModelProvider.GEMINI,
                context_length=1000000,
                supports_streaming=True,
                supports_functions=True,
                supports_vision=True,
                type=ModelType.CHAT,
                max_output_tokens=8192,
                recommended_max_tokens=4096,
                pricing={
                    "input_tokens_small": 0.075,  # <= 128k tokens
                    "input_tokens_large": 0.15,   # > 128k tokens
                    "output_tokens_small": 0.30,  # <= 128k tokens
                    "output_tokens_large": 0.60,  # > 128k tokens
                    "context_caching_small": 0.01875, # <= 128k tokens
                    "context_caching_large": 0.0375,  # > 128k tokens
                    "context_storage": 1.00,      # per 1M tokens per hour
                    "grounding_search": 35.0      # per 1k grounding requests
                },
                description="Fast and versatile performance across diverse tasks (legacy)"
            )
        ]

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=10,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=300,
                enable_cleanup_closed=True,
                force_close=False
            )
            # NO TIMEOUTS - allow unlimited response time like DeepSeek
            timeout = aiohttp.ClientTimeout(
                total=None,      # No total timeout
                connect=60,      # 60s to establish connection
                sock_read=None,  # NO read timeout - unlimited time between chunks
                sock_connect=60  # 60s socket connect timeout
            )
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "AI-Chat/2.0-Unlimited",
                    "Connection": "keep-alive",
                    "Keep-Alive": "timeout=300, max=1000"
                }
            )

    async def chat_completion(
        self,
        messages: List[Message],
        model: str = "gemini-2.5-flash",
        params: GenerationParams = None,
        **kwargs
    ) -> AsyncGenerator[ChatResponse, None]:
        if params is None:
            params = GenerationParams()

        await self._ensure_session()
        
        # Convert messages to Gemini API format
        contents = []
        for msg in messages:
            if msg.role == "system":
                # Gemini doesn't have system role, add as instruction to first user message
                if not contents:
                    contents.append({
                        "role": "user",
                        "parts": [{"text": f"System instruction: {msg.content}"}]
                    })
                else:
                    # Prepend to first user message
                    for content in contents:
                        if content["role"] == "user":
                            content["parts"][0]["text"] = f"System instruction: {msg.content}\n\n{content['parts'][0]['text']}"
                            break
            else:
                role = "model" if msg.role == "assistant" else "user"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })
        
        # Ensure we have at least one message
        if not contents:
            yield ChatResponse(
                error="No messages to process",
                meta={"provider": ModelProvider.GEMINI, "model": model}
            )
            return

        # Calculate input tokens (rough estimation)
        input_text = "\n".join([f"{msg.get('role', 'user')}: {msg.get('parts', [{}])[0].get('text', '')}" for msg in contents])
        input_tokens = self.estimate_tokens(input_text)

        # Gemini API payload
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": params.temperature,
                "maxOutputTokens": params.max_tokens,
                "topP": params.top_p,
            }
        }

        if params.stop_sequences:
            payload["generationConfig"]["stopSequences"] = params.stop_sequences

        self.logger.info(f"Sending request to Gemini API: {model}, temp={params.temperature}")

        accumulated_content = ""
        output_tokens = 0
        
        # Check if API key is available
        if not self.api_key:
            self.logger.error("Gemini API key not configured")
            yield ChatResponse(
                error="Gemini API key not configured. Please add your API key in settings.",
                meta={"provider": ModelProvider.GEMINI, "model": model}
            )
            return

        try:
            # Use streaming or non-streaming endpoint
            if params.stream:
                url = f"{self.base_url}/v1beta/models/{model}:streamGenerateContent?key={self.api_key}"
            else:
                url = f"{self.base_url}/v1beta/models/{model}:generateContent?key={self.api_key}"
            
            # Log URL for debugging (hide API key)
            safe_url = url.replace(self.api_key, "***API_KEY***") if self.api_key else url
            self.logger.info(f"Making request to: {safe_url}")
            
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"Gemini API error: {response.status} - {error_text}")
                    yield ChatResponse(
                        error=f"API Error {response.status}: {error_text}",
                        meta={"provider": ModelProvider.GEMINI, "model": model}
                    )
                    return

                if not params.stream:
                    # Handle non-streaming response
                    data = await response.json()
                    candidates = data.get("candidates", [])
                    if candidates and candidates[0].get("content"):
                        content = candidates[0]["content"]["parts"][0]["text"]
                        usage_metadata = data.get("usageMetadata", {})
                        
                        # Get accurate token counts from API or estimate
                        actual_input_tokens = usage_metadata.get("promptTokenCount", input_tokens)
                        actual_output_tokens = usage_metadata.get("candidatesTokenCount", self.estimate_tokens(content))
                        estimated_cost = self._calculate_cost(actual_input_tokens, actual_output_tokens, model, actual_input_tokens)
                        
                        yield ChatResponse(
                            content=content,
                            done=True,
                            meta={
                                "tokens_in": actual_input_tokens,
                                "tokens_out": actual_output_tokens,
                                "total_tokens": actual_input_tokens + actual_output_tokens,
                                "estimated_cost": estimated_cost,
                                "provider": ModelProvider.GEMINI,
                                "model": model
                            }
                        )
                    else:
                        yield ChatResponse(
                            error="No valid response from Gemini",
                            meta={"provider": ModelProvider.GEMINI, "model": model}
                        )
                    return

                # Handle streaming response - Gemini returns text fragments in JSON format
                # We need to buffer and parse complete JSON objects
                text_buffer = ""
                json_buffer = ""
                bracket_count = 0
                in_json = False
                
                async for chunk in response.content.iter_chunked(1024):
                    text_buffer += chunk.decode('utf-8', errors='ignore')
                    
                    # Process character by character to find complete JSON objects
                    i = 0
                    while i < len(text_buffer):
                        char = text_buffer[i]
                        
                        if char == '{' and not in_json:
                            # Start of a new JSON object
                            in_json = True
                            bracket_count = 1
                            json_buffer = char
                        elif in_json:
                            json_buffer += char
                            if char == '{':
                                bracket_count += 1
                            elif char == '}':
                                bracket_count -= 1
                                
                                # Complete JSON object found
                                if bracket_count == 0:
                                    try:
                                        json_response = json.loads(json_buffer)
                                        candidates = json_response.get("candidates", [])
                                        
                                        if candidates:
                                            candidate = candidates[0]
                                            content_part = candidate.get("content", {})
                                            parts = content_part.get("parts", [])
                                            
                                            if parts and "text" in parts[0]:
                                                content = parts[0]["text"]
                                                accumulated_content += content
                                                
                                                # Yield the streaming chunk immediately
                                                yield ChatResponse(
                                                    content=content,
                                                    done=False,
                                                    meta={
                                                        "tokens_in": input_tokens,
                                                        "tokens_out": self.estimate_tokens(accumulated_content),
                                                        "provider": ModelProvider.GEMINI,
                                                        "model": model
                                                    }
                                                )
                                            
                                            # Check for finish reason
                                            finish_reason = candidate.get("finishReason")
                                            if finish_reason:
                                                self.logger.info(f"Gemini response finished with reason: {finish_reason}")
                                                if finish_reason == "MAX_TOKENS":
                                                    self.logger.warning(f"Response was truncated due to max_tokens limit.")
                                                    yield ChatResponse(
                                                        content="\n\n⚠️ *Response was truncated due to token limit. You can increase max_tokens in settings for longer responses.*",
                                                        done=False,
                                                        meta={
                                                            "tokens_in": input_tokens,
                                                            "tokens_out": self.estimate_tokens(accumulated_content),
                                                            "provider": ModelProvider.GEMINI,
                                                            "model": model
                                                        }
                                                    )
                                                
                                                # Send final completion signal
                                                final_output_tokens = self.estimate_tokens(accumulated_content)
                                                estimated_cost = self._calculate_cost(input_tokens, final_output_tokens, model, input_tokens)
                                                yield ChatResponse(
                                                    content="",
                                                    done=True,
                                                    meta={
                                                        "tokens_in": input_tokens,
                                                        "tokens_out": final_output_tokens,
                                                        "total_tokens": input_tokens + final_output_tokens,
                                                        "estimated_cost": estimated_cost,
                                                        "provider": ModelProvider.GEMINI,
                                                        "model": model
                                                    }
                                                )
                                                return
                                        
                                    except json.JSONDecodeError as e:
                                        self.logger.warning(f"Failed to parse JSON object: {json_buffer[:100]}... - {e}")
                                    except Exception as e:
                                        self.logger.error(f"Error processing streaming response: {e}")
                                    
                                    # Reset for next JSON object
                                    in_json = False
                                    json_buffer = ""
                                    bracket_count = 0
                        
                        i += 1
                    
                    # Keep only the unprocessed part of the buffer
                    if in_json:
                        # We're in the middle of a JSON object, keep everything
                        text_buffer = ""
                    else:
                        # Remove processed characters, keep last incomplete part
                        text_buffer = text_buffer[i:]

        except Exception as e:
            self.logger.error(f"Error in Gemini API call: {e}")
            yield ChatResponse(
                error=f"API Error: {str(e)}",
                meta={"provider": ModelProvider.GEMINI, "model": model}
            )
            return

        # If we reach here, send a final response to ensure completion
        final_output_tokens = self.estimate_tokens(accumulated_content) if accumulated_content else output_tokens
        estimated_cost = self._calculate_cost(input_tokens, final_output_tokens, model, input_tokens)
        yield ChatResponse(
            content="",
            done=True,
            meta={
                "tokens_in": input_tokens,
                "tokens_out": final_output_tokens,
                "total_tokens": input_tokens + final_output_tokens,
                "estimated_cost": estimated_cost,
                "provider": ModelProvider.GEMINI,
                "model": model
            }
        )

    def _process_gemini_candidate(self, candidate, accumulated_content, input_tokens, model):
        """Process a single Gemini candidate and return the content"""
        content_part = candidate.get("content", {})
        parts = content_part.get("parts", [])
        
        if parts and "text" in parts[0]:
            content = parts[0]["text"]
            return content
        return None

    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models from Gemini"""
        return self.supported_models

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken (more accurate) or character-based approximation"""
        if not text:
            return 0
            
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                self.logger.warning(f"Tiktoken encoding failed: {e}, falling back to character estimation")
        
        # Fallback: Gemini uses different tokenization, rough estimate: 1 token ≈ 3.5 characters
        # This is more accurate than the previous 4 characters per token
        return max(1, len(text) // 4)

    async def estimate_tokens_accurate(self, text: str, model: str) -> int:
        """Get accurate token count using Gemini API if possible, fallback to estimation"""
        # Try to get accurate count from API first
        api_count = await self.count_tokens_api(text, model)
        if api_count is not None:
            return api_count
        
        # Fallback to local estimation
        return self.estimate_tokens(text)

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model: str, 
                       prompt_size: int = 0) -> float:
        """Calculate estimated cost based on Gemini model pricing"""
        
        # Get model info for pricing
        model_info = None
        for m in self.supported_models:
            if m.id == model or m.name == model:
                model_info = m
                break
        
        if not model_info or not model_info.pricing:
            # Default fallback pricing (Gemini 2.5 Flash)
            input_cost_per_million = 0.30
            output_cost_per_million = 2.50
        else:
            pricing = model_info.pricing
            
            # Determine pricing based on model and prompt size
            if model in ["gemini-2.5-pro"]:
                # Gemini 2.5 Pro has different pricing for <= 200k vs > 200k tokens
                if prompt_size <= 200_000:
                    input_cost_per_million = pricing.get("input_tokens_small", 1.25)
                    output_cost_per_million = pricing.get("output_tokens_small", 10.00)
                else:
                    input_cost_per_million = pricing.get("input_tokens_large", 2.50)
                    output_cost_per_million = pricing.get("output_tokens_large", 15.00)
            elif model in ["gemini-1.5-pro", "gemini-1.5-flash"]:
                # Gemini 1.5 models have different pricing for <= 128k vs > 128k tokens
                if prompt_size <= 128_000:
                    input_cost_per_million = pricing.get("input_tokens_small", 0.075)
                    output_cost_per_million = pricing.get("output_tokens_small", 0.30)
                else:
                    input_cost_per_million = pricing.get("input_tokens_large", 0.15)
                    output_cost_per_million = pricing.get("output_tokens_large", 0.60)
            else:
                # Simple pricing models (2.5 Flash, 2.5 Flash Lite, 2.0 Flash)
                input_cost_per_million = pricing.get("input_tokens", 0.30)
                output_cost_per_million = pricing.get("output_tokens", 2.50)
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * output_cost_per_million
        
        total_cost = input_cost + output_cost
        return round(total_cost, 6)

    async def close(self):
        """Clean up session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def count_tokens_api(self, text: str, model: str) -> Optional[int]:
        """
        Get accurate token count from Gemini API
        Gemini API provides a countTokens endpoint for precise token counting
        """
        if not self.api_key or not text:
            return None
            
        try:
            await self._ensure_session()
            
            url = f"{self.base_url}/v1beta/models/{model}:countTokens?key={self.api_key}"
            payload = {
                "contents": [{
                    "parts": [{"text": text}]
                }]
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("totalTokens", None)
                else:
                    self.logger.warning(f"Token counting API failed: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.warning(f"Failed to get accurate token count from API: {e}")
            return None

    async def get_model_pricing_info(self, model: str) -> Dict[str, Any]:
        """Get pricing information for a specific model"""
        for model_info in self.supported_models:
            if model_info.id == model or model_info.name == model:
                return {
                    "model": model,
                    "pricing": model_info.pricing,
                    "context_length": model_info.context_length,
                    "max_output_tokens": model_info.max_output_tokens,
                    "provider": "Gemini"
                }
        return {"model": model, "pricing": None, "provider": "Gemini"}