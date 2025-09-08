# Gemini Token Counting & Cost Calculation

## Overview

The Gemini provider now includes advanced token counting and cost calculation capabilities with support for all current Gemini models and their pricing tiers.

## Features

### 🔢 Accurate Token Counting
- **Tiktoken Integration**: Uses OpenAI's tiktoken library for more accurate token estimation
- **API-Based Counting**: Can use Gemini's native `countTokens` API endpoint for precise counts
- **Fallback Estimation**: Character-based estimation when other methods unavailable

### 💰 Comprehensive Cost Calculation
- **Real-time Pricing**: Updated pricing for all Gemini models as of September 2025
- **Context-Aware Pricing**: Handles different pricing tiers based on prompt size
- **Detailed Breakdown**: Separate costs for input tokens, output tokens, and special features

## Supported Models & Pricing

### Gemini 2.5 Pro (Most Powerful Thinking)
- **Context**: 2M tokens
- **Pricing**: 
  - Input: $1.25/1M (≤200k), $2.50/1M (>200k)
  - Output: $10.00/1M (≤200k), $15.00/1M (>200k)
  - Includes thinking tokens in output pricing

### Gemini 2.5 Flash (Best Value + Thinking)
- **Context**: 1M tokens
- **Pricing**:
  - Input: $0.30/1M (text/image/video), $1.00/1M (audio)
  - Output: $2.50/1M (includes thinking)
  - Context Caching: $0.075/1M (text/image/video)

### Gemini 2.5 Flash Lite (Fastest & Cheapest)
- **Context**: 1M tokens
- **Pricing**:
  - Input: $0.10/1M (text/image/video), $0.30/1M (audio)
  - Output: $0.40/1M
  - Most cost-effective option

### Gemini 2.0 Flash (Next-Gen)
- **Context**: 1M tokens
- **Pricing**:
  - Input: $0.10/1M (text/image/video), $0.70/1M (audio)
  - Output: $0.40/1M
  - Image Generation: $0.039/image

### Legacy Models (1.5 Pro & Flash)
- **Pricing tiers**: Different rates for ≤128k vs >128k tokens
- **Context caching**: Available with hourly storage costs

## Usage Examples

### Basic Token Counting
```python
from adapters.gemini_provider import GeminiAdapter

# Initialize adapter
gemini = GeminiAdapter(config)

# Count tokens locally
tokens = gemini.estimate_tokens("Your text here")

# Get accurate count from API
tokens = await gemini.estimate_tokens_accurate("Your text", "gemini-2.5-flash")
```

### Cost Calculation
```python
# Calculate cost for a chat completion
input_tokens = 1000
output_tokens = 500
model = "gemini-2.5-flash"

cost = gemini._calculate_cost(input_tokens, output_tokens, model, input_tokens)
print(f"Estimated cost: ${cost:.6f}")
```

### Model Pricing Information
```python
# Get detailed pricing info for a model
pricing_info = await gemini.get_model_pricing_info("gemini-2.5-pro")
print(pricing_info)
```

## Real-time Cost Tracking

The provider automatically includes cost information in chat responses:

```python
async for response in gemini.chat_completion(messages, params):
    if response.done and response.meta:
        tokens_in = response.meta.get('tokens_in', 0)
        tokens_out = response.meta.get('tokens_out', 0) 
        cost = response.meta.get('estimated_cost', 0)
        
        print(f"Tokens used: {tokens_in + tokens_out}")
        print(f"Estimated cost: ${cost:.6f}")
```

## Testing

Run the test script to verify token counting accuracy:

```bash
python test_gemini_tokens.py
```

## Notes

- **Thinking Tokens**: For Gemini 2.5 models, thinking tokens are included in output pricing
- **Context Caching**: Available for certain models with separate pricing
- **API Rate Limits**: Token counting API calls count against your rate limits
- **Pricing Updates**: Prices reflect September 2025 rates and may change

## Error Handling

The provider gracefully handles:
- Missing API keys (falls back to estimation)
- API failures (uses local estimation)
- Tiktoken import errors (uses character-based estimation)
- Invalid model names (uses default pricing)
