#!/usr/bin/env python3
"""
Test script for Gemini token counting and cost calculation
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.gemini_provider import GeminiAdapter
from adapters.base_provider import ProviderConfig, ModelProvider, Message, GenerationParams

async def test_gemini_token_counting():
    """Test Gemini token counting and cost calculation"""
    
    # Create a mock config (you'll need to set your actual API key)
    config = ProviderConfig(
        id=ModelProvider.GEMINI,
        name="Google Gemini",
        enabled=True,
        api_key=os.getenv("GEMINI_API_KEY", "your-api-key-here"),  # Replace with your API key
        base_url="https://generativelanguage.googleapis.com"
    )
    
    # Initialize Gemini adapter
    gemini = GeminiAdapter(config)
    
    print("🧪 Testing Gemini Token Counting & Cost Calculation")
    print("=" * 60)
    
    # Test different text lengths
    test_texts = [
        "Hello, world!",
        "This is a longer test message to see how token counting works with more text content.",
        """This is a very long message that contains multiple sentences and paragraphs.
        
        It includes various types of content like:
        - Lists and bullet points
        - Code snippets: print('hello world')
        - Numbers: 12345, 67890
        - Special characters: !@#$%^&*()
        
        The purpose is to test how accurately we can count tokens for different types of content
        and calculate the associated costs for various Gemini models."""
    ]
    
    models_to_test = [
        "gemini-2.5-pro",
        "gemini-2.5-flash", 
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test Text {i} ({len(text)} characters):")
        print(f"'{text[:50]}{'...' if len(text) > 50 else ''}'")
        print()
        
        # Test local token estimation
        local_tokens = gemini.estimate_tokens(text)
        print(f"🔢 Local Token Estimate: {local_tokens} tokens")
        
        # Test each model's pricing
        for model in models_to_test:
            # Calculate cost for input tokens
            input_cost = gemini._calculate_cost(local_tokens, 0, model, local_tokens)
            
            # Calculate cost for output tokens (assume same amount as input)
            total_cost = gemini._calculate_cost(local_tokens, local_tokens, model, local_tokens)
            
            # Get pricing info
            pricing_info = await gemini.get_model_pricing_info(model)
            
            print(f"  💰 {model}: Input Cost: ${input_cost:.6f}, Total Cost: ${total_cost:.6f}")
        
        print("-" * 40)
    
    print("\n🔍 Model Pricing Information:")
    print("=" * 60)
    
    for model in models_to_test:
        pricing_info = await gemini.get_model_pricing_info(model)
        print(f"\n📊 {model}:")
        print(f"  Context Length: {pricing_info.get('context_length', 'N/A'):,} tokens")
        print(f"  Max Output: {pricing_info.get('max_output_tokens', 'N/A'):,} tokens")
        
        pricing = pricing_info.get('pricing', {})
        if pricing:
            print("  Pricing (per 1M tokens):")
            for key, value in pricing.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: ${value:.2f}")
    
    # Test API token counting (if API key is valid)
    if config.api_key and config.api_key != "your-api-key-here":
        print(f"\n🌐 Testing API Token Counting:")
        print("=" * 60)
        
        test_text = "Hello, how are you doing today? This is a test message."
        model = "gemini-2.5-flash"
        
        try:
            api_tokens = await gemini.count_tokens_api(test_text, model)
            local_tokens = gemini.estimate_tokens(test_text)
            
            if api_tokens:
                print(f"Text: '{test_text}'")
                print(f"API Token Count: {api_tokens}")
                print(f"Local Estimate: {local_tokens}")
                print(f"Difference: {abs(api_tokens - local_tokens)} tokens")
                print(f"Accuracy: {(min(api_tokens, local_tokens) / max(api_tokens, local_tokens) * 100):.1f}%")
            else:
                print("❌ API token counting failed")
        except Exception as e:
            print(f"❌ Error testing API token counting: {e}")
    else:
        print("\n⚠️  Skipping API token counting test (no valid API key)")
    
    await gemini.close()
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    asyncio.run(test_gemini_token_counting())
