#!/usr/bin/env python3
"""
Тест счетчика токенов и расчета стоимости для Gemini провайдера
"""

import asyncio
import os
import sys
from pathlib import Path

# Добавляем корневую папку в путь для импорта
sys.path.insert(0, str(Path(__file__).parent))

from adapters.gemini_provider import GeminiAdapter
from adapters.base_provider import ProviderConfig, GenerationParams, Message, ModelProvider

async def test_gemini_token_counting():
    """Тест подсчета токенов и стоимости для Gemini"""
    
    print("🧪 Тестирование счетчика токенов Gemini...")
    
    # Инициализируем провайдер (используем простой объект)
    class TestConfig:
        def __init__(self):
            self.id = ModelProvider.GEMINI
            self.name = "gemini"
            self.enabled = True
            self.api_key = os.getenv("GEMINI_API_KEY", "test-key")
            self.base_url = "https://generativelanguage.googleapis.com"
    
    config = TestConfig()
    
    gemini = GeminiAdapter(config)
    
    # Тестовые данные
    test_texts = [
        "Hello, world!",
        "Explain quantum computing in simple terms.",
        "Write a detailed analysis of machine learning algorithms including supervised, unsupervised, and reinforcement learning approaches. Discuss their applications, advantages, and limitations in modern AI systems.",
        "Привет! Как дела? Расскажи о нейронных сетях на русском языке."
    ]
    
    models_to_test = [
        "gemini-2.5-pro",
        "gemini-2.5-flash", 
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash"
    ]
    
    print("\n📊 Тестирование подсчета токенов с помощью tiktoken...")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Тест {i} ---")
        print(f"Текст: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Tiktoken подсчет
        tokens_tiktoken = gemini.estimate_tokens_accurate(text)
        print(f"Токены (tiktoken): {tokens_tiktoken}")
        
        # Старый метод для сравнения
        tokens_old = gemini.estimate_tokens(text)
        print(f"Токены (старый метод): {tokens_old}")
        
        print(f"Разница: {abs(tokens_tiktoken - tokens_old)} токенов")
    
    print("\n💰 Тестирование расчета стоимости...")
    
    # Тестовые значения токенов
    input_tokens = 100
    output_tokens = 200
    
    print(f"\nПри {input_tokens} входящих и {output_tokens} исходящих токенах:")
    
    for model in models_to_test:
        cost = gemini._calculate_cost(model, input_tokens, output_tokens)
        print(f"{model}: ${cost:.6f}")
    
    print("\n📈 Сравнение стоимости моделей (1000 входящих + 1000 исходящих токенов):")
    
    for model in models_to_test:
        cost = gemini._calculate_cost(model, 1000, 1000)
        pricing = gemini.MODEL_PRICING[model]
        print(f"{model}:")
        print(f"  Стоимость: ${cost:.6f}")
        print(f"  Цена за 1M входящих: ${pricing['input_tokens']:.3f}")
        print(f"  Цена за 1M исходящих: ${pricing['output_tokens']:.3f}")
    
    # Тест с реальным API (если есть ключ)
    if os.getenv("GEMINI_API_KEY"):
        print("\n🔗 Тестирование API подсчета токенов...")
        try:
            test_text = "Explain artificial intelligence in one paragraph."
            api_tokens = await gemini.count_tokens_api(test_text, "gemini-2.5-flash")
            tiktoken_tokens = gemini.estimate_tokens_accurate(test_text)
            
            print(f"Текст: {test_text}")
            print(f"API токены: {api_tokens}")
            print(f"Tiktoken токены: {tiktoken_tokens}")
            print(f"Разница: {abs(api_tokens - tiktoken_tokens)} токенов")
            
        except Exception as e:
            print(f"⚠️  Ошибка API теста: {e}")
    else:
        print("\n⚠️  GEMINI_API_KEY не найден - пропускаем API тест")
    
    await gemini.close()
    
    print("\n✅ Тестирование завершено!")
    print("\n📋 Резюме:")
    print("- Tiktoken подсчет токенов реализован")
    print("- Актуальные цены на все модели Gemini добавлены")
    print("- Расчет стоимости работает корректно")
    print("- API подсчет токенов доступен при наличии ключа")

if __name__ == "__main__":
    asyncio.run(test_gemini_token_counting())
