# AI Chat

Многопровайдерный AI чат с поддержкой OpenAI, DeepSeek, Anthropic и других провайдеров.

## Структура проекта

```
ai-chat/
├── backend/          # Python FastAPI backend
├── frontend/         # React + TypeScript frontend  
├── adapters/         # AI provider adapters
├── storage/          # История чатов и сессии
├── data/            # Конфигурационные файлы и база данных
├── logs/            # Логи приложения
└── RUN_INSTRUCTIONS.md  # Подробные инструкции по запуску
```

## Быстрый старт

### Запуск Backend
```powershell
cd "c:\Users\Amo\Desktop\lobecopy\ai-chat\backend"; python main.py
```

### Запуск Frontend  
```powershell
cd "c:\Users\Amo\Desktop\lobecopy\ai-chat\frontend"; npm run dev
```

## Функциональность

✅ **Мультипровайдерная поддержка**: OpenAI, DeepSeek, Anthropic, **ChatGPT Pro**  
✅ **GPT-5 Pro интеграция**: Полная поддержка GPT-5 с extended reasoning  
✅ **Advanced reasoning модели**: o1-Pro, o3 Deep Research, o1, o1-mini  
✅ **Умная обработка больших текстов**: Автоматическая оптимизация для текстов >50k символов  
✅ **История разговоров**: Сохранение и управление чатами с поддержкой больших контекстов  
✅ **Настройка провайдеров**: API ключи, статус подключения, Pro features  
✅ **Проверка соединений**: Тест подключений и обновление моделей  
✅ **Потоковая передача**: Реальная обработка ответов с fallback механизмами  
✅ **Современный UI**: React с поддержкой темной темы и Pro Mode индикаторами  

## Подробная документация

- **Локальный запуск**: [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md) - инструкции для разработки
- **Деплой на сервер**: [DEPLOYMENT.md](DEPLOYMENT.md) - полный гид по развертыванию на VPS  
- **Render деплой**: [RENDER.md](RENDER.md) - автоматическое развертывание через Render (БЕСПЛАТНО)
- **Docker развертывание**: [DOCKER.md](DOCKER.md) - контейнеризация и Docker Compose
- **Быстрый деплой**: [QUICK_DEPLOY.md](QUICK_DEPLOY.md) - мгновенное развертывание на разных платформах

## Развертывание на сервере

### 🚀 Render.com (Рекомендуемый способ - БЕСПЛАТНО):

**⚡ БЫСТРЫЙ ДЕПЛОЙ:** [QUICK_START.md](QUICK_START.md) - 3 минуты до готового чата!

**GitHub → Render (автоматический деплой):**
1. Зайдите на [Render.com](https://render.com)
2. "New" → "Web Service" → "Connect Repository" 
3. Подключите `https://github.com/Amo808/mulitchat`
4. Render найдет Dockerfile и автоматически соберет
5. Добавьте API ключи в Environment Variables
6. Deploy! 🚀

**Прямая ссылка для деплоя:**
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Amo808/mulitchat)

### 🐳 Docker сборка (Рекомендуемый способ):
```bash
# Linux/macOS
wget https://raw.githubusercontent.com/YOUR_REPO/ai-chat/main/deploy-docker.sh
chmod +x deploy-docker.sh
./deploy-docker.sh

# Windows
# Скачайте deploy-docker.bat и запустите
```

### 📦 Традиционная установка на Ubuntu:
```bash
wget https://raw.githubusercontent.com/YOUR_REPO/ai-chat/main/install.sh
chmod +x install.sh
./install.sh
```

### Рекомендуемые VPS провайдеры:
- **Selectel** (Россия) - от 500₽/мес  
- **TimeWeb** (Россия) - от 300₽/мес
- **DigitalOcean** (США) - от $5/мес
- **Hetzner** (Германия) - от €3/мес

**Минимальные требования:** 2 CPU, 4GB RAM, 40GB SSD

## Последнее обновление

**20.08.2025** - Исправлена валидация OpenAI провайдера, добавлены эндпоинты тестирования, улучшен UI.

## 🚀 GPT-5 Pro Features

### **NEW!** Эксклюзивные возможности ChatGPT Pro:

🧠 **GPT-5 Pro Mode** - Передовая модель с расширенными возможностями reasoning  
🔍 **o1 Pro Mode** - Extended compute для максимальной надежности ответов  
📚 **o3 Deep Research** - Оптимизирована для веб-исследований и сложных задач  
⚡ **Unlimited Access** - Неограниченное использование o1, o1-mini моделей  
📝 **Smart Large Text Processing** - Автоматическая оптимизация для документов >50k символов  
🔄 **Adaptive Streaming** - Умное переключение между streaming и batch режимами  
⏱️ **Dynamic Timeouts** - Таймауты от 2 до 10 минут в зависимости от размера задачи  

### Настройка GPT-5 Pro:

1. **Получите ChatGPT Pro подписку** на [chat.openai.com](https://chat.openai.com)
2. **Добавьте CHATGPT_PRO_API_KEY** в настройки приложения
3. **Выберите GPT-5 Pro** в селекторе моделей
4. **Наслаждайтесь** передовыми AI возможностями! 🎉
