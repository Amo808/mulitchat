# 🚀 Быстрый деплой AI Chat

## Render.com - БЕСПЛАТНО и просто! (Рекомендуется)

### Автоматический деплой из GitHub:

1. **Зайдите на [Render.com](https://render.com)**
2. **New** → **Web Service** → **Connect Repository**
3. **Подключите GitHub и выберите `Amo808/mulitchat`**
4. **Настройки:**
   - **Name**: `ai-chat`
   - **Region**: `Oregon (US West)`
   - **Branch**: `main`
   - **Dockerfile**: будет найден автоматически
5. **Environment Variables:**
   ```
   OPENAI_API_KEY=sk-ваш_ключ_openai
   ANTHROPIC_API_KEY=ваш_ключ_anthropic
   DEEPSEEK_API_KEY=ваш_ключ_deepseek
   PORT=80
   ```
6. **Create Web Service** → Готово! 🎉

### Deploy кнопка (1 клик):
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Amo808/mulitchat)

## Результат деплоя

После успешного деплоя на Render:
- **Ваш сайт**: `https://ai-chat-xxxx.onrender.com`
- **API документация**: `https://ai-chat-xxxx.onrender.com/docs`
- **Health check**: `https://ai-chat-xxxx.onrender.com/health`

**Особенности Render бесплатного плана:**
- ✅ 750 часов в месяц (хватает на 24/7)
- ✅ HTTPS автоматически
- ✅ Автоматические деплои при push в GitHub
- ⚠️ Засыпает через 15 минут неактивности
- ⚠️ Первый запрос после сна ~30 секунд

## Альтернативы Docker Hub

### Railway
1. Подключите GitHub репозиторий
2. Railway автоматически найдет Dockerfile
3. Добавьте переменные окружения

### Vercel (только frontend)
1. Подключите репозиторий
2. Укажите папку `frontend`
3. Настройте build команды

### Heroku
1. Подключите репозиторий  
2. Добавьте heroku.yml файл
3. Настройте переменные окружения

Самый простой способ - использовать готовый образ `amochat/ai-chat:latest` на Render!
