# 🚀 Деплой на Render

## Автоматический деплой

1. **Подключите GitHub репозиторий к Render:**
   - Зайдите на https://render.com
   - Нажмите "New +" → "Web Service" 
   - Выберите "Build and deploy from a Git repository"
   - Подключите GitHub: https://github.com/Amo808/mulitchat

2. **Настройте сервис:**
   - **Name**: `multichat-app` (или любое другое)
   - **Runtime**: Docker
   - **Branch**: main  
   - **Root Directory**: оставьте пустым
   - **Build Command**: оставьте пустым (используется Dockerfile)
   - **Start Command**: оставьте пустым (используется Dockerfile)

3. **Environment Variables** (обязательно настроить):
   ```
   OPENAI_API_KEY=sk-proj-ваш_ключ_openai
   CHATGPT_PRO_API_KEY=sk-proj-ваш_ключ_chatgpt_pro  
   DEEPSEEK_API_KEY=ваш_ключ_deepseek
   ANTHROPIC_API_KEY=ваш_ключ_anthropic
   PORT=10000
   PYTHONUNBUFFERED=1
   DEBUG=False
   HOST=0.0.0.0
   ```

4. **Advanced Settings:**
   - **Health Check Path**: `/health`
   - **Auto-Deploy**: Yes (включено)

## После деплоя

- Ваше приложение будет доступно по адресу: `https://multichat-app.onrender.com` (или другому имени)
- Автоматический редеплой при push в main ветку
- Логи доступны в Dashboard Render

## Важные моменты для GPT-5 Pro

⚠️ **Обязательно настройте CHATGPT_PRO_API_KEY** - без него GPT-5 Pro не будет работать!

✅ **После деплоя проверьте:**
1. Откройте приложение в браузере
2. Зайдите в Provider Settings  
3. Убедитесь, что ChatGPT Pro показывает зеленый статус
4. Протестируйте GPT-5 Pro с коротким и длинным текстом

## Мониторинг

- Логи: Dashboard → Logs
- Статус: Dashboard → Events
- Метрики: Dashboard → Metrics

## Troubleshooting

Если приложение не запускается:
1. Проверьте Environment Variables в Render Dashboard
2. Посмотрите логи деплоя и runtime логи  
3. Убедитесь, что все API ключи правильные
4. Проверьте /health endpoint

## Локальное тестирование Docker

```bash
# Сборка
docker build -t multichat .

# Запуск с переменными окружения
docker run -p 10000:10000 \
  -e OPENAI_API_KEY=your_key \
  -e CHATGPT_PRO_API_KEY=your_key \
  -e DEEPSEEK_API_KEY=your_key \
  multichat
```

## Обновление деплоя

Просто сделайте git push - Render автоматически пересоберет и задеплоит приложение:

```bash
git add .
git commit -m "Update: новые изменения"
git push origin main
```
