# 🚀 Render.com Deployment Guide

## Почему Render?

✅ **БЕСПЛАТНО** - 750 часов в месяц (достаточно для постоянной работы)  
✅ **Автоматический деплой** - при push в GitHub  
✅ **HTTPS из коробки** - автоматические SSL сертификаты  
✅ **Docker поддержка** - найдет Dockerfile автоматически  
✅ **Логи в реальном времени** - удобная отладка  
✅ **Кастомные домены** - можно подключить свой домен  

## 🚀 Быстрый деплой (5 минут)

### Вариант 1: Deploy кнопка (1 клик)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Amo808/mulitchat)

### Вариант 2: Ручной деплой

1. **Зайдите на [Render.com](https://render.com)**
2. **Зарегистрируйтесь** (лучше через GitHub)
3. **New** → **Web Service**
4. **Connect Repository** → выберите `Amo808/mulitchat`
5. **Настройки:**
   - **Name**: `ai-chat-app`
   - **Environment**: `Docker`
   - **Dockerfile Path**: `./Dockerfile.render`
   - **Region**: `Oregon (US West)`
   - **Branch**: `main`
   - **Plan**: Free

## 🔑 Environment Variables (Обязательно!)

Добавьте переменные окружения в разделе Environment:

```env
# API ключи (получите на официальных сайтах)
OPENAI_API_KEY=sk-proj-ваш_ключ_openai
ANTHROPIC_API_KEY=ваш_ключ_anthropic
DEEPSEEK_API_KEY=ваш_ключ_deepseek

# Системные настройки
PORT=10000
PYTHONUNBUFFERED=1
```

### 📖 Как получить API ключи:

- **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic**: [console.anthropic.com](https://console.anthropic.com)  
- **DeepSeek**: [platform.deepseek.com](https://platform.deepseek.com)

## 🏗️ Архитектура деплоя

Проект использует единый Docker контейнер с:
- ✅ **Nginx** - веб-сервер и проксирование API
- ✅ **FastAPI Backend** - Python API сервер на порту 8000
- ✅ **React Frontend** - собранные статические файлы
- ✅ **Supervisord** - управление процессами

### Файлы для деплоя:
- `Dockerfile.render` - основной контейнер
- `render.yaml` - автоматическая конфигурация
- `supervisord.conf` - управление процессами
- `nginx.render.conf` - настройки веб-сервера

## ✅ После деплоя

### Ваш сайт будет доступен:
- **URL**: `https://your-app-name.onrender.com`
- **API**: `https://your-app-name.onrender.com/api/`
- **Health**: `https://your-app-name.onrender.com/health`
- **Docs**: `https://your-app-name.onrender.com/docs`

### Время деплоя:
- **Первый деплой**: ~8-12 минут (сборка и кеширование)
- **Последующие**: ~3-5 минут (используется кеш)

## 🔧 Обновления и редеплой

После изменений в коде:
```bash
git add .
git commit -m "Update app"
git push origin main
```

Render автоматически пересоберет и разместит обновленную версию.

## 📊 Мониторинг

В панели Render вы можете:
- ✅ Смотреть логи в реальном времени
- ✅ Отслеживать использование ресурсов
- ✅ Настроить алерты
- ✅ Смотреть метрики производительности

## ❓ Возможные проблемы

### Приложение "засыпает"
- **Проблема**: Free план засыпает через 15 минут
- **Решение**: Используйте [UptimeRobot](https://uptimerobot.com/) для пинга каждые 14 минут

### Медленный старт
- **Проблема**: Холодный старт ~30-60 секунд  
- **Решение**: Нормально для бесплатного плана

### API ключи не работают
- **Проблема**: Неверные или устаревшие ключи
- **Решение**: Проверьте ключи в Environment Variables

## 🆙 Upgrade планы

| План | Цена | RAM | CPU | Особенности |
|------|------|-----|-----|------------|
| **Free** | $0 | 512MB | 0.1 | Засыпает, медленный старт |
| **Starter** | $7/мес | 512MB | 0.5 | Не засыпает, быстрый старт |
| **Standard** | $25/мес | 2GB | 1.0 | Больше ресурсов, SLA |

## 🎯 Рекомендации

1. **Используйте Free план** для тестирования и демо
2. **Настройте UptimeRobot** чтобы избежать засыпания
3. **Мониторьте логи** для отладки проблем
4. **Обновляйте зависимости** регулярно
5. **Backup переменных окружения** в безопасном месте

---

💡 **Есть вопросы?** Проверьте логи в панели Render или создайте Issue в GitHub!
