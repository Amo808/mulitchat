# 🚀 БЫСТРЫЙ ДЕПЛОЙ - 3 МИНУТЫ

## Деплой на Render.com (БЕСПЛАТНО)

### 1. Переходите по ссылке:
👉 **[render.com](https://render.com)** 

### 2. Регистрация:
- **Sign up with GitHub** (используйте ваш GitHub аккаунт)

### 3. Создание сервиса:
- **New** → **Web Service**
- **Connect Repository** → выберите **`Amo808/mulitchat`**
- **Connect**

### 4. Настройки:
- **Name**: `ai-chat` (любое имя)
- **Region**: `Oregon (US West)`
- **Branch**: `main`
- **Build Command**: оставить пустым
- **Start Command**: оставить пустым

### 5. Environment Variables:
Добавьте ваши API ключи:

```
OPENAI_API_KEY
```
Значение: `sk-ваш-ключ-openai`

```
ANTHROPIC_API_KEY
```
Значение: `ваш-ключ-anthropic`

```
DEEPSEEK_API_KEY
```
Значение: `ваш-ключ-deepseek`

```
PORT
```
Значение: `80`

### 6. Деплой:
- **Create Web Service**
- Ждите 5-10 минут

### 7. Готово! 🎉
Ваш AI Chat будет доступен по адресу:
`https://ai-chat-xxxx.onrender.com`

---

## Что получите:
- ✅ Полнофункциональный AI чат в интернете
- ✅ Поддержка OpenAI, DeepSeek, Anthropic
- ✅ HTTPS из коробки
- ✅ Автоматические обновления при изменении кода
- ✅ Бесплатно (750 часов в месяц)

## Проверка работы:
- **Основной сайт**: ваша ссылка от Render
- **Health check**: добавьте `/health` к URL
- **API документация**: добавьте `/docs` к URL

## Если что-то не работает:
1. Проверьте логи в Render Dashboard
2. Убедитесь, что API ключи правильно указаны
3. Проверьте, что `PORT=80` установлен

**Готово! Теперь у вас есть свой AI чат в интернете! 🚀**
