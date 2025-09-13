# 🚨 URGENT: Force Render Redeploy

## Проблема
Несмотря на правильные исправления в коде, o3-deep-research все еще возвращает ошибку:
```
"Invalid type for 'prompt': expected an object, but got a string instead."
```

## Причина
Render может использовать кэшированную версию приложения и не подхватил последние изменения.

## ✅ Код исправлен корректно
```python
# ✅ ПРАВИЛЬНЫЙ PAYLOAD В КОДЕ
responses_payload = {
    "model": model,
    "messages": api_messages,  # ✅ Массив сообщений
    "stream": params.stream,
    "max_output_tokens": params.max_tokens  # ✅ Правильный параметр
}
```

## 🔧 Действия для принудительного деплоя

### Метод 1: Пустой коммит
```bash
git commit --allow-empty -m "Force redeploy for o3-deep-research fix"
git push
```

### Метод 2: Manual Deploy в Render Dashboard
1. Зайти в Render Dashboard
2. Найти сервис multiprovider  
3. Нажать "Manual Deploy"
4. Выбрать latest commit
5. Дождаться завершения деплоя

### Метод 3: Проверить Environment Variables
Убедиться что в Render установлены правильные переменные:
- `OPENAI_API_KEY`
- `NODE_ENV=production`

## 🧪 Как проверить что деплой прошел
1. Проверить логи в Render Dashboard
2. Найти строку: `🔍 [/responses] Sending X messages`
3. Убедиться что нет ошибок о `prompt` поле

## ⏰ Timeline
- **10:04** - Последняя ошибка с `prompt` 
- **Сейчас** - Код исправлен, но Render может использовать старую версию
- **Нужно** - Принудительный redeploy

## 🎯 Ожидаемый результат
После успешного деплоя o3-deep-research должен работать без ошибок 400.
