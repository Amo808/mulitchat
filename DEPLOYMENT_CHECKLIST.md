# Docker Hub & Render Deployment Checklist

## Подготовка к публикации на Docker Hub

### 1. Подготовка аккаунта и репозиториев

- [ ] Создать аккаунт на [Docker Hub](https://hub.docker.com)
- [ ] Создать репозитории:
  - `your-username/ai-chat` - полное приложение
  - `your-username/ai-chat-backend` - только backend
  - `your-username/ai-chat-frontend` - только frontend
- [ ] Войти в Docker: `docker login`

### 2. Сборка и публикация образов

**Автоматическая сборка (рекомендуется):**
```bash
# Linux/macOS
chmod +x docker-build.sh
./docker-build.sh v1.0.0 your-dockerhub-username

# Windows  
docker-build.bat v1.0.0 your-dockerhub-username
```

**Ручная сборка:**
```bash
# Backend
docker build -f backend/Dockerfile.production -t your-username/ai-chat-backend:latest ./backend/
docker push your-username/ai-chat-backend:latest

# Frontend
docker build -f frontend/Dockerfile.production -t your-username/ai-chat-frontend:latest ./frontend/

# Complete app
docker build -t your-username/ai-chat:latest .
docker push your-username/ai-chat:latest
```

### 3. Проверка опубликованных образов

- [ ] Проверить образы в Docker Hub веб-интерфейсе
- [ ] Протестировать локальный запуск:
  ```bash
  docker run -p 80:80 -p 8000:8000 your-username/ai-chat:latest
  ```

## Развертывание на Render

### Вариант 1: Использование готовых Docker Hub образов

1. **Подготовка конфига для Render:**
   - [ ] Отредактировать `render-dockerhub.yaml`
   - [ ] Заменить `your-username` на ваш Docker Hub username
   - [ ] Выбрать нужный тип сервиса (отдельные или полный)

2. **Деплой на Render:**
   - [ ] Зайти на [render.com](https://render.com)
   - [ ] Создать новый Web Service
   - [ ] Выбрать "Deploy an existing image from a registry"
   - [ ] Указать образ: `your-username/ai-chat:latest`
   - [ ] Установить переменные окружения
   - [ ] Установить порт: `80`

### Вариант 2: Автоматический деплой из GitHub

1. **Подготовка репозитория:**
   - [ ] Создать GitHub репозиторий
   - [ ] Скопировать `render-dockerhub.yaml` в корень
   - [ ] Отредактировать пути к образам

2. **Настройка Render:**
   - [ ] Подключить GitHub репозиторий
   - [ ] Render автоматически найдет `render-dockerhub.yaml`
   - [ ] Настроить переменные окружения
   - [ ] Запустить деплой

### Обязательные переменные окружения для Render:

```env
# API Keys (обязательные)
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key  
DEEPSEEK_API_KEY=your-deepseek-api-key

# Опциональные API Keys
GOOGLE_API_KEY=your-google-api-key
COHERE_API_KEY=your-cohere-api-key
TOGETHER_API_KEY=your-together-api-key

# Системные
PORT=80
PYTHONUNBUFFERED=1
```

## Проверка развертывания

### После публикации на Docker Hub:
- [ ] Образы доступны публично
- [ ] Размеры образов оптимальны (backend ~200MB, frontend ~25MB)
- [ ] Health checks работают
- [ ] Локальный тест прошел успешно

### После деплоя на Render:
- [ ] Сервис запущен и работает
- [ ] Health checks проходят (зеленый статус)
- [ ] API endpoints доступны
- [ ] Frontend загружается корректно
- [ ] Подключение к AI провайдерам работает

## Полезные команды

### Docker Hub:
```bash
# Просмотр локальных образов
docker images | grep ai-chat

# Удаление локальных образов
docker rmi your-username/ai-chat:latest

# Просмотр информации об образе
docker inspect your-username/ai-chat:latest
```

### Render:
```bash
# Проверка здоровья сервиса
curl https://your-app.onrender.com/health

# Проверка API
curl https://your-app.onrender.com/api/providers

# Проверка логов (в веб-интерфейсе Render)
```

## Решение проблем

### Проблемы Docker Hub:
- **Медленная загрузка**: Оптимизируйте слои Dockerfile
- **Большой размер**: Используйте multi-stage builds
- **Ошибки аутентификации**: `docker logout && docker login`

### Проблемы Render:
- **Долгий старт**: Увеличьте health check start period
- **Ошибки порта**: Убедитесь, что приложение слушает порт из `$PORT`
- **Timeout**: Проверьте health check endpoints
- **API ошибки**: Проверьте переменные окружения

## Готовые ссылки и команды для быстрого деплоя

### Для пользователей (после публикации):

**Быстрый запуск с Docker Hub:**
```bash
# Скачать и запустить
curl -o deploy-hub.sh https://raw.githubusercontent.com/YOUR_REPO/ai-chat/main/deploy-hub.sh
chmod +x deploy-hub.sh
./deploy-hub.sh your-dockerhub-username
```

**Render деплой одной командой:**
- Форкнуть репозиторий
- Подключить к Render
- Настроить переменные окружения
- Готово!

## Итоговый статус

После выполнения всех пунктов:
- ✅ **Docker Hub**: Готовые образы доступны всем
- ✅ **Render**: Приложение работает в продакшене
- ✅ **Автоматизация**: Простое развертывание для пользователей
- ✅ **Масштабируемость**: Легкое обновление и масштабирование
- ✅ **Надежность**: Health checks и мониторинг

Ваше приложение готово к использованию в продакшене! 🚀
