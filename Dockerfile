# Используем официальный образ Python
FROM python:3.11-slim

# Устанавливаем зависимости для работы с PostgreSQL (если используете)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && apt-get clean

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код приложения
COPY . .

# Открываем порт (указанный по умолчанию, можно изменить через переменную окружения)
EXPOSE 8000

# Определяем переменные окружения для настройки порта
ENV PORT 8000

# Запускаем Uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]