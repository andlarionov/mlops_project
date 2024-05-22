# Используем официальный Python образ
FROM python:3.8

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы с исходным кодом и requirements.txt в контейнер
COPY scr/ ./scr
COPY tests/ ./tests
COPY requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем команду для запуска приложения
CMD ["streamlit", "run", "scr/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
