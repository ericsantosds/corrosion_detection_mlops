# ==============================================================================
# 🐳 MLOPS PRODUCTION DOCKERFILE - CORROSION DETECTION
# ==============================================================================
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CORROSION_PROJECT_ROOT=/app \
    AIRFLOW_HOME=/opt/airflow \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# 1. Instala dependências de sistema e drivers gráficos para X11/Qt
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    curl \
    git \
    build-essential \
    libxcb-xinerama0 \
    libqt5gui5 \
    libgles2-mesa-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Instala o gerenciador 'uv' de forma global
RUN pip3 install --no-cache-dir uv

# 3. Copia os arquivos de mapeamento de dependências
COPY pyproject.toml uv.lock ./

# 4. Instala Python 3.11 e as dependências exatamente como travadas no uv.lock
RUN uv python install 3.11 \
    && uv sync --frozen --no-dev --no-install-project --python 3.11

# 5. Copia o restante do código fonte
COPY . .

# Mantém o AIRFLOW_HOME fora do volume de código montado pelo Compose.
RUN mkdir -p /opt/airflow/dags \
    && mkdir -p /app/online_data \
    && mkdir -p /app/review_data \
    && mkdir -p /app/runs

EXPOSE 8000 8080 5000

CMD ["python", "src/serving/app.py"]
