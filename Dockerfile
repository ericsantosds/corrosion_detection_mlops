# ==============================================================================
# 🐳 MLOPS PRODUCTION DOCKERFILE - CORROSION DETECTION
# ==============================================================================
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    AIRFLOW_HOME=/app/airflow_home

WORKDIR /app

# 1. Instala dependências de sistema e drivers gráficos para X11/Qt
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    curl \
    git \
    build-essential \
    libxcb-xinerama0 \
    libqt5gui5 \
    libgles2-mesa-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Configura o Python 3.11 como padrão
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# 2. Instala o gerenciador 'uv' de forma global
RUN pip install --no-cache-dir uv

# 3. Copia os arquivos de mapeamento de dependências
COPY pyproject.toml uv.lock ./

# 4. CORREÇÃO: Instala as dependências usando o ecossistema sincronizado do uv
RUN uv pip install --system --no-cache -e . || uv pip install --system --no-cache -r pyproject.toml --frozen

# 5. Copia o restante do código fonte
COPY . .

# Cria os diretórios necessários (Isolando o AIRFLOW_HOME para não ser esmagado por volumes)
RUN mkdir -p /app/airflow_home/dags \
    && mkdir -p /app/online_data \
    && mkdir -p /app/review_data \
    && cp /app/dags/*.py /app/airflow_home/dags/

EXPOSE 8000 8080 5000

CMD ["python", "src/serving/app.py"]
