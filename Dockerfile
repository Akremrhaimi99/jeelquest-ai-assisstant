FROM python:3.11-slim

WORKDIR /app

# Dépendances système nécessaires pour PDF
RUN apt-get update && apt-get install -y \
    gcc \
    poppler-utils \
    libgl1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# ⚠️ Installer PyTorch CPU AVANT tout
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Copier requirements
COPY requirements.txt .

# Installer dépendances sans réinstaller torch
RUN pip install --no-cache-dir --no-deps -r requirements.txt \
    && pip install --no-cache-dir unstructured[pdf]

# Copier le code
COPY . .

# Utiliser le port Railway
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
