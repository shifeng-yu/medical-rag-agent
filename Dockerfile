# ============================================================
# 多阶段构建：node 构建前端 → python 运行后端
#
# 前端产物（frontend/dist）打进镜像，由 FastAPI 直接托管；
# 不依赖 CDN，内网断网可用（与"数据不出域"同一约束）。
# ============================================================

# ---------- Stage 1: 前端构建 ----------
FROM node:20-alpine AS frontend

WORKDIR /web

# 先只拷依赖清单：改业务代码不会击穿 node_modules 这层缓存
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY frontend/ ./
RUN npm run build


# ---------- Stage 2: 运行时 ----------
FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY config/ ./config/
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY ragflow_plugins/ ./ragflow_plugins/
COPY scripts/ ./scripts/
COPY data/common_diseases_drugs.txt ./data/
COPY .env.example ./

# 前端构建产物（上面 node 阶段产出）
COPY --from=frontend /web/dist ./frontend/dist

# Create data/log directories
RUN mkdir -p /app/data /app/logs /app/volumes

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
