FROM python:3.14-slim

# 默认官方 PyPI；网络受限时构建期覆盖：
#   docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 依赖单独成层，改源码时无需重装
COPY requirements.txt ./
RUN pip install --index-url "${PIP_INDEX_URL}" -r requirements.txt

# data/ 是只读知识库，结构化工具运行期会读取原始 CSV/XLSX
COPY src ./src
COPY data ./data

# build/ 不入镜像，运行时用 volume 挂载；这里只创建挂载点
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/build \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 索引加载完成后 /api/health 才可用；启动期留足预热时间
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
