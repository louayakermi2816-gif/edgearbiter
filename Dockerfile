FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY edgearbiter ./edgearbiter
COPY results ./results
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install --no-cache-dir . python-multipart
EXPOSE 8000
CMD ["uvicorn", "edgearbiter.api:app", "--host", "0.0.0.0", "--port", "8000"]
