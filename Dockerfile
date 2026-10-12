FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY src/ src/
COPY sql/ sql/
COPY scripts/ scripts/

# Sample policy and case documents for the knowledge base
RUN python src/make_docs.py

EXPOSE 8501
CMD ["sh", "scripts/start.sh"]
