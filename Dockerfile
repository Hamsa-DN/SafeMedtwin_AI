# syntax=docker/dockerfile:1
# SafeMedTwin — container image
#
# Key design choice: the ML models are trained DURING the image build, not
# copied in from the host. This isn't just convenience — it permanently
# solves a real problem this project hit repeatedly during development:
# XGBoost model files pickled on one machine and loaded on another can
# fail with "input stream corrupted" or "data did not contain feature
# names" if the XGBoost version differs even slightly, or if the binary
# file gets mangled in transfer (Git line-ending conversion, antivirus,
# etc). Training inside the container guarantees the model is always
# built and loaded by the exact same XGBoost version, in the exact same
# environment — that whole class of bug becomes structurally impossible.
#
# Build arg WITH_SLM controls whether the real TinyLlama + DistilBERT
# models get baked in too (adds ~3-4GB and several extra minutes to the
# build, since it downloads real model weights):
#   docker compose build                     -> template SLM only (fast, default)
#   docker compose build --build-arg WITH_SLM=true   -> + real TinyLlama + DistilBERT

FROM python:3.12-slim

ARG WITH_SLM=false
ARG DISTILBERT_EPOCHS=5
WORKDIR /app

# Critical for diagnosability: Python buffers stdout by default when it's
# not attached to a real terminal (which `docker build` isn't), so a long-
# running step like DistilBERT training can go completely silent for its
# entire duration even though it's working fine — this is very likely
# what made earlier builds LOOK frozen for 2 hours. Unbuffering forces
# output (including the HF Trainer's progress bar) to stream immediately.
ENV PYTHONUNBUFFERED=1

# System packages: pypdfium2/pdfplumber's dependencies mostly ship as
# prebuilt wheels, but a C compiler is occasionally needed for source
# builds of transitive deps — cheap to include, saves a confusing failure.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (separate layer) so `docker build` doesn't
# reinstall everything just because you changed a .py file.
#
# --retries/--timeout: makes pip resilient to a slow/flaky connection
# instead of giving up on the first hiccup. (No BuildKit cache mount here
# on purpose — it was tried, but a cache mount that survives a failed
# build can end up holding truncated/corrupted files from the interrupted
# download, which then fails hash verification on the next attempt. A
# plain, uncached install is slower to retry but can't get into that
# state — worth the tradeoff after hitting that exact problem twice.)
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt

# Optional heavier stack (transformers/torch/accelerate) — only installed
# when WITH_SLM=true, so the default build stays small and fast.

# COPY requirements-slm.txt .
# RUN if [ "$WITH_SLM" = "true" ]; then pip install --no-cache-dir --retries 10 --timeout 120 -r requirements-slm.txt; fi

COPY requirements-slm.txt .

RUN if [ "$WITH_SLM" = "true" ]; then \
    pip install --no-cache-dir --retries 10 --timeout 120 torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --retries 10 --timeout 120 -r requirements-slm.txt; \
    fi

    
# Now copy the actual project code.
COPY . .

# Generate data + train all 5 XGBoost models AT BUILD TIME, inside this
# exact environment — see the note above for why this matters.
RUN python data/generate_datasets.py && python models/train_models.py

# Optional: pre-download TinyLlama and fine-tune the DistilBERT classifier
# during the build too, so the first real request during your demo isn't
# waiting on a multi-GB download — everything's already baked into the
# image and ready the instant the container starts.
#
# Split into two separate RUN steps on purpose, each with its own clearly
# labeled echo — if the build looks stuck, the LAST line printed tells you
# exactly which phase it's in (large download vs. CPU training), instead
# of one opaque combined step where you can't tell what's happening.
RUN if [ "$WITH_SLM" = "true" ]; then \
      echo ">>> Downloading TinyLlama (~2.2GB) — this can take a while depending on your connection..." && \
      python -c "from transformers import pipeline; pipeline('text-generation', model='TinyLlama/TinyLlama-1.1B-Chat-v1.0')" && \
      echo ">>> TinyLlama download complete."; \
    fi

ENV SAFEMEDTWIN_DISTILBERT_EPOCHS=${DISTILBERT_EPOCHS}
RUN if [ "$WITH_SLM" = "true" ]; then \
      echo ">>> Training DistilBERT classifier for ${SAFEMEDTWIN_DISTILBERT_EPOCHS} epochs — you should see loss/progress lines below within ~1 minute..." && \
      python models/train_distilbert_classifier.py && \
      echo ">>> DistilBERT training complete."; \
    fi
# Always point the app at the "transformers" SLM backend — this is safe
# even in the lightweight (WITH_SLM=false) build, since llm_slm/compare.py
# already catches the ImportError when transformers isn't installed and
# falls back to the template responder automatically. So this one setting
# correctly gives you the real TinyLlama when WITH_SLM=true, and the
# template (unchanged) when it's false — no fragile build-arg-to-ENV
# conditional logic needed.
ENV SAFEMEDTWIN_SLM_BACKEND=transformers

EXPOSE 8501

# Basic container health check so `docker ps` / orchestrators can tell if
# the app is actually up, not just that the process exists.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
