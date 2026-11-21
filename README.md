# SafeMedTwin

**Hallucination-Aware Medical AI with Small Language Models**
B.M.S. College of Engineering — Dept. of Machine Learning, Project Work-1

A working implementation of the system described in the accompanying
project report, plus several extensions: multi-disease ML risk prediction,
real-time SHAP explainability, a 3D anatomical Digital Twin, a rule-based
MedGuard safety layer, an LLM-vs-SLM evaluation framework, per-user login
with assessment history tracking, a context-aware chatbot, an
image-and-PDF medical report scanner, and personalized diet/lifestyle
recommendations.

**There are two ways to run this project** — pick one:

| | Two-service (original) | Unified web app (recommended) |
|---|---|---|
| Entry point | `backend/main.py` + `frontend/app.py` | `streamlit_app.py` |
| Processes | FastAPI + Streamlit, 2 terminals | Streamlit only, 1 terminal |
| Login / per-user data | No | Yes |
| Chatbot, report scanner, diet plans | No | Yes |
| Deployable to Streamlit Community Cloud | Not easily (needs 2 hosts) | Yes, directly |
| Has a standalone REST API other tools can call | Yes | No |

If you just want the app running and shareable as a URL, use
**`streamlit_app.py`**. Keep `backend/main.py` around if you specifically
need a REST API (e.g. for a separate mobile client).

## Architecture

```
data/           synthetic (or real) datasets for 5 diseases
models/         XGBoost training pipeline + SHAP explainability
digital_twin/   what-if scenario simulator + per-patient history
safety/         MedGuard validation + AlertEngine (threshold-based alerts)
llm_slm/        LLM (Anthropic API) vs SLM comparison, chatbot, consistency testing
diet/           rule-based + LLM-personalized diet & lifestyle recommendations
vision/         medical report scanner -- images and PDFs (Claude vision API + pdfplumber)
auth/           SQLite-backed login + per-user data isolation
backend/        FastAPI REST API (optional, two-service mode)
frontend/       Streamlit dashboard (two-service mode) + 3D body twin module
streamlit_app.py   unified single-process app (recommended entry point)
```

## Recent fixes

- **What-If sweep crash** (`TypeError: Invalid value '...' for dtype 'int64'`):
  the simulator built its feature row with pandas-inferred integer dtypes
  for whole-number fields (e.g. age), then choked when a sweep tried to
  assign a float into that column. Fixed by forcing float64 throughout
  `digital_twin/simulator.py` — this was also cascading and blocking every
  tab rendered after the crash point in that Streamlit run, which is why
  Diet/Chatbot/Report Scanner appeared broken too even though the bug was
  only in the simulator.
- **LLM vs SLM comparison was using the wrong "top factor"**: it silently
  defaulted to an arbitrary feature instead of the patient's actual top
  SHAP contributor. Fixed to compute it properly, and moved the comparison
  section out from behind the sweep so it's always visible once a
  prediction exists.
- **Consistency testing was built but never exposed in the UI** — now has
  its own button next to the LLM/SLM comparison.
- **Added PDF upload** to the Report Scanner (`vision/pdf_reader.py`) —
  text extraction first (fast), falling back to page-image rendering for
  scanned PDFs. Previously image-only.
- **Added a History tab** — per-disease risk-over-time chart plus a
  summary table across all 5 diseases, backed by the predictions already
  being saved to SQLite (the storage existed; it just wasn't surfaced in
  the UI before).

### If you see model-loading errors

Two related error classes can show up when the `.joblib` model files were
trained in a different environment than the one running the app:

- `ValueError: data did not contain feature names, but the following
  fields are expected: ...` — an XGBoost version mismatch between
  training and inference. `requirements.txt` now pins the exact ML-stack
  versions the shipped models were trained with, and `load_bundle()` /
  `get_bundle()` explicitly re-attach feature names after loading as a
  second layer of defense.
- `XGBoostError: input stream corrupted` — the `.joblib` file itself got
  mangled in transfer (corrupted download, antivirus interference, or —
  very commonly on Windows — Git rewriting line endings inside a binary
  file if you `git add` it without a `.gitattributes` marking it binary;
  this repo now ships one).

**The reliable fix for either one is to stop depending on the binary
files matching your machine at all — just retrain locally:**
```bash
rm models/*.joblib          # Windows: Remove-Item models\*.joblib
python data/generate_datasets.py
python models/train_models.py
```
This takes a few seconds and only depends on whatever's installed in your
own environment — no cross-machine pickle compatibility involved.

## Setup

```bash
pip install -r requirements.txt

# optional but recommended -- enables the LLM arm of every comparison,
# the chatbot, the report scanner, and LLM-personalized diet notes.
# Without it, everything still works via documented offline fallbacks.
# Set EITHER of these (both work -- see "Choosing a provider" below):
export ANTHROPIC_API_KEY=sk-ant-...
# or
export GEMINI_API_KEY=AIzaSy...
```

### Choosing a provider: Claude vs Gemini

This project supports both, through one shared abstraction
(`llm_slm/llm_provider.py`) -- every module that calls an LLM (chatbot,
diet notes, report scanner, LLM-vs-SLM comparison) goes through it, so
there's nothing else to change.

- **Auto-detected by default**: if `ANTHROPIC_API_KEY` is set, Claude is
  used; else if `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is set, Gemini is
  used.
- **Force a specific provider** (useful if you have both keys set and
  want to control which one runs, e.g. for a Claude-vs-Gemini comparison
  in your report):
  ```bash
  export SAFEMEDTWIN_LLM_PROVIDER=gemini    # or: anthropic
  ```

**Getting a Gemini key (free, no card needed for Flash-tier models):** go
to **aistudio.google.com** → sign in → **Get API key** → **Create API
key**. Gemini's free tier makes it the easier starting point if you don't
want to deal with billing setup.

**Getting a Claude key:** go to **console.anthropic.com** → **Settings →
API Keys** → **Create Key**. This is pay-as-you-go (no permanent free
tier), typically pennies for a project this size, but does need billing
set up first.

## Run (unified app -- recommended)

```bash
python data/generate_datasets.py   # run once
python models/train_models.py      # run once
streamlit run streamlit_app.py
```

First visit: create an account (any username/password, min 6 characters)
on the "Create account" tab, then log in. All your predictions, chat
history, and 3D twin state are scoped to your account.

## Run (two-service / API mode)

```bash
python data/generate_datasets.py && python models/train_models.py
uvicorn backend.main:app --reload --port 8000     # terminal 1
streamlit run frontend/app.py                       # terminal 2
```

*(Windows PowerShell doesn't support `&&` in older versions -- run each
line separately, or use `;`. If `uvicorn`/`streamlit` aren't recognized as
commands, run them as `python -m uvicorn ...` / `python -m streamlit ...`
instead -- this uses the same Python that has the packages installed.)*

---

## Deploying it as a real web app (Streamlit Community Cloud, free)

This is the easiest path to a public URL you can put in your report/demo.

1. **Push this project to a GitHub repo** (public or private).
   ```bash
   git init
   git add .
   git commit -m "SafeMedTwin"
   git remote add origin https://github.com/<you>/safemedtwin.git
   git push -u origin main
   ```
2. Go to **share.streamlit.io**, sign in with GitHub, click **"New app"**.
3. Pick your repo, branch `main`, and set **Main file path** to
   `streamlit_app.py`.
4. Under **Advanced settings -> Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Click **Deploy**. Make sure `models/*.joblib` and `data/raw/*.csv` are
   committed to the repo (not gitignored), so the deployed app doesn't
   need to retrain on first load.
6. You'll get a URL like `https://safemedtwin.streamlit.app` -- share that.

**Important:** the SQLite user database (`safemedtwin.db`) lives on the
Community Cloud container's disk, which is **not persistent** -- it resets
on redeploy/restart. Fine for a class demo; for anything longer-lived,
swap `auth/auth.py`'s SQLite connection for a hosted database (e.g.
Supabase/Postgres) -- the query logic wouldn't need to change much.

---

## Sharing this with your team (GitHub + Docker)

This is the setup for a team working on the same project, where everyone
should get an identical, working environment with one command rather than
each person debugging their own Python/package versions.

### 1. Push to GitHub (do this once)

```bash
cd safemedtwin
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/safemedtwin.git
git push -u origin main
```

`.gitignore` already excludes `.env` (secrets), `safemedtwin.db` (per-user
data), and `models/*.joblib` (binary files that don't survive Git's line-
ending handling reliably across teammates' OSes -- see "Recent fixes"
above for the exact bug this caused). `.gitattributes` also marks any
binary files that do get committed as binary, so Windows Git doesn't
silently corrupt them.

Add your teammates as collaborators: repo **Settings -> Collaborators ->
Add people**.

### 2. Each teammate: clone and run with Docker (one command)

This is the point of containerizing it -- nobody needs to install Python,
XGBoost, or debug version mismatches. Docker builds an identical
environment every time, and **trains the models inside the container
itself** -- which also permanently eliminates the cross-machine model-
corruption problem this project hit repeatedly during development, since
the model is now always trained and loaded by the exact same XGBoost
version, in the exact same environment.

**Install Docker Desktop first** (docker.com/products/docker-desktop) if
you don't have it -- one-time setup, works on Windows/Mac/Linux.

```bash
git clone https://github.com/<your-username>/safemedtwin.git
cd safemedtwin
cp .env.example .env
```
Open `.env` and paste in a real `GEMINI_API_KEY` (or `ANTHROPIC_API_KEY`)
-- see the earlier section on getting a Gemini key if you don't have one.
Then:
```bash
docker compose up --build
```
First run takes a few minutes (installing packages + training 5 models
inside the image). After that, open **http://localhost:8501** -- same app,
same login screen, same everything, on anyone's machine.

Subsequent runs (no code changes) are instant:
```bash
docker compose up
```

Only rebuild (`--build`) when the code or `requirements.txt` changes.

**Team accounts persist** across restarts via a Docker volume -- everyone
keeps their login and assessment history between sessions, but it does
**not** persist across `docker compose down -v` (which deletes volumes) or
switching machines. It's local to whoever's machine is running the
container -- this isn't a shared server everyone hits at once, each
person runs their own copy locally.

### 3. Making changes as a team

Normal Git workflow -- branch, commit, push, pull request:
```bash
git checkout -b your-feature-name
# make changes
git add .
git commit -m "what you changed"
git push -u origin your-feature-name
```
Then open a pull request on GitHub for teammates to review before merging
to `main`. Whoever pulls the updated `main` just runs
`docker compose up --build` again to pick up the changes -- no manual
dependency wrangling.

### 4. If you'd rather run it without Docker

Everything in the earlier "Run (unified app)" section still works exactly
as before -- Docker is the recommended path for a team, not a requirement.

---

## Your specific questions, answered

### "How are you getting the disease data, and is that good, or should I use Kaggle?"

Right now (`data/generate_datasets.py`) I generate **synthetic patients
using real clinical risk-factor logic** -- e.g. diabetes risk is a
weighted function of glucose + BMI + age + family history + activity
level, plus noise -- not random labels. This gives you a genuinely
explainable model (SHAP finds real patterns) without needing a dataset,
but it's not as good as real data: it can't capture the messy
correlations, missing values, and edge cases real patients have.

**Use real data if you can** -- it'll make your results and your report
stronger. Good public options for your 5 diseases:
- Diabetes: Pima Indians Diabetes Dataset (Kaggle)
- CVD: UCI Heart Disease Dataset
- CKD: UCI/Kaggle Chronic Kidney Disease Dataset
- Liver: Indian Liver Patient Dataset (ILPD)
- Hypertension: several exist on Kaggle -- search "hypertension risk dataset"

To use one: download the CSV, rename its columns to match what each
`generate_X()` function in `data/generate_datasets.py` produces (or adjust
the column list in `models/train_models.py`), and drop it at
`data/raw/<disease>.csv`. The loader (`load_or_generate()`) automatically
prefers a real file over synthetic generation -- no other code changes
needed. Then rerun `python models/train_models.py`.

### "How do I add a real DistilBERT SLM?"

**TinyLlama (generative SLM)** -- already wired up. `llm_slm/slm_transformers.py`
is a real `transformers` backend. On your machine (which has normal
internet access, unlike the sandbox this was built in):
```bash
pip install -r requirements-slm.txt   # transformers, torch, accelerate, scikit-learn
export SAFEMEDTWIN_SLM_BACKEND=transformers
export SAFEMEDTWIN_SLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0   # default
streamlit run streamlit_app.py
```
That's it -- `llm_slm/compare.py` automatically routes through the real
model instead of the offline template. Downloads ~2.2GB the first time.

**DistilBERT (classifier -- the actual right job for it)** -- DistilBERT
is an *encoder*, built for classification, not text generation, so it
can't fill the SLM's role above. It has its own, complementary role: a
fine-tuned text classifier that predicts risk band (low/moderate/high)
directly from a natural-language description of a patient -- a second,
independent way of arriving at the same label space as the XGBoost
models, and a genuine demonstration of what small encoder models are
actually fast and cheap at.
```bash
pip install -r requirements-slm.txt
python models/train_distilbert_classifier.py   # fine-tunes on ~18k examples, a few minutes on CPU
streamlit run streamlit_app.py
```
Downloads `distilbert-base-uncased` (~268MB) the first time, then trains
for 3 epochs on a text dataset built from all 5 diseases (see
`models/text_dataset.py` -- converts each patient's structured features
into a sentence like *"Patient, age 55, BMI 34.0, glucose 165 mg/dL,
family history of the condition. Being assessed for diabetes."*). Once
trained, a **"🔤 DistilBERT Text Classifier"** section appears on the 3D
Twin & What-If tab, showing DistilBERT's classification alongside the
XGBoost model's, with confidence and latency -- agreement/disagreement
between the two is itself a legitimate finding for your report, not
something that needs to match.

**What I could and couldn't verify for these, honestly:** this sandbox
can't reach huggingface.co, so I couldn't download the real TinyLlama or
DistilBERT weights to test end-to-end here. What I *did* verify: I
installed the real `transformers`/`torch`/`accelerate` stack and ran the
exact training code (dataset building, tokenization, `Trainer`, save,
reload, inference) against a tiny randomly-initialized model with a
locally-built vocabulary -- no download needed for that, since the goal
was testing the code's mechanics, not the specific pretrained weights.
Every step passed, including the save/load round-trip and label
resolution. The API calls (`TrainingArguments`, `AutoModelForSequenceClassification`,
etc.) are confirmed correct against the actual installed library version
-- not guessed from memory. The only untested part is the real
`distilbert-base-uncased`/TinyLlama weights themselves downloading and
performing well, which needs your machine's internet access to check.

### Chatbot, report image scanner, diet plans, login -- all built

- **Login** (`auth/auth.py`): SQLite-backed, hashed passwords (PBKDF2,
  200k iterations, per-user salt), each user's predictions/chat history
  are private to their account.
- **Chatbot** (`llm_slm/chatbot.py`, "Chatbot" tab): sees the patient's own
  latest risk results as context, answers questions grounded in their
  actual data, never diagnoses, MedGuard-validated.
- **Report scanner** (`vision/report_reader.py` for images,
  `vision/pdf_reader.py` for PDFs, "Report Scanner" tab): upload a
  photo/screenshot OR a PDF of a lab report. PDFs are read via direct text
  extraction when possible (fast, cheap), falling back to rendering pages
  as images for scanned/image-only PDFs. Claude extracts legible values
  and writes a cautious plain-language summary (never a diagnosis).
  Extracted values can prefill the assessment form -- nothing is
  auto-submitted.
- **History** ("History" tab): every assessment you've run is saved to
  your account; view risk-over-time per disease plus a summary table
  across all 5 diseases.
- **Diet & lifestyle** (`diet/recommendations.py`, "Diet & Lifestyle" tab):
  rule-based plan per disease + risk band, with an optional
  LLM-personalized note woven around the patient's specific top SHAP risk
  factors.

## 3D anatomical Digital Twin

`frontend/body_twin.py` renders an interactive 3D body (drag to rotate,
scroll to zoom, auto-rotating) with the organ tied to each disease
positioned anatomically and colored/pulsing by that disease's current risk
band -- pancreas (diabetes), heart (cardiovascular disease), kidneys (CKD),
liver (liver disease), arterial system (hypertension). Updates as you
assess more diseases for the same account.

It embeds a real three.js engine with no external CDN dependency -- the
self-contained CommonJS bundle (`frontend/assets/three.bundle.js`) is
inlined into the HTML and executed via a small `module.exports` shim, so
it works fully offline too. Verified by constructing every geometry the
scene uses against the real library in Node, and by running the app
headlessly with zero exceptions. Not verified: actual pixel-level WebGL
rendering in a browser (this sandbox can't run one) -- give it a visual
check the first time you run it.

## What's real vs. a documented stand-in

Everything is real and functional except two things, both because this
was built in a sandbox without Hugging Face/Kaggle access -- both are
structured as clean drop-in swaps (see above), not hacks:

- **Datasets** are synthetic-with-real-risk-logic, not downloaded from
  Kaggle/UCI -- swap in real CSVs any time.
- **The SLM's default backend** is a fast offline template responder, not
  a downloaded DistilBERT/TinyLlama -- one env var switches to the real
  `transformers` backend, already built.

The LLM arm calls the real Anthropic API. The ML models, SHAP
explainability, MedGuard rules, AlertEngine, Digital Twin simulator, 3D
visualization, auth system, chatbot, report scanner, and diet engine are
all fully real, not mocked.

## What each report requirement maps to in code

| Report item | Implementation |
|---|---|
| 5 diseases simultaneously | `models/train_models.py`, one XGBoost model per disease |
| Real-time SHAP per prediction | `models/explainability.py` |
| Digital Twin engine | `digital_twin/simulator.py` |
| 3D anatomical Digital Twin *(bonus, not in report)* | `frontend/body_twin.py` |
| MedGuard safety layer | `safety/medguard.py` |
| AlertEngine (threshold-based alerts) | `safety/alert_engine.py` |
| LLM + SLM comparison framework | `llm_slm/compare.py` |
| Consistency testing (variance across repeated outputs) | `llm_slm/compare.py::consistency_test` |
| PDF/report analysis | `vision/report_reader.py` (images) + `vision/pdf_reader.py` (PDFs) |
| Personalized recommendations | `diet/recommendations.py` |
| Streamlit interactive dashboard | `streamlit_app.py` / `frontend/app.py` |
| Login / per-user access *(bonus, not in report)* | `auth/auth.py` |
| Chatbot *(bonus, not in report)* | `llm_slm/chatbot.py` |

## Extending this

- **Retrain on real data**: drop CSVs in `data/raw/`, rerun `train_models.py`.
- **Add a 6th disease**: write a `generate_X()` in `data/generate_datasets.py`,
  add it to `GENERATORS`, and add its label + organ mapping in
  `streamlit_app.py` / `frontend/body_twin.py`.
- **Move off SQLite** for a persistent multi-user deployment -- see the
  deployment note above.
