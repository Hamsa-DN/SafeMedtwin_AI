# Decision map
repo: safemedtwin | built from: code-only | date: 2026-11-18
root problem: Usually all 5 diseases are chronic diseases where earlier stage of prediction is difficult in these generation so we have developed it.

## D1. Turn patient measurements into a comparable disease-risk estimate
- Problem: Given a patient's measurements, decide which numerical estimate represents the chance of a disease and keep the result usable for comparison across diseases.
- Given: Five disease assessments begin with numeric patient features, and no trained system exists yet.
- Constraints, in order:
  1. Different diseases have different required measurements -> breaks: one universal flat feature input for every disease
  2. Missing required measurements make a result unreliable -> breaks: silent filling of omitted patient values
  3. The same patient should receive a probability, not only a class label -> breaks: a classifier that exposes only yes-or-no output
- Decision: disease-specific supervised risk estimators; one fitted model bundle per disease contains the model, ordered feature schema, and evaluation metrics.
- Options: 2 to 4 approaches including the real one, each with its tradeoff
  - One shared model for every disease: simpler but cannot represent different feature sets cleanly.
  - Five independent binary classifiers: clear disease boundaries and the chosen design, but require five training and maintenance paths.
  - A rule-score table: transparent and lightweight, but less able to learn feature interactions.
- Evidence: code only; models/train_models.py, models/metrics.json

## D2. Preserve measurement meaning and support retraining from local data
- Problem: Create reusable patient records and keep a trained model tied to the exact feature names, order, and data source used to produce it.
- Given: Numerical feature schemas exist only after the disease data format is established.
- Constraints, in order:
  1. A model trained on named columns must receive the same columns later -> breaks: passing dictionaries or rows in arbitrary order
  2. Binary files can become unusable across environments -> breaks: depending on a prebuilt model bundle surviving machine changes
  3. Real patient datasets may be unavailable -> breaks: assuming the application can always train from downloaded records
- Decision: serialized model bundles plus dataset fallback; training writes named feature columns and metrics beside each model, while data loading prefers real CSV data before synthetic generation.
- Options: 2 to 4 approaches including the real one, each with its tradeoff
  - Download fixed datasets at setup: realistic but outside availability and transfer fragility.
  - Train and keep only in memory: portable source but no instant or repeatable deployment.
  - Serialize a bundle containing model, feature names, explainer, and metrics: repeatable and chosen, but environment and binary compatibility still require care.
- Evidence: code only; models/train_models.py, data/generate_datasets.py

## D3. Explain why one patient received this estimate
- Problem: Show which measurements pushed a result upward or downward instead of presenting an opaque score.
- Given: A fitted disease model can score an ordered patient feature vector.
- Constraints, in order:
  1. A global model summary does not explain this patient -> breaks: display only learned feature importance
  2. The presentation must identify both direction and relative influence -> breaks: ranking raw feature values instead of model contributions
  3. The explanation must fit a short patient-facing summary -> breaks: exposing every coefficient or all SHAP values
- Decision: contribution ranking and summarization; compute per-patient feature contributions, rank by magnitude, and convert the strongest positive and negative factors into plain text.
- Options: 2 to 4 approaches including the real one, each with its tradeoff
  - Raw feature thresholds: understandable but not tied to the actual model.
  - Static feature importance: fast but misleading for an individual patient.
  - Per-patient contribution ranking: personalized and the chosen design, but harder to interpret and potentially overclaimed.
- Evidence: code only; models/explainability.py

## D4. Prevent implausible scores and unsafe generated advice
- Problem: Keep a prediction or generated recommendation from crossing basic clinical safety limits before it reaches the patient.
- Given: The system now produces probabilities, ranked factors, and text recommendations.
- Constraints, in order:
  1. Physiologically impossible combinations must stop the prediction, not just receive a warning -> breaks: warning-only validation
  2. Generated text can give medication or dosage guidance even when the underlying score is based on data -> breaks: relying on model prompt wording alone
  3. A large number of safety problems requires withholding rather than piecemeal editing -> breaks: always showing the response with a few notes
- Decision: auditable validation gates; rule checks block invalid structured predictions, and text scans sanitize or withhold unsafe recommendations.
- Options: 2 to 4 approaches including the real one, each with its tradeoff
  - A second trained safety model: adaptable but opaque and harder to audit.
  - Prompt-only safety instructions: easy to add but not guaranteed after generation.
  - Explicit auditable rules plus sanitization: chosen and understandable, but narrower than the full clinical space.
- Evidence: code only; safety/medguard.py, safety/alert_engine.py

## D5. Keep one person's assessments and conversations separate
- Problem: Retain useful history while ensuring that users cannot read or continue another user's assessments.
- Given: The application has patients, predictions, and chat interactions that may span repeated visits.
- Constraints, in order:
  1. Shared deployment without an external identity provider needs a small user boundary -> breaks: one global history table
  2. Passwords must not be recoverable from stored account records -> breaks: storing plaintext passwords
  3. Assessment history must remain tied to the acting account across screens -> breaks: using browser state as the access control
- Decision: local account records and relational ownership; SQLite stores hashed credentials and user-owned prediction/chat rows queried by user identifier.
- Options: 2 to 4 approaches including the real one, each with its tradeoff
  - No login: simplest but merges every user's sensitive history.
  - External identity provider: stronger and scalable but adds deployment and provider dependencies.
  - Local SQLite accounts: small, deployable, and chosen, but limited for multi-host persistence and high trust.
- Evidence: code only; auth/auth.py

## D6. Represent repeated changes as a patient's evolving scenario history
- Problem: Explore how a patient's predicted risk changes under hypothetical measurements and preserve observations across time.
- Given: A disease model can re-score a complete feature vector, and a user has a baseline patient record.
- Constraints, in order:
  1. A what-if change must leave every other feature fixed -> breaks: rebuilding a partial feature row
  2. Results must be comparable with the baseline and across sweep values -> breaks: reporting scenario outcomes without a baseline
  3. Longitudinal history must not require a separate database in the simulator -> breaks: retaining only the latest scenario
- Decision: baseline-relative scenario replay and lightweight patient history; copy the full baseline row, replace selected features, re-score, and append timestamped outcomes.
- Options: 2 to 4 approaches including the real one, each with its tradeoff
  - A calibrated biophysical model: richer but vastly more assumptions and data.
  - One prediction per changed value: simple but loses the baseline relationship.
  - Full-row replay with delta and history: the chosen design, more accurate than a label-only tool but still not causal.
- Evidence: code only; digital_twin/simulator.py

## D7. Make generated text available without coupling it to one provider
- Problem: Let several features call an external text or vision model while preserving usable behavior if no provider or network is available.
- Given: Chat, diet notes, report interpretation, and model comparison all need similar generation operations.
- Constraints, in order:
  1. Each feature cannot embed provider-specific request code -> breaks: direct vendor SDK calls throughout the app
  2. Rate limits and transient network failures should not fail immediately -> breaks: a single provider request
  3. Missing credentials or provider outages must not disable the whole application -> breaks: exposing an uncaught generation error
- Decision: provider abstraction with retry and fallback; a shared interface selects a configured provider, normalizes text/chat/vision calls, retries transient errors, and lets callers use offline responses.
- Options: 2 to 4 approaches including the real one, each with its tradeoff
  - Vendor calls in every feature: fastest to write but duplicated and hard to switch.
  - One provider only: simpler but unavailable during key or service failures.
  - A shared provider interface with retries and caller fallbacks: chosen and resilient, with per-provider SDK and response differences to manage.
- Evidence: code only; llm_slm/llm_provider.py, llm_slm/compare.py

## D8. Extract report values through the cheapest reliable input path
- Problem: Turn uploaded lab reports into reviewable values and summaries without assuming every document contains usable text.
- Given: Reports arrive as images or PDFs, and the system already has text and vision generation capabilities plus safety validation.
- Constraints, in order:
  1. A text PDF should avoid expensive vision processing -> breaks: always rasterizing and sending every page
  2. Scanned pages with no extractable text need another representation -> breaks: returning extraction failure as the only outcome
  3. Extracted values must remain untrusted until the user reviews them -> breaks: automatically changing the assessment inputs
- Decision: staged document interpretation; extract PDF text first, invoke vision on rasterized pages as a fallback, validate generated text, and return values for manual review.
- Options: 2 to 4 approaches including the real one, each with its tradeoff
  - Vision for every report: robust to layout but slower and more costly.
  - Text extraction only: cheap but fails for scanned reports.
  - Text-first with scanned-page vision fallback: chosen and efficient, but parsing quality depends on source and model output.
- Evidence: code only; vision/report_reader.py, vision/pdf_reader.py
