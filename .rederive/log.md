## D1. Turn patient measurements into a comparable disease-risk estimate
struggled with: Initially named evaluation metrics as the model's training target; a short sketch helped distinguish the known outcome from the scores used to evaluate predictions.
found on their own: Chose five separate disease classifiers, identified the low/moderate/high categories as groups derived from probabilities, and recognized that medically validated cutoffs may not fit synthetic data.
did well: Explained the project uses disease-specific clinical measurements and synthetic training data, and clarified that it is an educational screening prototype rather than a clinical tool.
did ok: First preferred medically validated cutoffs, then chose clearly labeled demo cutoffs when considering the validation limitation.

## D2. Preserve measurement meaning and support retraining from local data
struggled with: Was unsure how to preserve feature-to-model alignment until comparing fixed order, named inputs, and model metadata.
found on their own: Chose to keep expected feature names and order with each model, suggested Docker to make the runtime portable, and identified GitHub as a possible data source.
did well: Connected disease-specific dashboards to collecting the appropriate measurements and had already explained that the project generates synthetic data from clinical mathematical formulas.
did ok: GitHub can be one source, but it does not by itself guarantee data availability, fit, or reproducibility; local synthetic-data generation can reduce that dependency.
better idea: Docker can improve environment reproducibility when the image pins dependencies and includes the required model bundle.

## D3. Explain why one patient received this estimate
struggled with: Initially confused account access with explaining an individual result, then needed choices to decide how to present contribution values.
found on their own: Chose SHAP and per-person feature contributions, then preferred summarizing the strongest influences in plain language with their direction.
did well: Recognized that individual explanations should clarify the factors behind a particular risk estimate rather than only describe the whole model.
did ok: Needed options to distinguish a global feature ranking from a patient-specific explanation and a readable summary.

## D4. Prevent implausible scores and unsafe generated advice
struggled with: Initially preferred warnings for out-of-range values and edited chatbot advice with a doctor note; needed to consider cases where a warning or edit would leave serious risk.
found on their own: Chose explicit input-validation rules and identified MedGuard as the component for checking generated advice; ultimately chose to edit minor issues but withhold serious or repeated unsafe content.
did well: Kept the prototype's educational scope in view and included consultation with a doctor in the user-facing advice.
did ok: Needed tradeoff options to distinguish warning-only handling from blocking invalid predictions and withholding unsafe responses.

## D5. Keep one person's assessments and conversations separate
struggled with: Initially chose reversible password encryption and gave only a broad "backend" answer for access control; constraints and choices clarified the difference between storing credentials and enforcing identity.
found on their own: Chose one-way password hashing and a verified login session, alongside storing each person's history separately.
did well: Recognized that the database should associate predictions and history with the particular person.
did ok: Needed options to identify why backend enforcement must rely on verified identity rather than a browser-provided user identifier.

## D6. Represent repeated changes as a patient's evolving scenario history
struggled with: Initially mentioned only tracking the latest scenario, then considered preserving earlier comparisons; also mixed the simulation's causal caveat with the separate LLM-versus-SLM feature.
found on their own: Chose to change one measurement while holding others fixed, compare the scenario with the original, and store scenario history.
did well: Connected scenario changes with the simulated organ visualization and recognized the value of comparing against the baseline.
did ok: Needed a prompt to distinguish a hypothetical model result from proof of a real-world organ effect.

## D7. Make generated text available without coupling it to one provider
struggled with: Initially proposed only telling users to try later, then favored waiting and retrying without addressing persistent outages.
found on their own: Chose a shared provider layer, trying another available model, and surfacing an error if no provider can respond.
did well: Recognized that the app should try alternatives rather than depend on one model provider.
did ok: Needed a flow sketch to reason about the case where every provider fails; an explicit error is safer than showing nothing.

## D8. Extract report values through the cheapest reliable input path
struggled with: Initially proposed requiring a text version, which would block people who only have a scan.
found on their own: Identified PDF text extraction, then chose reading scanned pages as images when needed and waiting for the user's review before using extracted values.
did well: Considered both text PDFs and scanned reports, and kept the person in control of their assessment inputs.
did ok: Needed the distinction between embedded PDF text and a scanned image to reason through fallback behavior.
