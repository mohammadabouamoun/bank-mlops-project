# Design Decisions

1. **Folder name `platform_service`** – avoids conflict with Python's standard library `platform` module.

2. **Operating threshold** – chosen by the week‑5 day‑2 rule: highest threshold with recall ≥ 0.75 on the validation set. Final value: **0.3527**.

3. **Model selection** – RandomForest chosen over LogisticRegression and GradientBoosting based on validation AUC (0.8087). `class_weight='balanced'` applied to handle class imbalance.

4. **Model serving** – A `ModelManager` class wraps the sklearn pipeline, threshold, and feature names. This allows `/predict` to return both the binary prediction and the calibrated probability.

5. **Drift detection** – PSI for numeric features, chi‑squared for categoricals, and PSI on predicted probabilities. Severity thresholds: PSI ≥ 0.15 → medium, ≥ 0.25 → high; chi‑squared p‑value < 0.05 → medium.

6. **Promotion gate** – Programmatic checks: recall ≥ 0.75, AUC ≥ 0.70, threshold tag present, and no regression against the current Production model. All thresholds are configurable in `contracts/settings.py`.

7. **Webhooks are fire‑and‑forget** – The platform never blocks on an unreachable agent; connection errors are logged gracefully.

8. **All prompts stored as files** (agent side) – never inline strings.

9. **LLM calls are mocked in CI** – snapshot trajectory tests run without an API key.
