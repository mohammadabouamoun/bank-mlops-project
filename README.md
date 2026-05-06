# Drift Triage Co‑Pilot

Week 5 MLOps & Agentic Systems project.

## Team
- **Platform (Person A):** Mohammad Abou Amoun
- **Agent (Person B):** Aya Baghdady

## Documentation
- [Architecture](ARCH.md)
- [Design Decisions](DECISIONS.md)
- [Runbook](RUNBOOK.md)

## Model
- **Name:** bank_marketing_classifier (version **4** in Production)
- **Type:** RandomForest with `class_weight='balanced'`
- **Test AUC:** 0.8115
- **Test F1:** 0.3752
- **Operating threshold:** 0.3527 (highest threshold with recall ≥ 0.75)
