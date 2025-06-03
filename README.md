# Fake News Detection

## Project Overview

This application uses machine learning to detect whether a given news headline or content is real or fake. It consists of:
1. **ML Training** (Phase 1) – Preprocesses data, trains multiple models, and saves the best model.
2. **Flask Backend** (Phase 2) – Exposes a `/predict` endpoint and server-side rendering for the UI.
3. **Frontend UI** (Phase 3) – Uses Tailwind CSS for a responsive design that shows predictions.
4. **Real‐time Fetching** (Phase 4) – Retrieves live news via NewsAPI, appends to CSVs, and runs predictions.
5. **Final Polish** (Phase 5) – Comments, code cleanup, and deployment configuration.
