# EstateIQ - Sustainable Facility and Estate Intelligence Dashboard (India)

AI-assisted decision dashboard for hospitals, colleges, industrial estates and company campuses.
Covers air quality, energy, water, waste-bin overflow, parking hotspots, scorecards, forecasts, anomaly detection and scenario simulation.

## Run locally
    docker compose up --build        # open http://localhost:8080
Without Docker:
    cd backend && pip install -r requirements.txt && uvicorn main:app --port 8000
    cd frontend && python -m http.server 8080   # set window.API_BASE='http://localhost:8000/api' in config.js

## Deploy
- Single VPS (AWS/Azure/DigitalOcean): install Docker, clone repo, `docker compose up -d --build`, point a domain at port 8080 (add Caddy/nginx for HTTPS).
- Render/Railway: backend = Web Service from `backend/` (Dockerfile, port 8000); frontend = Static Site from `frontend/`; set `API_BASE` in `frontend/config.js` to `https://<backend-url>/api`.

## How it maps to the problem statement
| Requirement | Where |
|---|---|
| Synthetic IoT data, configurable facility types | `generate()` and `FAC` in `backend/main.py` |
| Trends, 3D campus, maps of hotspots | `frontend/index.html` (Three.js, Chart.js) |
| Anomaly detection | robust z-score (median/MAD) per zone and signal |
| Short-term forecast | Ridge regression, 24h ahead with 80% band |
| Plain-language actions | `dashboard()` recommendations with why, action and impact |
| Scenario simulation | `POST /api/simulate` |
| Role-based views | Administrator, Operations, Sustainability tabs |
| Explainability | every recommendation shows its evidence; assumptions panel |

## Next steps
Replace `generate()` with real sensors or open data (CPCB AQI, utility meters), add safety-incident module, auth for roles, PostgreSQL storage.
Outputs are decision-support insights on synthetic data, not official measurements.
