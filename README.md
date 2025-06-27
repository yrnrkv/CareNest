# CareNest – Incubator Monitoring & Maintenance Prediction

**CareNest** is an AI toolkit that keeps low‑cost neonatal incubators running in remote clinics.  
It combines two complementary models:

| Model | Purpose | File |
|-------|---------|------|
| Random Forest | Classify failure mode (`heater_issue`, `sensor_drift`, `air_circulation`, `none`) | `src/care_nest.py` |
| Isolation Forest | Detect any temperature anomaly without labels | `src/isolation_forest.py` |

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## 🔧 Quick start

```bash
pip install -r requirements.txt
python src/care_nest.py          # generates synthetic data, trains RF, saves plots & model
python src/isolation_forest.py   # trains IF on the same data, saves anomaly plot
