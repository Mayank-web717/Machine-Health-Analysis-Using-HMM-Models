# Data Directory

## Contents

| File | Description |
|------|-------------|
| `sample_sensor_data.csv` | Auto-generated synthetic machine sensor dataset (created on first run) |

## Synthetic Dataset

The synthetic dataset simulates **10 industrial machines** running to failure, each with:
- `machine_id`: Unique integer ID (1–10)
- `cycle`: Operating cycle number (1 to max_cycles per machine)
- `temperature`: Simulated temperature (°K units, approx 350–430)
- `vibration`: Simulated vibration amplitude (approx 0.5–2.5)
- `pressure`: Simulated operating pressure (approx 30–52 bar)
- `rotation_speed`: Simulated RPM (approx 2600–3020)
- `health_stage`: Ground-truth label (0=Healthy, 1=Degrading, 2=Critical) — **for evaluation only**

### Health Stage Progression

Each machine follows a three-phase trajectory:
1. **Healthy (stage 0)**: Stable sensor readings with low noise
2. **Degrading (stage 1)**: Gradual drift in temperature (+), vibration (+), pressure (−)
3. **Critical (stage 2)**: High amplitude fluctuations, clear degradation trend

### NASA C-MAPSS Dataset (Optional)

For real-world data, download the NASA CMAPSS dataset:
- URL: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- File: `CMAPSSData.zip`
- After download, place `train_FD001.txt` or similar files in this directory.

> ⚠️ The NASA dataset has a different column structure. A custom loader would be needed for direct use.
> The synthetic dataset provides a realistic alternative for course demonstration purposes.

## Disclaimer

> **The synthetic dataset is computer-generated for educational purposes only.**
> Sensor values, progression rates, and noise levels are chosen to illustrate
> HMM concepts and do not represent real industrial measurements.
> No real equipment failure predictions should be made based on this project.
