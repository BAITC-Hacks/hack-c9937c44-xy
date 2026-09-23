# Handoff: ALEM WIND

**GitHub:** `BAITC-Hacks/hack-c9937c44-xy`  
**Branch to check out:** `codex/initial-wind-forecast`  
**Current baseline commit:** `5806523`

The initial prototype is pushed. Start with [README.md](README.md) for the architecture, setup, backtest rules and known assumptions.

## What is in the repo

- `data_agent.py`: 10-minute CSV aggregation, hourly features, and archived Open-Meteo fixed-lead forecasts with availability metadata.
- `model_agent.py`: a joint two-turbine PyTorch encoder–decoder Transformer.
- `validator_agent.py`: normalized power bounds and wind cut-in/cut-out checks.
- `main.py`: daily February simulation, audit sidecars, and checkpoint output.
- `preview/`: standalone UI prototype. Its values are synthetic; it does not read model output or call the API.
- `tests/`: CPU tests for chronology, data gaps, the model, validation, and simulation.

## Start the UI and offline example

Python 3.11 or later:

```bash
python -m venv .venv
source .venv/bin/activate                 # Linux / WSL2
python -m pip install -r requirements-demo.txt
python main.py --demo
python -m http.server 8765 --directory preview
```

Open `http://127.0.0.1:8765`. The offline simulation writes clearly labelled synthetic files under `outputs/demo/`; they are not competition forecasts.

## Important before running the real model

Both provided CSVs actually end at **2026-01-31 23:50**. They contain no observed February SCADA, despite the date in their filenames. The default `--history-policy frozen` trains once before February and reuses the last observed context while fetching each day's archived forecast. `--history-policy expanding` needs new SCADA readings and fails closed when they are missing. Frozen-context predictions are a baseline with increasingly stale history, not validated February results.

The source CSV timezone is undocumented; confirm it before setting `--data-timezone`. The eight-hour weather publication lag is an assumption recorded in each audit, not a confirmed provider publication log. The forecast wind is for 10 m; hub height is not supplied. Do not describe the current February output as scored or validated. No GPU/cuDF run or accuracy evaluation has been completed.

For the real path, install compatible NVIDIA RAPIDS and CUDA-enabled PyTorch on a Linux or WSL2 GPU machine, then run:

```bash
python -m pip install -r requirements.txt
python -c "import cudf, torch; print(cudf.__version__, torch.__version__); assert torch.cuda.is_available()"
python main.py --turbine-1 "data/raw/turbine_1.csv" --turbine-2 "data/raw/turbine_2.csv" --data-timezone YOUR_CONFIRMED_IANA_TIMEZONE --history-policy frozen --backend cudf --device cuda --start 2026-02-01 --end 2026-02-28 --horizon 48
```

The sample command assumes the two CSVs are copied into `data/raw/`, which is gitignored. Replace `YOUR_CONFIRMED_IANA_TIMEZONE` with the timezone verified for the measurements.

## Suggested next steps

1. Verify the CSV timestamp timezone/convention, turbine coordinates, hub height and submission schema with the case owner.
2. Reproduce the CPU checks with `python -m unittest discover -s tests -v`, then run the GPU/cuDF smoke check and one-day real forecast.
3. Build a chronological rolling-origin evaluation and persistence/power-curve baselines; report MAE/RMSE per turbine and lead time.
4. Decide how missing February SCADA should be handled for the competition; never fill it with future observations or synthetic data.
5. Connect actual CSV/JSON forecast results to `preview/` only after the pipeline has been independently validated.
