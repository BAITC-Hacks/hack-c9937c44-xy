# Handoff: ALEM WIND

**GitHub:** `BAITC-Hacks/hack-c9937c44-xy`

**Branch to check out:** `codex/initial-wind-forecast`

**Initial baseline commit:** `5806523`; check out the latest branch tip for `main_simulation.py` and the cumulative submission.

From the IDE terminal, clone this branch directly:

```bash
git clone -b codex/initial-wind-forecast --single-branch https://github.com/BAITC-Hacks/hack-c9937c44-xy.git
```

The initial prototype is pushed. Start with [README.md](README.md) for the architecture, setup, backtest rules and known assumptions.

## What is in the repo

- `data_agent.py`: 10-minute CSV aggregation, hourly features, and archived Open-Meteo fixed-lead forecasts with availability metadata.
- `model_agent.py`: a joint two-turbine PyTorch encoder–decoder Transformer, CUDA/cuML feature normalization, and time-validated `train_model` / `predict` functions.
- `validator_agent.py`: normalized power bounds and wind cut-in/cut-out checks.
- `main_simulation.py`: daily February simulation, audit sidecars, checkpoints and cumulative `submission.csv`. `main.py` remains a compatible entrypoint.
- `preview/`: standalone UI prototype. Its values are synthetic; it does not read model output or call the API.
- `tests/`: CPU tests for chronology, data gaps, the model, validation, and simulation.

## Start the UI and offline example

Python 3.11 or later:

```bash
python -m venv .venv
source .venv/bin/activate                 # Linux / WSL2
python -m pip install -r requirements-demo.txt
python main_simulation.py --demo
python -m http.server 8765 --directory preview
```

Open `http://127.0.0.1:8765`. The offline simulation writes clearly labelled synthetic files under `outputs/demo/`; they are not competition forecasts.

## Important before running the real model

Both provided CSVs actually end at **2026-01-31 23:50**. They contain no observed February SCADA, despite the date in their filenames. The default `--history-policy expanding` retrains daily and needs new SCADA readings; it fails when they are missing. Explicitly select `--history-policy frozen` for the supplied files to train once before February and reuse the last observed context while fetching each day's archived forecast. Frozen-context predictions are a baseline with increasingly stale history, not validated February results.

The source CSV timezone is undocumented; confirm it before setting `--data-timezone`. The eight-hour weather publication lag is an assumption recorded in each audit, not a confirmed provider publication log. The forecast wind is for 10 m; hub height is not supplied. Do not describe the current February output as scored or validated. No GPU/cuDF run or accuracy evaluation has been completed.

For the real path, install compatible NVIDIA RAPIDS (cuDF, cuML, CuPy) and CUDA-enabled PyTorch on a Linux or WSL2 GPU machine, then run:

```bash
python -m pip install -r requirements.txt
python -c "import cudf, cuml, cupy, torch; print(cudf.__version__, cuml.__version__, torch.__version__); assert torch.cuda.is_available()"
python main_simulation.py --data-dir "data/raw" --data-timezone YOUR_CONFIRMED_IANA_TIMEZONE --history-policy frozen --backend cudf --device cuda --start 2026-02-01 --end 2026-02-28 --horizon 48
```

The sample command assumes the two CSVs keep their exact original Russian filenames in `data/raw/`, which is gitignored. Replace `YOUR_CONFIRMED_IANA_TIMEZONE` with the timezone verified for the measurements. Custom paths are supported via `--turbine-1` and `--turbine-2`.

The cumulative CSV retains all forecast origins, including overlapping target hours. Only `status: completed` in `run.json` marks a completed run; a failure can leave a partial CSV whose completed origins are listed in the manifest.

Latest local validation: 38 CPU tests passed, and the February offline demo completed all 28 origins with 2,688 rows for a 48-hour horizon. CUDA/cuDF/cuML execution and forecast accuracy remain unverified.

## Suggested next steps

1. Verify the CSV timestamp timezone/convention, turbine coordinates, hub height and submission schema with the case owner.
2. Reproduce the CPU checks with `python -m unittest discover -s tests -v`, then run the GPU/cuDF smoke check and one-day real forecast.
3. Build a chronological rolling-origin evaluation and persistence/power-curve baselines; report MAE/RMSE per turbine and lead time.
4. Decide how missing February SCADA should be handled for the competition; never fill it with future observations or synthetic data.
5. Connect actual CSV/JSON forecast results to `preview/` only after the pipeline has been independently validated.
