# Prompt for OpenDesign

Paste the text below into [OpenDesign](https://open-design.ai/), with this repository or the current `preview/` folder attached if your setup supports local files. The reference links are in [DESIGN_REFERENCES.md](DESIGN_REFERENCES.md).

```text
Design and build a high-fidelity, responsive web dashboard prototype for ALEM WIND, an internal wind-power forecasting workspace for a two-turbine wind farm near Astana, Kazakhstan. This is an operator tool used to inspect daily, hourly forecasts over 24 or 48 hours and audit how each forecast was produced. It is not a marketing landing page.

Start from the existing ALEM WIND prototype in this repository: preview/index.html, preview/styles.css, and preview/app.js. It already uses an Electricity Maps-inspired navigation rail, detail panel, schematic map, and hourly time dock. Keep its restrained forest-green/ivory identity, Russian interface copy, and working controls, then refine the hierarchy and visual quality. Create any exploratory redesign as a separate runnable prototype so the current preview remains available for comparison. Provide desktop (1440px) and mobile (390px) layouts, reusable visual tokens, and accessible interactions.

Use these products as inspiration for specific patterns, not as visual templates to copy:
- OpenWindCast: https://openwindcast.com/wind-power-forecast/great-britain/ — clear forecast issue, pending observations, and dated score record.
- Meteomatics: https://www.meteomatics.com/en/energy-forecasting/wind-power/ — connection between weather and power, physical wind thresholds.
- Electricity Maps: https://app.electricitymaps.com/map/zone/DE/72h/fifteen_minutes — time navigation and detail panel.
- MetX: https://www.meteomatics.com/en/weather-visualization/getting-started/ — complementary map, plot, and table views.

Design the following experience:
1. A compact sidebar: Overview, Forecast, Turbines, Evaluation, and Data & audit. Header identifies the wind farm and simulation origin (1–28 February 2026), shows UTC explicitly, and has a persistent “DEMO DATA” or real-run status badge.
2. At the top, date picker for forecast issue date, 24h/48h segmented control, and a clear export CSV action. Summary cards show forecast mean power and peak power for each turbine, plus the number of forecast hours affected by the 3 m/s cut-in or 25 m/s cut-out rule. Use only metrics derivable from the available forecast.
3. Make the map-and-detail area the first major surface: a two-marker schematic map, a side panel showing the selected hour and turbine values, a Power/Wind layer switch, and a docked hourly scrubber. Marker selection and the scrubber must update the detail panel and charts together. The map is illustrative, not verified topography or a weather field.
4. Directly below the map, show a large interactive hourly chart with clearly distinguished lines for T1 (51.04, 71.46) and T2 (51.05, 71.45). Values are normalized active power, 0–1 p.u.; label the unit. Show the selected UTC target hour and make the forecast issue date distinct from target time. Include a turbine filter. Place a smaller aligned wind-speed chart in m/s below it, with a synchronized cursor and 3/25 m/s thresholds. Then show two concise turbine cards and an expandable hourly table with valid hour, power, wind, and any physical constraint applied. On mobile, stack the detail panel, map, scrubber, charts, and table without tiny text.
5. Provide a “Data & audit” drawer or panel with forecast origin, source forecast model, assumed weather-availability bound, latest observed SCADA hour, training cutoff, model/checkpoint, simulation policy, and data completeness status. Treat provenance as useful operator information rather than decoration. Explain that the weather publication lag is an assumption, not a verified provider issue timestamp.
6. The Evaluation view can compare a past forecast with later observed power and show MAE/RMSE by turbine and lead hour when observations exist. For February 2026 in the supplied files, show a thoughtful empty state: “Фактические данные за февраль пока недоступны”. Never invent observed production, accuracy scores, uncertainty bands, turbine capacity in MW, or live telemetry.

Interactions must work in the prototype: change date and 24h/48h range, filter turbines, synchronize chart tooltip and hourly table, open audit details, and export the currently shown rows. Use deterministic illustrative data and label every demo value clearly. No external API calls, authentication, or backend is needed in this design prototype. Later, the design should be able to consume daily forecast CSVs with columns forecast_origin, valid_time, turbine_id, power_normalized, and wind_speed_ms, plus run.json and daily audit JSON files.

Use typography that is readable at normal desktop and mobile sizes, strong focus states, color contrast, a legend that does not rely on color alone, and reduced-motion support. Deliver the runnable HTML/CSS/JS files, a short design-system note (color, type, spacing, chart rules), and a brief explanation of the most important design choices. Show realistic Russian UI labels, but do not imply the synthetic prototype is a validated February forecast.
```
