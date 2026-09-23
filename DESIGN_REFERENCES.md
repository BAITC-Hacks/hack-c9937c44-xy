# ALEM WIND — interface references

Reviewed 23 September 2026. These are references for product decisions and interaction patterns, not templates to copy. The target is an operator dashboard for two turbines and a 24–48 hour hourly forecast. Our [current prototype](preview/index.html) already has the core chart, date and horizon controls, turbine cards, and hourly table.

| Reference | What to inspect | Application to ALEM WIND |
| --- | --- | --- |
| [OpenWindCast — Great Britain wind forecast](https://openwindcast.com/wind-power-forecast/great-britain/) | Forecast value, explicit “metered not yet” state, recent score, forecast-vs-observation chart, and a dated record table. | Make forecast issue time and later verification visible. Keep the current February actuals in an “unavailable” state. Its 80% interval is **not** a feature of our model and must not be fabricated. |
| [Meteomatics — wind power forecasting](https://www.meteomatics.com/en/energy-forecasting/wind-power/) | Public screenshots of forecast power against actual output and wind-threshold maps. | Let the operator relate the two power curves to forecast wind and cut-in/cut-out periods. Use p.u. until turbine rated capacities are known. |
| [Electricity Maps — Germany, 72-hour view](https://app.electricitymaps.com/map/zone/DE/72h/fifteen_minutes) | Time selection, a geographic overview, and detailed charts in a zone panel. | Use the timeline and drilldown hierarchy. Since our turbines are only about 1 km apart, make the chart the main surface and a two-marker map secondary. Live page content may depend on JavaScript. |
| [Meteomatics MetX — dashboard walkthrough](https://www.meteomatics.com/en/weather-visualization/getting-started/) | Public screenshots of customizable map, plot, table, and location tools. The product itself requires login. | Group weather, power, and audit views without crowding one screen; provide clear location selection and weather-source context. |
| [Windy map embed configurator](https://embed.windy.com/config/map) | Forecast-model and wind-layer selectors, location marker, and animated time navigation. | If we add a weather map, give it a time cursor tied to the forecast chart. Do not make a large decorative map the primary content. |

[OpenDesign’s dashboard workflow](https://open-design.ai/solutions/dashboard/) accepts a metrics brief and data source and produces an HTML dashboard. The companion [OPEN_DESIGN_PROMPT.md](OPEN_DESIGN_PROMPT.md) is written for that workflow.

## Recommended direction

1. Lead with the decision: what power is expected from each turbine, at which hour, and what forecast wind supports it.
2. Show the issue date, forecast horizon, UTC timezone, source status, and simulation mode near the chart. Separate forecast and observed periods by labeling rather than visually blending them.
3. Keep the two power series legible. Show wind speed in an aligned second chart with synchronized hover instead of mixing p.u. and m/s on one unlabeled axis.
4. Offer a small provenance drawer for training cutoff, weather source and availability assumption, missing observations, and physical constraints.
5. Show MAE/RMSE and comparisons only when later observed power exists. The provided turbine CSVs stop on 31 January 2026, so no February evaluation can currently be displayed as measured performance.
6. Keep the current restrained green/ivory identity, but improve contrast, typography, chart hierarchy, and mobile layout. Use third-party sites for ideas, not copied branding or components.
