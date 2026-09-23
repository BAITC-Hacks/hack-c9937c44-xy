# ALEM WIND — interface references

Reviewed 23 September 2026. These are references for product decisions and interaction patterns, not templates to copy. The target is an operator dashboard for two turbines and a 24–48 hour hourly forecast. Our [current prototype](preview/index.html) now has a map and detail panel, date and horizon controls, a synchronized hourly timeline, two charts, and an hourly table.

| Reference | What to inspect | Application to ALEM WIND |
| --- | --- | --- |
| [OpenWindCast — Great Britain wind forecast](https://openwindcast.com/wind-power-forecast/great-britain/) | Forecast value, explicit “metered not yet” state, recent score, forecast-vs-observation chart, and a dated record table. | Make forecast issue time and later verification visible. Keep the current February actuals in an “unavailable” state. Its 80% interval is **not** a feature of our model and must not be fabricated. |
| [Meteomatics — wind power forecasting](https://www.meteomatics.com/en/energy-forecasting/wind-power/) | Public screenshots of forecast power against actual output and wind-threshold maps. | Let the operator relate the two power curves to forecast wind and cut-in/cut-out periods. Use p.u. until turbine rated capacities are known. |
| [Electricity Maps — Germany, 72-hour view](https://app.electricitymaps.com/map/zone/DE/72h/fifteen_minutes) | Slim navigation rail, scrollable detail panel, large map, time dock, and Electricity/Emissions switch. | Use this split layout and the time/layer interaction for our two turbines. Keep a readable power chart below the map and label the map as a schematic. Live page content may depend on JavaScript. |
| [Meteomatics MetX — dashboard walkthrough](https://www.meteomatics.com/en/weather-visualization/getting-started/) | Public screenshots of customizable map, plot, table, and location tools. The product itself requires login. | Group weather, power, and audit views without crowding one screen; provide clear location selection and weather-source context. |
| [Windy map embed configurator](https://embed.windy.com/config/map) | Forecast-model and wind-layer selectors and location marker. | A future real weather map would need an explicit source/model label and a time cursor tied to the power forecast. |

[OpenDesign’s dashboard workflow](https://open-design.ai/solutions/dashboard/) accepts a metrics brief and data source and produces an HTML dashboard. The companion [OPEN_DESIGN_PROMPT.md](OPEN_DESIGN_PROMPT.md) is written for that workflow.

## Recommended direction

1. Use the map-and-detail split to orient the operator, then lead into the hourly power and wind charts. Two map pins remain selectable even though the locations are close.
2. Show the issue date, forecast horizon, UTC timezone, source status, and simulation mode beside the map and charts. Separate forecast and observed periods by labeling rather than visually blending them.
3. Keep the two power series legible. Show wind speed in an aligned second chart with synchronized hover instead of mixing p.u. and m/s on one unlabeled axis.
4. Offer a small provenance drawer for training cutoff, weather source and availability assumption, missing observations, and physical constraints.
5. Show MAE/RMSE and comparisons only when later observed power exists. The provided turbine CSVs stop on 31 January 2026, so no February evaluation can currently be displayed as measured performance.
6. Keep the current restrained green/ivory identity, but improve contrast, typography, chart hierarchy, and mobile layout. Use third-party sites for ideas, not copied branding or components.
