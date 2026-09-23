# Open Design → ALEM WIND

Integrated 2026-09-23 on `codex/wind-ui-opendesign`, based on Elias's published
forecast, evaluation and economics implementation. Default entry: `/preview/`.

Source: Open Design project **Интерактивная карта ALEM WIND**,
ID `67fc1b57-2a7a-4e82-8768-1b88e149e7a3`.

The generated folder contained product/brand specifications, MapLibre GL JS 4.7.1,
map assets and a report adapter. It did **not** contain the advertised
`alem-wind.html`, `alem-wind.js`, or `alem-wind.css`. The two specification files
beside this document are unmodified provenance snapshots, not current product
documentation: their statement about absent backtests is now outdated.

Integrated the supplied brand tokens and vendored MapLibre JS/CSS, with its license
in `preview/assets/maplibre-LICENSE.txt`. Implemented the missing map adapter and
responsive layout in `preview/map-ui.js` and `preview/map-ui.css` while preserving
the current shared forecast state, all CPU/GPU sources, quality metrics, CSV export,
economic scenarios and Python report endpoints.

Rejected the generated CARTO raster tile cache: visual inspection showed
**API KEY REQUIRED** placeholder images. No placeholder tiles are committed.
The basemap uses OpenFreeMap's Positron style instead:
[official integration guide](https://openfreemap.org/quick_start/).
Attribution comes from the style and stays visible on the map. Basemap loading
requires internet; forecast values and economic inputs are not sent to the map
provider. Map source requests convey the viewed public map area.

The generated `reports.js` / `launch_preview.py` were not copied because they
only supported three older sources and omitted Elias's economics integration.
The existing report service remains authoritative.

## Behaviour

- Two geographic markers at the case coordinates, labelled as unconfirmed.
- Power / wind share the selected hour with inspector, charts and table.
- Marker fill uses fixed scales 0–1 p.u. and 0–25 m/s (saturated above 25).
- Zoom, pan, north reset, fullscreen, turbine lookup and fit both.
- 0.5× / 1× / 2× playback; stops on source/date/horizon change or tab hide.
- Reduced-motion support and an explicit camera animation switch.
- On WebGL failure or a 20-second initial map timeout, accessible turbine cards
  remain available alongside calculations. A reload retries the map.
- Mobile stacks the map and timeline above the inspector.
- No fabricated wind direction, terrain simulation or uncertainty bands.

## Run / check

```powershell
python -X utf8 -B serve_preview.py --port 8767
python -X utf8 -m unittest discover -s tests
node tests/map-ui.test.cjs
node tests/test_preview.cjs
node tests/report-ui.test.cjs
node tests/economics-ui.test.cjs
```

Use a separate feature branch when collaborating with Elias. This integration
does not require replacing his backend, rebasing his branch, or force-pushing.
