const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const script = fs.readFileSync(path.join(__dirname, "../preview/economics-ui.js"), "utf8");

function harness(fetch) {
  const elements = new Map(), events = {};
  const $ = id => {
    if (!elements.has(id)) elements.set(id, { value: "", listeners: {}, checked: true,
      addEventListener(name, handler) { this.listeners[name] = handler; },
      checkValidity() { return true; }, replaceChildren() {}, append() {} });
    return elements.get(id);
  };
  for (const [key, value] of Object.entries({ capacity_1_mw: 2, capacity_2_mw: 2, plan_mw: 3, tariff_kzt_kwh: 25 })) {
    $("eco-" + key).value = String(value); $("eco-" + key).valueAsNumber = value;
  }
  $("eco-tariff_note").value = "demo";
  $("source").value = "rolling-january-gpu";
  const state = { rows: [{}], horizon: 24, audit: { forecast_origin: "2026-01-29T00:00:00Z" } };
  vm.runInNewContext(script, { $, state, fetch, window: {}, Intl, TypeError, Event,
    setTimeout: () => 1, clearTimeout() {}, document: {
      addEventListener: (name, callback) => { events[name] = callback; },
      dispatchEvent() {}, createElement: () => ({ append() {} }),
    } });
  return { $, state, events };
}

const response = () => ({ ok: true, headers: { get: () => "application/json" }, json: async () => ({
  totals: { forecast_mwh: 30, planned_mwh: 72, shortfall_mwh: 42, shortfall_kwh: 42000,
    shortfall_hours: 24, lost_energy_revenue_kzt: 1050000 },
  recommendation: "Scenario", procurement: null, rows: [], horizon_hours: 24, forecast_origin: "2026-01-29T00:00:00Z",
}) });
const flush = () => new Promise(resolve => setImmediate(resolve));

test("model economics sends only assumptions and artifact identity; clearing a forecast clears money", async () => {
  let payload;
  const ui = harness(async (_, options) => { payload = JSON.parse(options.body); return response(); });
  await flush();
  assert.equal(payload.rows, undefined);
  assert.equal(payload.origin, ui.state.audit.forecast_origin);
  assert.equal(payload.economics.tariff_kzt_kwh, 25);
  assert.equal(ui.$("economics-result").hidden, false);
  ui.state.rows = [];
  await ui.events["forecast-updated"]();
  assert.equal(ui.$("economics-result").hidden, true);
  assert.equal(ui.$("economics-download").disabled, true);
});

test("changing a tariff prevents an old monetary result from reappearing", async () => {
  let resolve;
  const pending = new Promise(done => { resolve = done; });
  const ui = harness(() => pending);
  ui.$("economics-form").listeners.input();
  resolve(response());
  await flush();
  assert.equal(ui.$("economics-result").hidden, true);
  assert.equal(ui.$("economics-download").disabled, true);
});
