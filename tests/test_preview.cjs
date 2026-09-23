const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

async function main() {
  const origin = Date.parse("2026-01-25T19:00:00Z");
  const manifest = { created_at: "run-1", source_data_timezone: "UTC", horizon_hours: 24, completed_origins: [new Date(origin).toISOString()] };
  const report = { schema_version: 1, run_created_at: "run-1", source_data_timezone: "UTC", horizon_hours: 24, completed_origins: manifest.completed_origins, summary: [], daily_metrics: [], assumptions: [], rows: [] };
  for (let hour = 0; hour < 24; hour++) {
    for (let id = 1; id <= 2; id++) report.rows.push({ forecast_origin: manifest.completed_origins[0], valid_time: new Date(origin + hour * 3600000).toISOString(), turbine_id: `turbine_${id}`, power_normalized: 0.5, observed: hour === 5 ? null : 0.8, persistence: 0.2, eligible: hour !== 5 });
  }
  let response = { ok: true, status: 200, json: async () => report };
  const context = vm.createContext({ URL, fetch: async () => response });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../preview/evaluation.js"), "utf8"), context);
  const load = vm.runInContext("loadEvaluation", context);
  const rows = () => Array.from({ length: 24 }, () => ({ power1: 0.5, power2: 0.5 }));
  const loaded = rows();
  await load("http://localhost/outputs/check/", manifest, origin, loaded);
  assert.equal(loaded[0].observed1, 0.8);
  assert.equal(loaded[5].observed1, null);
  assert.equal(loaded[5].persistence1, 0.2);
  report.run_created_at = "stale";
  await assert.rejects(load("http://localhost/", manifest, origin, rows()), /другому запуску/);
  report.run_created_at = "run-1";
  report.rows[0].power_normalized = 0.9;
  await assert.rejects(load("http://localhost/", manifest, origin, rows()), /не совпадает/);
  report.rows[0].power_normalized = 0.5;
  report.rows[1] = report.rows[0];
  await assert.rejects(load("http://localhost/", manifest, origin, rows()), /Некорректные часы/);
  response = { ok: false, status: 404 };
  assert.equal(await load("http://localhost/", manifest, origin, rows()), null);
  console.log("PASS: preview evaluation accepts gaps/non-midnight origins and rejects stale, mismatched, duplicate data");
}

main().catch(error => { console.error(error); process.exitCode = 1; });
