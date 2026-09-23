/* Run with: node tests/report-ui.test.cjs (no dependencies or child processes). */
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const script = fs.readFileSync(path.join(__dirname, "../preview/report-ui.js"), "utf8");

class Element {
  constructor() { this.children = []; this.listeners = {}; this.textContent = ""; }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  replaceChildren(...children) { this.children = children; }
  append(...children) { this.children.push(...children); }
  removeAttribute(name) { delete this[name]; }
}

function harness(fetch, url = "http://127.0.0.1:8767/preview/#reports", economicSettings) {
  const elements = new Map();
  const $ = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  $("source").value = "synthetic";
  $("date").value = "2026-02-14";
  const events = {};
  const state = { rows: [{ time: "2026-02-14T00:00:00Z", power: 0.5 }], horizon: 48 };
  vm.runInNewContext(script, {
    fetch, state, $, TypeError, window: { location: new URL(url), economicSettings },
    document: {
      getElementById: $, createElement: () => new Element(),
      addEventListener: (name, handler) => { events[name] = handler; },
    },
  });
  return { $, state, events, click: () => $("generate-report").listeners.click(), reset: events["forecast-updated"] };
}

test("report sends economic assumptions and discards a response after tariff changes", async () => {
  const pending = deferred();
  let payload;
  const ui = harness(async (_, options) => { payload = JSON.parse(options.body); return pending.promise; }, undefined,
    () => ({ tariff_kzt_kwh: 25, tariff_note: "demo" }));
  const request = ui.click();
  assert.equal(payload.economics.tariff_kzt_kwh, 25);
  ui.events["economics-updated"]();
  pending.resolve(response());
  await request;
  assert.equal(ui.$("report-result").hidden, true);
});

function response({ ok = true, status = 200, body = null, contentType = "application/json" } = {}) {
  return {
    ok, status, headers: { get: () => contentType },
    json: async () => body || {
      base_url: "/outputs/reports/example/",
      report: {
        html: "report.html", pdf: "report.pdf", bundle: "report.zip", figures: [],
        n_observed: 0, mode: "synthetic-demo", horizon_hours: 48, source_sha256: "0123456789abcdef",
      },
    },
  };
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

test("model reports use the selected artifact origin instead of the hidden demo date", async () => {
  let payload;
  const ui = harness(async (_, options) => { payload = JSON.parse(options.body); return response(); });
  ui.$("source").value = "rolling-january-gpu";
  ui.state.audit = { forecast_origin: "2026-01-29T19:00:00+00:00" };
  await ui.click();
  assert.equal(payload.source, "rolling-january-gpu");
  assert.equal(payload.origin, ui.state.audit.forecast_origin);
  assert.equal(payload.rows, undefined);
});

test("offline report request shows a local recovery command and restores the button", async () => {
  const ui = harness(async () => { throw new TypeError("Failed to fetch"); }, "http://localhost:8799/preview/");
  await ui.click();
  assert.match(ui.$("report-status").textContent, /Не удалось подключиться/);
  assert.match(ui.$("report-status").textContent, /python serve_preview\.py --port 8799/);
  assert.match(ui.$("report-status").textContent, /http:\/\/127\.0\.0\.1:8799\/preview\/#reports/);
  assert.doesNotMatch(ui.$("report-status").textContent, /Failed to fetch/);
  assert.equal(ui.$("generate-report").disabled, false);
  assert.equal(ui.$("report-result").hidden, true);
});

test("nonlocal and privileged origins use the known local development port", async () => {
  for (const url of ["https://example.com:9000/preview/", "http://127.0.0.1/preview/", "file:///tmp/preview/index.html"]) {
    const ui = harness(async () => { throw new TypeError("Failed to fetch"); }, url);
    await ui.click();
    assert.match(ui.$("report-status").textContent, /--port 8767/);
  }
});

test("a dropped response body gives connection recovery guidance", async () => {
  const ui = harness(async () => ({
    ...response(), json: async () => { throw new TypeError("Failed to fetch"); },
  }));
  await ui.click();
  assert.match(ui.$("report-status").textContent, /Соединение прервалось/);
  assert.doesNotMatch(ui.$("report-status").textContent, /Failed to fetch/);
  assert.equal(ui.$("generate-report").disabled, false);
  assert.equal(ui.$("report-result").hidden, true);
});

test("ordinary static servers receive the service startup instructions", async () => {
  const ui = harness(async () => response({ ok: false, status: 501, contentType: "text/html" }), "http://127.0.0.1:8766/preview/");
  await ui.click();
  assert.match(ui.$("report-status").textContent, /По этому адресу сервис отчётов недоступен/);
  assert.match(ui.$("report-status").textContent, /serve_preview\.py/);
  assert.match(ui.$("report-status").textContent, /--port 8767/);
  assert.equal(ui.$("report-result").hidden, true);
});

test("server validation errors retain their useful message", async () => {
  const ui = harness(async () => response({ ok: false, status: 400, body: { error: "Прогноз не найден." } }));
  await ui.click();
  assert.equal(ui.$("report-status").textContent, "Прогноз не найден.");
  assert.equal(ui.$("report-result").hidden, true);
  assert.equal(ui.$("generate-report").disabled, false);
});

test("successful generation exposes links; failed regeneration clears old results immediately", async () => {
  const pending = deferred();
  let calls = 0;
  const ui = harness(() => ++calls === 1 ? Promise.resolve(response()) : pending.promise);
  await ui.click();
  assert.equal(ui.$("report-result").hidden, false);
  assert.equal(ui.$("report-pdf").href, "/outputs/reports/example/report.pdf");
  assert.match(ui.$("report-status").textContent, /отчёт готов/);
  const retry = ui.click();
  assert.equal(ui.$("report-result").hidden, true);
  assert.equal(ui.$("report-outline").hidden, false);
  assert.equal(ui.$("report-pdf").href, undefined);
  assert.equal(ui.$("report-figures").children.length, 0);
  pending.reject(new TypeError("Failed to fetch"));
  await retry;
  assert.equal(ui.$("report-result").hidden, true);
  assert.doesNotMatch(ui.$("report-status").textContent, /отчёт готов/);
});

test("changing the forecast prevents a stale success from reappearing", async () => {
  const pending = deferred();
  const ui = harness(() => pending.promise);
  const generating = ui.click();
  ui.reset();
  const resetStatus = ui.$("report-status").textContent;
  pending.resolve(response());
  await generating;
  assert.equal(ui.$("report-status").textContent, resetStatus);
  assert.equal(ui.$("report-result").hidden, true);
  assert.equal(ui.$("report-pdf").href, undefined);
});

test("a stale failed request does not hide a newer successful report", async () => {
  const oldRequest = deferred();
  let calls = 0;
  const ui = harness(() => ++calls === 1 ? oldRequest.promise : Promise.resolve(response()));
  const first = ui.click();
  ui.reset();
  await ui.click();
  const successStatus = ui.$("report-status").textContent;
  oldRequest.reject(new TypeError("Failed to fetch"));
  await first;
  assert.equal(ui.$("report-status").textContent, successStatus);
  assert.equal(ui.$("report-result").hidden, false);
  assert.equal(ui.$("report-pdf").href, "/outputs/reports/example/report.pdf");
});
