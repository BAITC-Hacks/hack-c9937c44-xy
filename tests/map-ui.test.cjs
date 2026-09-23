const assert = require("node:assert/strict");
const test = require("node:test");
const vm = require("node:vm");
const fs = require("node:fs");

function setup({ webgl = true, reduced = false } = {}) {
  const elements = new Map(), listeners = {}, timers = [];
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      attrs: {}, styles: {}, events: {}, classes: new Set(),
      setAttribute(k, v) { this.attrs[k] = v; },
      addEventListener(k, fn) { this.events[k] = fn; },
      append(pin) { pin.parent = this; },
      style: { setProperty(k, v) { element(id).styles[k] = v; } },
      classList: {
        add(k) { element(id).classes.add(k); }, remove(k) { element(id).classes.delete(k); },
        toggle(k, value) { value ? this.add(k) : this.remove(k); },
      },
    });
    return elements.get(id);
  }
  let instance, playbackStopped = false;
  class MapMock {
    constructor(options) { this.options = options; this.events = {}; instance = this; }
    addControl() {}
    getCanvas() { return element("canvas"); }
    on(k, fn) { this.events[k] = fn; }
    fitBounds(bounds, options) { this.fitted = { bounds, options }; }
    easeTo(options) { this.camera = options; }
    remove() { this.removed = true; }
  }
  class Marker {
    constructor({ element }) { this.element = element; element.setAttribute("aria-label", "Map marker"); }
    setLngLat(point) { this.element.point = point; return this; }
    addTo() { return this; }
  }
  const gl = { Map: MapMock, Marker, NavigationControl: class {}, FullscreenControl: class {}, ScaleControl: class {}, AttributionControl: class {} };
  const state = { hour: 0, turbine: "both", layer: "power", rows: [{ power1: .4, power2: .6, wind1: 4, wind2: 30 }] };
  const document = {
    getElementById: element, hidden: false,
    addEventListener(k, fn) { listeners[k] = fn; },
    querySelector(selector) { return { click() { state.turbine = selector.match(/="([^"]+)"/)[1]; } }; },
  };
  vm.runInNewContext(fs.readFileSync("preview/map-ui.js", "utf8"), {
    state, document, window: { maplibregl: webgl ? gl : undefined, matchMedia: () => ({ matches: reduced, addEventListener() {} }) },
    maplibregl: gl, ResizeObserver: class { observe() {} },
    setTimeout(fn) { timers.push(fn); return timers.length; }, clearTimeout() {},
    stopPlayback() { playbackStopped = true; },
  });
  return { element, state, listeners, timers, instance, document, stopped: () => playbackStopped };
}

test("map uses case coordinates, labels and a fixed metric scale; empty forecasts disable stale markers", () => {
  const f = setup();
  assert.deepEqual(Array.from(f.element("pin-1").point), [71.46, 51.04]);
  assert.match(f.element("pin-1").attrs["aria-label"], /0.400 p.u./);
  f.state.layer = "wind"; f.listeners["forecast-frame"]();
  assert.equal(f.element("pin-2").styles["--level"], "100%");
  assert.equal(f.element("map-scale-max").textContent, "25+ м/с");
  f.state.rows = []; f.listeners["forecast-updated"]();
  assert.equal(f.element("pin-1").disabled, true);
  assert.equal(f.element("pin-1").styles["--level"], "0%");
  assert.match(f.element("pin-1").attrs["aria-label"], /нет данных/);
});
test("WebGL unavailable and style load timeout preserve accessible turbine choices", () => {
  for (const webgl of [false, true]) {
    const f = setup({ webgl });
    if (webgl) f.timers[0]();
    assert.equal(f.element("map-fallback").hidden, false);
    assert.equal(f.element("pin-1").parent, f.element("map-fallback"));
    assert.equal(f.element("geographic-map").hidden, true);
    assert.equal(f.element("map-fit").disabled, true);
    assert.equal(f.element("pin-1").disabled, false);
  }
});
test("reduced motion, explicit no-animation and hidden tab are respected", () => {
  const f = setup({ reduced: true });
  f.instance.events.load();
  assert.equal(f.instance.fitted.options.duration, 0);
  f.element("map-find").events.change({ target: { value: "2" } });
  assert.equal(f.state.turbine, "2");
  assert.equal(f.instance.camera.duration, 0);
  assert.deepEqual(Array.from(f.instance.camera.center), [71.45, 51.05]);
  f.document.hidden = true; f.listeners.visibilitychange();
  assert.equal(f.stopped(), true);
  const g = setup();
  g.element("map-motion").checked = false;
  g.element("map-fit").events.click();
  assert.equal(g.instance.fitted.options.duration, 0);
});
test("late timeout cannot remove a loaded map; context loss switches to cards", () => {
  const f = setup();
  f.instance.events.load(); f.timers[0]();
  assert.equal(f.instance.removed, undefined);
  f.element("canvas").events.webglcontextlost({ preventDefault() {} });
  assert.equal(f.instance.removed, true);
  assert.equal(f.element("map-fallback").hidden, false);
});
