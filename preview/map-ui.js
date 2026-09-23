/* Open Design map assets; forecast state and calculations remain owned by app.js. */
"use strict";
(() => {
  const coordinates = { 1: [71.46, 51.04], 2: [71.45, 51.05] };
  const pins = [1, 2].map(id => document.getElementById(`pin-${id}`));
  const host = document.getElementById("geographic-map");
  const fallback = document.getElementById("map-fallback");
  const status = document.getElementById("map-status");
  const motion = document.getElementById("map-motion");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
  motion.checked = !reduced.matches;
  let map;
  const duration = () => motion.checked && !reduced.matches ? 650 : 0;
  function unavailable(message) {
    status.textContent = message;
    fallback.hidden = false;
    pins.forEach(pin => { pin.classList.remove("geographic-pin"); fallback.append(pin); });
    host.hidden = true;
    const previousMap = map;
    map = null;
    if (previousMap) previousMap.remove();
    document.getElementById("map-fit").disabled = true;
    document.getElementById("map-find").disabled = true;
  }
  function fit() {
    if (!map) return;
    map.fitBounds([[71.45, 51.04], [71.46, 51.05]], {
      padding: { top: host.clientWidth < 360 ? 300 : 230, bottom: 135, left: 30, right: 140 },
      maxZoom: 14, duration: duration(),
    });
  }
  function update() {
    const row = state.rows[state.hour];
    pins.forEach((pin, index) => {
      const id = index + 1;
      const metric = state.layer === "wind" ? "wind" : "power";
      const value = row?.[`${metric}${id}`];
      const unit = metric === "wind" ? "м/с" : "p.u.";
      pin.disabled = !row;
      pin.classList.toggle("selected", state.turbine === String(id));
      pin.style.opacity = "1";
      pin.style.setProperty("--level", `${row ? Math.min(100, Math.max(0, value / (metric === "wind" ? 25 : 1) * 100)) : 0}%`);
      pin.setAttribute("aria-label", `Выбрать турбину ${id}${row ? `: ${value.toFixed(3)} ${unit}` : ": нет данных"}`);
    });
    document.getElementById("map-scale-max").textContent = state.layer === "wind" ? "25+ м/с" : "1 p.u.";
    document.getElementById("map-find").value = state.turbine;
  }
  document.addEventListener("forecast-frame", update);
  document.addEventListener("forecast-updated", update);
  document.getElementById("map-fit").addEventListener("click", fit);
  document.getElementById("map-find").addEventListener("change", event => {
    const id = event.target.value;
    document.querySelector(`[data-turbine="${id}"]`).click();
    if (id === "both") fit();
    else if (map) map.easeTo({ center: coordinates[id], zoom: 14, duration: duration() });
  });
  reduced.addEventListener("change", () => { motion.checked = !reduced.matches; });
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopPlayback(); });
  update();
  try {
    if (!window.maplibregl) throw new Error("Map library unavailable");
    map = new maplibregl.Map({
      container: host,
      center: [71.455, 51.045], zoom: 13,
      minZoom: 10, maxZoom: 16,
      maxBounds: [[71.34, 50.98], [71.57, 51.12]],
      attributionControl: false,
      // Generated CARTO tiles were API-key placeholders, not geographic imagery.
      // Only public map tiles are fetched; forecast data never leave our server.
      style: "https://tiles.openfreemap.org/styles/positron",
      locale: {
        "NavigationControl.ZoomIn": "Увеличить масштаб", "NavigationControl.ZoomOut": "Уменьшить масштаб",
        "NavigationControl.ResetBearing": "Ориентировать на север", "FullscreenControl.Enter": "Полноэкранная карта",
        "FullscreenControl.Exit": "Выйти из полноэкранного режима", "AttributionControl.ToggleAttribution": "Источники карты",
      },
    });
    map.addControl(new maplibregl.NavigationControl(), "top-right");
    map.addControl(new maplibregl.FullscreenControl({ container: document.getElementById("site-map") }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
    map.addControl(new maplibregl.AttributionControl({ compact: false }), "bottom-right");
    map.getCanvas().setAttribute("aria-label", "Карта двух турбин. Стрелки — перемещение, плюс и минус — масштаб.");
    pins.forEach((pin, index) => {
      pin.classList.add("geographic-pin");
      // Anchor at the circular turbine symbol, rather than the width of its label.
      new maplibregl.Marker({ element: pin, anchor: "left", offset: [-25, 0] }).setLngLat(coordinates[index + 1]).addTo(map);
    });
    fallback.hidden = true;
    update(); // MapLibre adds its own marker label during construction.
    let loaded = false;
    const timeout = setTimeout(() => {
      if (!loaded && map) unavailable("Не удалось загрузить карту. Проверьте интернет; прогноз доступен в карточках.");
    }, 20000);
    map.on("load", () => {
      loaded = true;
      clearTimeout(timeout);
      status.textContent = "OpenFreeMap · координаты из кейса не подтверждены";
      fit();
    });
    map.on("error", () => { status.textContent = "Часть подложки недоступна. Координаты и прогноз доступны в карточках."; });
    map.getCanvas().addEventListener("webglcontextlost", event => {
      if (!map) return;
      event.preventDefault();
      unavailable("Графический контекст недоступен. Выберите турбину в списке; расчёты работают.");
    });
    new ResizeObserver(() => { if (map) { map.resize(); fit(); } }).observe(host);
  } catch (_) {
    unavailable("Карта недоступна в этом браузере. Выберите турбину в списке; расчёты работают.");
  }
})();
