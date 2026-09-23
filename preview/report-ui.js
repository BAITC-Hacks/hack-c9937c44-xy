/* Scientific reports use Matplotlib on the local report service. */
"use strict";
(() => {
  const button = document.getElementById("generate-report");
  const status = document.getElementById("report-status");
  let generation = 0;

  function reset() {
    generation += 1;
    button.disabled = !state.rows.length;
    button.textContent = "Сформировать отчёт ↗";
    document.getElementById("report-result").hidden = true;
    document.getElementById("report-outline").hidden = false;
    document.getElementById("report-figures").replaceChildren();
    status.textContent = state.rows.length
      ? "Графики ошибок и метрики добавляются только при наличии фактических измерений."
      : "Сначала загрузите доступный прогноз.";
  }

  function link(label, url, download = false) {
    const element = document.createElement("a");
    element.textContent = label;
    element.href = url;
    if (download) element.download = "";
    else { element.target = "_blank"; element.rel = "noopener"; }
    return element;
  }

  button.addEventListener("click", async () => {
    if (!state.rows.length) return;
    const request = ++generation;
    button.disabled = true;
    button.textContent = "Формируем рисунки…";
    status.textContent = "Подготовка PDF, векторных рисунков и исходных данных…";
    try {
      const response = await fetch("/api/report", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: $("source").value, date: $("date").value,
          horizon: state.horizon, rows: $("source").value === "synthetic" ? state.rows : undefined }),
      });
      if (response.status === 501 || !(response.headers.get("content-type") || "").includes("application/json")) {
        throw new Error("Сервис отчётов недоступен. Запустите python serve_preview.py --port 8767 и откройте http://127.0.0.1:8767/preview/.");
      }
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Не удалось сформировать отчёт.");
      if (request !== generation) return;
      const base = result.base_url, report = result.report;
      $("report-html").href = base + report.html;
      $("report-pdf").href = base + report.pdf;
      $("report-bundle").href = base + report.bundle;
      $("report-figures").replaceChildren();
      report.figures.forEach((figure, index) => {
        const card = document.createElement("article");
        card.className = "report-figure";
        const imageLink = link("", base + figure.svg);
        const image = document.createElement("img");
        image.src = base + figure.svg;
        image.alt = `Рисунок ${index + 1}. ${figure.title}`;
        image.loading = "lazy";
        image.width = 1123;
        image.height = 797;
        imageLink.append(image);
        const heading = document.createElement("h3");
        heading.textContent = `${String(index + 1).padStart(2, "0")} / ${figure.title}`;
        const caption = document.createElement("p");
        caption.textContent = figure.caption;
        const exports = document.createElement("div");
        exports.className = "figure-exports";
        exports.append(link("SVG ↓", base + figure.svg, true), link("PNG 300 dpi ↓", base + figure.png, true));
        card.append(imageLink, heading, caption, exports);
        $("report-figures").append(card);
      });
      $("report-validation").textContent = report.n_observed
        ? `Оценка по ${report.n_observed} совпавшим турбино-часам. MAE, RMSE и смещение приведены в полном отчёте; пропуски не заполнялись.`
        : "Оценка точности недоступна: фактические значения для этого отчёта не предоставлены или не совпали по времени.";
      $("report-outline").hidden = true;
      $("report-result").hidden = false;
      status.textContent = `${report.mode === "synthetic-demo" ? "Демонстрационный" : "Научный"} отчёт готов · ${report.horizon_hours} ч · ${report.figures.length} ${report.figures.length === 4 ? "рисунка" : "рисунков"} · SHA-256: ${report.source_sha256.slice(0, 12)}`;
    } catch (error) {
      if (request === generation) status.textContent = error.message;
    } finally {
      if (request === generation) { button.disabled = !state.rows.length; button.textContent = "Сформировать заново ↗"; }
    }
  });
  document.addEventListener("forecast-updated", reset);
  reset();
})();
