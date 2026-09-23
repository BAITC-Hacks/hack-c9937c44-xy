/* Scientific reports use Matplotlib on the local report service. */
"use strict";
(() => {
  const button = document.getElementById("generate-report");
  const status = document.getElementById("report-status");
  let generation = 0;

  function clearReport() {
    document.getElementById("report-result").hidden = true;
    document.getElementById("report-outline").hidden = false;
    document.getElementById("report-figures").replaceChildren();
    ["report-html", "report-pdf", "report-bundle"].forEach(id => $(id).removeAttribute("href"));
    $("report-validation").textContent = "";
  }

  function serviceHelp(message, usePagePort = true) {
    const { hostname, port: pagePort } = window.location;
    const local = ["localhost", "127.0.0.1", "[::1]", "::1"].includes(hostname);
    const port = usePagePort && local && /^\d+$/.test(pagePort) && Number(pagePort) >= 1024 && Number(pagePort) <= 65535
      ? Number(pagePort) : 8767;
    return `${message} Повторите попытку. Если ошибка сохраняется, в папке проекта запустите python serve_preview.py --port ${port}, `
      + `затем откройте http://127.0.0.1:${port}/preview/#reports. `
      + "Если этот порт занят обычным HTTP-сервером, остановите его перед запуском сервиса отчётов.";
  }

  function reset() {
    generation += 1;
    button.disabled = !state.rows.length;
    button.textContent = "Сформировать отчёт ↗";
    clearReport();
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
    clearReport();
    button.disabled = true;
    button.textContent = "Формируем рисунки…";
    status.textContent = "Подготовка PDF, векторных рисунков и исходных данных…";
    try {
      let response;
      try {
        response = await fetch("/api/report", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source: $("source").value, date: $("date").value,
            origin: $("source").value === "synthetic" ? undefined : state.audit?.forecast_origin,
            horizon: state.horizon, rows: $("source").value === "synthetic" ? state.rows : undefined }),
        });
      } catch {
        throw new Error(serviceHelp("Не удалось подключиться к сервису отчётов."));
      }
      if (response.status === 501 || !(response.headers.get("content-type") || "").includes("application/json")) {
        throw new Error(serviceHelp("По этому адресу сервис отчётов недоступен.", false));
      }
      let result;
      try {
        result = await response.json();
      } catch (error) {
        if (error instanceof TypeError) throw new Error(serviceHelp("Соединение прервалось при получении отчёта."));
        throw new Error("Сервис вернул некорректный ответ. Повторите попытку или проверьте журнал сервера.");
      }
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
      if (request === generation) {
        clearReport();
        status.textContent = error.message;
      }
    } finally {
      if (request === generation) { button.disabled = !state.rows.length; button.textContent = "Сформировать заново ↗"; }
    }
  });
  document.addEventListener("forecast-updated", reset);
  reset();
})();
