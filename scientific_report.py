"""Reproducible scientific figures and PDF/HTML reports from audited hourly forecasts.

This module only describes forecasts and optionally scores later observations.
It neither trains models nor substitutes observations into forecast features.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import textwrap
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from standards_profile import wind_standards_profile
from economics import calculate_economics, economics_csv, economics_summary

IDS = ("turbine_1", "turbine_2")
COLORS = ("#177565", "#35649a")
FIELDS = ("forecast_origin", "valid_time", "turbine_id", "power_normalized", "wind_speed_ms")
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13,
         "axes.labelsize": 10, "axes.linewidth": .8, "pdf.fonttype": 42,
         "svg.fonttype": "none", "savefig.facecolor": "white"}


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must include an explicit UTC offset")
    return result.astimezone(timezone.utc)


@dataclass
class Forecast:
    origin: datetime
    times: list[datetime]
    power: np.ndarray
    wind: np.ndarray
    audit: dict

    @property
    def demo(self) -> bool:
        return self.audit["mode"] == "synthetic-demo"


def validate_forecast(records: list[dict], audit: dict, horizon: int | None = None) -> Forecast:
    """Reject partial, mixed-origin, duplicated or nonfinite forecast artifacts."""
    if audit.get("mode") not in ("synthetic-demo", "historical-backtest"):
        raise ValueError("Unsupported forecast audit mode")
    if audit["mode"] == "historical-backtest" and audit.get("trained_model") is not True:
        raise ValueError("Model forecasts must have a trained-model audit")
    full_horizon = audit.get("horizon_hours")
    if type(full_horizon) is not int or full_horizon not in (24, 48):
        raise ValueError("Forecast horizon must be 24 or 48 hours")
    horizon = full_horizon if horizon is None else horizon
    if type(horizon) is not int or horizon not in (24, 48) or horizon > full_horizon:
        raise ValueError("Requested horizon exceeds the available forecast")
    origin = timestamp(audit["forecast_origin"])
    if origin.minute or origin.second or origin.microsecond:
        raise ValueError("Forecast origin must align to an hour")
    if len(records) != full_horizon * 2:
        raise ValueError("Exactly two turbine values are required per forecast hour")
    power = np.full((full_horizon, 2), np.nan)
    wind = np.full_like(power, np.nan)
    for row in records:
        if set(row) != set(FIELDS) or timestamp(row["forecast_origin"]) != origin:
            raise ValueError("Unexpected columns or mixed forecast origins")
        if row["turbine_id"] not in IDS:
            raise ValueError("Unknown turbine")
        lead = (timestamp(row["valid_time"]) - origin).total_seconds() / 3600
        if not lead.is_integer() or not 0 <= lead < full_horizon:
            raise ValueError("Forecast timestamp is outside the hourly horizon")
        i, j = int(lead), IDS.index(row["turbine_id"])
        p, v = float(row["power_normalized"]), float(row["wind_speed_ms"])
        if not np.isfinite([p, v]).all() or not 0 <= p <= 1 or v < 0:
            raise ValueError("Power must be finite in [0,1]; wind must be finite and nonnegative")
        if np.isfinite(power[i, j]):
            raise ValueError("Duplicate forecast turbine/hour")
        power[i, j], wind[i, j] = p, v
    if not np.isfinite(power).all():
        raise ValueError("Missing forecast turbine/hour")
    return Forecast(origin, [origin + timedelta(hours=i) for i in range(horizon)],
                    power[:horizon], wind[:horizon], audit)


def load_forecast(path: Path, horizon: int | None = None) -> Forecast:
    audit = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with path.open(encoding="utf-8-sig", newline="") as file:
        records = list(csv.DictReader(file))
    return validate_forecast(records, audit, horizon)


def load_observations(path: Path, forecast: Forecast) -> np.ndarray:
    """Match later hourly normalized observations by UTC timestamp and turbine.

    Missing hours stay NaN. No interpolation or filling of validation targets.
    Input schema: valid_time,turbine_id,power_normalized.
    """
    if forecast.demo:
        raise ValueError("Synthetic forecasts cannot be scored against observed SCADA")
    observed = np.full_like(forecast.power, np.nan)
    lookup = {time: i for i, time in enumerate(forecast.times)}
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if set(reader.fieldnames or []) != {"valid_time", "turbine_id", "power_normalized"}:
            raise ValueError("Unexpected observation columns")
        for row in reader:
            time, turbine = timestamp(row["valid_time"]), row["turbine_id"]
            if turbine not in IDS or time.minute or time.second or time.microsecond:
                raise ValueError("Observations must identify a turbine and an aligned UTC hour")
            key = (time, turbine)
            if key in seen:
                raise ValueError("Duplicate observed turbine/hour")
            seen.add(key)
            value = float(row["power_normalized"])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Observed power must be finite in [0,1]")
            if time in lookup:
                observed[lookup[time], IDS.index(turbine)] = value
    return observed


def statistics(forecast: Forecast, observed: np.ndarray | None = None) -> list[dict]:
    if observed is not None:
        if observed.shape != forecast.power.shape or np.isinf(observed).any():
            raise ValueError("Observation matrix must match forecast shape and contain no infinities")
        finite = observed[np.isfinite(observed)]
        if ((finite < 0) | (finite > 1)).any() or forecast.demo:
            raise ValueError("Cannot score these observations")
    result = []
    for j, turbine in enumerate(IDS):
        p, wind = forecast.power[:, j], forecast.wind[:, j]
        row = {"turbine_id": turbine, "n_forecast": len(p), "mean_power_pu": float(p.mean()),
               "peak_power_pu": float(p.max()), "mean_wind_ms": float(wind.mean()),
               "max_abs_ramp_pu_per_hour": float(np.abs(np.diff(p)).max()),
               "integrated_power_pu_h": float(p.sum()), "n_observed": 0,
               "mae_pu": None, "rmse_pu": None, "bias_pu": None}
        if observed is not None:
            mask = np.isfinite(observed[:, j])
            error = p[mask] - observed[mask, j]
            row["n_observed"] = int(mask.sum())
            if len(error):
                row.update(mae_pu=float(np.abs(error).mean()), rmse_pu=float(np.sqrt(np.mean(error**2))),
                           bias_pu=float(error.mean()))
        result.append(row)
    return result


def forecast_csv(forecast: Forecast) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(FIELDS)
    for i, time in enumerate(forecast.times):
        for j, turbine in enumerate(IDS):
            writer.writerow([forecast.origin.isoformat(), time.isoformat(), turbine,
                             forecast.power[i, j], forecast.wind[i, j]])
    return stream.getvalue()


def grid(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(which="major", color="#bdc4c9", linewidth=.65)
    ax.grid(which="minor", color="#e6e9eb", linewidth=.4)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(direction="in", which="both", top=True, right=True)


def time_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m\n%H:%M", tz=timezone.utc))


def figures(forecast: Forecast, observed: np.ndarray | None):
    """Yield original, data-derived figures; no manufacturer curves are fabricated."""
    t, p, v = forecast.times, forecast.power, forecast.wind
    fig, axes = plt.subplots(2, 1, figsize=(11.7, 8.3), sharex=True)
    for j in range(2):
        style = "-" if j == 0 else "--"
        axes[0].plot(t, p[:, j], style, color=COLORS[j], label=f"T{j+1} · прогноз", linewidth=1.6)
        axes[1].plot(t, v[:, j], style, color=COLORS[j], label=f"T{j+1}", linewidth=1.4)
        if observed is not None:
            axes[0].plot(t, observed[:, j], ".", color=COLORS[j], label=f"T{j+1} · факт")
    grid(axes[0], "", "Нормализованная мощность, p.u.")
    grid(axes[1], "Время действия прогноза, UTC", "Скорость ветра, м/с")
    axes[0].set_ylim(-.03, 1.06)
    axes[0].legend(ncol=2, fontsize=9)
    axes[1].legend(fontsize=9)
    time_axis(axes[1])
    yield fig, "trajectory", "Почасовые траектории мощности и ветра", "Каждая точка соответствует одному часу. Разрывы фактических измерений не заполняются."

    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    for j in range(2):
        ax.scatter(v[:, j], p[:, j], s=30, marker="o" if j == 0 else "^", facecolors="none",
                   edgecolors=COLORS[j], linewidths=1.1, label=f"T{j+1} · {len(t)} прогнозных точек")
    if forecast.audit.get("physical_validation", {}).get("wind_limits_applied", forecast.demo):
        for threshold, label in ((3, "Включение 3 м/с"), (25, "Отключение 25 м/с")):
            ax.axvline(threshold, color="#9b7546", linestyle=":", linewidth=1, label=label)
    grid(ax, "Прогноз скорости ветра, м/с", "Прогноз мощности, p.u.")
    ax.set_ylim(-.03, 1.06)
    ax.set_xlim(left=0)
    ax.legend(fontsize=9)
    yield fig, "power_wind", "Рабочие точки: мощность — скорость ветра", "Зависимость двух прогнозируемых величин; не паспортная и не измеренная кривая мощности. Высота ветра указана в методике."

    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    for j in range(2):
        ax.step(np.arange(1, len(t) + 1) / len(t) * 100, np.sort(p[:, j])[::-1],
                where="post", color=COLORS[j], linestyle="-" if j == 0 else "--", label=f"T{j+1}")
    grid(ax, "Доля прогнозных часов с мощностью не ниже указанной, %", "Нормализованная мощность, p.u.")
    ax.set_xlim(0, 100)
    ax.set_ylim(-.03, 1.06)
    ax.legend()
    yield fig, "duration", "Кривые обеспеченности прогнозной мощности", "Мощность отсортирована по убыванию внутри выбранного горизонта. Это характеристика прогноза, а не годовая статистика."

    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    for j in range(2):
        ax.plot(t[1:], np.diff(p[:, j]), "o-" if j == 0 else "^--", color=COLORS[j],
                markersize=3, linewidth=1.2, label=f"T{j+1}")
    ax.axhline(0, color="#343f46", linewidth=.8)
    grid(ax, "Время окончания часового интервала, UTC", "Изменение мощности ΔP/Δt, p.u./ч")
    time_axis(ax)
    ax.legend()
    yield fig, "ramps", "Почасовые изменения мощности", "ΔP = P(t) − P(t−1), Δt = 1 ч. Положительные значения означают рост прогнозной мощности."

    if observed is not None and np.isfinite(observed).any():
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3), sharex=True, sharey=True)
        stats = statistics(forecast, observed)
        for j, ax in enumerate(axes):
            mask = np.isfinite(observed[:, j])
            ax.plot([0, 1], [0, 1], "k--", linewidth=.8, label="Идеальное совпадение")
            ax.scatter(observed[mask, j], p[mask, j], facecolors="none", edgecolors=COLORS[j], s=35)
            ax.set_aspect("equal")
            grid(ax, "Фактическая мощность, p.u.", "Прогнозная мощность, p.u.")
            ax.set_xlim(-.03, 1.03)
            ax.set_ylim(-.03, 1.03)
            row = stats[j]
            label = f"T{j+1} · n = {row['n_observed']}"
            if row["n_observed"]:
                label += f"\nMAE = {row['mae_pu']:.4f}; RMSE = {row['rmse_pu']:.4f} p.u."
            ax.set_title(label, fontsize=11)
        yield fig, "validation", "Сопоставление прогноза с измерениями", "Используются только совпавшие UTC-часы и турбины. Измерения применяются после прогноза исключительно для оценки."

        fig, axes = plt.subplots(2, 1, figsize=(11.7, 8.3))
        for j in range(2):
            error = p[:, j] - observed[:, j]
            axes[0].plot(t, error, "o-" if j == 0 else "^--", color=COLORS[j], markersize=3, label=f"T{j+1}")
            finite = error[np.isfinite(error)]
            if len(finite):
                axes[1].hist(finite, bins=np.linspace(-1, 1, 21), histtype="step", color=COLORS[j], label=f"T{j+1}")
        axes[0].axhline(0, color="#343f46", linewidth=.8)
        grid(axes[0], "Время действия прогноза, UTC", "Ошибка Pпрогноз − Pфакт, p.u.")
        time_axis(axes[0])
        grid(axes[1], "Ошибка прогноза, p.u.", "Число совпавших часов")
        for ax in axes:
            ax.legend(fontsize=9)
        yield fig, "residuals", "Временная структура и распределение ошибок", "Положительная ошибка означает завышение прогноза. Пропуски измерений исключены; выборка ограничена выбранным горизонтом."


def make_report(forecast: Forecast, output_root: Path, observed: np.ndarray | None = None, economic_settings: dict | None = None) -> Path:
    economics = calculate_economics(forecast, economic_settings) if economic_settings is not None else None
    economic_text = economics_summary(economics) if economics else []
    stats = statistics(forecast, observed)
    payload = forecast_csv(forecast)
    canonical = payload + json.dumps(forecast.audit, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if observed is not None:
        canonical += json.dumps(np.where(np.isfinite(observed), observed, -1).tolist())
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    standards = wind_standards_profile()
    report_canonical = canonical + json.dumps(standards, sort_keys=True, ensure_ascii=False)
    if economics:
        report_canonical += json.dumps(economics, sort_keys=True, ensure_ascii=False, allow_nan=False)
    report_digest = hashlib.sha256(report_canonical.encode("utf-8")).hexdigest()
    report_id = f"forecast_{forecast.origin:%Y%m%dT%H%MZ}_{len(forecast.times)}h_{report_digest[:12]}"
    destination = output_root / report_id
    destination.mkdir(parents=True, exist_ok=True)
    observed_n = sum(row["n_observed"] for row in stats)
    status = "СИНТЕТИЧЕСКИЙ ПРИМЕР" if forecast.demo else "ПРОГНОЗ МОДЕЛИ"
    wind_height = forecast.audit.get("wind_height_m")
    wind_text = f"{wind_height} м" if wind_height is not None else "не указана в аудите"
    notes = [
        "Выбранный горизонт содержит полные почасовые пары T1 и T2. Часовой пояс отчёта — UTC.",
        "Мощность задана в p.u. Пересчёт в МВт и МВт·ч невозможен без подтверждённой базы нормировки и номинальной мощности.",
        f"Высота ветра: {wind_text}. Диаграмма P(v) описывает прогнозные рабочие точки, а не паспортную характеристику.",
        "Наблюдения используются только для последующей оценки. Никакой подстановки в признаки модели здесь нет.",
        "MAE = mean(|Pпрогноз − Pфакт|); RMSE = sqrt(mean((Pпрогноз − Pфакт)²)); bias = mean(Pпрогноз − Pфакт).",
        "Доверительные интервалы, аэродинамический КПД и карты компрессора не рассчитываются: необходимых данных и калибровки нет.",
    ]
    if forecast.demo:
        notes.insert(0, "Все значения синтетические. Этот документ демонстрирует формат, а не точность модели или свойства реального оборудования.")
    if not observed_n:
        notes.append("Фактические значения для оценки не предоставлены или не совпали по времени. Ошибки и метрики качества недоступны.")
    bounds = forecast.audit.get("physical_validation", {})
    limits_applied = bounds.get("wind_limits_applied", forecast.demo)
    notes.append("Пороги 3/25 м/с применены источником прогноза." if limits_applied
                 else "Пороги 3/25 м/с не применены источником; ветер на 10 м не равен ветру на высоте ступицы.")
    metadata = {"schema_version": 2, "report_id": report_id, "source_sha256": digest, "report_sha256": report_digest,
                "generated_at": datetime.now(timezone.utc).isoformat(), "mode": forecast.audit["mode"],
                "forecast_origin": forecast.origin.isoformat(), "horizon_hours": len(forecast.times),
                "n_observed": observed_n, "statistics": stats, "methodology": notes, "audit": forecast.audit,
                "standards_profile": standards, "standards_file": "standards.json",
                "references": [{"title": "NREL: Fundamentals of Wind Energy", "url": "https://www.nrel.gov/docs/fy23osti/84501.pdf"}]
                              + [{"title": item["designation"], "url": item["source"]} for item in standards["selected"]],
                "figures": [], "pdf": "report.pdf", "html": "report.html", "bundle": "report_bundle.zip"}
    if economics:
        metadata["economics"] = economics
    provenance = [
        ("Выпуск прогноза", forecast.origin.strftime("%d.%m.%Y %H:%M UTC")),
        ("Горизонт", f"{len(forecast.times)} ч · {len(forecast.times)*2} турбино-часов"),
        ("Источник погоды", forecast.audit.get("weather_source", "не указан")),
        ("Политика истории", forecast.audit.get("history_policy", "не применимо")),
        ("Наблюдения до (исключая)", forecast.audit.get("observation_cutoff_exclusive", "не указано")),
        ("Лаг доступности погоды", str(forecast.audit.get("publication_lag_hours_assumption", "не указан")) + " ч (допущение)"),
        ("Checkpoint", forecast.audit.get("checkpoint", "не применимо")),
        ("SHA-256 источника", digest),
    ]
    svg_sections = []
    with plt.rc_context(STYLE), PdfPages(destination / "report.pdf", metadata={"Title": "ALEM WIND — научный отчёт", "Author": "ALEM WIND", "Subject": status}) as pdf:
        cover = plt.figure(figsize=(11.7, 8.3))
        cover.text(.075, .92, "ALEM WIND / НАУЧНЫЙ ОТЧЁТ", fontsize=20, weight="bold")
        cover.text(.075, .87, status, fontsize=11, color=COLORS[0])
        y = .81
        for key, value in provenance:
            line = f"{key}: {value}"
            for wrapped in textwrap.wrap(line, 100):
                cover.text(.075, y, wrapped, fontsize=9)
                y -= .026
        columns = ["Турбина", "n", "Средняя P\np.u.", "Пик P\np.u.", "Средний v\nм/с", "max |ΔP|\np.u./ч", "Факт n", "MAE\np.u.", "RMSE\np.u.", "Bias\np.u."]
        table_rows = [[f"T{j+1}", s["n_forecast"], f"{s['mean_power_pu']:.4f}", f"{s['peak_power_pu']:.4f}", f"{s['mean_wind_ms']:.2f}", f"{s['max_abs_ramp_pu_per_hour']:.4f}", s["n_observed"], "—" if s["mae_pu"] is None else f"{s['mae_pu']:.4f}", "—" if s["rmse_pu"] is None else f"{s['rmse_pu']:.4f}", "—" if s["bias_pu"] is None else f"{s['bias_pu']:.4f}"] for j, s in enumerate(stats)]
        ax = cover.add_axes([.075, .30, .85, .18])
        ax.axis("off")
        table = ax.table(cellText=table_rows, colLabels=columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        cover.text(.075, .24, "Оценка: " + (f"{observed_n} совпавших турбино-часов; неполная выборка возможна." if observed_n else "фактические данные недоступны; точность не оценена."), fontsize=10)
        cover.text(.075, .16, textwrap.fill(notes[0], 105), fontsize=9)
        cover.text(.075, .06, f"ID: {report_id} · значения округлены только для отображения", fontsize=8, color="#56636d")
        pdf.savefig(cover)
        plt.close(cover)
        if economics:
            business = plt.figure(figsize=(11.7, 8.3))
            business.text(.075, .92, "Lost Energy Revenue / СЦЕНАРИЙ", fontsize=18, weight="bold")
            y = .85
            for paragraph in economic_text:
                lines = textwrap.wrap(paragraph, 112)
                business.text(.075, y, "\n".join(lines), fontsize=10, va="top", linespacing=1.5)
                y -= .025 * len(lines) + .02
            business.text(.075, .035, "Почасовой расчёт: economics.csv. Входные параметры и допущения: economics.json.", fontsize=9)
            pdf.savefig(business)
            plt.close(business)
        for number, (fig, name, title, caption) in enumerate(figures(forecast, observed), 1):
            fig.suptitle(f"Рисунок {number}. {title}", x=.075, ha="left", y=.97, fontsize=15, weight="bold")
            fig.text(.075, .92, f"{status} · выпуск {forecast.origin:%d.%m.%Y %H:%M} UTC · горизонт {len(forecast.times)} ч", fontsize=9, color="#52606b")
            fig.tight_layout(rect=(.025, .13, .98, .89), h_pad=2)
            fig.text(.075, .045, textwrap.fill(caption, 115), fontsize=9)
            fig.savefig(destination / f"{name}.svg")
            fig.savefig(destination / f"{name}.png", dpi=300)
            pdf.savefig(fig)
            plt.close(fig)
            metadata["figures"].append({"id": name, "title": title, "caption": caption, "svg": f"{name}.svg", "png": f"{name}.png"})
            svg = (destination / f"{name}.svg").read_text(encoding="utf-8")
            svg = svg[svg.index("<svg"):]
            svg_sections.append(f'<figure>{svg}<figcaption>Рисунок {number}. {html.escape(caption)}</figcaption></figure>')
        method = plt.figure(figsize=(11.7, 8.3))
        method.text(.075, .91, "Методика, происхождение и ограничения", fontsize=18, weight="bold")
        y = .84
        for i, note in enumerate(notes, 1):
            lines = textwrap.wrap(f"{i}. {note}", 112)
            method.text(.075, y, "\n".join(lines), fontsize=10, va="top", linespacing=1.6)
            y -= .03 * len(lines) + .019
        method.text(.075, .10, "Справочный материал: NREL, Fundamentals of Wind Energy\nhttps://www.nrel.gov/docs/fy23osti/84501.pdf", fontsize=9)
        method.text(.075, .045, "Аудит источника и машинные значения приложены в metadata.json и forecast.csv.", fontsize=9)
        pdf.savefig(method)
        plt.close(method)
        normative = plt.figure(figsize=(11.7, 8.3))
        normative.text(.075, .92, "Нормативная основа ВЭС", fontsize=18, weight="bold")
        normative.text(.075, .87, standards["statement"], fontsize=10, color="#78591f")
        y = .80
        for item in standards["selected"]:
            normative.text(.075, y, item["designation"], fontsize=12, weight="bold")
            lines = textwrap.wrap(item["application"] + " " + item["implementation"], 114)
            y -= .035
            normative.text(.075, y, "\n".join(lines), fontsize=10, va="top", linespacing=1.5)
            y -= .025 * len(lines) + .035
        normative.text(.075, y, "Что требуется для дальнейшего применения", fontsize=12, weight="bold")
        y -= .04
        for item in standards["open_items"]:
            lines = textwrap.wrap("• " + item, 114)
            normative.text(.075, y, "\n".join(lines), fontsize=9, va="top", linespacing=1.4)
            y -= .025 * len(lines) + .015
        normative.text(.075, .075, "Области применения и ссылки проверены 23.09.2026. Перечень источников и статусы — в standards.json.", fontsize=8)
        normative.text(.075, .04, "Объект: ветроустановки. Классы и обозначения паровых турбин к этим данным не применяются.", fontsize=8)
        pdf.savefig(normative)
        plt.close(normative)
    cells = "".join("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>" for row in table_rows)
    provenance_html = "".join(f"<dt>{html.escape(str(k))}</dt><dd>{html.escape(str(v))}</dd>" for k, v in provenance)
    standards_html = "".join(
        f'<li><a href="{html.escape(item["source"], quote=True)}">{html.escape(item["designation"])}</a>: '
        f'{html.escape(item["application"])} {html.escape(item["implementation"])}</li>'
        for item in standards["selected"]
    )
    document = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ALEM WIND — научный отчёт</title>
<style>body{{font:15px/1.6 Arial,sans-serif;color:#172630;margin:40px auto;padding:0 24px;max-width:1100px}}h1{{font-size:30px}}.status{{color:#177565;font-weight:bold}}dl{{display:grid;grid-template-columns:240px 1fr;gap:8px}}dt{{font-weight:bold}}dd{{margin:0;overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border:1px solid #bac4ca;padding:10px}}th{{background:#f0f3f4}}figure{{margin:40px 0}}svg{{width:100%;height:auto}}figcaption{{font-size:13px;color:#475963}}li{{margin:10px 0}}@media print{{@page{{size:A4 landscape;margin:14mm}}body{{margin:0;max-width:none}}figure{{break-before:page;break-inside:avoid}}a.download{{display:none}}}}</style>
<h1>ALEM WIND / Научный отчёт</h1><p class="status">{status}</p><p><a class="download" href="report.pdf">PDF</a> · <a class="download" href="report_bundle.zip">Все файлы и графики</a></p><dl>{provenance_html}</dl>
<table><thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in columns)}</tr></thead><tbody>{cells}</tbody></table>
<p>Фактических совпавших турбино-часов: {observed_n}. {'Точность не оценена.' if not observed_n else 'Метрики рассчитаны только по совпавшим значениям.'}</p>
{'<h2>Lost Energy Revenue / Сценарий</h2>' if economics else ''}{''.join(f'<p>{html.escape(n)}</p>' for n in economic_text)}
{''.join(svg_sections)}<h2>Методика и ограничения</h2><ol>{''.join(f'<li>{html.escape(n)}</li>' for n in notes)}</ol>
<h2>Нормативная основа ВЭС</h2><p>{html.escape(standards['statement'])}</p><ul>{standards_html}</ul>
<p>Для дальнейшего применения:</p><ol>{''.join(f'<li>{html.escape(n)}</li>' for n in standards['open_items'])}</ol>
<p>Справочный материал: <a href="https://www.nrel.gov/docs/fy23osti/84501.pdf">NREL: Fundamentals of Wind Energy</a>.</p></html>'''
    (destination / "report.html").write_text(document, encoding="utf-8")
    (destination / "forecast.csv").write_text(payload, encoding="utf-8")
    if observed is not None:
        with (destination / "observations.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["valid_time", "turbine_id", "power_normalized"])
            for i, time in enumerate(forecast.times):
                for j, turbine in enumerate(IDS):
                    if np.isfinite(observed[i, j]):
                        writer.writerow([time.isoformat(), turbine, observed[i, j]])
    (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (destination / "standards.json").write_text(json.dumps(standards, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    names = ["report.pdf", "report.html", "forecast.csv", "metadata.json", "standards.json"]
    if economics:
        (destination / "economics.json").write_text(json.dumps(economics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (destination / "economics.csv").write_text(economics_csv(economics), encoding="utf-8")
        names += ["economics.json", "economics.csv"]
    names += [figure[extension] for figure in metadata["figures"] for extension in ("svg", "png")]
    if observed is not None:
        names.append("observations.csv")
    with zipfile.ZipFile(destination / "report_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        sums = []
        for name in names:
            content = (destination / name).read_bytes()
            archive.writestr(name, content)
            sums.append(f"{hashlib.sha256(content).hexdigest()}  {name}")
        archive.writestr("SHA256SUMS.txt", "\n".join(sums) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, required=True, help="Daily CSV with adjacent JSON audit")
    parser.add_argument("--horizon", type=int, choices=(24, 48))
    parser.add_argument("--observations", type=Path, help="UTC-hourly observed normalized power CSV")
    parser.add_argument("--output", type=Path, default=Path("outputs/reports"))
    args = parser.parse_args()
    forecast = load_forecast(args.forecast, args.horizon)
    observed = load_observations(args.observations, forecast) if args.observations else None
    print(make_report(forecast, args.output, observed))


if __name__ == "__main__":
    main()
