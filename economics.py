"""Scenario economics for a validated hourly forecast; no market orders or live prices."""
import csv
import hashlib
import io
import json
import math


def number(value, label, maximum, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label}: требуется конечное число")
    if not 0 <= value <= maximum or (positive and value == 0):
        raise ValueError(f"{label}: значение вне допустимого диапазона")
    return float(value)


def calculate_economics(forecast, settings):
    if not isinstance(settings, dict):
        raise ValueError("Укажите параметры экономического сценария")
    capacity = [number(settings.get(f"capacity_{i}_mw"), f"Мощность T{i}", 100_000, True) for i in (1, 2)]
    plan = number(settings.get("plan_mw"), "План отпуска", sum(capacity))
    tariff = number(settings.get("tariff_kzt_kwh"), "Тариф продажи", 1_000_000)
    tariff_note = settings.get("tariff_note")
    if not isinstance(tariff_note, str) or not tariff_note.strip() or len(tariff_note) > 160:
        raise ValueError("Укажите источник/дату тарифа или отметку «демо» (до 160 символов)")
    prices = [settings.get(key) for key in ("advance_kzt_kwh", "balancing_kzt_kwh")]
    if (prices[0] is None) != (prices[1] is None):
        raise ValueError("Для сравнения закупки нужны обе цены")
    if prices[0] is not None:
        prices = [number(p, "Цена закупки", 1_000_000) for p in prices]
    inputs = {"capacity_1_mw": capacity[0], "capacity_2_mw": capacity[1], "plan_mw": plan,
              "tariff_kzt_kwh": tariff, "tariff_note": tariff_note.strip(),
              "advance_kzt_kwh": prices[0], "balancing_kzt_kwh": prices[1]}
    rows = []
    for time, powers in zip(forecast.times, forecast.power, strict=True):
        power = sum(float(p) * c for p, c in zip(powers, capacity, strict=True))
        shortfall = max(plan - power, 0.0)
        rows.append({"valid_time": time.isoformat(), "forecast_mw": power, "plan_mw": plan,
                     "forecast_mwh": power, "shortfall_mwh": shortfall, "surplus_mwh": max(power - plan, 0.0),
                     "lost_energy_revenue_kzt": shortfall * 1000 * tariff})
    total = {name: sum(row[name] for row in rows) for name in
             ("forecast_mwh", "shortfall_mwh", "surplus_mwh", "lost_energy_revenue_kzt")}
    total["planned_mwh"] = plan * len(rows)
    total["shortfall_kwh"] = total["shortfall_mwh"] * 1000
    total["shortfall_hours"] = sum(row["shortfall_mwh"] > 1e-9 for row in rows)
    total["peak_shortfall_mw"] = max(row["shortfall_mwh"] for row in rows)
    comparison = None
    if prices[0] is not None:
        advance, balancing = [total["shortfall_kwh"] * p for p in prices]
        comparison = {"advance_cost_kzt": advance, "balancing_cost_kzt": balancing,
                      "potential_saving_kzt": balancing - advance}
    deficit = total["shortfall_mwh"]
    if deficit <= 1e-9:
        recommendation = "В этом сценарии прогноз покрывает заданный план во все часы. Оснований для закупки по этому расчёту нет."
    else:
        recommendation = (f"Прогнозируется недобор {deficit:,.2f} МВт·ч относительно заданного плана "
                          f"за {len(rows)} ч. Под риском {total['lost_energy_revenue_kzt']:,.0f} ₸ выручки. ")
        if comparison and comparison["potential_saving_kzt"] > 0:
            recommendation += (f"Рассмотрите предварительную закупку энергии для часов дефицита: при заданных ценах "
                               f"разница затрат составит {comparison['potential_saving_kzt']:,.0f} ₸. "
                               "Перед решением подтвердите котировку, почасовой объём и условия договора.")
        elif comparison:
            recommendation += "При заданных ценах предварительная закупка не дешевле балансирования. Согласуйте план покрытия с провайдером баланса."
        else:
            recommendation += "Запросите почасовые котировки на покрытие дефицита у провайдера баланса. Без двух цен экономия ранней закупки не определена."
    notes = [
        "Сценарная оценка, не фактический убыток и не подтверждённый дефицит энергосистемы. Охват: обе турбины.",
        "Допущение: 1 p.u. соответствует введённой номинальной мощности. База нормировки SCADA и номиналы требуют подтверждения.",
        "P = p.u.(T1) × номинал T1 + p.u.(T2) × номинал T2; E = P × 1 ч. План отпуска постоянный на весь горизонт.",
        "Недобор = сумма max(план − прогноз, 0) × 1 ч. Избыток в другом часу не компенсирует дефицит без накопителя.",
        "Lost Energy Revenue = недобор МВт·ч × 1000 × тариф ₸/кВт·ч. Это риск выручки относительно плана, не потеря относительно полной мощности.",
        "Цена продажи и цены закупки задаются пользователем без НДС; котировки не загружаются. Исторический прогноз оценивается по заданному сценарию тарифа.",
        "Разница затрат на закупку не складывается с Lost Energy Revenue. Комиссии, штрафы, потери сети, ограничения рынка и ошибка прогноза не моделируются.",
        "Ветер на высоте 10 м не доказывает штиль/шторм на роторе. Причина недобора не установлена; мощность прогноза повторно по ветру не обнуляется.",
    ]
    result = {"schema_version": 1, "status": "scenario", "forecast_origin": forecast.origin.isoformat(),
              "forecast_mode": forecast.audit["mode"], "horizon_hours": len(rows), "interval_hours": 1,
              "currency": "KZT", "vat_included": False, "live_market_prices": False, "order_placed": False,
              "inputs": inputs, "totals": total, "procurement": comparison, "recommendation": recommendation,
              "methodology": notes, "rows": rows}
    canonical = json.dumps({"result": result, "forecast_audit": forecast.audit}, sort_keys=True, ensure_ascii=False, allow_nan=False)
    result["scenario_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return result


def economics_csv(result):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(result["rows"][0]))
    writer.writeheader()
    writer.writerows(result["rows"])
    return output.getvalue()


def economics_summary(result):
    total, settings = result["totals"], result["inputs"]
    return [
        result["recommendation"],
        f"Прогноз: {total['forecast_mwh']:,.3f} МВт·ч; план: {total['planned_mwh']:,.3f} МВт·ч; недобор: {total['shortfall_mwh']:,.3f} МВт·ч.",
        f"Lost Energy Revenue: {total['lost_energy_revenue_kzt']:,.2f} ₸ без НДС. Тариф: {settings['tariff_kzt_kwh']:g} ₸/кВт·ч.",
        f"Номинал T1/T2: {settings['capacity_1_mw']:g}/{settings['capacity_2_mw']:g} МВт. Постоянный план: {settings['plan_mw']:g} МВт.",
        f"Источник/дата тарифа: {settings['tariff_note']}",
        result["methodology"][0], result["methodology"][1], result["methodology"][3], result["methodology"][6],
    ]
