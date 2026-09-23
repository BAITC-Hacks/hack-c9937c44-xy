"""Reviewed scope of standards for ALEM WIND, not a conformity certificate.

The project forecasts wind-turbine power. Measurement standards do not certify
forecast accuracy, and selecting a report standard does not complete its layout.
"""
from __future__ import annotations


def wind_standards_profile() -> dict:
    """Return a fresh, versioned profile with explicitly limited implementation status."""
    return {
        "profile_version": "wind-references-2026-09-23-v1",
        "reviewed_on": "2026-09-23",
        "equipment": "wind_turbine",
        "report_purpose": "hourly_power_forecast_analysis",
        "conformity_claim": False,
        "conformity_status": "not_assessed",
        "statement": "Подобраны нормативные ориентиры для ВЭС. Полное соответствие стандартам не установлено.",
        "selected": [
            {
                "designation": "ГОСТ 7.32-2017",
                "title": "Отчёт о научно-исследовательской работе. Структура и правила оформления",
                "role": "research_report_format",
                "status": "reference_selected_layout_not_fully_implemented",
                "application": "Основа для структуры и оформления отчёта о НИР.",
                "implementation": "Текущий PDF — аналитический отчёт. Полная вёрстка и реквизиты отчёта о НИР ещё не реализованы и не проверены.",
                "source": "https://protect.gost.ru/gost/details/7d280e43-7036-4a69-8e6e-d15867028343",
                "kazakhstan_catalog_status": "listed_as_active",
                "kazakhstan_source": "https://new-shop.ksm.kz/catalog/?PAGEN_1=346&arrFilter_pf%5BCATEGORY%5D=&page=360&set_filter=Y&view=cards",
            },
            {
                "designation": "IEC 61400-12-1:2022 + COR1:2025",
                "title": "Измерение энергетических характеристик ветроустановок",
                "role": "power_performance_measurements",
                "status": "reference_selected_measurements_not_performed",
                "application": "Методический ориентир для будущих измерений характеристики отдельной ВЭУ и оценки неопределённости.",
                "implementation": "Испытания по IEC не выполнялись. Прогнозные точки P(v) и MAE/RMSE модели не заменяют измеренную кривую и бюджет неопределённости.",
                "source": "https://webstore.iec.ch/en/publication/68499",
                "corrigendum_source": "https://webstore.iec.ch/en/publication/106400",
                "kazakhstan_adoption": "not_verified",
            },
        ],
        "related_reference": {
            "designation": "ГОСТ Р 54418.12.1-2011 (МЭК 61400-12-1:2005)",
            "role": "russian_national_reference_only",
            "note": "Российский национальный документ на основе редакции IEC 2005 года; не подменяет IEC 2022 и не назначен обязательным для Казахстана.",
            "source": "https://protect.gost.ru/gost/details/15493e9c-75f3-4c7d-80fa-042f908fe2b3",
        },
        "excluded": [
            {
                "designation": "ГОСТ 3618-2016",
                "reason": "Область — стационарные паровые турбины до 50 МВт; оборудование проекта — ветроустановки.",
                "source": "https://protect.gost.ru/gost/details/07e4e8d9-b4b7-4059-9140-50dfb72bfbe3",
            },
            {
                "designation": "ГОСТ 24278-2016",
                "reason": "Область — стационарные паротурбинные установки ТЭС 50–1600 МВт; к ВЭС проекта не относится.",
                "source": "https://protect.gost.ru/gost/details/06e66c55-66ea-4d7c-83c3-e20cd141a1a0",
            },
        ],
        "open_items": [
            "Подтвердить применяемую в Казахстане редакцию стандарта испытаний и требования заказчика.",
            "Подготовить полный шаблон НИР по ГОСТ 7.32: структуру, вёрстку и организационные реквизиты; выполнить нормоконтроль.",
            "Для измерительной программы получить паспорт ВЭУ, номинальную мощность, геометрию и высоту ступицы.",
            "Получить измерения ветра и мощности, сведения о средствах измерений, калибровке, площадке и влияющих условиях.",
            "Разработать и выполнить программу испытаний с оценкой неопределённости по полному тексту выбранной редакции IEC.",
        ],
    }
