# ruff: noqa: RUF001  (Turkish UI text: dotless i is correct here)
"""Single-file HTML report: Jinja2 with autoescape, inline CSS and SVG, no external resources.

The wording is probabilistic on purpose (PRD §8, ADR-001): the label is a *sign* of
possible memorization ("ezber şüphesi"), never a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined

from ccrbench.report.models import ProblemCard, Report, ReportWarning, WarningCode
from ccrbench.scoring.axes import AxisStatus
from ccrbench.scoring.labels import Label

TEMPLATE_NAME = "report.html.j2"

# Plot geometry (SVG user units).
_WIDTH = 560.0
_HEIGHT = 400.0
_LEFT = 56.0
_RIGHT = 20.0
_TOP = 20.0
_BOTTOM = 48.0
_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)

LABEL_TEXT = {
    Label.SUSPECTED: "ezber şüphesi",
    Label.NOT_SUSPECTED: "işaret yok",
    Label.UNDETERMINED: "belirlenemedi",
}

_WARNING_TITLES = {
    WarningCode.CACHE_NOT_DISABLED: "Sağlayıcı önbelleği kapatılamadı",
    WarningCode.IGNORED_PARAMS: "Sağlayıcı bir parametreyi yok saydı",
    WarningCode.LOW_SPLIT_CONFIDENCE: "Adım bölme güveni düşük",
    WarningCode.METRIC_EXCLUDED: "Metrik çeşitlilik skoruna katılmadı",
    WarningCode.DIVERSITY_UNAVAILABLE: "Çeşitlilik ekseni hesaplanamadı, etiket verilmedi",
    WarningCode.VARIANT_AXIS_UNAVAILABLE: "Varyant ekseni hesaplanamadı, etiket verilmedi",
}
_WARNING_NOTES = {
    WarningCode.CACHE_NOT_DISABLED: (
        "Aynı istemlerin tekrarında sağlayıcı önbelleği çeşitliliği yapay olarak düşürmüş olabilir."
    ),
    WarningCode.IGNORED_PARAMS: (
        "Bu eksen uygulanamadı: ilgili varyasyon (ör. sıcaklık, seed) sağlayıcıda "
        "uygulanmadığı için çeşitlilik yalnız kalan varyasyonlardan gelir."
    ),
    WarningCode.LOW_SPLIT_CONFIDENCE: (
        "Transkriptlerde belirgin adım işareti yok; adım tabanlı metrikler kaba kalabilir."
    ),
    WarningCode.METRIC_EXCLUDED: (
        'Güveni sıfır olan metrik "düşük çeşitlilik" sayılmaz, ortalamanın dışında bırakılır.'
    ),
    WarningCode.DIVERSITY_UNAVAILABLE: (
        "Üç koşuldan biri değerlendirilemediği için işaret verilmez."
    ),
    WarningCode.VARIANT_AXIS_UNAVAILABLE: (
        "Üç koşuldan biri değerlendirilemediği için işaret verilmez."
    ),
}


@dataclass(frozen=True)
class _Point:
    x: str
    y: str
    diamond: str
    title: str
    label: Label


@dataclass(frozen=True)
class _Tick:
    position: str
    text: str
    label_x: str
    label_y: str


def _num(value: float) -> str:
    return f"{value:g}"


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _x(value: float) -> float:
    return _LEFT + value * (_WIDTH - _LEFT - _RIGHT)


def _y(value: float) -> float:
    return _TOP + (1.0 - value) * (_HEIGHT - _TOP - _BOTTOM)


def _coord(value: float) -> str:
    return f"{value:.1f}"


def _diamond(cx: float, cy: float, radius: float = 7.0) -> str:
    return (
        f"M {_coord(cx)} {_coord(cy - radius)} L {_coord(cx + radius)} {_coord(cy)} "
        f"L {_coord(cx)} {_coord(cy + radius)} L {_coord(cx - radius)} {_coord(cy)} Z"
    )


def _chart(report: Report) -> dict[str, Any]:
    thresholds = report.thresholds
    low_div = thresholds.low_diversity.value
    high_drop = thresholds.high_variant_drop.value
    points: list[_Point] = []
    not_plotted: list[str] = []
    for card in report.problems:
        div = card.diversity.score
        drop = card.variant_robustness.variant_drop
        if (
            card.diversity.status is AxisStatus.AVAILABLE
            and card.variant_robustness.status is AxisStatus.AVAILABLE
            and div is not None
            and drop is not None
        ):
            points.append(
                _Point(
                    x=_coord(_x(div)),
                    y=_coord(_y(drop)),
                    diamond=_diamond(_x(div), _y(drop)),
                    title=(
                        f"{card.problem_id}: çeşitlilik {_fmt(div)}, varyant düşüşü {_fmt(drop)}, "
                        f"{LABEL_TEXT[card.label.label]}"
                    ),
                    label=card.label.label,
                )
            )
        else:
            not_plotted.append(card.problem_id)
    return {
        "width": _num(_WIDTH),
        "height": _num(_HEIGHT),
        "plot_left": _coord(_x(0.0)),
        "plot_right": _coord(_x(1.0)),
        "plot_top": _coord(_y(1.0)),
        "plot_bottom": _coord(_y(0.0)),
        "plot_width": _coord(_x(1.0) - _x(0.0)),
        "plot_height": _coord(_y(0.0) - _y(1.0)),
        # Suspicion zone: low diversity and high variant drop.
        "zone_x": _coord(_x(0.0)),
        "zone_y": _coord(_y(1.0)),
        "zone_width": _coord(_x(low_div) - _x(0.0)),
        "zone_height": _coord(_y(high_drop) - _y(1.0)),
        "low_diversity_x": _coord(_x(low_div)),
        "high_drop_y": _coord(_y(high_drop)),
        "x_ticks": [
            _Tick(_coord(_x(t)), _num(t), _coord(_x(t)), _coord(_y(0.0) + 16.0)) for t in _TICKS
        ],
        "y_ticks": [
            _Tick(_coord(_y(t)), _num(t), _coord(_x(0.0) - 8.0), _coord(_y(t) + 4.0))
            for t in _TICKS
        ],
        "x_title_y": _coord(_HEIGHT - 8.0),
        "axis_label_x": _coord((_x(0.0) + _x(1.0)) / 2),
        "axis_label_y": _coord((_y(0.0) + _y(1.0)) / 2),
        "points": points,
        "not_plotted": not_plotted,
    }


def _card_notes(card: ProblemCard) -> dict[str, str | None]:
    """Turkish explanations for the unavailable axes, derived from the card's data.

    The model's ``unavailable_reason`` / ``reason`` strings are English (JSON consumers);
    the HTML words the same facts itself, like it does for the warnings.
    """
    diversity: str | None = None
    if card.diversity.status is AxisStatus.UNAVAILABLE:
        diversity = (
            "Bu probleme uygulanan metrik yok."
            if not card.diversity.metrics
            else "Tüm metriklerin güveni sıfır (çok az koşum ya da başarısız metrik)."
        )
    robustness = card.variant_robustness
    variant: str | None = None
    if robustness.status is AxisStatus.UNAVAILABLE:
        if robustness.original_runs == 0:
            variant = "Orijinal problemin koşumu yok."
        elif robustness.variants:
            variant = "Varyantlar var ama hiçbirinin koşumu yok."
        else:
            variant = "Çalıştırılan varyant yok (türetilemedi, atlandı ya da tanımlı değil)."
    label: str | None = None
    if card.label.label is Label.UNDETERMINED:
        conditions = card.label.conditions
        missing = []
        if conditions.low_diversity is None:
            missing.append("çeşitlilik ekseni hesaplanamadı")
        if conditions.high_accuracy is None or conditions.high_variant_drop is None:
            missing.append("varyant ekseni hesaplanamadı")
        label = "Etiket verilmedi: " + "; ".join(missing) + "."
    return {"label": label, "diversity": diversity, "variant": variant}


def _warning_view(warning: ReportWarning) -> dict[str, Any]:
    return {
        "title": _WARNING_TITLES[warning.code],
        "note": _WARNING_NOTES[warning.code],
        "detail": warning.detail,
        "problem_ids": warning.problem_ids,
    }


def _environment() -> Environment:
    env = Environment(
        loader=PackageLoader("ccrbench.report", "templates"),
        autoescape=True,  # the template name ends in .j2, so enable it unconditionally
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["fmt"] = _fmt
    env.filters["pct"] = _pct
    return env


def render_html(report: Report) -> str:
    """The whole report as one self-contained HTML document."""
    template = _environment().get_template(TEMPLATE_NAME)
    return template.render(
        report=report,
        chart=_chart(report),
        label_text={label.value: text for label, text in LABEL_TEXT.items()},
        warnings=[_warning_view(w) for w in report.warnings],
        notes={card.problem_id: _card_notes(card) for card in report.problems},
        generated_at=report.generated_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )


def write_html(report: Report, path: Path) -> Path:
    """Write the HTML report to ``path`` (parent directories are created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(report), encoding="utf-8")
    return path
