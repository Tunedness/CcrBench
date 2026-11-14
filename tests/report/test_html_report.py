# ruff: noqa: RUF001  (Turkish UI text: dotless i is correct here)
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from ccrbench.report import Report, render_html, write_html
from tests.report import SCRIPT_PROMPT, build_scenario

SNAPSHOT = Path(__file__).parent / "snapshots" / "report.html"


@pytest.fixture
def report(tmp_path: Path) -> Report:
    return build_scenario(tmp_path)[1]


@pytest.fixture
def html(report: Report) -> str:
    return render_html(report)


def test_html_matches_the_snapshot(html: str) -> None:
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.write_text(html, encoding="utf-8")
    assert SNAPSHOT.exists(), "snapshot missing: run with UPDATE_SNAPSHOTS=1"
    expected = SNAPSHOT.read_text(encoding="utf-8")
    assert html == expected, "HTML changed: review the diff, then refresh with UPDATE_SNAPSHOTS=1"


def test_prompt_markup_is_escaped(html: str) -> None:
    assert "<script" not in html.lower()
    assert "<b>bold</b>" not in html
    assert "&lt;script&gt;alert(&#34;x&#34;)&lt;/script&gt;" in html
    assert "&amp; explain" in html
    assert SCRIPT_PROMPT not in html


def test_suite_and_model_names_are_escaped(report: Report) -> None:
    hostile = report.model_copy(
        update={
            "suite": report.suite.model_copy(update={"name": "<img src=x onerror=alert(1)>"}),
            "model": "fake:<i>m</i>",
        }
    )
    page = render_html(hostile)
    assert "<img src=x" not in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page
    assert "<i>m</i>" not in page


def test_page_is_self_contained(html: str) -> None:
    assert not re.search(r"https?://", html)
    assert not re.search(r"<link\b|@import|url\(|<script\b|<iframe\b|\bsrc=", html, re.IGNORECASE)
    assert html.count("<style>") == 1
    assert "<svg" in html


def test_light_and_dark_themes_are_defined(html: str) -> None:
    assert "prefers-color-scheme: dark" in html
    assert "color-scheme: light dark" in html


def test_wording_is_probabilistic(html: str) -> None:
    lowered = html.lower()
    assert "ezber şüphesi" in lowered
    assert "işaret" in lowered
    assert "ezberlemiş" not in lowered
    assert "ezberliyor" not in lowered
    assert "olasılıksal" in lowered


def test_page_shows_every_problem_warning_and_source(html: str, report: Report) -> None:
    for card in report.problems:
        assert f"<code>{card.problem_id}</code>" in html
    assert "Bu eksen uygulanamadı" in html
    assert "Varyant ekseni hesaplanamadı, etiket verilmedi" in html
    assert "no code block to rename" in html
    assert "test-embed-v1" in html
    assert f"{report.suite.hash}/fake__memorize/memo/run-0.json" in html


def test_chart_marks_the_suspect_zone_and_plots_only_complete_points(html: str) -> None:
    assert 'class="zone"' in html
    # memo (suspected diamond) + reason, cached, script (circles); novariant is not plotted
    assert html.count('class="pt-suspected"') == 1
    assert html.count("<circle") == 3
    assert "Çizilemeyen problemler" in html
    assert re.search(r"Çizilemeyen problemler[^<]*<code>novariant</code>", html)


def test_unavailable_reasons_are_shown_in_turkish(html: str) -> None:
    assert "Etiket verilmedi: varyant ekseni hesaplanamadı." in html
    assert "Çalıştırılan varyant yok" in html
    for english in ("no label given", "no variant was run", "variant axis unavailable"):
        assert english not in html


def test_templates_load_through_the_package_loader() -> None:
    from ccrbench.report import html_report

    loader = html_report._environment().loader
    assert type(loader).__name__ == "PackageLoader"


def test_write_html_creates_parents(report: Report, tmp_path: Path) -> None:
    target = write_html(report, tmp_path / "out" / "report.html")
    assert target.read_text(encoding="utf-8") == render_html(report)
