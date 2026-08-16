"""Self-contained HTML run report (PNGs embedded as base64)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from jinja2 import Template

_TEMPLATE = Template(
    """<!doctype html>
<html><head><meta charset="utf-8"><title>{{ title }}</title>
<style>
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px;
       color: #222; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
table { border-collapse: collapse; margin: 1rem 0; }
td, th { border: 1px solid #ccc; padding: 0.35rem 0.7rem; font-size: 0.9rem; }
th { background: #f2f2f2; text-align: left; }
img { max-width: 100%; margin: 0.5rem 0; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
pre { background: #f7f7f7; padding: 0.8rem; overflow-x: auto; font-size: 0.8rem; }
</style></head><body>
<h1>{{ title }}</h1>
<h2>Summary metrics</h2>
<table><tr><th>metric</th><th>value</th></tr>
{% for k, v in summary.items() %}<tr><td>{{ k }}</td><td>{{ "%.4g"|format(v) }}</td></tr>
{% endfor %}</table>
{% if figures %}<h2>Figures</h2><div class="grid">
{% for name, b64 in figures %}<figure><img src="data:image/{{ 'gif' if name.endswith('.gif')
 else 'png' }};base64,{{ b64 }}" alt="{{ name }}"><figcaption>{{ name }}</figcaption></figure>
{% endfor %}</div>{% endif %}
<h2>Configuration</h2><pre>{{ config_json }}</pre>
</body></html>"""
)


def write_report(
    out_path: str | Path,
    title: str,
    summary: dict[str, float],
    figure_paths: list[Path],
    config: dict[str, Any],
) -> Path:
    figures = []
    for fp in figure_paths:
        if fp.exists():
            figures.append((fp.name, base64.b64encode(fp.read_bytes()).decode("ascii")))
    html = _TEMPLATE.render(
        title=title,
        summary=summary,
        figures=figures,
        config_json=json.dumps(config, indent=2, default=str),
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
