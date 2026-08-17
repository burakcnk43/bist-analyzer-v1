from jinja2 import Template

REPORT_TEMPLATE = """
# FIRST RESEARCH REPORT - BCBIST ML Research Engine
**Generated on:** {{ timestamp }}

## 1. Executive Summary
- **Stocks Analyzed:** {{ stocks_analyzed }}
- **Total Samples:** {{ total_samples }}
- **Sectors Covered:** {{ sectors | join(', ') }}
- **Regimes Identified:** {{ regimes | join(', ') }}

## 2. Top Global Predictive Patterns
These patterns are derived from all stocks in the universe.

| Feature | Bin | Count | Mean Return (5d) | Positive Rate | CI (Mean) |
|---------|-----|-------|------------------|---------------|-----------|
{% for p in global_patterns %}
| {{ p.feature }} | {{ p.bin }} | {{ p.sample_count }} | {{ "%.2f%%"|format(p.mean_forward_return * 100) }} | {{ "%.2f%%"|format(p.positive_rate * 100) }} | [{{ "%.2f%%"|format(p.mean_ci_lower*100) }}, {{ "%.2f%%"|format(p.mean_ci_upper*100) }}] |
{% endfor %}

## 3. Sector-Specific Discoveries
Top patterns for major sectors.

{% for sector, patterns in sector_analysis.items() %}
### Sector: {{ sector }}
| Feature | Bin | Mean Return (5d) | Positive Rate |
|---------|-----|------------------|---------------|
{% for p in patterns %}
| {{ p.feature }} | {{ p.bin }} | {{ "%.2f%%"|format(p.mean_forward_return * 100) }} | {{ "%.2f%%"|format(p.positive_rate * 100) }} |
{% endfor %}
{% endfor %}

## 4. Regime Analysis
How relationships change under different market conditions.

{% for regime, patterns in regime_analysis.items() %}
### Regime: {{ regime }}
| Feature | Bin | Mean Return (5d) | Positive Rate |
|---------|-----|------------------|---------------|
{% for p in patterns %}
| {{ p.feature }} | {{ p.bin }} | {{ "%.2f%%"|format(p.mean_forward_return * 100) }} | {{ "%.2f%%"|format(p.positive_rate * 100) }} |
{% endfor %}
{% endfor %}

## 5. Feature Stability (Out-of-Sample)
Consistency of features across walk-forward folds.

| Feature | Bin | Avg Effect | Fold Count | Effect Std |
|---------|-----|------------|------------|------------|
{% for feat, stabs in stability_analysis.items() %}
{% for s in stabs %}
| {{ feat }} | {{ s.bin }} | {{ "%.2f%%"|format(s.avg_effect * 100) }} | {{ s.fold_count }} | {{ "%.4f"|format(s.effect_std) }} |
{% endfor %}
{% endfor %}

## 6. Research Conclusions & Limitations
- **Strongest Signals:** Identified in Global Analysis.
- **Stability:** Refer to Section 5 for fold-level consistency.
- **Limitations:** Fundamental and News data were not included in this cycle (Market-only).
- **Leakage Audit:** Purged walk-forward ensures out-of-sample validity.

> [!CAUTION]
> All findings are historical associations and do not imply causal relationships or guarantee future performance.
"""

def generate_markdown_report(data: dict) -> str:
    template = Template(REPORT_TEMPLATE)
    return template.render(**data)
