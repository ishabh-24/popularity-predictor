"""
Generate Hot 100 chart `date=` strings aligned with a year range (e.g. 2000–2019).

Billboard’s chart is weekly (typically dated by the chart week). We use **Saturdays**
in each period so `billboard.ChartData(..., date=...)` resolves to a valid week.

Modes:
- **yearly**: one Saturday per year (mid‑June) → matches the dataset era with few API calls
- **monthly**: first Saturday of each month → better coverage (~240 calls for 2000–2019)
- **weekly**: every Saturday in range → maximum coverage (~1040 calls; slow)
"""

from __future__ import annotations

from datetime import date, timedelta


def parse_year_range(s: str) -> tuple[int, int]:
    s = s.strip().replace(":", "-")
    if "-" not in s:
        raise ValueError(f"Expected YEAR_START-YEAR_END, got: {s!r}")
    a, b = s.split("-", 1)
    y0, y1 = int(a.strip()), int(b.strip())
    if y0 > y1:
        y0, y1 = y1, y0
    return y0, y1


def first_saturday_on_or_after(d: date) -> date:
    while d.weekday() != 5:  # Saturday
        d += timedelta(days=1)
    return d


def first_saturday_of_month(year: int, month: int) -> date:
    return first_saturday_on_or_after(date(year, month, 1))


def iter_saturdays_yearly(year_start: int, year_end: int) -> list[str]:
    """One chart week per year: first Saturday on or after June 15."""
    out: list[str] = []
    for year in range(year_start, year_end + 1):
        d = first_saturday_on_or_after(date(year, 6, 15))
        out.append(d.isoformat())
    return out


def iter_saturdays_monthly(year_start: int, year_end: int) -> list[str]:
    """First Saturday of each calendar month."""
    out: list[str] = []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            try:
                d = first_saturday_of_month(year, month)
            except ValueError:
                continue
            out.append(d.isoformat())
    return out


def iter_saturdays_weekly(year_start: int, year_end: int) -> list[str]:
    """Every Saturday from first Saturday on/after Jan 1 of start year through end year."""
    start = first_saturday_on_or_after(date(year_start, 1, 1))
    end = date(year_end, 12, 31)
    out: list[str] = []
    d = start
    while d <= end:
        out.append(d.isoformat())
        d += timedelta(days=7)
    return out


def billboard_dates_for_dataset_years(
    year_start: int,
    year_end: int,
    sample: str,
) -> list[str]:
    sample = sample.lower().strip()
    if sample == "yearly":
        return iter_saturdays_yearly(year_start, year_end)
    if sample == "monthly":
        return iter_saturdays_monthly(year_start, year_end)
    if sample == "weekly":
        return iter_saturdays_weekly(year_start, year_end)
    raise ValueError(f"sample must be yearly|monthly|weekly, got {sample!r}")
