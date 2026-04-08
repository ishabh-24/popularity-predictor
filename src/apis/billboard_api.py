from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

@dataclass(frozen=True)
class BillboardConfig:
    """
    `billboard.py` scrapes Billboard charts 
    It can be slower / sensitive to HTML changes, so we keep a wrapper for future caching.
    """
    chart_name: str = "hot-100"


class BillboardClient:
    def __init__(self, cfg: BillboardConfig | None = None):
        self.cfg = cfg or BillboardConfig()

    def get_chart(self, chart_date: str) -> list[dict[str, Any]]:
        """
        Fetch a specific chart snapshot by date (YYYY-MM-DD).

        Returns a list of normalized entries:
        - rank
        - track_name
        - artist_name
        - weeks_on_chart
        - peak_rank
        - last_week_rank
        - chart_date
        """
        try:
            import billboard
        except Exception as e:
            raise RuntimeError("Missing dependency for Billboard client. Install `billboard.py`.") from e

        chart = billboard.ChartData(self.cfg.chart_name, date=chart_date)
        out: list[dict[str, Any]] = []

        for entry in chart:
            out.append(
                {
                    "chart_name": self.cfg.chart_name,
                    "chart_date": chart_date,
                    "rank": entry.rank,
                    "track_name": entry.title,
                    "artist_name": entry.artist,
                    "weeks_on_chart": entry.weeks,
                    "peak_rank": entry.peakPos,
                    "last_week_rank": entry.lastPos,
                }
            )

        return out


def main():
    client = BillboardClient()

    test_date = "2008-02-08"  # known valid chart date
    print(f"Fetching Billboard chart for {test_date}...")

    data = client.get_chart(test_date)

    print(f"\nTotal entries fetched: {len(data)}")

    if data:
        print("\nFirst entry:")
        first = data[0]
        for k, v in first.items():
            print(f"{k}: {v}")

        print("\nTop 5 songs:")
        for entry in data[:5]:
            print(f"{entry['rank']}. {entry['track_name']} - {entry['artist_name']}")
    else:
        print("No data returned.")


if __name__ == "__main__":
    main()