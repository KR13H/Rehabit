# trackers/march.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Any, Optional
import json
import os


__all__ = ["MarchTracker"]


def _ensure_march(d: date) -> None:
    if d.month != 3:
        raise ValueError(f"MarchTracker only accepts dates in March. Got {d.isoformat()}.")


@dataclass
class Entry:
    day: date
    habit: str
    value: Any = 1  # e.g., count/boolean/score
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day": self.day.isoformat(),
            "habit": self.habit,
            "value": self.value,
            "note": self.note,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Entry":
        return Entry(
            day=datetime.fromisoformat(d["day"]).date(),
            habit=d["habit"],
            value=d.get("value", 1),
            note=d.get("note", ""),
        )


@dataclass
class MarchTracker:
    """Simple habit/metric tracker scoped to March.

    Use:
        mt = MarchTracker()
        mt.add_entry(date(2025, 3, 1), "meditate", 1, "5 mins")
        mt.add_entry(date(2025, 3, 1), "steps", 6532)
        print(mt.summary())  # per-habit totals for March
    """

    year: int = field(default_factory=lambda: date.today().year)
    _entries: List[Entry] = field(default_factory=list)

    # ---- Core API ----
    def add_entry(
        self,
        day: date,
        habit: str,
        value: Any = 1,
        note: str = "",
    ) -> None:
        """Record a habit/value for a March day."""
        _ensure_march(day)
        self._entries.append(Entry(day=day, habit=habit, value=value, note=note))

    def get_entries(
        self,
        habit: Optional[str] = None,
        day: Optional[date] = None,
    ) -> List[Entry]:
        """Fetch entries, optionally filtered by habit and/or day."""
        def ok(e: Entry) -> bool:
            return (
                (habit is None or e.habit == habit) and
                (day is None or e.day == day)
            )
        return [e for e in self._entries if ok(e)]

    def remove_entries(self, habit: str, day: Optional[date] = None) -> int:
        """Remove entries for a habit (optionally only for a specific day). Returns count removed."""
        before = len(self._entries)
        self._entries = [
            e for e in self._entries
            if not (e.habit == habit and (day is None or e.day == day))
        ]
        return before - len(self._entries)

    # ---- Reports ----
    def summary(self) -> Dict[str, Any]:
        """Aggregate values per habit across March."""
        agg: Dict[str, Any] = {}
        for e in self._entries:
            if isinstance(e.value, (int, float)):
                agg[e.habit] = agg.get(e.habit, 0) + e.value
            else:
                # Non-numeric values are counted as occurrences
                agg[e.habit] = agg.get(e.habit, 0) + 1
        return agg

    def by_day(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return entries keyed by ISO day string."""
        out: Dict[str, List[Dict[str, Any]]] = {}
        for e in self._entries:
            out.setdefault(e.day.isoformat(), []).append(e.to_dict())
        return out

    # ---- Persistence ----
    def save(self, path: str) -> None:
        """Save tracker to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        blob = {
            "year": self.year,
            "entries": [e.to_dict() for e in self._entries],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "MarchTracker":
        """Load tracker from a JSON file. If missing, returns empty tracker."""
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        mt = cls(year=blob.get("year", date.today().year))
        mt._entries = [Entry.from_dict(d) for d in blob.get("entries", [])]
        return mt
