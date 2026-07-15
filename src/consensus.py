"""
consensus.py — the S&OP consensus layer of PlanningCopilot (PRD Section 9, FR-5/FR-6/FR-8).

WHY THIS EXISTS:
forecast_engine.py produces a baseline number. The PRD's S&OP process (Section 9)
requires Sales, Marketing, and Product to submit adjustments to that baseline
WITH documented rationale (FR-5), reconciled into a single consensus number
signed off by the S&OP lead (FR-6), with the whole trail retained for later
accuracy review (FR-8). This module is the deterministic, no-LLM bookkeeping
for that process — it never touches the forecast model itself, it only records
who changed the number, by how much, and why, then does the arithmetic to
combine baseline + adjustments into a consensus total.

Persistence is two flat CSVs under data/ (sop_adjustments.csv, sop_signoffs.csv)
rather than an in-memory/session store, specifically so the audit trail survives
across app restarts — a planner adjustment made today must still be there next
week when actuals come in and someone wants to know why the number moved.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import pandas as pd

FUNCTIONS = ["Sales", "Marketing", "Product"]

ADJUSTMENTS_COLUMNS = ["cycle", "sku_id", "function", "delta_units", "rationale", "submitted_at"]
SIGNOFFS_COLUMNS = ["cycle", "sku_id", "signed_off_at", "signed_off_by"]


@dataclass
class ConsensusResult:
    cycle: str
    sku_id: str
    baseline_total: float
    adjustments: list  # list of dicts: function, delta_units, rationale, submitted_at
    consensus_total: float
    is_signed_off: bool
    signed_off_at: Optional[str]
    signed_off_by: Optional[str]


class ConsensusStore:
    """
    Reads/writes the two CSVs backing the S&OP consensus workflow.

    cycle: a free-form label for the forecast cycle being reconciled, e.g.
           "2026-07" — matches the PRD's monthly cadence, one cycle per month.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.adjustments_path = self.data_dir / "sop_adjustments.csv"
        self.signoffs_path = self.data_dir / "sop_signoffs.csv"

    @staticmethod
    def _read(path: Path, columns: list) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame(columns=columns)

    def add_adjustment(self, cycle: str, sku_id: str, function: str, delta_units: float, rationale: str) -> None:
        """
        Records one function's adjustment to the baseline, with rationale.
        Refuses a missing rationale (FR-5 requires it) and refuses to add
        to a cycle/SKU that's already been signed off (FR-6 — sign-off means
        the number is final until the next cycle).
        """
        if function not in FUNCTIONS:
            raise ValueError(f"Unknown function: {function!r}. Must be one of {FUNCTIONS}")
        if not rationale or not rationale.strip():
            raise ValueError("Rationale is required for every adjustment.")
        if self.is_signed_off(cycle, sku_id) is not None:
            raise ValueError(
                f"Cycle {cycle!r} for {sku_id} is already signed off — adjustments are locked."
            )

        df = self._read(self.adjustments_path, ADJUSTMENTS_COLUMNS)
        new_row = pd.DataFrame([{
            "cycle": cycle,
            "sku_id": sku_id,
            "function": function,
            "delta_units": float(delta_units),
            "rationale": rationale.strip(),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.adjustments_path, index=False)

    def get_adjustments(self, cycle: str, sku_id: str) -> pd.DataFrame:
        df = self._read(self.adjustments_path, ADJUSTMENTS_COLUMNS)
        if df.empty:
            return df
        return df[(df["cycle"] == cycle) & (df["sku_id"] == sku_id)].reset_index(drop=True)

    def is_signed_off(self, cycle: str, sku_id: str) -> Optional[dict]:
        df = self._read(self.signoffs_path, SIGNOFFS_COLUMNS)
        if df.empty:
            return None
        match = df[(df["cycle"] == cycle) & (df["sku_id"] == sku_id)]
        if match.empty:
            return None
        row = match.iloc[-1]
        return {"signed_off_at": row["signed_off_at"], "signed_off_by": row["signed_off_by"]}

    def sign_off(self, cycle: str, sku_id: str, signed_off_by: str = "S&OP Lead") -> None:
        """Marks a cycle/SKU's consensus as final. Idempotent — re-signing just
        appends a new record and is_signed_off() reports the latest one."""
        df = self._read(self.signoffs_path, SIGNOFFS_COLUMNS)
        new_row = pd.DataFrame([{
            "cycle": cycle,
            "sku_id": sku_id,
            "signed_off_at": datetime.now(timezone.utc).isoformat(),
            "signed_off_by": signed_off_by,
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.signoffs_path, index=False)

    def get_consensus(self, cycle: str, sku_id: str, baseline_total: float) -> ConsensusResult:
        """
        Reconciles the baseline with every recorded adjustment for this
        cycle/SKU into a single consensus number (PRD Section 9, step 4-5).
        Reconciliation here is a straight sum of deltas — the PRD's S&OP
        meeting is where humans negotiate any conflicting adjustments down
        to the numbers actually recorded here.
        """
        adj_df = self.get_adjustments(cycle, sku_id)
        adjustments = adj_df.to_dict("records")
        total_delta = float(adj_df["delta_units"].sum()) if not adj_df.empty else 0.0
        consensus_total = round(baseline_total + total_delta, 1)

        signoff = self.is_signed_off(cycle, sku_id)

        return ConsensusResult(
            cycle=cycle,
            sku_id=sku_id,
            baseline_total=round(baseline_total, 1),
            adjustments=adjustments,
            consensus_total=consensus_total,
            is_signed_off=signoff is not None,
            signed_off_at=signoff["signed_off_at"] if signoff else None,
            signed_off_by=signoff["signed_off_by"] if signoff else None,
        )


if __name__ == "__main__":
    # Quick smoke test — no API keys or model needed, same as the other modules'.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = ConsensusStore(data_dir=tmp)
        cycle, sku = "2026-07", "SKU-1003"

        store.add_adjustment(cycle, sku, "Sales", 120, "Regional account confirmed a bulk reorder.")
        store.add_adjustment(cycle, sku, "Marketing", -40, "Planned promo pulled forward to June.")

        result = store.get_consensus(cycle, sku, baseline_total=1000.0)
        print(f"Baseline: {result.baseline_total}  Consensus: {result.consensus_total}")
        for adj in result.adjustments:
            print(f"  {adj['function']:10s} {adj['delta_units']:+.0f}  {adj['rationale']}")

        store.sign_off(cycle, sku, signed_off_by="Jane (S&OP Lead)")
        result = store.get_consensus(cycle, sku, baseline_total=1000.0)
        print(f"Signed off: {result.is_signed_off} by {result.signed_off_by} at {result.signed_off_at}")
