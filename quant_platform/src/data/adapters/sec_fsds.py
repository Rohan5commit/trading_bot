from __future__ import annotations

import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import requests
from pydantic import BaseModel, Field

from ..utils import ensure_dir, update_progress

DEFAULT_TAGS = [
    "Assets",
    "AssetsCurrent",
    "Liabilities",
    "LiabilitiesCurrent",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "ShortTermBorrowings",
    "DebtCurrent",
    "AccountsReceivableNetCurrent",
    "InventoryNet",
    "AccountsPayableCurrent",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "IncomeLossFromContinuingOperations",
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
    "WeightedAverageNumberOfSharesOutstandingBasic",
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsOfDividends",
    "ShareBasedCompensation",
    "ResearchAndDevelopmentExpense",
    "SellingGeneralAndAdministrativeExpense",
    "OperatingExpenses",
]


def _snake_case(value: str) -> str:
    out = []
    for i, ch in enumerate(value):
        if ch.isupper() and i > 0 and (
            value[i - 1].islower() or (i + 1 < len(value) and value[i + 1].islower())
        ):
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


class SecFsdsConfig(BaseModel):
    enabled: bool = False
    years: List[int] = Field(default_factory=lambda: list(range(2010, 2025)))
    forms: List[str] = Field(default_factory=lambda: ["10-Q", "10-K"])
    tags: List[str] = Field(default_factory=lambda: list(DEFAULT_TAGS))
    uoms: List[str] = Field(default_factory=lambda: ["USD", "USD/shares", "shares"])
    raw_dir: str = "data/raw/sec_fsds"
    processed_dir: str = "data/processed/sec_fsds"
    refresh: bool = False
    download_pause: float = 0.2
    download_workers: int = 2
    chunk_size: int = 200_000


class SecFsdsAdapter:
    def __init__(self, config: SecFsdsConfig | None = None) -> None:
        self.config = config or SecFsdsConfig()
        self.session = requests.Session()

    def prepare(self) -> None:
        if not self.config.enabled:
            return
        raw_dir = Path(ensure_dir(self.config.raw_dir))
        processed_dir = Path(ensure_dir(self.config.processed_dir))
        progress_path = Path(self.config.processed_dir) / "progress.json"
        cik_map = self._load_cik_map()
        quarters = [(year, quarter) for year in self.config.years for quarter in range(1, 5)]
        total_quarters = len(quarters)
        to_process = []
        for year, quarter in quarters:
            processed_path = processed_dir / f"sec_fsds_{year}q{quarter}.parquet"
            if processed_path.exists() and not self.config.refresh:
                continue
            to_process.append((year, quarter))
        completed_quarters = total_quarters - len(to_process)
        downloaded = completed_quarters
        processed = completed_quarters
        update_progress(
            progress_path,
            {"fsds": {"status": "downloading", "total_quarters": total_quarters, "completed_quarters": completed_quarters, "downloaded_quarters": downloaded, "processed_quarters": processed}},
        )

        def _prep(year: int, quarter: int) -> None:
            zip_path = raw_dir / f"{year}q{quarter}.zip"
            extract_dir = raw_dir / f"{year}q{quarter}"
            if not zip_path.exists() or self.config.refresh:
                self._download_zip(year, quarter, zip_path)
            if not extract_dir.exists() or self.config.refresh:
                self._extract_zip(zip_path, extract_dir)

        if self.config.download_workers > 1 and len(to_process) > 1:
            with ThreadPoolExecutor(max_workers=self.config.download_workers) as ex:
                futures = [ex.submit(_prep, year, quarter) for year, quarter in to_process]
                for future in as_completed(futures):
                    future.result()
                    downloaded += 1
                    update_progress(progress_path, {"fsds": {"downloaded_quarters": downloaded}})
        else:
            for year, quarter in to_process:
                _prep(year, quarter)
                downloaded += 1
                update_progress(progress_path, {"fsds": {"downloaded_quarters": downloaded}})

        for year, quarter in to_process:
            self._process_quarter(year, quarter, raw_dir, cik_map)
            processed += 1
            update_progress(
                progress_path,
                {"fsds": {"status": "processing", "processed_quarters": processed}},
            )
        update_progress(progress_path, {"fsds": {"status": "done"}})


    def get_fsds_features(self, ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
        if not self.config.enabled:
            return pd.DataFrame()
        processed_dir = Path(self.config.processed_dir)
        if not processed_dir.exists():
            self.prepare()
        files = sorted(processed_dir.glob("sec_fsds_*q*.parquet"))
        if not files:
            self.prepare()
            files = sorted(processed_dir.glob("sec_fsds_*q*.parquet"))
        frames = []
        for path in files:
            try:
                df = pd.read_parquet(path, filters=[("ticker", "=", ticker)])
            except Exception:
                continue
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        if df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df.drop(columns=["ticker", "cik"], errors="ignore")
        df = df.set_index("date")
        df = df[~df.index.duplicated(keep="last")]
        df = self._add_derived(df)
        df = df.reindex(dates).ffill()
        return df

    def _add_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        revenue = self._pick_first(
            df,
            [
                "fsds_revenues",
                "fsds_sales_revenue_net",
                "fsds_revenue_from_contract_with_customer_excluding_assessed_tax",
            ],
        )
        if revenue is not None:
            df["fsds_revenue"] = revenue
            df["fsds_revenue_yoy"] = revenue.pct_change(4)
            df["fsds_revenue_qoq"] = revenue.pct_change(1)
        if "fsds_gross_profit" in df.columns and "fsds_revenue" in df.columns:
            df["fsds_gross_margin"] = df["fsds_gross_profit"] / df["fsds_revenue"].replace(0, pd.NA)
        if "fsds_operating_income_loss" in df.columns and "fsds_revenue" in df.columns:
            df["fsds_operating_margin"] = (
                df["fsds_operating_income_loss"] / df["fsds_revenue"].replace(0, pd.NA)
            )
        if "fsds_net_income_loss" in df.columns and "fsds_revenue" in df.columns:
            df["fsds_net_margin"] = df["fsds_net_income_loss"] / df["fsds_revenue"].replace(0, pd.NA)
        if (
            "fsds_net_cash_provided_by_used_in_operating_activities" in df.columns
            and "fsds_payments_to_acquire_property_plant_and_equipment" in df.columns
        ):
            df["fsds_free_cash_flow"] = (
                df["fsds_net_cash_provided_by_used_in_operating_activities"]
                - df["fsds_payments_to_acquire_property_plant_and_equipment"]
            )
        if "fsds_long_term_debt" in df.columns and "fsds_stockholders_equity" in df.columns:
            df["fsds_debt_to_equity"] = df["fsds_long_term_debt"] / df[
                "fsds_stockholders_equity"
            ].replace(0, pd.NA)
        if "fsds_assets_current" in df.columns and "fsds_liabilities_current" in df.columns:
            df["fsds_current_ratio"] = df["fsds_assets_current"] / df[
                "fsds_liabilities_current"
            ].replace(0, pd.NA)
        return df

    def _pick_first(self, df: pd.DataFrame, cols: List[str]) -> pd.Series | None:
        available = [c for c in cols if c in df.columns]
        if not available:
            return None
        return df[available].bfill(axis=1).iloc[:, 0]

    def _process_quarter(
        self, year: int, quarter: int, raw_dir: Path, cik_map: Dict[int, str]
    ) -> Path | None:
        processed_dir = ensure_dir(self.config.processed_dir)
        processed_path = Path(processed_dir) / f"sec_fsds_{year}q{quarter}.parquet"
        if processed_path.exists() and not self.config.refresh:
            return processed_path
        zip_path = raw_dir / f"{year}q{quarter}.zip"
        extract_dir = raw_dir / f"{year}q{quarter}"
        if not zip_path.exists() or self.config.refresh:
            self._download_zip(year, quarter, zip_path)
        if not extract_dir.exists() or self.config.refresh:
            self._extract_zip(zip_path, extract_dir)
        num_path = extract_dir / "num.txt"
        sub_path = extract_dir / "sub.txt"
        if not num_path.exists() or not sub_path.exists():
            return processed_path
        frame = self._build_quarter_frame(num_path, sub_path, cik_map)
        if frame.empty:
            return processed_path
        frame.to_parquet(processed_path, index=False)
        return processed_path

    def _build_quarter_frame(
        self, num_path: Path, sub_path: Path, cik_map: Dict[int, str]
    ) -> pd.DataFrame:
        sub = pd.read_csv(
            sub_path,
            sep="	",
            usecols=["adsh", "cik", "form", "period", "filed"],
            dtype={"adsh": "string", "cik": "int64", "form": "string", "period": "string", "filed": "string"},
            low_memory=False,
        )
        sub = sub[sub["form"].isin(self.config.forms)]
        if sub.empty:
            return pd.DataFrame()
        adsh_set = set(sub["adsh"].dropna().unique().tolist())
        tags = set(self.config.tags)
        chunks = []
        for chunk in pd.read_csv(
            num_path,
            sep="	",
            usecols=["adsh", "tag", "ddate", "uom", "value"],
            dtype={
                "adsh": "string",
                "tag": "string",
                "ddate": "string",
                "uom": "string",
                "value": "float64",
            },
            chunksize=self.config.chunk_size,
            low_memory=False,
        ):
            chunk = chunk[chunk["adsh"].isin(adsh_set)]
            if tags:
                chunk = chunk[chunk["tag"].isin(tags)]
            if self.config.uoms:
                chunk = chunk[chunk["uom"].isin(self.config.uoms)]
            if not chunk.empty:
                chunks.append(chunk)
        if not chunks:
            return pd.DataFrame()
        num = pd.concat(chunks, ignore_index=True)
        merged = num.merge(sub, on="adsh", how="left")
        merged["date"] = pd.to_datetime(merged["ddate"], format="%Y%m%d", errors="coerce")
        merged["filed"] = pd.to_datetime(merged["filed"], errors="coerce")
        merged = merged.dropna(subset=["date", "cik", "tag"])
        merged = merged.sort_values(["cik", "date", "tag", "filed"])
        merged = merged.drop_duplicates(subset=["cik", "date", "tag"], keep="last")
        pivot = merged.pivot_table(index=["cik", "date"], columns="tag", values="value", aggfunc="last")
        pivot = pivot.reset_index()
        filed = merged.groupby(["cik", "date"], as_index=False)["filed"].max()
        pivot = pivot.merge(filed, on=["cik", "date"], how="left")
        pivot["ticker"] = pivot["cik"].map(cik_map)
        pivot = pivot.dropna(subset=["ticker"])
        if "filed" in pivot.columns:
            pivot["fsds_filing_lag_days"] = (pivot["filed"] - pivot["date"]).dt.days
            pivot = pivot.drop(columns=["filed"])
        rename = {tag: f"fsds_{_snake_case(tag)}" for tag in tags if tag in pivot.columns}
        if rename:
            pivot = pivot.rename(columns=rename)
        return pivot

    def _download_zip(self, year: int, quarter: int, dest: Path) -> None:
        url = f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{year}q{quarter}.zip"
        resp = self._get(url, stream=True)
        ensure_dir(dest.parent)
        with open(dest, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        time.sleep(self.config.download_pause)

    def _extract_zip(self, zip_path: Path, dest_dir: Path) -> None:
        ensure_dir(dest_dir)
        with zipfile.ZipFile(zip_path) as zf:
            for name in ("num.txt", "sub.txt"):
                if name in zf.namelist():
                    zf.extract(name, dest_dir)

    def _load_cik_map(self) -> Dict[int, str]:
        resp = self._get("https://www.sec.gov/files/company_tickers.json", stream=False)
        data = resp.json()
        mapping: Dict[int, str] = {}
        for item in data.values():
            cik = item.get("cik_str")
            ticker = item.get("ticker")
            if cik is None or ticker is None:
                continue
            mapping[int(cik)] = str(ticker).upper()
        return mapping

    def _get(self, url: str, stream: bool = False) -> requests.Response:
        user_agent = os.getenv("SEC_USER_AGENT")
        if not user_agent:
            user_agent = "train-once-quant-platform/0.1 (Rohan; rohan.santhoshkumar1@gmail.com)"
        headers = {"User-Agent": user_agent}
        for attempt in range(5):
            resp = self.session.get(url, headers=headers, stream=stream, timeout=60)
            if resp.status_code in (429, 503):
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp
        resp.raise_for_status()
        return resp
