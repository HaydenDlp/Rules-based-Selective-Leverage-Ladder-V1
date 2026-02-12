# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 19:16:54 2026

@author: user
"""

import os
import time
import numpy as np
import pandas as pd
import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed

CSV_PATH   = r"\your_file.csv"
START_DATE = "1957-01-02"

NUM_WINDOWS = 10000
SEED = 123

N_WORKERS = max(1, (os.cpu_count() or 4) - 1)
CHUNKSIZE = 25

TAX_RATE = 0.25
DO_TLH = True
TLH_TRIGGER = -0.10
TLH_COOLDOWN_DAYS = 31

MAX_LEVER_FRAC = 0.50
MAX_SSO_FRAC   = 0.50

CASH_REDEPLOY_DD = -0.10
INITIAL = 10000.0


df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df = df[df["date"] >= pd.to_datetime(START_DATE)].copy().reset_index(drop=True)

df["ret_sp"] = df["return"].fillna(0.0)

df["idx_1x"] = (1.0 + df["ret_sp"]).cumprod()
df["idx_2x"] = (1.0 + 2.0 * df["ret_sp"]).cumprod()
df["idx_3x"] = (1.0 + 3.0 * df["ret_sp"]).cumprod()

dates_all = df["date"].values

if len(df) < 3000:
    print(f"Warning: Only {len(df)} rows after START_DATE={START_DATE}. "
          "Monte Carlo windows may fail or be limited.")


def max_drawdown(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    peaks = np.maximum.accumulate(arr)
    dd = (arr / peaks) - 1.0
    return float(dd.min())


def decile_medians_and_mean(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    xs = np.sort(x)
    n = xs.size
    k = max(1, int(np.floor(0.10 * n)))
    bottom = xs[:k]
    top = xs[-k:]
    return float(np.median(bottom)), float(np.median(xs)), float(np.mean(xs)), float(np.median(top))


def chunker(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]


class LotPositionHIFO:
    def __init__(self):
        self.lots = []

    def shares(self) -> float:
        return float(sum(sh for sh, _c in self.lots))

    def value(self, price: float) -> float:
        return float(self.shares() * price)

    def _insert_lot_desc(self, shares: float, cost: float):
        keys = [-lot[1] for lot in self.lots]
        idx = bisect.bisect_left(keys, -cost)
        self.lots.insert(idx, [shares, cost])

    def buy_dollars(self, dollars: float, price: float):
        if dollars <= 0 or price <= 0:
            return 0.0
        sh = float(dollars / price)
        self._insert_lot_desc(sh, float(price))
        return sh

    def sell_dollars_hifo(self, dollars: float, price: float):
        if dollars <= 0 or price <= 0:
            return 0.0, 0.0
        return self.sell_shares_hifo(float(dollars / price), float(price))

    def sell_shares_hifo(self, shares_to_sell: float, price: float):
        if shares_to_sell <= 0 or price <= 0:
            return 0.0, 0.0

        proceeds = 0.0
        realized = 0.0
        remaining = shares_to_sell

        i = 0
        while remaining > 1e-14 and i < len(self.lots):
            lot_sh, lot_cost = self.lots[i]
            sell_sh = min(lot_sh, remaining)

            proceeds_piece = sell_sh * price
            basis_piece = sell_sh * lot_cost
            realized_piece = proceeds_piece - basis_piece

            proceeds += proceeds_piece
            realized += realized_piece

            lot_sh -= sell_sh
            remaining -= sell_sh

            if lot_sh <= 1e-14:
                self.lots.pop(i)
            else:
                self.lots[i][0] = lot_sh
                i += 1

        return float(proceeds), float(realized)


class TaxLedger:
    def __init__(self, tax_rate=0.25):
        self.tax_rate = float(tax_rate)
        self.carry_loss = 0.0

        self.gain_unlev = 0.0
        self.loss_unlev = 0.0
        self.gain_lev = 0.0
        self.loss_lev = 0.0

        self.taxes_paid_total = 0.0

    def record_realized(self, sleeve: str, realized_pl: float):
        if abs(realized_pl) < 1e-14:
            return
        if sleeve == "unlev":
            if realized_pl > 0:
                self.gain_unlev += realized_pl
            else:
                self.loss_unlev += -realized_pl
        elif sleeve == "lev":
            if realized_pl > 0:
                self.gain_lev += realized_pl
            else:
                self.loss_lev += -realized_pl
        else:
            raise ValueError("sleeve must be 'unlev' or 'lev'")

    def compute_tax_due(self):
        gains = self.gain_unlev + self.gain_lev
        losses = self.loss_unlev + self.loss_lev
        net = gains - losses - self.carry_loss
        tax_due = self.tax_rate * net if net > 0 else 0.0
        return float(tax_due), float(gains), float(losses), float(net)

    def finalize_year(self):
        tax_due, gains, losses, net = self.compute_tax_due()
        carry_end = 0.0 if net > 0 else -net

        self.carry_loss = carry_end
        self.gain_unlev = self.loss_unlev = 0.0
        self.gain_lev = self.loss_lev = 0.0


STRATEGIES = [
    "SPY_ONLY",
    "SSO_BH",
    "LADDER_1_2",
    "LADDER_2_3",
    "LADDER_3_4",
    "SSO_LADDER_1_2",
    "SSO_LADDER_2_3",
    "SSO_LADDER_3_4",
    "SSO_LADDER_4_5",
]


def sso_only_rungs(strategy_name: str):
    shallow_dds = [-0.20, -0.25, -0.30, -0.35]
    deep_dds    = [-0.40, -0.45, -0.50, -0.55, -0.60]

    if strategy_name == "SSO_LADDER_1_2":
        a, b = 0.01, 0.02
    elif strategy_name == "SSO_LADDER_2_3":
        a, b = 0.02, 0.03
    elif strategy_name == "SSO_LADDER_3_4":
        a, b = 0.03, 0.04
    elif strategy_name == "SSO_LADDER_4_5":
        a, b = 0.04, 0.05
    else:
        return []
    return [(thr, a) for thr in shallow_dds] + [(thr, b) for thr in deep_dds]


def mixed_ladder_rungs(strategy_name: str):
    sso_dds  = [-0.20, -0.25, -0.30, -0.35]
    upro_dds = [-0.40, -0.45, -0.50, -0.55]

    if strategy_name == "LADDER_1_2":
        sso_pct, upro_pct = 0.01, 0.02
    elif strategy_name == "LADDER_2_3":
        sso_pct, upro_pct = 0.02, 0.03
    elif strategy_name == "LADDER_3_4":
        sso_pct, upro_pct = 0.03, 0.04
    else:
        return [], []

    sso_rungs  = [(thr, sso_pct) for thr in sso_dds]
    upro_rungs = [(thr, upro_pct) for thr in upro_dds]
    return sso_rungs, upro_rungs


def run_strategy_on_window(sub: pd.DataFrame, strategy_name: str):
    sub = sub.copy().reset_index(drop=True)

    sub["idx_1n"] = sub["idx_1x"] / sub["idx_1x"].iloc[0]
    sub["idx_2n"] = sub["idx_2x"] / sub["idx_2x"].iloc[0]
    sub["idx_3n"] = sub["idx_3x"] / sub["idx_3x"].iloc[0]

    spy  = LotPositionHIFO()
    sso  = LotPositionHIFO()
    upro = LotPositionHIFO()
    cash = float(INITIAL)

    ledger = TaxLedger(tax_rate=TAX_RATE)

    years_with_tax_due = 0
    tax_years_count = 0
    realized_gains_total = 0.0
    realized_losses_total = 0.0

    last_tlh_ordinal = {
        "unlev": -10**18,
        "sso":   -10**18,
        "upro":  -10**18,
    }

    def record_realized(sleeve: str, realized_pl: float):
        nonlocal realized_gains_total, realized_losses_total
        ledger.record_realized(sleeve, realized_pl)
        if realized_pl > 0:
            realized_gains_total += realized_pl
        elif realized_pl < 0:
            realized_losses_total += -realized_pl

    idx1_0 = float(sub["idx_1n"].iloc[0])
    idx2_0 = float(sub["idx_2n"].iloc[0])

    if strategy_name == "SPY_ONLY":
        spy.buy_dollars(cash, idx1_0)
        cash = 0.0
    elif strategy_name == "SSO_BH":
        sso.buy_dollars(cash, idx2_0)
        cash = 0.0
    else:
        spy.buy_dollars(cash, idx1_0)
        cash = 0.0

    is_sso_only = strategy_name.startswith("SSO_LADDER_")
    is_mixed    = strategy_name.startswith("LADDER_")

    if is_sso_only:
        rungs_sso_only = sso_only_rungs(strategy_name)
        rung_fired_sso = {thr: False for thr, _ in rungs_sso_only}

    if is_mixed:
        sso_rungs, upro_rungs = mixed_ladder_rungs(strategy_name)
        rung_state = {("SSO", thr): False for thr, _ in sso_rungs}
        rung_state.update({("UPRO", thr): False for thr, _ in upro_rungs})

    last_ath_price = float(sub["SP500"].iloc[0])
    totals = []
    prev_year = int(sub["date"].iloc[0].year)

    def vals(idx1, idx2, idx3):
        spy_val  = spy.value(idx1)
        sso_val  = sso.value(idx2)
        upro_val = upro.value(idx3)
        total = spy_val + sso_val + upro_val + cash
        return spy_val, sso_val, upro_val, total

    def lever_room(total, sso_val, upro_val):
        max_lev_val = MAX_LEVER_FRAC * total
        lev_val = sso_val + upro_val
        return max(0.0, max_lev_val - lev_val)

    def sso_room(total, sso_val):
        max_sso_val = MAX_SSO_FRAC * total
        return max(0.0, max_sso_val - sso_val)

    def tlh_harvest_underwater_lots(pos: LotPositionHIFO, price: float, sleeve: str) -> bool:
        nonlocal cash
        if (not DO_TLH) or price <= 0 or not pos.lots:
            return False

        trigger = abs(float(TLH_TRIGGER))
        if trigger <= 0 or trigger >= 1:
            return False

        threshold_cost = price / (1.0 - trigger)

        proceeds_total = 0.0
        realized_total = 0.0
        harvested_any = False

        while pos.lots and pos.lots[0][1] >= threshold_cost:
            harvested_any = True
            lot_sh, lot_cost = pos.lots.pop(0)
            proceeds = lot_sh * price
            basis = lot_sh * lot_cost
            realized = proceeds - basis
            proceeds_total += proceeds
            realized_total += realized

        if not harvested_any or proceeds_total <= 1e-14:
            return False

        cash += proceeds_total
        record_realized(sleeve, realized_total)

        buy_amt = min(cash, proceeds_total)
        pos.buy_dollars(buy_amt, price)
        cash -= buy_amt

        return True

    def reinvest_leftover_cash(idx1: float, idx2: float):
        nonlocal cash
        if cash <= 1e-12:
            return
        if strategy_name == "SSO_BH":
            sso.buy_dollars(cash, idx2)
        else:
            spy.buy_dollars(cash, idx1)
        cash = 0.0

    def settle_tax_pay_on_first_day_of_new_year(idx1: float, idx2: float):
        nonlocal cash, years_with_tax_due, tax_years_count

        tax_due, _, _, _ = ledger.compute_tax_due()

        tax_years_count += 1
        if tax_due > 1e-12:
            years_with_tax_due += 1

        ledger.finalize_year()

        if tax_due <= 1e-12:
            reinvest_leftover_cash(idx1, idx2)
            return

        need = max(0.0, tax_due - cash)
        if need > 1e-12:
            if strategy_name != "SSO_BH":
                proceeds, realized = spy.sell_dollars_hifo(need, idx1)
                cash += proceeds
                record_realized("unlev", realized)
            else:
                proceeds, realized = sso.sell_dollars_hifo(need, idx2)
                cash += proceeds
                record_realized("lev", realized)

        pay = min(cash, tax_due)
        cash -= pay
        ledger.taxes_paid_total += pay

        reinvest_leftover_cash(idx1, idx2)

    def settle_tax_terminal_same_day(idx1: float, idx2: float):
        nonlocal cash, years_with_tax_due, tax_years_count

        tax_years_count += 1

        for _ in range(60):
            tax_due, _, _, _ = ledger.compute_tax_due()
            if tax_due <= 1e-12:
                break

            need = max(0.0, tax_due - cash)
            if need <= 1e-12:
                break

            if strategy_name != "SSO_BH":
                proceeds, realized = spy.sell_dollars_hifo(need, idx1)
                if proceeds <= 1e-12:
                    break
                cash += proceeds
                record_realized("unlev", realized)
            else:
                proceeds, realized = sso.sell_dollars_hifo(need, idx2)
                if proceeds <= 1e-12:
                    break
                cash += proceeds
                record_realized("lev", realized)

        tax_due, _, _, _ = ledger.compute_tax_due()
        if tax_due > 1e-12:
            years_with_tax_due += 1

        pay = min(cash, tax_due)
        cash -= pay
        ledger.taxes_paid_total += pay

        ledger.finalize_year()
        reinvest_leftover_cash(idx1, idx2)

    for row in sub.itertuples(index=False):
        date = row.date
        year = int(pd.Timestamp(date).year)

        sp_price = float(row.SP500)
        idx1 = float(row.idx_1n)
        idx2 = float(row.idx_2n)
        idx3 = float(row.idx_3n)

        if year != prev_year:
            settle_tax_pay_on_first_day_of_new_year(idx1, idx2)
            prev_year = year

        if DO_TLH:
            ord_today = pd.Timestamp(date).to_pydatetime().date().toordinal()

            if (ord_today - last_tlh_ordinal["unlev"]) >= TLH_COOLDOWN_DAYS:
                if tlh_harvest_underwater_lots(spy, idx1, "unlev"):
                    last_tlh_ordinal["unlev"] = ord_today

            if (ord_today - last_tlh_ordinal["sso"]) >= TLH_COOLDOWN_DAYS:
                if tlh_harvest_underwater_lots(sso, idx2, "lev"):
                    last_tlh_ordinal["sso"] = ord_today

            if (ord_today - last_tlh_ordinal["upro"]) >= TLH_COOLDOWN_DAYS:
                if tlh_harvest_underwater_lots(upro, idx3, "lev"):
                    last_tlh_ordinal["upro"] = ord_today

        if sp_price > last_ath_price:
            last_ath_price = sp_price
            if is_sso_only:
                for thr in rung_fired_sso:
                    rung_fired_sso[thr] = False
            if is_mixed:
                for k in rung_state:
                    rung_state[k] = False

            if cash > 1e-12 and strategy_name != "SSO_BH":
                spy.buy_dollars(cash, idx1)
                cash = 0.0

        dd = (sp_price - last_ath_price) / last_ath_price

        if cash > 1e-12 and dd >= CASH_REDEPLOY_DD and strategy_name != "SSO_BH":
            spy.buy_dollars(cash, idx1)
            cash = 0.0

        spy_val, sso_val, upro_val, total = vals(idx1, idx2, idx3)
        if total <= 0:
            totals.append(0.0)
            continue

        if is_sso_only:
            for thr, pct in rungs_sso_only:
                if rung_fired_sso[thr] or dd > thr:
                    continue

                spy_val, sso_val, upro_val, total = vals(idx1, idx2, idx3)
                room = sso_room(total, sso_val)
                move_val = min(pct * total, room, spy_val)

                if move_val <= 0:
                    rung_fired_sso[thr] = True
                    continue

                proceeds, realized = spy.sell_dollars_hifo(move_val, idx1)
                cash += proceeds
                record_realized("unlev", realized)

                buy_amt = min(cash, proceeds)
                if buy_amt > 0:
                    sso.buy_dollars(buy_amt, idx2)
                    cash -= buy_amt

                rung_fired_sso[thr] = True

        elif is_mixed:
            for thr, pct in sso_rungs:
                key = ("SSO", thr)
                if rung_state[key] or dd > thr:
                    continue

                spy_val, sso_val, upro_val, total = vals(idx1, idx2, idx3)
                avail = lever_room(total, sso_val, upro_val)
                move_val = min(pct * total, avail, spy_val)

                if move_val <= 0:
                    rung_state[key] = True
                    continue

                proceeds, realized = spy.sell_dollars_hifo(move_val, idx1)
                cash += proceeds
                record_realized("unlev", realized)

                buy_amt = min(cash, proceeds)
                if buy_amt > 0:
                    sso.buy_dollars(buy_amt, idx2)
                    cash -= buy_amt

                rung_state[key] = True

            for thr, pct in upro_rungs:
                key = ("UPRO", thr)
                if rung_state[key] or dd > thr:
                    continue

                spy_val, sso_val, upro_val, total = vals(idx1, idx2, idx3)
                avail = lever_room(total, sso_val, upro_val)
                move_val = min(pct * total, avail, spy_val)

                if move_val <= 0:
                    rung_state[key] = True
                    continue

                proceeds, realized = spy.sell_dollars_hifo(move_val, idx1)
                cash += proceeds
                record_realized("unlev", realized)

                buy_amt = min(cash, proceeds)
                if buy_amt > 0:
                    upro.buy_dollars(buy_amt, idx3)
                    cash -= buy_amt

                rung_state[key] = True

        spy_val, sso_val, upro_val, total = vals(idx1, idx2, idx3)
        totals.append(total)

    last_idx1 = float(sub["idx_1n"].iloc[-1])
    last_idx2 = float(sub["idx_2n"].iloc[-1])
    settle_tax_terminal_same_day(last_idx1, last_idx2)

    totals = np.asarray(totals, dtype=float)

    last_idx3 = float(sub["idx_3n"].iloc[-1])
    spy_val, sso_val, upro_val, total = vals(last_idx1, last_idx2, last_idx3)

    years = (sub["date"].iloc[-1] - sub["date"].iloc[0]).days / 365.25
    years = max(years, 1e-9)
    cagr = (total / INITIAL) ** (1.0 / years) - 1.0

    w_spy  = spy_val  / total if total > 0 else 0.0
    w_sso  = sso_val  / total if total > 0 else 0.0
    w_upro = upro_val / total if total > 0 else 0.0
    w_cash = cash     / total if total > 0 else 0.0

    pct_years_tax_due = (years_with_tax_due / tax_years_count) if tax_years_count > 0 else 0.0

    return {
        "strategy": strategy_name,
        "cagr": float(cagr),
        "final_total": float(total),
        "dd": float(max_drawdown(totals)),
        "taxes_paid_total": float(ledger.taxes_paid_total),

        "end_spy_val": float(spy_val),
        "end_sso_val": float(sso_val),
        "end_upro_val": float(upro_val),
        "end_cash": float(cash),
        "end_spy_w": float(w_spy),
        "end_sso_w": float(w_sso),
        "end_upro_w": float(w_upro),
        "end_cash_w": float(w_cash),

        "pct_years_tax_due": float(pct_years_tax_due),
        "carry_loss_end": float(ledger.carry_loss),
        "realized_gains_total": float(realized_gains_total),
        "realized_losses_total": float(realized_losses_total),
        "tax_years_count": int(tax_years_count),
    }


def generate_windows(num_windows: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(df)
    windows = []

    for _ in range(num_windows):
        ok = False
        for _try in range(500):
            start_idx = int(rng.integers(0, n - 1))
            start_date = dates_all[start_idx]

            target_years = float(rng.uniform(15.0, 30.0))
            horizon_days = int(target_years * 365.25)
            end_date = start_date + np.timedelta64(horizon_days, "D")

            end_idx = int(np.searchsorted(dates_all, end_date, side="right") - 1)

            if not (0 <= end_idx < n and end_idx > start_idx + 252):
                continue

            actual_years = (dates_all[end_idx] - dates_all[start_idx]) / np.timedelta64(1, "D") / 365.25
            if actual_years < 15.0:
                continue

            windows.append((start_idx, end_idx))
            ok = True
            break

        if not ok:
            pass

    return windows


def run_window(start_end):
    s_idx, e_idx = start_end
    sub = df.iloc[s_idx:e_idx + 1].copy()
    out = []
    for strat in STRATEGIES:
        out.append(run_strategy_on_window(sub, strat))
    return out


def run_chunk(window_chunk):
    out = []
    for w in window_chunk:
        out.extend(run_window(w))
    return out


def run_monte_carlo_parallel(num_windows: int, seed: int, print_progress: bool = True):
    windows = generate_windows(num_windows, seed)
    rows = []

    total_chunks = (len(windows) + CHUNKSIZE - 1) // CHUNKSIZE
    done_chunks = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = []
        for chunk in chunker(windows, CHUNKSIZE):
            futures.append(ex.submit(run_chunk, chunk))

        for fut in as_completed(futures):
            rows.extend(fut.result())
            done_chunks += 1

            if print_progress and total_chunks > 0:
                step = max(1, total_chunks // 20)
                if done_chunks % step == 0 or done_chunks == total_chunks:
                    elapsed = time.time() - t0
                    pct = done_chunks / total_chunks
                    eta = (elapsed / pct) - elapsed if pct > 0 else float("inf")
                    print(f"Progress: {pct*100:5.1f}% | elapsed {elapsed/60:6.1f} min | ETA {eta/60:6.1f} min")

    return pd.DataFrame(rows)


def summarize_strategy(df_res: pd.DataFrame) -> pd.DataFrame:
    out = []

    perf_metrics = [
        ("cagr", "CAGR"),
        ("final_total", "Final($)"),
        ("dd", "MaxDD"),
        ("taxes_paid_total", "TaxesPaid($)"),
    ]

    alloc_metrics = [
        ("end_spy_val", "EndSPY($)"),
        ("end_sso_val", "EndSSO($)"),
        ("end_upro_val", "EndUPRO($)"),
        ("end_cash", "EndCash($)"),
        ("end_spy_w", "EndSPY(%)"),
        ("end_sso_w", "EndSSO(%)"),
        ("end_upro_w", "EndUPRO(%)"),
        ("end_cash_w", "EndCash(%)"),
    ]

    tax_diag_metrics = [
        ("pct_years_tax_due", "PctYearsTaxDue"),
        ("carry_loss_end", "CarryLossEnd($)"),
        ("realized_gains_total", "RealizedGains($)"),
        ("realized_losses_total", "RealizedLosses($)"),
        ("tax_years_count", "TaxYearsCount"),
    ]

    for strat, g in df_res.groupby("strategy"):
        row = {"strategy": strat, "trials": int(len(g))}

        for col, name in perf_metrics:
            b10, med, mean, t10 = decile_medians_and_mean(g[col].values)
            row[f"{name}_bot10_med"] = b10
            row[f"{name}_median"] = med
            row[f"{name}_mean"] = mean
            row[f"{name}_top10_med"] = t10

        for col, name in alloc_metrics:
            v = g[col].astype(float).values
            row[f"{name}_median"] = float(np.median(v))
            row[f"{name}_mean"] = float(np.mean(v))

        for col, name in tax_diag_metrics:
            v = g[col].astype(float).values
            row[f"{name}_median"] = float(np.median(v))
            row[f"{name}_mean"] = float(np.mean(v))

        out.append(row)

    return pd.DataFrame(out).sort_values("strategy").reset_index(drop=True)


def print_tables(summary: pd.DataFrame):
    perf_cols = [
        "strategy","trials",
        "CAGR_bot10_med","CAGR_median","CAGR_mean","CAGR_top10_med",
        "Final($)_bot10_med","Final($)_median","Final($)_mean","Final($)_top10_med",
        "MaxDD_bot10_med","MaxDD_median","MaxDD_mean","MaxDD_top10_med",
        "TaxesPaid($)_bot10_med","TaxesPaid($)_median","TaxesPaid($)_mean","TaxesPaid($)_top10_med",
    ]
    perf = summary[perf_cols].copy()

    for c in ["CAGR_bot10_med","CAGR_median","CAGR_mean","CAGR_top10_med"]:
        perf[c] = (100 * perf[c]).map(lambda v: f"{v:6.2f}%")
    for c in ["MaxDD_bot10_med","MaxDD_median","MaxDD_mean","MaxDD_top10_med"]:
        perf[c] = (100 * perf[c]).map(lambda v: f"{v:7.2f}%")
    for c in ["Final($)_bot10_med","Final($)_median","Final($)_mean","Final($)_top10_med",
              "TaxesPaid($)_bot10_med","TaxesPaid($)_median","TaxesPaid($)_mean","TaxesPaid($)_top10_med"]:
        perf[c] = perf[c].map(lambda v: f"${v:,.2f}")

    print("\n=== PERFORMANCE (bottom10 median / median / mean / top10 median) ===")
    print(perf.to_string(index=False))

    alloc_cols = [
        "strategy",
        "EndSPY($)_median","EndSPY($)_mean",
        "EndSSO($)_median","EndSSO($)_mean",
        "EndUPRO($)_median","EndUPRO($)_mean",
        "EndCash($)_median","EndCash($)_mean",
        "EndSPY(%)_median","EndSPY(%)_mean",
        "EndSSO(%)_median","EndSSO(%)_mean",
        "EndUPRO(%)_median","EndUPRO(%)_mean",
        "EndCash(%)_median","EndCash(%)_mean",
    ]
    alloc = summary[alloc_cols].copy()

    for c in ["EndSPY($)_median","EndSPY($)_mean","EndSSO($)_median","EndSSO($)_mean",
              "EndUPRO($)_median","EndUPRO($)_mean","EndCash($)_median","EndCash($)_mean"]:
        alloc[c] = alloc[c].map(lambda v: f"${v:,.2f}")
    for c in ["EndSPY(%)_median","EndSPY(%)_mean","EndSSO(%)_median","EndSSO(%)_mean",
              "EndUPRO(%)_median","EndUPRO(%)_mean","EndCash(%)_median","EndCash(%)_mean"]:
        alloc[c] = (100 * alloc[c]).map(lambda v: f"{v:7.2f}%")

    print("\n=== ENDING ALLOCATIONS (median and mean) ===")
    print(alloc.to_string(index=False))

    tax_cols = [
        "strategy",
        "PctYearsTaxDue_median","PctYearsTaxDue_mean",
        "CarryLossEnd($)_median","CarryLossEnd($)_mean",
        "RealizedGains($)_median","RealizedGains($)_mean",
        "RealizedLosses($)_median","RealizedLosses($)_mean",
        "TaxYearsCount_median","TaxYearsCount_mean",
    ]
    tax = summary[tax_cols].copy()

    tax["PctYearsTaxDue_median"] = (100 * tax["PctYearsTaxDue_median"]).map(lambda v: f"{v:7.2f}%")
    tax["PctYearsTaxDue_mean"]   = (100 * tax["PctYearsTaxDue_mean"]).map(lambda v: f"{v:7.2f}%")

    for c in ["CarryLossEnd($)_median","CarryLossEnd($)_mean",
              "RealizedGains($)_median","RealizedGains($)_mean",
              "RealizedLosses($)_median","RealizedLosses($)_mean"]:
        tax[c] = tax[c].map(lambda v: f"${v:,.2f}")

    tax["TaxYearsCount_median"] = tax["TaxYearsCount_median"].map(lambda v: f"{v:6.1f}")
    tax["TaxYearsCount_mean"]   = tax["TaxYearsCount_mean"].map(lambda v: f"{v:6.1f}")

    print("\n=== TAX DIAGNOSTICS (median and mean) ===")
    print(tax.to_string(index=False))


if __name__ == "__main__":
    print(f"START_DATE={START_DATE} | windows={NUM_WINDOWS} | workers={N_WORKERS} | chunksize={CHUNKSIZE}")
    print(f"MAX_LEVER_FRAC={MAX_LEVER_FRAC} | MAX_SSO_FRAC={MAX_SSO_FRAC} | "
          f"TLH={DO_TLH} @ {TLH_TRIGGER*100:.0f}% | TLH_COOLDOWN={TLH_COOLDOWN_DAYS}d | "
          f"TAX_RATE={TAX_RATE*100:.0f}%")

    df_res = run_monte_carlo_parallel(NUM_WINDOWS, SEED, print_progress=True)
    summary = summarize_strategy(df_res)
    print_tables(summary)
