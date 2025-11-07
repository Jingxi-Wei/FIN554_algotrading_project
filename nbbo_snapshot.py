#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import pyreadr
from typing import Optional
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
REQUIRED_COLS = [
    'date','time_m','sym_root','sym_suffix','ex','bid','bidsiz','ask','asksiz',
    'best_bidex','best_bid','best_bidsiz','best_askex','best_ask','best_asksiz',
    'qu_cond','qu_seqnum','natbbo_ind'
]


def _list_rda_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in {'.rda', '.rdata', '.rdata'}:
        return [path]
    if path.is_dir():
        files = []
        files += sorted(path.glob("*.rda"))
        files += sorted(path.glob("*.RData"))
        files += sorted(path.glob("*.rdata"))
        return files
    raise FileNotFoundError(f"Path not found: {path}")


def load_nbbo_frames(path: Path, object_name: str = "nbbo") -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    rda_files = _list_rda_files(path)
    if not rda_files:
        raise FileNotFoundError(f"No .rda/.RData files found in {path}")
    for fp in rda_files:
        res = pyreadr.read_r(str(fp))
        if object_name not in res:
            raise KeyError(f"{fp} does not contain object '{object_name}'")
        df = res[object_name]
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise KeyError(f"{fp} missing columns: {missing}")
        frames.append(df)
    return frames


def clean_and_enrich(df: pd.DataFrame, symbol: str = "AAPL") -> pd.DataFrame:
    cols_needed = [
        'date', 'time_m', 'sym_root',
        'best_bid', 'best_bidsiz', 'best_ask', 'best_asksiz'
    ]
    df = df[cols_needed].copy()

    # Filter symbol
    df = df[df['sym_root'].astype(str) == symbol]

    # Build timestamp: ts = to_datetime(date) + to_timedelta(time_m, 's')
    date_dt = pd.to_datetime(df['date'], errors='coerce', utc=False)
    time_td = pd.to_timedelta(pd.to_numeric(df['time_m'], errors='coerce'), unit='s')
    df['ts'] = date_dt + time_td
    df = df.dropna(subset=['ts'])

    # Coerce numerics
    for col in ['best_bid', 'best_ask', 'best_bidsiz', 'best_asksiz']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Valid NBBO: best_bid < best_ask and sizes > 0
    df = df[
        (df['best_bid'] < df['best_ask']) &
        (df['best_bidsiz'] > 0) &
        (df['best_asksiz'] > 0)
    ].copy()

    # Derived fields
    df['mid'] = (df['best_bid'] + df['best_ask']) / 2.0
    df['spread'] = df['best_ask'] - df['best_bid']
    df['micro'] = (
        (df['best_ask'] * df['best_bidsiz'] + df['best_bid'] * df['best_asksiz']) /
        (df['best_bidsiz'] + df['best_asksiz'])
    )

    # Ensure time order for .last()
    df = df.sort_values('ts')
    return df


class AlphaCalculator:
    """Alpha formula container.

    Implement alphas as `alpha_<name>(df, **ctx) -> pd.Series` returning RAW series.
    Use `score(raw)` to transform raw alpha into a bounded score.
    """



    # ----- Utilities -----
    @staticmethod
    def _rolling_std(s: pd.Series, w: int) -> pd.Series:
        sd = s.rolling(w, min_periods=max(10, w // 6)).std().shift(1)
        return sd.mask((sd.isna()) | (sd < 1e-12), 1e-12)

    @staticmethod
    def score(raw: pd.Series, std: Union[pd.Series, float, int], k: float = 3.0) -> pd.Series:
        if not isinstance(std, pd.Series):
            std = pd.Series(float(std), index=raw.index)
        std = std.mask((std.isna()) | (std < 1e-12), 1e-12)
        return np.tanh(raw / (k * std))

    # ----- Precompute context -----
    def _ctx(self, df: pd.DataFrame, *, window: int, tick: float, jump_ticks: int) -> dict:
        mid = df['mid']
        spread = df['spread']
        bid = df.get('best_bid', mid)
        ask = df.get('best_ask', mid)
        bidsiz = df.get('best_bidsiz', pd.Series(0.0, index=df.index))
        asksiz = df.get('best_asksiz', pd.Series(0.0, index=df.index))
        total_sz = (bidsiz + asksiz).replace(0, np.nan)
        ret = mid.pct_change()
        dmid = mid.diff()

        spread_ma = spread.rolling(window).mean().shift(1)

        weights = (bidsiz + asksiz).fillna(0.0)
        vwap_num = (mid * weights).rolling(window).sum().shift(1)
        vwap_den = weights.rolling(window).sum().shift(1).replace(0, np.nan)
        vwap_past = (vwap_num / vwap_den).fillna(method='bfill')

        quotes = df.get('quote_count', pd.Series(0.0, index=df.index))
        quotes_w = quotes.rolling(window).sum().shift(1).fillna(0.0)
        quote_arrival_rate = quotes_w / float(window)
        quote_lifetime = (window / quotes_w.replace(0.0, np.nan)).fillna(method='bfill')

        dir1 = np.sign(dmid)
        next_dir_up = (dir1.shift(-1) > 0).astype(float)
        down_now = (dir1 < 0).astype(float)
        trans_num = (next_dir_up * down_now).rolling(window).sum().shift(1)
        trans_den = down_now.rolling(window).sum().shift(1).replace(0, np.nan)
        ba_trans_prob = (trans_num / trans_den).fillna(0.0)

        jump = ((dmid.abs() >= (jump_ticks * tick)) & (dir1 == dir1.shift(1)) & (dir1 != 0)).astype(float)
        price_jump_intensity = jump.rolling(window).sum().shift(1).fillna(0.0)

        dq_bid = bidsiz.diff()
        dq_ask = asksiz.diff()
        dp_bid = bid.diff()
        dp_ask = ask.diff()
        ofi_raw = dq_bid.where(dp_bid >= 0, 0.0) - dq_ask.where(dp_ask <= 0, 0.0)

        return dict(
            mid=mid, spread=spread, bid=bid, ask=ask,
            bidsiz=bidsiz, asksiz=asksiz, total_sz=total_sz,
            ret=ret, dmid=dmid, spread_ma=spread_ma,
            vwap_past=vwap_past,
            quote_arrival_rate=quote_arrival_rate, quote_lifetime=quote_lifetime,
            ba_trans_prob=ba_trans_prob,
            price_jump_intensity=price_jump_intensity,
            dq_bid=dq_bid, dq_ask=dq_ask,
            ofi_raw=ofi_raw,
        )

    # ----- Alpha definitions (RAW) -----
    def alpha_mid_pct(self, df_or_mid, *, window: int) -> pd.Series:
        if isinstance(df_or_mid, pd.Series):
            mid = df_or_mid
        else:
            mid = df_or_mid['mid']
        return mid.pct_change().fillna(0.0)

    def alpha_spread_dev(self, df: pd.DataFrame, *, window: int) -> pd.Series:
        ctx = self._ctx(df, window=window, tick=0.01, jump_ticks=1)
        base = (ctx['spread'] - ctx['spread_ma']) / ctx['spread_ma'].replace(0, np.nan)
        return base.fillna(0.0)

    def alpha_eff_spread(self, df: pd.DataFrame, *, window: int) -> pd.Series:
        ctx = self._ctx(df, window=window, tick=0.01, jump_ticks=1)
        return (ctx['spread'] / ctx['mid'].replace(0, np.nan)).fillna(0.0)

    def alpha_price_dist_vwap(self, df: pd.DataFrame, *, window: int) -> pd.Series:
        ctx = self._ctx(df, window=window, tick=0.01, jump_ticks=1)
        return ((ctx['mid'] - ctx['vwap_past']) / ctx['spread'].replace(0, np.nan)).fillna(0.0)

    def alpha_liq_imbalance(self, df: pd.DataFrame, *, window: int) -> pd.Series:
        ctx = self._ctx(df, window=window, tick=0.01, jump_ticks=1)
        return ((ctx['bidsiz'] - ctx['asksiz']) / (ctx['bidsiz'] + ctx['asksiz']).replace(0, np.nan)).fillna(0.0)

    def alpha_depth_ratio(self, df: pd.DataFrame, *, window: int) -> pd.Series:
        ctx = self._ctx(df, window=window, tick=0.01, jump_ticks=1)
        return (ctx['bidsiz'] / ctx['asksiz'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def alpha_spread_depth_coupling(self, df: pd.DataFrame, *, window: int) -> pd.Series:
        ctx = self._ctx(df, window=window, tick=0.01, jump_ticks=1)
        return (ctx['spread'] / (ctx['bidsiz'] + ctx['asksiz']).replace(0, np.nan)).fillna(0.0)

    def alpha_size_vol(self, df: pd.DataFrame, *, window: int) -> pd.Series:
        ctx = self._ctx(df, window=window, tick=0.01, jump_ticks=1)
        mean = ctx['total_sz'].rolling(window).mean().shift(1)
        std = ctx['total_sz'].rolling(window).std().shift(1)
        return (std / mean.replace(0, np.nan)).fillna(0.0)

    def alpha_ofi(self, df: pd.DataFrame, *, window: int) -> pd.Series:
        ctx = self._ctx(df, window=window, tick=0.01, jump_ticks=1)
        return ctx['ofi_raw'].fillna(0.0)

    def alpha_signed_vol_change(self, df: pd.DataFrame, *, window: int) -> pd.Series:
        ctx = self._ctx(df, window=window, tick=0.01, jump_ticks=1)
        return (np.sign(ctx['dmid']).fillna(0.0) * (ctx['dq_bid'].fillna(0.0) + ctx['dq_ask'].fillna(0.0))).fillna(0.0)

    def alpha_price_jump_intensity(self, df: pd.DataFrame, *, window: int, tick: float, jump_ticks: int) -> pd.Series:
        ctx = self._ctx(df, window=window, tick=tick, jump_ticks=jump_ticks)
        return ctx['price_jump_intensity']

    def alpha_quote_arrival_rate(self, df: pd.DataFrame, *, window: int) -> pd.Series:
        ctx = self._ctx(df, window=window, tick=0.01, jump_ticks=1)
        return ctx['quote_arrival_rate']

    def alpha_quote_lifetime(self, df: pd.DataFrame, *, window: int) -> pd.Series:
        ctx = self._ctx(df, window=window, tick=0.01, jump_ticks=1)
        return ctx['quote_lifetime']

    def alpha_micro_volatility(self, df: pd.DataFrame, *, window: int) -> pd.Series:
        ctx = self._ctx(df, window=window, tick=0.01, jump_ticks=1)
        return ctx['ret'].rolling(window).std().shift(1).fillna(0.0)

    def alpha_realized_spread_decay(self, df: pd.DataFrame, *, window: int) -> pd.Series:
        ctx = self._ctx(df, window=window, tick=0.01, jump_ticks=1)
        s = ctx['spread'].shift(1)
        r = ctx['ret']
        mean_s = s.rolling(window).mean()
        mean_r = r.rolling(window).mean()
        cov = ((s - mean_s) * (r - mean_r)).rolling(window).mean()
        var_s = ((s - mean_s) ** 2).rolling(window).mean()
        beta = (cov / var_s.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        return beta.fillna(0.0)

    def alpha_quote_staleness(self, df: pd.DataFrame, *, window: int) -> pd.Series:
        return df.get('quote_staleness', pd.Series(0.0, index=df.index))

    def alpha_ba_transition_prob(self, df: pd.DataFrame, *, window: int) -> pd.Series:
        ctx = self._ctx(df, window=window, tick=0.01, jump_ticks=1)
        return ctx['ba_trans_prob']

    # ----- Dispatcher -----
    def compute(self, name: str, data, **kwargs) -> pd.Series:
        fn = getattr(self, f"alpha_{name}", None)
        if fn is None:
            available = sorted(m.replace('alpha_', '') for m in dir(self) if m.startswith('alpha_'))
            raise ValueError(f"Unknown alpha '{name}'. Available: {available}")
        return fn(data, **kwargs)

    def compute_all(self, df: pd.DataFrame, *, window: int, tick: float, jump_ticks: int, names: Optional[list[str]] = None, compress_k: float = 3.0) -> dict[str, pd.Series]:
        # Known alphas
        known = [
            'mid_pct', 'spread_dev', 'eff_spread', 'price_dist_vwap',
            'liq_imbalance', 'depth_ratio', 'spread_depth_coupling', 'size_vol',
            'ofi', 'signed_vol_change', 'price_jump_intensity',
            'quote_arrival_rate', 'quote_lifetime', 'micro_volatility',
            'realized_spread_decay', 'quote_staleness', 'ba_transition_prob',
        ]
        if names is None:
            names = known
        else:
            # ensure we still compute everything for CSV if requested names is subset
            names = list(dict.fromkeys([*names]))

        out: dict[str, pd.Series] = {}
        for n in names:
            raw = self.compute(n, df, window=window, tick=tick, jump_ticks=jump_ticks) if n in {'price_jump_intensity'} else self.compute(n, df, window=window)
            std = self._rolling_std(raw, window)
            score = self.score(raw, std, k=compress_k).fillna(0.0)
            out[f'alpha_{n}'] = score
        return out


def second_snapshot(
    df: pd.DataFrame,
    alpha_name: Union[str, list[str]] = 'mid_pct',
    window: int = 60,
    tick: float = 0.01,
    jump_ticks: int = 1,
) -> pd.DataFrame:
    # Per-second last quote in each second
    per_sec = df.groupby(df['ts'].dt.floor('S'), as_index=True).last()

    if per_sec.empty:
        # Return empty with expected columns
        out = pd.DataFrame(columns=['mid', 'micro', 'spread', 'alpha', 'vol'])
        out.index.name = 'ts'
        return out

    # Reindex to continuous seconds and forward-fill (limit 3 seconds)
    full_index = pd.date_range(per_sec.index.min(), per_sec.index.max(), freq='S')
    per_sec = per_sec.reindex(full_index)

    # Forward-fill core fields and sizes/prices needed for advanced alphas
    cols_ffill = ['mid', 'micro', 'spread', 'best_bid', 'best_ask', 'best_bidsiz', 'best_asksiz']
    existing = [c for c in cols_ffill if c in per_sec.columns]
    per_sec[existing] = per_sec[existing].ffill(limit=3)

    # Quote counts per second and staleness
    per_sec_count = df.groupby(df['ts'].dt.floor('S')).size().reindex(full_index, fill_value=0)
    time_index_series = pd.Series(full_index, index=full_index)
    last_update_time = time_index_series.where(per_sec_count > 0).ffill()
    staleness_sec = (time_index_series - last_update_time).dt.total_seconds().astype(float)

    # Compute features via alpha calculator (supports multiple alpha names)
    calc = AlphaCalculator()
    if isinstance(alpha_name, str):
        names = [n.strip() for n in alpha_name.split(',') if n.strip()]
    else:
        names = list(alpha_name)

    base_cols = [c for c in ['mid','micro','spread','best_bid','best_ask','best_bidsiz','best_asksiz'] if c in per_sec.columns]
    out = per_sec[base_cols].copy()
    out['quote_count'] = per_sec_count.astype(float)
    out['quote_staleness'] = staleness_sec.fillna(0.0)

    # Compute all alphas via AlphaCalculator and 3-sigma scoring
    # Always compute full set for CSV; use provided names only for the composite average
    all_alpha = calc.compute_all(out, window=window, tick=tick, jump_ticks=jump_ticks, names=None, compress_k=3.0)
    for col, ser in all_alpha.items():
        out[col] = ser
    alpha_cols = sorted(all_alpha.keys())

    # Determine which alphas to combine for composite
    # Supports weights via: name:weight or name*weight (e.g., eff_spread:-0.5 or ofi*2)
    def _parse_alpha_and_weights(spec: Union[str, list[str]]) -> tuple[list[str], dict[str, float]]:
        names: list[str] = []
        weights: dict[str, float] = {}
        tokens = [t.strip() for t in (spec.split(',') if isinstance(spec, str) else spec) if t and t.strip()]
        for t in tokens:
            if ':' in t:
                n, w = t.split(':', 1)
            elif '*' in t:
                n, w = t.split('*', 1)
            else:
                n, w = t, '1'
            n = n.strip()
            try:
                weights[n] = float(w)
            except Exception:
                weights[n] = 1.0
            names.append(n)
        return names, weights

    sel_names, sel_weights = _parse_alpha_and_weights(alpha_name)
    sel_cols = [(n, f'alpha_{n}') for n in sel_names if f'alpha_{n}' in out.columns]
    if sel_cols:
        num = None
        denom = 0.0
        for n, col in sel_cols:
            w = float(sel_weights.get(n, 1.0))
            if num is None:
                num = w * out[col]
            else:
                num = num + w * out[col]
            denom += abs(w)
        if denom <= 1e-12 or num is None:
            out['alpha'] = 0.0
        else:
            out['alpha'] = (num / denom).clip(-1.0, 1.0)
    else:
        out['alpha'] = out[alpha_cols].mean(axis=1, skipna=True) if alpha_cols else 0.0

    vol = out['alpha'].rolling(window).std().bfill()
    out['vol'] = vol
    out.index.name = 'ts'
    return out


def backtest_mm(
    ticks: pd.DataFrame,
    snap: pd.DataFrame,
    *,
    gamma: float = 0.3,
    kappa: float = 0.00001,
    delta0: float = 0.02,
    c: float = 20.0,
    tick: float = 0.01,
    qty: int = 100,
    qmax: int = 2000,
    plot: bool = True,
    max_fills_per_sec: int = 1,
    alpha_names: Optional[Union[str, list[str]]] = None,
) -> pd.DataFrame:
    """Simple per-second market making backtest using cross-tick fills.

    - Quotes each second with skewed microprice quotes.
    - Fills if within (t, t+1] any tick crosses our quotes.
    - Resets inventory/cash at the start of each day; concatenates results across days.
    """
    if ticks.empty or snap.empty:
        return pd.DataFrame(columns=['ts','bid','ask','skew','delta','filled_buy','filled_sell','q','cash','equity'])

    # Ensure required fields
    for col in ['ts', 'best_bid', 'best_ask']:
        if col not in ticks.columns:
            raise KeyError(f"ticks is missing required column: {col}")
    for col in ['mid', 'micro', 'vol']:
        if col not in snap.columns:
            raise KeyError(f"snapshot is missing required column: {col}")
    if not (('alpha' in snap.columns) or any(c.startswith('alpha_') for c in snap.columns)):
        raise KeyError("snapshot must contain 'alpha' or at least one 'alpha_*' column")

    rows = []

    # Prepare copies
    ticks = ticks.copy().sort_values('ts')
    snap = snap.copy().sort_index()
    # Single pass over all seconds, carrying state across days
    cash = 0.0
    q = 0
    idx = snap.index
    snap = snap.copy().sort_index()

    snap['mid_diff'] = snap['mid'].diff()

    snap['sigma_2m'] = (
        snap['mid_diff']
        .rolling(120, min_periods=60)
        .std()
        .shift(1)
    )    # Ensure a combined alpha (already scored in snapshot). Fallback if missing.
    alpha_cols = [c for c in snap.columns if c.startswith('alpha_') and c != 'alpha']
    if 'alpha' not in snap.columns:
        if alpha_cols:
            snap['alpha'] = snap[alpha_cols].mean(axis=1, skipna=True).fillna(0.0)
        else:
            snap['alpha'] = 0.0
    for t in idx:
        micro = snap.at[t, 'micro'] if pd.notna(snap.at[t, 'micro']) else snap.at[t, 'mid']
        a = snap.at[t, 'alpha']
        v = snap.at[t, 'sigma_2m']
        if pd.isna(v):
            v = 0.0
        if pd.isna(a):
            a = 0.0
        # alpha already scored in snapshot; clip defensively to [-1,1]
        score = float(np.clip(a, -1.0, 1.0))

        # 3️⃣
        tau = 10.0
        m = c * score * v * np.sqrt(tau)
        # 4️⃣ 库存风险修正
        beta = 20 * tick / qmax  # 极限20个depth
        skew = m - beta * q

        delta = max(delta0 + c * v, tick)
        bid_cont = micro - delta + skew
        ask_cont = micro + delta + skew
        bid = np.floor(bid_cont / tick) * tick
        ask = np.ceil(ask_cont / tick) * tick
        if not pd.isna(bid) and not pd.isna(ask) and ask <= bid:
            ask = bid + tick

        t_next = t + pd.Timedelta(seconds=1)
        interval = ticks[(ticks['ts'] > t) & (ticks['ts'] <= t_next)]

        filled_buy = False 
        filled_sell = False

        if not interval.empty and not pd.isna(bid) and q + qty <= qmax:
            if (interval['best_ask'] <= bid).any():
                filled_buy = True
                cash -= float(bid) * qty
                q += qty

        if not interval.empty and not pd.isna(ask) and q - qty >= -qmax:
            if (interval['best_bid'] >= ask).any():
                filled_sell = True
                cash += float(ask) * qty
                q -= qty

        next_mid = snap['mid'].get(t_next, snap.at[t, 'mid'])
        if pd.isna(next_mid):
            next_mid = snap.at[t, 'mid']
        equity = cash + q * float(next_mid)

        rows.append({
            'ts': t,
            'bid': bid,
            'ask': ask,
            'skew': skew,
            'delta': delta,
            'filled_buy': int(filled_buy),
            'filled_sell': int(filled_sell),
            'q': q,
            'cash': cash,
            'equity': equity,
        })

    bt = pd.DataFrame(rows)

    # Compute alpha win rates against next-mid direction
    try:
        def _compute_win_rate(pred: pd.Series, mid: pd.Series) -> tuple[float, int]:
            future = mid.shift(-1) - mid
            s_future = np.sign(future)
            s_pred = np.sign(pred)
            valid = (s_future != 0) & (s_pred != 0) & s_future.notna() & s_pred.notna()
            n = int(valid.sum())
            if n == 0:
                return float('nan'), 0
            rate = float((s_future[valid] == s_pred[valid]).mean())
            return rate, n

        # Determine selected alpha component columns (with optional weights)
        all_alpha_cols = sorted([c for c in snap.columns if c.startswith('alpha_') and c != 'alpha'])

        def _parse_weight_spec(spec) -> tuple[list[str], dict[str, float]]:
            if spec is None:
                return [], {}
            if isinstance(spec, list):
                tokens = [str(x).strip() for x in spec if str(x).strip()]
            else:
                tokens = [t.strip() for t in str(spec).split(',') if t.strip()]
            names: list[str] = []
            weights: dict[str, float] = {}
            for t in tokens:
                if ':' in t:
                    n, w = t.split(':', 1)
                elif '*' in t:
                    n, w = t.split('*', 1)
                else:
                    n, w = t, '1'
                n = n.strip()
                try:
                    weights[n] = float(w)
                except Exception:
                    weights[n] = 1.0
                names.append(n)
            return names, weights

        sel_names, weight_map = _parse_weight_spec(alpha_names)
        if sel_names:
            sel_cols = [(n, f'alpha_{n}', float(weight_map.get(n, 1.0))) for n in sel_names if f'alpha_{n}' in snap.columns]
        else:
            sel_cols = [(c.replace('alpha_',''), c, 1.0) for c in all_alpha_cols]

        win_rates: list[tuple[str, float, int]] = []
        # Composite alpha
        if 'alpha' in snap.columns:
            wr, n = _compute_win_rate(snap['alpha'], snap['mid'])
            win_rates.append(('alpha(composite)', wr, n))
        # Component alphas (respect sign of weights by multiplying prediction)
        for name, col, w in sel_cols:
            wr, n = _compute_win_rate(w * snap[col], snap['mid'])
            label = f"{col} (w={w:+g})"
            win_rates.append((label, wr, n))
    except Exception as _:
        win_rates = []
    print()
    if plot and not bt.empty:
        try:
            # 横轴：优先使用 ts 列，否则 index
            x = bt['ts'] if 'ts' in bt.columns else bt.index

            # 固定三行：Equity / Inventory / Win Rates
            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 2, 1]})
            try:
                fig.set_dpi(150)
            except Exception:
                pass

            axes = np.atleast_1d(axes).ravel()

            # === 1) Equity ===
            axes[0].plot(x, bt['equity'], label='Equity', linewidth=1.2)
            axes[0].set_ylabel('Equity')
            axes[0].legend(loc='best')

            # === 2) Inventory ===
            axes[1].plot(x, bt['q'], label='Inventory', color='tab:orange', linewidth=1.2)
            axes[1].set_ylabel('Inventory (q)')
            axes[1].legend(loc='best')
            axes[-1].set_xlabel('Time')
            for ax_i in axes:
                try:
                    ax_i.tick_params(labelsize=10)
                except Exception:
                    pass

            # === 3) Win Rate 面板 ===
            ax_text = axes[2]
            lines = ["Win rate (next-mid):"]
            row = []
            for i, (name, wr, n) in enumerate(win_rates):
                val = f"{wr * 100:.1f}%" if np.isfinite(wr) else "n/a"
                row.append(f"{name}: {val} (n={n})")
                if (i + 1) % 3 == 0:  # 每行显示 3 个
                    lines.append("   ".join(row))
                    row = []
            if row:
                lines.append("   ".join(row))
            txt = "\n".join(lines)
            ax_text.set_axis_off()
            ax_text.text(
                0.01, 0.98, txt,
                va='top', ha='left',
                fontsize=10, family='monospace',
                transform=ax_text.transAxes
            )
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")

    return bt





def main():
    parser = argparse.ArgumentParser(description="Build per-second NBBO snapshot and run MM backtest.")
    parser.add_argument("--input", "-i", default=".", help="Path to .rda file or directory (default: .)")
    parser.add_argument("--symbol", "-s", default="AAPL", help="Symbol root to filter (default: AAPL)")
    parser.add_argument("--output", "-o", default="./aapl_snap.csv", help="Optional CSV output path for snapshot")
    parser.add_argument(
        "--alpha",
        default="mid_pct",
        help="Alpha name(s), comma-separated. Supports weights via name:weight or name*weight (e.g., ofi:2,eff_spread:-0.5). Default: mid_pct",
    )

    # Backtest flags and params
    parser.add_argument("--backtest", action="store_true", help="Run market-making backtest")
    parser.add_argument("--gamma", type=float, default=0.3, help="Skew gamma (default: 0.3)")
    parser.add_argument("--kappa", type=float, default=0.002, help="Inventory penalty kappa (default: 0.002)")
    parser.add_argument("--delta0", type=float, default=0.01, help="Base half-spread (default: 0.01)")
    parser.add_argument("--c", type=float, default=20.0, help="Vol scaling (default: 20.0)")
    parser.add_argument("--tick", type=float, default=0.01, help="Tick size (default: 0.01)")
    parser.add_argument("--qty", type=int, default=100, help="Trade size per fill (default: 100)")
    parser.add_argument("--qmax", type=int, default=2000, help="Inventory limit abs (default: 2000)")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting for backtest")
    parser.add_argument("--reset-daily", action="store_true", help="Reset cash/inventory at each day start")
    parser.add_argument("--max-fills-per-sec", type=int, default=1, help="Cap fills per second (1=default, -1 for unlimited)")

    args = parser.parse_args()

    # Build snapshot
    target = Path(args.input) if args.input else Path(".")
    frames = load_nbbo_frames(target, object_name="nbbo")
    raw = pd.concat(frames, ignore_index=True) if isinstance(frames, list) and len(frames) > 1 else (frames[0] if isinstance(frames, list) else frames)
    cleaned = clean_and_enrich(raw, symbol=args.symbol)
    snap = second_snapshot(cleaned, alpha_name=args.alpha)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        snap.to_csv(args.output)

    print(snap.head())
    print(f"\nRows: {len(snap)}; Time range: {snap.index.min()} -> {snap.index.max()}")
    if args.output:
        print(f"Saved to: {args.output}")

    if args.backtest:
        bt = backtest_mm(
            ticks=cleaned,
            snap=snap,
            gamma=args.gamma,
            kappa=args.kappa,
            delta0=args.delta0,
            c=args.c,
            tick=args.tick,
            qty=args.qty,
            qmax=args.qmax,
            plot=(not args.no_plots),
            reset_daily=args.reset_daily,
            max_fills_per_sec=(None if args.max_fills_per_sec == -1 else args.max_fills_per_sec),
            alpha_names=args.alpha,
        )
        print(bt.head())
        if not bt.empty:
            final_pnl = bt['equity'].iloc[-1]
            trades = int(bt['filled_buy'].sum() + bt['filled_sell'].sum())
            print(f"Final PnL: {final_pnl:.2f}; Trades: {trades}")


# if __name__ == "__main__":
#     main()


if __name__ == "__main__":
    # 手动设置参数
    input_path = "./"
    output_path = "./aapl_snap.csv"
    # alpha = [
    #     "mid_pct", "spread_dev", "eff_spread", "price_dist_vwap",
    #     "liq_imbalance", "depth_ratio", "spread_depth_coupling", "size_vol",
    #     "ofi", "signed_vol_change", "price_jump_intensity",
    #     "quote_arrival_rate", "quote_lifetime", "micro_volatility",
    #     "realized_spread_decay", "quote_staleness", "ba_transition_prob"
    # ]
    alpha = [
        "liq_imbalance",
        "ofi",
        "price_dist_vwap:-1"
    ]


    # backtest 参数
    run_backtest = True
    gamma = 0.03
    kappa = 0.002
    delta0 = 0.01
    c = 2.0
    tick = 0.01
    qty = 100
    qmax = 2000
    plot = True
    symbol= "AAPL"
    # === Snapshot部分 ===
    target = Path(input_path)
    frames = load_nbbo_frames(target, object_name="nbbo")
    raw = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    cleaned = clean_and_enrich(raw, symbol=symbol)
    snap = second_snapshot(cleaned, alpha_name=alpha)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        snap.to_csv(output_path)
    print(snap.head())
    print(f"\nRows: {len(snap)}; Time range: {snap.index.min()} -> {snap.index.max()}")
    if output_path:
        print(f"Saved to: {output_path}")

    # === Backtest部分 ===
    if run_backtest:
        bt = backtest_mm(
            ticks=cleaned,
            snap=snap,
            gamma=gamma,
            kappa=kappa,
            delta0=delta0,
            c=c,
            tick=tick,
            qty=qty,
            qmax=qmax,
            plot=plot,
            alpha_names=alpha,
        )
        print(bt.head())
        if not bt.empty:
            final_pnl = bt['equity'].iloc[-1]
            trades = int(bt['filled_buy'].sum() + bt['filled_sell'].sum())
            print(f"Final PnL: {final_pnl:.2f}; Trades: {trades}")
