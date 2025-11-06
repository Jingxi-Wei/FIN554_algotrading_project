#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import pyreadr
from typing import Optional
from typing import Union
import numpy as np

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
    """Alpha formula container for easy extension.

    Add new methods as `alpha_<name>(mid: pd.Series) -> pd.Series`.
    """

    def __init__(self, clip: float = 0.02):
        self.clip = float(clip)

    def alpha_mid_pct(self, mid: pd.Series) -> pd.Series:
        """alpha = mid.pct_change().clip(-clip, clip).fillna(0)"""
        return mid.pct_change().clip(-self.clip, self.clip).fillna(0)

    def compute(self, name: str, mid: pd.Series) -> pd.Series:
        fn = getattr(self, f"alpha_{name}", None)
        if fn is None:
            available = sorted(m.replace('alpha_', '') for m in dir(self) if m.startswith('alpha_'))
            raise ValueError(f"Unknown alpha '{name}'. Available: {available}")
        return fn(mid)


def second_snapshot(df: pd.DataFrame, alpha_name: str = 'mid_pct', clip: float = 0.02) -> pd.DataFrame:
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

    per_sec[['mid', 'micro', 'spread']] = per_sec[['mid', 'micro', 'spread']].ffill(limit=3)

    # Compute features via alpha calculator
    calc = AlphaCalculator(clip=clip)
    alpha = calc.compute(alpha_name, per_sec['mid'])
    vol = alpha.rolling(60).std().bfill()

    out = per_sec[['mid', 'micro', 'spread']].copy()
    out['alpha'] = alpha
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
    for col in ['mid', 'micro', 'alpha', 'vol']:
        if col not in snap.columns:
            raise KeyError(f"snapshot is missing required column: {col}")

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

    # 前 120 秒滚动 std（min_periods=60 可按需调），并 shift(1) 防未来函数
    snap['sigma_2m'] = (
        snap['mid_diff']
        .rolling(120, min_periods=60)
        .std()
        .shift(1)
    )
    snap['alpha_std_60'] = (
        snap['alpha']
        .rolling(60, min_periods=10)
        .std()
        .shift(1)
    )
    for t in idx:
        micro = snap.at[t, 'micro'] if pd.notna(snap.at[t, 'micro']) else snap.at[t, 'mid']
        a = snap.at[t, 'alpha']
        v = snap.at[t, 'sigma_2m']
        if pd.isna(v):
            v = 0.0
        if pd.isna(a):
            a = 0.0
        # Use precomputed, label-indexed rolling std to avoid .iloc with Timestamp
        std_alpha = snap.at[t, 'alpha_std_60']
        if np.isnan(std_alpha) or std_alpha < 1e-12:
            std_alpha = 1e-12

        # 2️⃣ alpha → [-1,1] 的 score (4σ 压缩)
        score = np.tanh(a / (4 * std_alpha))

        # 3️⃣ 公平价偏移
        tau = 10.0  # 时间窗口（秒）
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

    if plot and not bt.empty:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax[0].plot(bt['ts'], bt['equity'], label='Equity')
            ax[0].set_ylabel('Equity')
            ax[0].legend(loc='best')
            ax[1].plot(bt['ts'], bt['q'], label='Inventory', color='tab:orange')
            ax[1].set_ylabel('Inventory (q)')
            ax[1].legend(loc='best')
            ax[1].set_xlabel('Time')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")

    return bt


def backtest_mm_from_path(
    path: Union[str, Path],
    symbol: str = "AAPL",
    *,
    alpha: str = 'mid_pct',
    clip: float = 0.02,
    gamma: float = 0.3,
    kappa: float = 0.002,
    delta0: float = 0.01,
    c: float = 20.0,
    tick: float = 0.01,
    qty: int = 100,
    qmax: int = 2000,
    plot: bool = True,
    reset_daily: bool = False,
) -> pd.DataFrame:
    """Convenience: load multi-day .rda files from a path and run backtest."""
    target = Path(path)
    frames = load_nbbo_frames(target, object_name="nbbo")
    raw = pd.concat(frames, ignore_index=True) if isinstance(frames, list) and len(frames) > 1 else (frames[0] if isinstance(frames, list) else frames)
    cleaned = clean_and_enrich(raw, symbol=symbol)
    snap = second_snapshot(cleaned, alpha_name=alpha, clip=clip)
    return backtest_mm(
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
        reset_daily=reset_daily,
    )
def run(path: Optional[str] = None,
        symbol: str = "AAPL",
        output: Optional[str] = None,
        alpha: str = 'mid_pct',
        clip: float = 0.02) -> pd.DataFrame:
    target = Path(path) if path else Path(".")
    frames = load_nbbo_frames(target, object_name="nbbo")
    raw = pd.concat(frames, ignore_index=True) if isinstance(frames, list) and len(frames) > 1 else (frames[0] if isinstance(frames, list) else frames)
    cleaned = clean_and_enrich(raw, symbol=symbol)
    snap = second_snapshot(cleaned, alpha_name=alpha, clip=clip)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        snap.to_csv(output, index=False)
    return snap


def main():
    parser = argparse.ArgumentParser(description="Build per-second NBBO snapshot and run MM backtest.")
    parser.add_argument("--input", "-i", default=".", help="Path to .rda file or directory (default: .)")
    parser.add_argument("--symbol", "-s", default="AAPL", help="Symbol root to filter (default: AAPL)")
    parser.add_argument("--output", "-o", default="./aapl_snap.csv", help="Optional CSV output path for snapshot")
    parser.add_argument("--alpha", default="mid_pct", help="Alpha formula name (default: mid_pct)")
    parser.add_argument("--clip", type=float, default=0.02, help="Clip threshold for alpha (default: 0.02)")

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
    snap = second_snapshot(cleaned, alpha_name=args.alpha, clip=args.clip)

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
    input_path = "./"          # .rda 或目录路径
    symbol = "AAPL"            # 股票代码
    output_path = "./aapl_snap.csv"
    alpha = "mid_pct"
    clip = 0.1

    # backtest 参数
    run_backtest = True         # ✅ 改这里控制是否跑回测
    gamma = 0.03
    kappa = 0.002
    delta0 = 0.01
    c = 20.0
    tick = 0.01
    qty = 100
    qmax = 2000
    plot = True

    # === Snapshot部分 ===
    target = Path(input_path)
    frames = load_nbbo_frames(target, object_name="nbbo")
    raw = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    cleaned = clean_and_enrich(raw, symbol=symbol)
    snap = second_snapshot(cleaned, alpha_name=alpha, clip=clip)

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
        )
        print(bt.head())
        if not bt.empty:
            final_pnl = bt['equity'].iloc[-1]
            trades = int(bt['filled_buy'].sum() + bt['filled_sell'].sum())
            print(f"Final PnL: {final_pnl:.2f}; Trades: {trades}")
