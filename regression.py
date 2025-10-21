# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

csv_path = "orderbook_aggregated.csv"
ofi_col  = "MULTI_LEVEL_OFI"
mid_col  = "MID_PRICE"
min_obs_per_bin = 180   # 每个30min分组最少样本，按你的采样改：10秒一笔的话180更合适

# ========= 1) 读取 =========
df = pd.read_csv(csv_path)

# ========= 2) 构造时间戳 t（稳健处理多种字段） =========
t = None
if "t" in df.columns:
    t = pd.to_datetime(df["t"], errors="coerce")
elif "TIME_BUCKET" in df.columns:
    # TIME_BUCKET 可能是“当日秒数”或已是时间字符串
    if np.issubdtype(df["TIME_BUCKET"].dtype, np.number):
        # 如果有 DATE，拼成“日期 + 当日秒数”
        if "DATE" in df.columns:
            # DATE 可能是 'YYYY-MM-DD' 或 'YYYYMMDD'
            try:
                date_parsed = pd.to_datetime(df["DATE"], errors="coerce")
            except:
                date_parsed = pd.to_datetime(df["DATE"].astype(str), errors="coerce")
            t = date_parsed + pd.to_timedelta(df["TIME_BUCKET"].astype(float), unit="s")
        else:
            # 没有 DATE 就用同一天的“虚拟日期”，也能分桶
            base = pd.Timestamp("2000-01-01")
            t = base + pd.to_timedelta(df["TIME_BUCKET"].astype(float), unit="s")
    else:
        t = pd.to_datetime(df["TIME_BUCKET"], errors="coerce")
elif {"DATE","TIME_END"}.issubset(df.columns):
    # DATE + TIME_END 组合
    t = pd.to_datetime(df["DATE"].astype(str) + " " + df["TIME_END"].astype(str), errors="coerce")
else:
    # 尝试任何可能的时间列
    for cand in ["TIMESTAMP","DATETIME","TIME","time","datetime"]:
        if cand in df.columns:
            t = pd.to_datetime(df[cand], errors="coerce")
            break

if t is None:
    raise ValueError("找不到可解析的时间列，请检查：TIME_BUCKET / DATE+TIME_END / t 等列。")
df["t"] = t
df = df.dropna(subset=["t", ofi_col, mid_col]).sort_values("t")

# ========= 3) 构造下一期价格变化 =========
df["MID_NEXT"] = df[mid_col].shift(-1)
df["price_change"] = df["MID_NEXT"] - df[mid_col]
df = df.dropna(subset=["price_change"])

# （可选）去掉明显脏点：mid=0
df = df[df[mid_col] != 0]

# ========= 4) 直接“取整分桶”到30分钟 =========
df["bin30"] = df["t"].dt.floor("30T")   # 每条样本落到所属的 30min 桶

# ========= 5) 按桶回归：Δp = α + β·OFI =========
rows = []
for btime, g in df.groupby("bin30", sort=True, observed=True):
    if len(g) < min_obs_per_bin:
        continue
    X = sm.add_constant(g[ofi_col].astype(float))
    y = g["price_change"].astype(float)
    # Newey-West / HAC 稳健标准误
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags":5})
    rows.append({
        "bin_start": btime,
        "label": btime.strftime("%H:%M"),
        "n": len(g),
        "beta": float(res.params[ofi_col]),
        "t_beta": float(res.tvalues[ofi_col]),
        "pval": float(res.pvalues[ofi_col]),
        "r2": float(res.rsquared)
    })

results = pd.DataFrame(rows).sort_values("bin_start")
if results.empty:
    raise RuntimeError("没有任何30min分组满足样本阈值 min_obs_per_bin，请调小阈值或检查时间解析。")

print(results)

# ========= 6) 画图（横轴用当天时间标签）=========
plt.figure(figsize=(10,4))
plt.plot(results["label"], results["beta"], marker="o", label="β (OFI impact)")
plt.axhline(0, color="gray", linestyle="--")
plt.title("β (OFI impact) per 30-min bin")
plt.xlabel("Time of day (HH:MM)")
plt.ylabel("β coefficient")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ========= 7) 额外：显示每个桶的 R² 或显著性（可选）=========
plt.figure(figsize=(10,3))
plt.bar(results["label"], results["r2"])
plt.title("R² per 30-min bin")
plt.xlabel("Time of day")
plt.ylabel("R²")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
