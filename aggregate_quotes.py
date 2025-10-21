import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def aggregate_orderbook_data(df, time_interval=10):
    """
    聚合每N秒的订单簿数据，通过累计所有买卖订单重构订单簿
    df 需包含: ['DATE','TIME_M','SYM_ROOT','BID','BIDSIZ','ASK','ASKSIZ']
    """
    print(f"开始聚合每{time_interval}秒的订单簿数据...")
    df['TIME_BUCKET'] = (df['TIME_M'] // time_interval) * time_interval
    grouped = df.groupby(['DATE', 'SYM_ROOT', 'TIME_BUCKET'])

    orderbook_results = []
    prev_best_bid, prev_best_ask = {}, {}  # 跨桶回退

    # >>> CHANGE: 辅助函数——在一个分组内“向后/向前”寻找最近的有效（非0且非NaN）的价格
    def find_last_valid(group, price_col, size_col, fallback_prev=np.nan):
        for _, r in group.iloc[::-1].iterrows():  # 末尾→开头
            p, s = r[price_col], r[size_col]
            if pd.notna(p) and p != 0:
                return float(p), (float(s) if pd.notna(s) else 0.0)
        # 分组内没有有效的 → 回退到上一个桶的 best
        if not np.isnan(fallback_prev):
            return float(fallback_prev), 0.0
        return np.nan, 0.0

    def find_first_valid(group, price_col, size_col, fallback_next=np.nan):
        for _, r in group.iterrows():  # 开头→末尾
            p, s = r[price_col], r[size_col]
            if pd.notna(p) and p != 0:
                return float(p), (float(s) if pd.notna(s) else 0.0)
        if not np.isnan(fallback_next):
            return float(fallback_next), 0.0
        return np.nan, 0.0

    for (date, symbol, time_bucket), group in grouped:
        group = group.sort_values('TIME_M')
        symbol_key = f"{date}_{symbol}"
        prev_bid = prev_best_bid.get(symbol_key, np.nan)
        prev_ask = prev_best_ask.get(symbol_key, np.nan)

        # >>> CHANGE: 最后一个 quote 若为0/NaN，向前找；再不行用上一个桶的 best
        best_bid, best_bid_size = find_last_valid(group, 'BID', 'BIDSIZ', fallback_prev=prev_bid)
        best_ask, best_ask_size = find_last_valid(group, 'ASK', 'ASKSIZ', fallback_prev=prev_ask)

        # >>> CHANGE: 构造订单簿用到的所有报价，过滤 0 与 NaN
        bid_orders = group[['BID', 'BIDSIZ']].copy()
        bid_orders = bid_orders[(bid_orders['BID'].notna()) & (bid_orders['BID'] != 0)]
        ask_orders = group[['ASK', 'ASKSIZ']].copy()
        ask_orders = ask_orders[(ask_orders['ASK'].notna()) & (ask_orders['ASK'] != 0)]

        if not bid_orders.empty:
            bid_book = bid_orders.groupby('BID')['BIDSIZ'].sum().sort_index(ascending=False)
            total_bid_volume = float(bid_book.sum())
        else:
            bid_book = pd.Series(dtype=float)
            total_bid_volume = 0.0

        if not ask_orders.empty:
            ask_book = ask_orders.groupby('ASK')['ASKSIZ'].sum().sort_index(ascending=True)
            total_ask_volume = float(ask_book.sum())
        else:
            ask_book = pd.Series(dtype=float)
            total_ask_volume = 0.0

        # ========== 构建多level ==========
        bid_levels, ask_levels = {}, {}

        if not np.isnan(best_bid) and not bid_book.empty:
            bid_prices = bid_book.index.tolist()
            if best_bid not in bid_prices:
                bid_prices.append(best_bid)
            bid_prices = sorted(set(bid_prices), reverse=True)
            pos_bb = bid_prices.index(best_bid)
            for i in range(1, 11):
                if pos_bb + i - 1 < len(bid_prices):
                    price = bid_prices[pos_bb + i - 1]
                    bid_levels[f'BID_L{i}'] = price
                    bid_levels[f'BIDSIZ_L{i}'] = float(bid_book.get(price, 0.0))
                else:
                    bid_levels[f'BID_L{i}'] = np.nan
                    bid_levels[f'BIDSIZ_L{i}'] = 0.0
        else:
            for i in range(1, 11):
                bid_levels[f'BID_L{i}'] = np.nan
                bid_levels[f'BIDSIZ_L{i}'] = 0.0

        if not np.isnan(best_ask) and not ask_book.empty:
            ask_prices = ask_book.index.tolist()
            if best_ask not in ask_prices:
                ask_prices.append(best_ask)
            ask_prices = sorted(set(ask_prices))
            pos_ba = ask_prices.index(best_ask)
            for i in range(1, 11):
                if pos_ba + i - 1 < len(ask_prices):
                    price = ask_prices[pos_ba + i - 1]
                    ask_levels[f'ASK_L{i}'] = price
                    ask_levels[f'ASKSIZ_L{i}'] = float(ask_book.get(price, 0.0))
                else:
                    ask_levels[f'ASK_L{i}'] = np.nan
                    ask_levels[f'ASKSIZ_L{i}'] = 0.0
        else:
            for i in range(1, 11):
                ask_levels[f'ASK_L{i}'] = np.nan
                ask_levels[f'ASKSIZ_L{i}'] = 0.0

        # ========== 基础指标 ==========
        if not np.isnan(best_bid) and not np.isnan(best_ask):
            mid_price = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
            spread_bps = (spread / mid_price) * 10000.0 if mid_price != 0 else np.nan
        else:
            mid_price, spread, spread_bps = np.nan, np.nan, np.nan

        order_imbalance = (total_bid_volume - total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0.0

        # ========== multi-level OFI（用你原来的思路，仅避免0价）==========
        multi_level_ofi = 0.0
        ofi_bid = 0.0
        ofi_ask = 0.0

        prev_bid_for_ofi = prev_best_bid.get(symbol_key, np.nan)
        prev_ask_for_ofi = prev_best_ask.get(symbol_key, np.nan)
        if not np.isnan(best_bid) and not np.isnan(prev_bid_for_ofi) and not np.isnan(best_ask) and not np.isnan(prev_ask_for_ofi):
            bid_price_change = best_bid - prev_bid_for_ofi
            ask_price_change = best_ask - prev_ask_for_ofi
            for level in range(1, 4):
                bid_col, ask_col = f'BID_L{level}', f'ASK_L{level}'
                bidsiz_col, asksiz_col = f'BIDSIZ_L{level}', f'ASKSIZ_L{level}'
                bid_price = bid_levels.get(bid_col, np.nan)
                ask_price = ask_levels.get(ask_col, np.nan)
                bid_size = bid_levels.get(bidsiz_col, 0.0)
                ask_size = ask_levels.get(asksiz_col, 0.0)
                if (pd.isna(bid_price) or bid_price == 0) or (pd.isna(ask_price) or ask_price == 0):
                    continue
                bchg = bid_price_change if bid_price_change != 0 else 1
                achg = ask_price_change if ask_price_change != 0 else 1
                level_ofi_bid = np.sign(bchg) * bid_size
                level_ofi_ask = -np.sign(achg) * ask_size
                level_ofi = level_ofi_bid - level_ofi_ask
                weight = 0.5 ** (level - 1)
                multi_level_ofi += level_ofi * weight

        # >>> CHANGE: 价格变动用“桶内第一条有效 mid 与最后一条有效 mid”
        first_bid, _ = find_first_valid(group, 'BID', 'BIDSIZ', fallback_next=best_bid)
        first_ask, _ = find_first_valid(group, 'ASK', 'ASKSIZ', fallback_next=best_ask)
        if not np.isnan(first_bid) and not np.isnan(first_ask) and not np.isnan(best_bid) and not np.isnan(best_ask):
            first_mid = (first_bid + first_ask) / 2.0
            last_mid = (best_bid + best_ask) / 2.0
            price_change = last_mid - first_mid
        else:
            price_change = 0.0

        # 更新跨桶 best（供下个桶回退用）
        prev_best_bid[symbol_key] = best_bid
        prev_best_ask[symbol_key] = best_ask

        result_dict = {
            'DATE': date,
            'SYM_ROOT': symbol,
            'TIME_BUCKET': time_bucket,
            'TIME_START': group['TIME_M'].min(),
            'TIME_END': group['TIME_M'].max(),
            'TICK_COUNT': len(group),
            'BEST_BID': best_bid,
            'BEST_BID_SIZE': best_bid_size,
            'BEST_ASK': best_ask,
            'BEST_ASK_SIZE': best_ask_size,
            'TOTAL_BID_VOLUME': total_bid_volume,
            'TOTAL_ASK_VOLUME': total_ask_volume,
            'MID_PRICE': mid_price,
            'SPREAD': spread,
            'SPREAD_BPS': spread_bps,
            'ORDER_IMBALANCE': order_imbalance,
            'PRICE_CHANGE': price_change,
            'BID_LEVELS': len(bid_book),
            'ASK_LEVELS': len(ask_book),
            'OFI_BID': ofi_bid,
            'OFI_ASK': ofi_ask,
            'MULTI_LEVEL_OFI': multi_level_ofi
        }
        result_dict.update(bid_levels)
        result_dict.update(ask_levels)

        # 填满缺失 level
        for i in range(1, 11):
            result_dict.setdefault(f'BID_L{i}', np.nan)
            result_dict.setdefault(f'BIDSIZ_L{i}', 0.0)
            result_dict.setdefault(f'ASK_L{i}', np.nan)
            result_dict.setdefault(f'ASKSIZ_L{i}', 0.0)

        orderbook_results.append(result_dict)

    orderbook_df = pd.DataFrame(orderbook_results)
    print(f"聚合完成，共生成 {len(orderbook_df)} 个时间桶")
    return orderbook_df


def main():
    """
    主函数：聚合quotes数据
    """
    
    print("=== Quotes数据聚合 ===")
    
    # 读取处理后的数据
    try:
        df = pd.read_csv('quotes_processed.csv')
        print(f"成功读取数据，共 {len(df)} 条记录")
    except FileNotFoundError:
        print("未找到quotes_processed.csv文件，请先运行process_quotes.py")
        return
    
    # 检查必要的列
    required_cols = ['DATE', 'TIME_M', 'SYM_ROOT', 'BID', 'BIDSIZ', 'ASK', 'ASKSIZ']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"缺少必要的列: {missing_cols}")
        return
    
    # 聚合订单簿数据（每10秒）
    orderbook_agg = aggregate_orderbook_data(df, time_interval=10)
    
    # 保存结果
    output_file = 'orderbook_aggregated.csv'
    orderbook_agg.to_csv(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")
    
    # 显示统计信息
    print("\n=== 统计摘要 ===")
    print(f"总记录数: {len(orderbook_agg)}")
    print(f"股票数量: {orderbook_agg['SYM_ROOT'].nunique()}")
    print(f"日期范围: {orderbook_agg['DATE'].min()} 到 {orderbook_agg['DATE'].max()}")
    
    # 显示前几行数据
    print(f"\n数据预览:")
    display_cols = ['DATE', 'SYM_ROOT', 'TIME_BUCKET', 'BEST_BID', 'BEST_ASK', 'MID_PRICE', 'SPREAD', 'ORDER_IMBALANCE', 'OFI_BID', 'OFI_ASK', 'MULTI_LEVEL_OFI']
    available_cols = [col for col in display_cols if col in orderbook_agg.columns]
    print(orderbook_agg[available_cols].head(10))
    
    # 显示多level数据示例
    print(f"\n多level订单簿数据示例:")
    level_cols = ['DATE', 'SYM_ROOT', 'TIME_BUCKET'] + [f'BID_L{i}' for i in range(1, 4)] + [f'ASK_L{i}' for i in range(1, 4)]
    available_level_cols = [col for col in level_cols if col in orderbook_agg.columns]
    if available_level_cols:
        print(orderbook_agg[available_level_cols].head(5))
    
    return orderbook_agg

if __name__ == "__main__":
    result = main()
