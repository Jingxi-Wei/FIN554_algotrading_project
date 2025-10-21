import pandas as pd
import numpy as np
from datetime import datetime, time

def process_trade_data(input_file, output_file):
    """
    处理trade.csv数据：
    1. 筛选正常交易时间（9:30 AM - 4:00 PM ET）
    2. 将时间转换为秒数
    3. 只保留指定的列
    """
    
    print("开始读取trade数据...")
    
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    print(f"原始数据行数: {len(df)}")
    print(f"原始数据列数: {len(df.columns)}")
    print(f"列名: {list(df.columns)}")
    
    # 筛选指定列（根据实际trade数据调整）
    required_columns = ['DATE', 'TIME_M', 'SYM_ROOT', 'PRICE', 'SIZE']
    
    # 检查哪些列存在
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"警告: 以下列不存在: {missing_columns}")
        print("可用的列:", list(df.columns))
        return None
    
    # 只保留存在的列
    available_columns = [col for col in required_columns if col in df.columns]
    df_filtered = df[available_columns].copy()
    
    print(f"筛选列后数据行数: {len(df_filtered)}")
    
    # 处理时间筛选
    if 'TIME_M' in df_filtered.columns:
        print("开始筛选正常交易时间...")
        
        # 将TIME_M转换为时间对象
        time_objects = pd.to_datetime(df_filtered['TIME_M'], format='%H:%M:%S.%f').dt.time
        
        # 定义正常交易时间范围（9:30 AM - 4:00 PM）
        market_open = time(9, 30)  # 9:30 AM
        market_close = time(16, 0)  # 4:00 PM
        
        # 筛选正常交易时间
        before_time_filter = len(df_filtered)
        time_mask = (time_objects >= market_open) & (time_objects <= market_close)
        df_filtered = df_filtered[time_mask]
        after_time_filter = len(df_filtered)
        
        print(f"筛选交易时间后数据行数: {after_time_filter} (移除了 {before_time_filter - after_time_filter} 行)")
        
        # 将时间转换为从每天0时开始的秒数
        print("将时间转换为秒数...")
        time_objects_filtered = time_objects[time_mask]
        seconds_from_midnight = []
        
        for t in time_objects_filtered:
            # 计算从午夜开始的秒数
            total_seconds = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1000000
            seconds_from_midnight.append(total_seconds)
        
        df_filtered['TIME_M'] = seconds_from_midnight
        
        print(f"时间转换完成，范围: {min(seconds_from_midnight):.3f} - {max(seconds_from_midnight):.3f} 秒")
    
    # 保存处理后的数据
    print(f"保存处理后的数据到 {output_file}...")
    df_filtered.to_csv(output_file, index=False)
    
    print(f"处理完成！")
    print(f"最终数据行数: {len(df_filtered)}")
    print(f"最终数据列数: {len(df_filtered.columns)}")
    
    # 显示前几行数据
    print("\n处理后的数据预览:")
    print(df_filtered.head())
    
    return df_filtered

if __name__ == "__main__":
    # 处理数据
    processed_df = process_trade_data('trade.csv', 'trade_processed.csv')

