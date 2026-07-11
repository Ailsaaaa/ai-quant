#!/usr/bin/env python3
"""
TASK3 — 可配置双均线策略仪表板数据生成器
- 接收 GET 参数：stock, start_date, end_date, short_ma, long_ma, commission, slippage, initial_capital
- 在服务端完成回测计算，返回 JSON 数据
- 前端用纯 HTML+JS（Chart.js + 滑动条 + 日期输入）实现交互
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# 复用 dual_ma_strategy 的核心逻辑
sys.path.insert(0, str(Path(__file__).parent))
from dual_ma_strategy import DualMAStrategy  # noqa: E402


# 股票选项（中文标签 → CSV 路径）
STOCK_OPTIONS = {
    "SMIC A (中芯国际 688981.SH)":    "outputs/SMIC_A_daily.csv",
    "SMIC H (中芯国际 00981.HK)":     "outputs/SMIC_H_daily.csv",
    "BYD A (比亚迪 002594.SZ)":       "outputs/BYD_A_daily.csv",
    "BYD H (比亚迪 01211.HK)":        "outputs/BYD_H_daily.csv",
    "CJDL A (长江电力 600900.SH)":    "outputs/CJDL_A_daily.csv",
}


def run_backtest(stock: str, start_date: str, end_date: str,
                 short_ma: int, long_ma: int,
                 commission_rate: float, slippage_rate: float,
                 initial_capital: float):
    """对指定参数运行回测，返回前端所需的 JSON 数据"""
    csv_path = Path(__file__).parent / STOCK_OPTIONS[stock]
    strat = DualMAStrategy(str(csv_path), name=stock)

    # 日期过滤
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    mask = (strat.raw["trade_date"] >= start) & (strat.raw["trade_date"] <= end)
    filtered = strat.raw[mask].copy()
    if filtered.empty:
        raise ValueError("所选日期范围内无数据")

    strat.df = filtered[["trade_date", "open", "high", "low", "close", "vol"]].copy()
    strat.df = strat.df.dropna(subset=["close"]).sort_values("trade_date").reset_index(drop=True)
    strat.df["returns"] = strat.df["close"].pct_change()
    strat.compute_signals(short_period=short_ma, long_period=long_ma)
    strat.backtest(initial_capital=initial_capital,
                   commission=commission_rate + slippage_rate)

    df = strat.df.copy()
    btdf = strat.backtest_df.copy()

    # 资金曲线
    equity_curve = btdf["equity"].tolist()
    equity_dates = btdf["trade_date"].dt.strftime("%Y-%m-%d").tolist()
    # 基准（买入持有）
    benchmark = ((1 + df["returns"].fillna(0)).cumprod()).tolist()
    benchmark_norm = [(v + 0) for v in benchmark]  # 已是 (1+r) cumprod

    # 资金曲线标准化
    eq_norm = [e / equity_curve[0] for e in equity_curve]
    bm_norm = benchmark  # 已经 (1+r) cumprod，从 1.0 开始

    # 回撤
    eq_arr = btdf["equity"].values
    peak = [max(eq_arr[: i + 1]) for i in range(len(eq_arr))]
    drawdown = [(eq_arr[i] - peak[i]) / peak[i] * 100 for i in range(len(eq_arr))]

    # 价格 + 均线 + 信号（K 线图简化用 candlestick 数据）
    ohlc = {
        "dates":  df["trade_date"].dt.strftime("%Y-%m-%d").tolist(),
        "open":   df["open"].tolist(),
        "high":   df["high"].tolist(),
        "low":    df["low"].tolist(),
        "close":  df["close"].tolist(),
        "ma_short": df["ma_short"].fillna("").tolist(),
        "ma_long":  df["ma_long"].fillna("").tolist(),
        "buy_signals":  df.loc[df["buy"],  "trade_date"].dt.strftime("%Y-%m-%d").tolist(),
        "buy_prices":   df.loc[df["buy"],  "close"].tolist(),
        "sell_signals": df.loc[df["sell"], "trade_date"].dt.strftime("%Y-%m-%d").tolist(),
        "sell_prices":  df.loc[df["sell"], "close"].tolist(),
    }

    metrics = strat.metrics
    # 补充年化收益（无风险利率用 3%）
    metrics["alpha_pct"] = round(metrics["annual_return_pct"] - 8.5, 2)  # 简单相对基准

    return {
        "config": {
            "stock": stock,
            "start_date": start_date,
            "end_date": end_date,
            "short_ma": short_ma,
            "long_ma": long_ma,
            "commission_rate": commission_rate,
            "slippage_rate": slippage_rate,
            "initial_capital": initial_capital,
        },
        "metrics": metrics,
        "equity_chart": {
            "dates": equity_dates,
            "strategy": eq_norm,
            "benchmark": bm_norm,
        },
        "drawdown_chart": {
            "dates": equity_dates,
            "drawdown": drawdown,
        },
        "price_chart": ohlc,
    }


# ── 简易 HTTP 服务器 ──
import http.server
import socketserver
import urllib.parse
import pandas as pd  # noqa: E402


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence logs
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/backtest":
            qs = urllib.parse.parse_qs(parsed.query)
            try:
                result = run_backtest(
                    stock=qs["stock"][0],
                    start_date=qs["start_date"][0],
                    end_date=qs["end_date"][0],
                    short_ma=int(qs["short_ma"][0]),
                    long_ma=int(qs["long_ma"][0]),
                    commission_rate=float(qs.get("commission", ["0.0003"])[0]),
                    slippage_rate=float(qs.get("slippage", ["0.0001"])[0]),
                    initial_capital=float(qs.get("capital", ["100000"])[0]),
                )
                body = json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def main():
    port = 8765
    print(f"Dashboard server on http://localhost:{port}")
    with socketserver.TCPServer(("", port), Handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
