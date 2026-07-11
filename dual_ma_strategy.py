#!/usr/bin/env python3
"""
TASK3 — 双均线交叉策略（Dual Moving Average Crossover Strategy）
================================================================
- 加载股价数据
- 计算短/长均线
- 生成金叉/死叉买卖信号
- 模拟交易回测 & 计算量化指标（累计回报 / 夏普比率 / 最大回撤）
- 可视化：K线 + 均线 + 交易信号
- 多股票、多周期对比分析
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import warnings
import os

warnings.filterwarnings("ignore")

# ── 中国股市配色：红涨绿跌 ──
RED_UP   = "#DC143C"
GREEN_DN = "#228B22"
BG_COLOR = "#FAFAFA"


# ============================================================
#  PART 1: 策略引擎
# ============================================================

class DualMAStrategy:
    """
    双均线交叉策略
    - 金叉 (short_ma 上穿 long_ma) → 买入信号
    - 死叉 (short_ma 下穿 long_ma) → 卖出信号
    """

    def __init__(self, csv_path: str, name: str = ""):
        self.name = name
        self.raw = pd.read_csv(csv_path, parse_dates=["trade_date"])
        self.df = self.raw[["trade_date", "open", "high", "low", "close", "vol"]].copy()
        self.df = self.df.dropna(subset=["close"]).sort_values("trade_date").reset_index(drop=True)
        self.df["returns"] = self.df["close"].pct_change()
        self.results = None

    # ── 均线 & 信号 ──

    def compute_signals(self, short_period: int = 5, long_period: int = 15):
        df = self.df.copy()
        df["ma_short"] = df["close"].rolling(short_period).mean()
        df["ma_long"]  = df["close"].rolling(long_period).mean()
        # 交叉信号
        df["cross"] = df["ma_short"] - df["ma_long"]
        df["signal"] = 0
        df.loc[df["cross"] > 0, "signal"] = 1   # 短均线在上 → 持仓
        df.loc[df["cross"] < 0, "signal"] = -1   # 短均线在下 → 空仓
        df["trade_signal"] = df["signal"].diff()
        # trade_signal: +2 = 金叉买入, -2 = 死叉卖出
        df["buy"]  = df["trade_signal"] == 2
        df["sell"] = df["trade_signal"] == -2
        self.short_period = short_period
        self.long_period  = long_period
        self.df = df

    # ── 回测（单边做多） ──

    def backtest(self, initial_capital: float = 1_000_000.0, commission: float = 0.0003):
        df = self.df.copy()
        cash = initial_capital
        shares = 0
        equity_curve = []
        in_position = False

        for i in range(len(df)):
            price = df.loc[i, "close"]
            # 买入信号
            if df.loc[i, "buy"] and not in_position:
                cost = commission * price
                shares = cash / (price + cost) if price > 0 else 0
                cash = 0
                in_position = True
            # 卖出信号
            elif df.loc[i, "sell"] and in_position:
                cash = shares * price * (1 - commission)
                shares = 0
                in_position = False

            equity = cash + shares * price
            equity_curve.append(equity)

        df["equity"] = equity_curve
        df["equity_returns"] = pd.Series(equity_curve).pct_change().fillna(0)
        df["in_position"] = np.nan
        pos_start = None
        for i in range(len(df)):
            if df.loc[i, "buy"] and pos_start is None:
                pos_start = i
            elif df.loc[i, "sell"] and pos_start is not None:
                df.loc[pos_start:i, "in_position"] = True
                pos_start = None
            if pos_start is not None:
                df.loc[pos_start, "in_position"] = True

        self.backtest_df = df
        self.final_equity = equity_curve[-1]
        self.equity_curve = np.array(equity_curve)
        self.metrics = self._calc_metrics(initial_capital)
        self.results = {
            "name": self.name,
            "short": self.short_period,
            "long": self.long_period,
            **self.metrics,
        }
        return self.results

    def _calc_metrics(self, initial_capital):
        eq = self.equity_curve
        eq_ret = pd.Series(eq).pct_change().fillna(0)
        benchmark_ret = self.df["returns"].fillna(0)
        total_return = (eq[-1] / initial_capital - 1) * 100
        benchmark_return = (1 + benchmark_ret).prod() - 1
        benchmark_return_pct = benchmark_return * 100

        # 年化收益率（假设 252 个交易日）
        n_days = len(eq)
        years = n_days / 252
        annual_return = (eq[-1] / initial_capital) ** (1 / max(years, 1e-6)) - 1

        # 最大回撤 (MDD)
        peak = np.maximum.accumulate(eq)
        drawdown = (eq - peak) / peak
        mdd = drawdown.min() * 100

        # 夏普比率（无风险利率设为 0.03）
        rf_daily = 0.03 / 252
        excess = eq_ret - rf_daily
        sharpe = excess.mean() / max(excess.std(), 1e-9) * np.sqrt(252)

        # 胜率 & 交易次数
        buys  = self.df["buy"].sum()
        sells = self.df["sell"].sum()
        trades = min(buys, sells)

        # 每个交易对的盈亏
        buy_prices = self.df.loc[self.df["buy"], "close"].values
        sell_prices = self.df.loc[self.df["sell"], "close"].values
        n_pairs = min(len(buy_prices), len(sell_prices))
        if n_pairs > 0:
            pair_returns = (sell_prices[:n_pairs] / buy_prices[:n_pairs] - 1)
            win_rate = (pair_returns > 0).mean() * 100
            avg_win  = pair_returns[pair_returns > 0].mean() * 100 if (pair_returns > 0).any() else 0
            avg_loss = pair_returns[pair_returns < 0].mean() * 100 if (pair_returns < 0).any() else 0
        else:
            win_rate = avg_win = avg_loss = np.nan

        # 波动率
        volatility = eq_ret.std() * np.sqrt(252) * 100

        return {
            "cumulative_return_pct": round(total_return, 2),
            "annual_return_pct": round(annual_return * 100, 2),
            "benchmark_return_pct": round(benchmark_return_pct, 2),
            "mdd_pct": round(mdd, 2),
            "sharpe_ratio": round(sharpe, 3),
            "volatility_pct": round(volatility, 2),
            "n_trades": int(trades),
            "win_rate_pct": round(win_rate, 1) if not np.isnan(win_rate) else None,
            "avg_win_pct": round(avg_win, 2) if not np.isnan(avg_win) else None,
            "avg_loss_pct": round(avg_loss, 2) if not np.isnan(avg_loss) else None,
        }


# ============================================================
#  PART 2: 可视化
# ============================================================

def plot_strategy(strategy: DualMAStrategy, output_path: str):
    """绘制完整的策略可视化：K线+均线+信号+资金曲线+回撤"""
    df = strategy.df.copy()
    btdf = getattr(strategy, "backtest_df", None)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                             gridspec_kw={"height_ratios": [3, 1.2, 1]},
                             sharex=True)
    fig.patch.set_facecolor(BG_COLOR)

    title = (f"{strategy.name} 双均线策略  |  "
             f"短均线 MA({strategy.short_period})  长均线 MA({strategy.long_period})")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)

    # ── 子图1：股价 + 均线 + 买卖信号 ──
    ax1 = axes[0]
    ax1.set_facecolor(BG_COLOR)
    valid = df.dropna(subset=["ma_short", "ma_long"])
    ax1.plot(df["trade_date"], df["close"], color="#333333", linewidth=0.9, alpha=0.7, label="Close Price")
    ax1.plot(valid["trade_date"], valid["ma_short"], color=RED_UP, linewidth=1.3, label=f"MA({strategy.short_period})")
    ax1.plot(valid["trade_date"], valid["ma_long"], color="#1565C0", linewidth=1.3, label=f"MA({strategy.long_period})")

    # 买入/卖出标记
    buy_dates  = df.loc[df["buy"],  "trade_date"]
    buy_prices = df.loc[df["buy"],  "close"]
    sell_dates = df.loc[df["sell"], "trade_date"]
    sell_prices = df.loc[df["sell"], "close"]

    ax1.scatter(buy_dates, buy_prices, marker="^", color=RED_UP, s=80,
                zorder=5, edgecolors="white", linewidths=0.5, label=f"Buy ({len(buy_dates)})")
    ax1.scatter(sell_dates, sell_prices, marker="v", color=GREEN_DN, s=80,
                zorder=5, edgecolors="white", linewidths=0.5, label=f"Sell ({len(sell_dates)})")

    # 持仓区间浅色填充
    if btdf is not None and "in_position" in btdf.columns:
        pos_mask = btdf["in_position"].fillna(False)
        in_pos = False
        start_i = None
        for i in range(len(btdf)):
            if pos_mask.iloc[i] and not in_pos:
                start_i = i
                in_pos = True
            elif not pos_mask.iloc[i] and in_pos:
                ax1.axvspan(btdf.loc[start_i, "trade_date"],
                            btdf.loc[i, "trade_date"],
                            color=RED_UP, alpha=0.06, lw=0)
                in_pos = False
        if in_pos and start_i is not None:
            ax1.axvspan(btdf.loc[start_i, "trade_date"],
                        btdf.loc[len(btdf)-1, "trade_date"],
                        color=RED_UP, alpha=0.06, lw=0)

    ax1.set_ylabel("Price (¥)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9, ncol=4, framealpha=0.9)
    ax1.grid(True, alpha=0.25)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"¥{x:.1f}"))

    # ── 子图2：资金曲线 vs 买入持有 ──
    ax2 = axes[1]
    ax2.set_facecolor(BG_COLOR)
    if btdf is not None:
        eq_norm = btdf["equity"] / btdf["equity"].iloc[0]
        # benchmark
        bm = (1 + df["returns"].fillna(0)).cumprod()
        ax2.plot(btdf["trade_date"], eq_norm, color=RED_UP, linewidth=1.5, label="Strategy Equity")
        ax2.plot(df["trade_date"], bm, color="#555555", linewidth=0.8, linestyle="--", alpha=0.7, label="Buy & Hold")
        ax2.fill_between(btdf["trade_date"], eq_norm, 1.0,
                         where=(eq_norm >= 1.0), color=RED_UP, alpha=0.08)
        ax2.fill_between(btdf["trade_date"], eq_norm, 1.0,
                         where=(eq_norm < 1.0), color=GREEN_DN, alpha=0.08)

    ax2.axhline(1.0, color="black", linewidth=0.5, linestyle="-", alpha=0.5)
    ax2.set_ylabel("Equity (×)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.25)

    # ── 子图3：回撤曲线 ──
    ax3 = axes[2]
    ax3.set_facecolor(BG_COLOR)
    if btdf is not None:
        eq = btdf["equity"].values
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        ax3.fill_between(btdf["trade_date"], dd, 0, color=GREEN_DN, alpha=0.35, label="Drawdown")
        ax3.plot(btdf["trade_date"], dd, color=GREEN_DN, linewidth=0.7)
    ax3.set_ylabel("Drawdown (%)", fontsize=11)
    ax3.set_xlabel("Date", fontsize=11)
    ax3.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.25)
    ax3.axhline(0, color="black", linewidth=0.5)

    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  图表已保存: {output_path}")


def plot_comparison_heatmap(results_df: pd.DataFrame, output_path: str):
    """多参数组合对比热力图"""
    if results_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Multi-Stock / Multi-Period Strategy Comparison", fontsize=14, fontweight="bold")

    metrics_config = [
        ("cumulative_return_pct", "Cumulative Return (%)", "RdBu_r"),
        ("sharpe_ratio",            "Sharpe Ratio",           "RdBu_r"),
        ("mdd_pct",                 "Max Drawdown (%)",       "RdBu"),
        ("win_rate_pct",            "Win Rate (%)",           "RdBu_r"),
    ]

    for ax, (col, title, cmap) in zip(axes.flat, metrics_config):
        pivot = results_df.pivot_table(values=col, index="name", columns="period_label", aggfunc="first")
        if not pivot.empty and col == "mdd_pct":
            # MDD 越小越好，反转颜色映射
            cmap = "RdBu_r"
        if not pivot.empty:
            im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, fontsize=8, rotation=30, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)
            ax.set_title(title, fontsize=11, fontweight="bold")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = pivot.iloc[i, j]
                    txt = f"{v:.1f}" if not pd.isna(v) else "N/A"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                            color="white" if not pd.isna(v) and abs(v) > 10 else "black")
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=12)
            ax.set_title(title, fontsize=11)
        ax.set_facecolor(BG_COLOR)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  对比热力图已保存: {output_path}")


# ============================================================
#  PART 3: 主流程
# ============================================================

def main():
    OUTPUT_DIR = Path("/Users/ailsa/Desktop/ai  working/outputs")
    STRATEGY_DIR = Path("/Users/ailsa/Desktop/ai  working/outputs/task3_strategy")
    STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

    # ── 数据文件列表 ──
    stock_files = [
        ("SMIC A",    OUTPUT_DIR / "SMIC_A_daily.csv"),
        ("SMIC H",    OUTPUT_DIR / "SMIC_H_daily.csv"),
        ("BYD A",     OUTPUT_DIR / "BYD_A_daily.csv"),
        ("BYD H",     OUTPUT_DIR / "BYD_H_daily.csv"),
        ("CJDL A",    OUTPUT_DIR / "CJDL_A_daily.csv"),
    ]

    # ── 均线周期组合 ──
    period_pairs = [
        (5, 15),
        (5, 20),
        (10, 30),
        (10, 60),
    ]

    all_results = []

    print("=" * 70)
    print("  TASK3 — 双均线交叉策略：全股票 / 全周期回测")
    print("=" * 70)

    for name, csv_path in stock_files:
        if not csv_path.exists():
            print(f"\n  ⚠ 跳过 {name}: 文件不存在 {csv_path}")
            continue

        print(f"\n{'─'*60}")
        print(f"  股票: {name}")
        print(f"{'─'*60}")

        for short_p, long_p in period_pairs:
            label = f"MA{short_p}/{long_p}"
            print(f"\n  ▶ {name}  {label} ...", end=" ")

            strat = DualMAStrategy(str(csv_path), name=name)
            strat.compute_signals(short_period=short_p, long_period=long_p)
            result = strat.backtest()

            period_label = f"MA({short_p},{long_p})"
            result["period_label"] = period_label
            all_results.append(result)

            # 指标摘要
            m = result
            print(f"✓")
            print(f"    累计回报: {m['cumulative_return_pct']:+.2f}%  |  "
                  f"基准: {m['benchmark_return_pct']:+.2f}%")
            print(f"    年化收益: {m['annual_return_pct']:+.2f}%  |  "
                  f"夏普比率: {m['sharpe_ratio']:.3f}  |  "
                  f"MDD: {m['mdd_pct']:.2f}%")
            print(f"    交易次数: {m['n_trades']}  |  "
                  f"胜率: {m['win_rate_pct']:.1f}%  |  "
                  f"波动率: {m['volatility_pct']:.2f}%")

            # 绘制图表
            chart_path = STRATEGY_DIR / f"{name.replace(' ','_')}_{short_p}_{long_p}.png"
            plot_strategy(strat, str(chart_path))

    # ── 汇总表 ──
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df = results_df.sort_values(["name", "short"])
        csv_path = STRATEGY_DIR / "all_results.csv"
        results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n{'='*70}")
        print(f"  全部结果汇总: {csv_path}")
        print(f"{'='*70}")

        # 热力图
        heatmap_path = STRATEGY_DIR / "comparison_heatmap.png"
        plot_comparison_heatmap(results_df, str(heatmap_path))

    print(f"\n✅ TASK3 完成 — 输出目录: {STRATEGY_DIR}")
    return results_df


if __name__ == "__main__":
    results = main()
