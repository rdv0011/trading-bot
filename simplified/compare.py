"""
Comparison module for Sim vs Demo Trading.
Compares simulation trades against real demo trading logs.
"""

import os
import re
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from logging import log_info, log_debug, log_warning


# ── Demo Log Parser ─────────────────────────────────────────────────────
def parse_demo_logs(
    log_dir: str = "logs",
    pattern: str = "trading_*.log",
) -> pd.DataFrame:
    """
    Parse demo trading log files into trades DataFrame.

    Expected log format (from current system):
    2024-01-15 14:30:00 | INFO    | TRADE | ENTER LONG @ 42000.50 | Qty: 0.01 | PnL: 0.0

    Returns DataFrame with columns:
    timestamp, symbol, side, entry_price, exit_price, qty, pnl, regime, exit_reason
    """
    log_files = sorted(Path(log_dir).glob(pattern))

    if not log_files:
        log_warning(f"No demo log files found matching {pattern} in {log_dir}")
        return pd.DataFrame()

    log_info(f"Found {len(log_files)} demo log files")

    trades = []

    for log_file in log_files:
        log_info(f"Parsing {log_file.name}...")

        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse trades using regex patterns
        # Pattern 1: Trade entries
        entry_pattern = re.compile(
            r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?"
            r"TRADE.*?"
            r"ENTER\s+(LONG|SHORT)\s+@\s+([\d.]+).*?"
            r"Qty:\s+([\d.]+).*?"
            r"PnL:\s+(-?[\d.]+)",
            re.IGNORECASE
        )

        # Pattern 2: Trade exits
        exit_pattern = re.compile(
            r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?"
            r"TRADE.*?"
            r"EXIT\s+(LONG|SHORT)\s+@\s+([\d.]+).*?"
            r"Regime:\s+(\w+).*?"
            r"Reason:\s+(\w+)",
            re.IGNORECASE
        )

        # Pattern 3: Meta-params
        meta_pattern = re.compile(
            r"Meta:\s+({.*?})",
            re.IGNORECASE
        )

        entries = entry_pattern.findall(content)
        exits = exit_pattern.findall(content)

        # Match entries with exits (FIFO assumption)
        entry_idx = 0
        for exit_match in exits:
            exit_time, exit_side, exit_price, regime, reason = exit_match

            if entry_idx < len(entries):
                entry_time, entry_side, entry_price, qty, _ = entries[entry_idx]

                if entry_side.lower() == exit_side.lower():
                    # Calculate PnL from prices
                    entry_price = float(entry_price)
                    exit_price = float(exit_price)
                    qty = float(qty)

                    if entry_side.upper() == "LONG":
                        pnl = (exit_price - entry_price) * qty
                    else:
                        pnl = (entry_price - exit_price) * qty

                    trades.append({
                        "timestamp": entry_time,
                        "exit_timestamp": exit_time,
                        "symbol": "BTCUSDT",
                        "side": entry_side.upper(),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "qty": qty,
                        "pnl": round(pnl, 6),
                        "regime": regime,
                        "exit_reason": reason,
                        "source": "demo",
                    })

                    entry_idx += 1

        # Handle any remaining entries without exits
        while entry_idx < len(entries):
            entry_time, entry_side, entry_price, qty, _ = entries[entry_idx]
            trades.append({
                "timestamp": entry_time,
                "exit_timestamp": None,
                "symbol": "BTCUSDT",
                "side": entry_side.upper(),
                "entry_price": float(entry_price),
                "exit_price": None,
                "qty": float(qty),
                "pnl": None,
                "regime": None,
                "exit_reason": "open",
                "source": "demo",
            })
            entry_idx += 1

    df = pd.DataFrame(trades)

    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'exit_timestamp' in df.columns:
            df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'], errors='coerce')

    log_info(f"Parsed {len(df)} demo trades")
    return df


# ── Simulation Log Parser ───────────────────────────────────────────────
def parse_sim_logs(
    log_dir: str = "logs",
    pattern: str = "trades_sim_*.csv",
) -> pd.DataFrame:
    """
    Parse simulation trade CSV logs.
    Expected CSV format from simulate.py's MockBroker.

    Returns DataFrame with same schema as demo trades.
    """
    csv_files = sorted(Path(log_dir).glob(pattern))

    if not csv_files:
        log_warning(f"No sim log files found matching {pattern} in {log_dir}")
        return pd.DataFrame()

    log_info(f"Found {len(csv_files)} sim log files")

    dfs = []
    for csv_file in csv_files:
        log_info(f"Loading {csv_file.name}...")
        try:
            df = pd.read_csv(csv_file)
            df['source'] = 'sim'
            dfs.append(df)
        except Exception as e:
            log_warning(f"Error loading {csv_file}: {e}")

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Ensure required columns
    required = ['timestamp', 'symbol', 'side', 'entry_price', 'exit_price', 'qty', 'pnl', 'regime', 'exit_reason']
    for col in required:
        if col not in df.columns:
            df[col] = None

    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    log_info(f"Loaded {len(df)} sim trades")
    return df


# ── Trade Alignment ─────────────────────────────────────────────────────
def align_trades(
    demo_trades: pd.DataFrame,
    sim_trades: pd.DataFrame,
    time_tolerance_seconds: int = 300,  # 5 minutes
) -> pd.DataFrame:
    """
    Align demo and sim trades by timestamp for comparison.

    Returns DataFrame with columns:
    - demo_*: Demo trade metrics
    - sim_*: Sim trade metrics
    - diff_*: Differences between demo and sim
    """
    if demo_trades.empty or sim_trades.empty:
        log_warning("One or both trade datasets are empty")
        return pd.DataFrame()

    log_info(f"Aligning trades: {len(demo_trades)} demo vs {len(sim_trades)} sim")

    # Ensure sorted by timestamp
    demo = demo_trades.sort_values('timestamp').reset_index(drop=True)
    sim = sim_trades.sort_values('timestamp').reset_index(drop=True)

    aligned = []

    for _, demo_row in demo.iterrows():
        # Find closest sim trade within tolerance
        time_diffs = abs((sim['timestamp'] - demo_row['timestamp']).dt.total_seconds())
        best_idx = time_diffs.idxmin()
        best_diff = time_diffs.min()

        if best_diff <= time_tolerance_seconds:
            sim_row = sim.iloc[best_idx]

            # Calculate differences
            entry_diff = 0.0
            exit_diff = 0.0
            pnl_diff = 0.0

            if pd.notna(demo_row['entry_price']) and pd.notna(sim_row.get('entry_price')):
                entry_diff = sim_row['entry_price'] - demo_row['entry_price']

            if pd.notna(demo_row['exit_price']) and pd.notna(sim_row.get('exit_price')):
                exit_diff = sim_row['exit_price'] - demo_row['exit_price']

            if pd.notna(demo_row['pnl']) and pd.notna(sim_row.get('pnl')):
                pnl_diff = sim_row['pnl'] - demo_row['pnl']

            aligned.append({
                'demo_timestamp': demo_row['timestamp'],
                'sim_timestamp': sim_row['timestamp'] if 'timestamp' in sim_row else None,
                'time_diff_seconds': best_diff,
                'side': demo_row['side'],
                'demo_entry': demo_row['entry_price'],
                'sim_entry': sim_row.get('entry_price'),
                'entry_diff': entry_diff,
                'demo_exit': demo_row['exit_price'],
                'sim_exit': sim_row.get('exit_price'),
                'exit_diff': exit_diff,
                'demo_pnl': demo_row['pnl'],
                'sim_pnl': sim_row.get('pnl'),
                'pnl_diff': pnl_diff,
                'demo_regime': demo_row['regime'],
                'sim_regime': sim_row.get('regime'),
                'demo_reason': demo_row['exit_reason'],
                'sim_reason': sim_row.get('exit_reason'),
            })

    df_aligned = pd.DataFrame(aligned)
    log_info(f"Aligned {len(df_aligned)} trade pairs")
    return df_aligned


# ── Metrics Comparison ──────────────────────────────────────────────────
def compare_metrics(
    demo_trades: pd.DataFrame,
    sim_trades: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compare aggregate metrics between demo and sim.
    Returns dict with demo, sim, and diff metrics.
    """
    metrics = {
        'demo': {},
        'sim': {},
        'diff': {},
    }

    for name, df in [('demo', demo_trades), ('sim', sim_trades)]:
        if df.empty or 'pnl' not in df.columns:
            metrics[name] = {
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_trade_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
            }
            continue

        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] <= 0]

        total_pnl = df['pnl'].sum()
        win_rate = len(wins) / len(df) if len(df) > 0 else 0.0
        avg_trade = df['pnl'].mean()
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0.0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0.0
        gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 1.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Max drawdown from cumulative PnL
        cum_pnl = df['pnl'].cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        metrics[name] = {
            'total_return': round(total_pnl, 6),
            'total_return_pct': round(total_pnl / 1000, 6) if total_pnl != 0 else 0.0,  # Assuming $1000 initial
            'num_trades': len(df),
            'win_rate': round(win_rate, 4),
            'avg_trade_pnl': round(avg_trade, 6),
            'avg_win': round(avg_win, 6),
            'avg_loss': round(avg_loss, 6),
            'profit_factor': round(profit_factor, 4),
            'max_drawdown': round(max_dd, 6),
        }

    # Calculate differences
    for key in metrics['demo']:
        if isinstance(metrics['demo'][key], (int, float)):
            metrics['diff'][key] = round(
                metrics['sim'][key] - metrics['demo'][key], 6
            )
        else:
            metrics['diff'][key] = None

    return metrics


# ── Regime Breakdown ────────────────────────────────────────────────────
def regime_breakdown(
    demo_trades: pd.DataFrame,
    sim_trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare performance by regime (trend, chop, high_vol).
    Returns DataFrame with regime-level metrics for both demo and sim.
    """
    regimes = ['trend', 'chop', 'high_vol']
    rows = []

    for regime in regimes:
        demo_r = demo_trades[demo_trades['regime'] == regime]
        sim_r = sim_trades[sim_trades['regime'] == regime]

        demo_win = len(demo_r[demo_r['pnl'] > 0]) / len(demo_r) if len(demo_r) > 0 else 0
        sim_win = len(sim_r[sim_r['pnl'] > 0]) / len(sim_r) if len(sim_r) > 0 else 0

        rows.append({
            'regime': regime,
            'demo_count': len(demo_r),
            'sim_count': len(sim_r),
            'demo_pnl': round(demo_r['pnl'].sum(), 6) if len(demo_r) > 0 else 0,
            'sim_pnl': round(sim_r['pnl'].sum(), 6) if len(sim_r) > 0 else 0,
            'demo_win_rate': round(demo_win, 4),
            'sim_win_rate': round(sim_win, 4),
            'pnl_diff': round(
                (sim_r['pnl'].sum() if len(sim_r) > 0 else 0) -
                (demo_r['pnl'].sum() if len(demo_r) > 0 else 0),
                6
            ),
        })

    return pd.DataFrame(rows)


# ── Report Generation ───────────────────────────────────────────────────
def generate_comparison_report(
    demo_trades: pd.DataFrame,
    sim_trades: pd.DataFrame,
    aligned: pd.DataFrame,
    metrics: Dict[str, Any],
    output_path: str = "logs/comparison_report.html",
) -> str:
    """
    Generate HTML comparison report with:
    - Summary metrics table
    - Equity curves overlay
    - Trade-by-trade comparison
    - Regime breakdown
    - Slippage/fee analysis
    """
    regime_df = regime_breakdown(demo_trades, sim_trades)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Sim vs Demo Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .positive {{ color: green; font-weight: bold; }}
        .negative {{ color: red; font-weight: bold; }}
        .metric-box {{ display: inline-block; margin: 10px; padding: 15px; background: #f9f9f9; border-radius: 5px; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Simulation vs Demo Trading Comparison</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <h2>Summary Metrics</h2>
    <div>
        <div class="metric-box">
            <div class="metric-label">Demo Trades</div>
            <div class="metric-value">{metrics['demo']['num_trades']}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Sim Trades</div>
            <div class="metric-value">{metrics['sim']['num_trades']}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Demo Total PnL</div>
            <div class="metric-value {'positive' if metrics['demo']['total_return'] > 0 else 'negative'}">
                ${metrics['demo']['total_return']:.2f}
            </div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Sim Total PnL</div>
            <div class="metric-value {'positive' if metrics['sim']['total_return'] > 0 else 'negative'}">
                ${metrics['sim']['total_return']:.2f}
            </div>
        </div>
    </div>

    <h2>Detailed Comparison</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Demo</th>
            <th>Sim</th>
            <th>Difference</th>
        </tr>"""

    for key in ['total_return', 'num_trades', 'win_rate', 'avg_trade_pnl', 'profit_factor', 'max_drawdown']:
        demo_val = metrics['demo'][key]
        sim_val = metrics['sim'][key]
        diff_val = metrics['diff'][key]

        # Format based on type
        if isinstance(demo_val, float):
            if 'pct' in key or 'rate' in key or 'drawdown' in key:
                fmt = lambda x: f"{x*100:.2f}%"
            else:
                fmt = lambda x: f"${x:.2f}"
        else:
            fmt = lambda x: f"{x}"

        html += f"""
        <tr>
            <td>{key}</td>
            <td>{fmt(demo_val)}</td>
            <td>{fmt(sim_val)}</td>
            <td class="{'positive' if diff_val > 0 else 'negative' if diff_val < 0 else ''}">{fmt(diff_val)}</td>
        </tr>"""

    html += """
    </table>

    <h2>Regime Breakdown</h2>
    <table>
        <tr>
            <th>Regime</th>
            <th>Demo Count</th>
            <th>Sim Count</th>
            <th>Demo PnL</th>
            <th>Sim PnL</th>
            <th>Demo Win Rate</th>
            <th>Sim Win Rate</th>
        </tr>"""

    for _, row in regime_df.iterrows():
        html += f"""
        <tr>
            <td>{row['regime']}</td>
            <td>{row['demo_count']}</td>
            <td>{row['sim_count']}</td>
            <td class="{'positive' if row['demo_pnl'] > 0 else 'negative'}">${row['demo_pnl']:.2f}</td>
            <td class="{'positive' if row['sim_pnl'] > 0 else 'negative'}">${row['sim_pnl']:.2f}</td>
            <td>{row['demo_win_rate']*100:.1f}%</td>
            <td>{row['sim_win_rate']*100:.1f}%</td>
        </tr>"""

    html += """
    </table>

    <h2>Trade-by-Trade Comparison</h2>
    <table>
        <tr>
            <th>Time</th>
            <th>Side</th>
            <th>Demo Entry</th>
            <th>Sim Entry</th>
            <th>Entry Diff</th>
            <th>Demo Exit</th>
            <th>Sim Exit</th>
            <th>Exit Diff</th>
            <th>Demo PnL</th>
            <th>Sim PnL</th>
            <th>PnL Diff</th>
            <th>Reason</th>
        </tr>"""

    if not aligned.empty:
        for _, row in aligned.head(100).iterrows():  # Limit to 100 trades for HTML
            html += f"""
        <tr>
            <td>{row['demo_timestamp'].strftime('%Y-%m-%d %H:%M')}</td>
            <td>{row['side']}</td>
            <td>${row['demo_entry']:.2f}</td>
            <td>${row['sim_entry']:.2f if pd.notna(row['sim_entry']) else 'N/A'}</td>
            <td class="{'positive' if row['entry_diff'] > 0 else 'negative'}">${row['entry_diff']:.2f}</td>
            <td>${row['demo_exit']:.2f if pd.notna(row['demo_exit']) else 'N/A'}</td>
            <td>${row['sim_exit']:.2f if pd.notna(row['sim_exit']) else 'N/A'}</td>
            <td class="{'positive' if row['exit_diff'] > 0 else 'negative'}">${row['exit_diff']:.2f}</td>
            <td class="{'positive' if row['demo_pnl'] > 0 else 'negative'}">${row['demo_pnl']:.2f if pd.notna(row['demo_pnl']) else 'N/A'}</td>
            <td class="{'positive' if row['sim_pnl'] > 0 else 'negative'}">${row['sim_pnl']:.2f if pd.notna(row['sim_pnl']) else 'N/A'}</td>
            <td class="{'positive' if row['pnl_diff'] > 0 else 'negative'}">${row['pnl_diff']:.2f}</td>
            <td>{row['demo_reason']}</td>
        </tr>"""

    html += """
    </table>

    <h2>Analysis Notes</h2>
    <ul>
        <li><strong>Entry Price Diff:</strong> Positive = Sim got better entry, Negative = Demo got better entry</li>
        <li><strong>Exit Price Diff:</strong> Positive = Sim got better exit, Negative = Demo got better exit</li>
        <li><strong>PnL Diff:</strong> Positive = Sim performed better, Negative = Demo performed better</li>
    </ul>
</body>
</html>"""

    # Write report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    log_info(f"Comparison report saved: {output_path}")
    return output_path


# ── Console Summary ─────────────────────────────────────────────────────
def print_comparison_summary(
    demo_trades: pd.DataFrame,
    sim_trades: pd.DataFrame,
    metrics: Dict[str, Any],
) -> None:
    """Print brief comparison summary to console."""
    log_info("=" * 60)
    log_info("COMPARISON SUMMARY: Demo vs Simulation")
    log_info("=" * 60)

    log_info(f"Demo Trades:  {metrics['demo']['num_trades']}")
    log_info(f"Sim Trades:   {metrics['sim']['num_trades']}")

    # Color coding
    demo_pnl = metrics['demo']['total_return']
    sim_pnl = metrics['sim']['total_return']

    log_info(f"Demo Total PnL: ${demo_pnl:.2f} ({'+' if demo_pnl > 0 else ''}{demo_pnl*100:.1f}%)")
    log_info(f"Sim Total PnL:  ${sim_pnl:.2f} ({'+' if sim_pnl > 0 else ''}{sim_pnl*100:.1f}%)")

    # Win rates
    log_info(f"Demo Win Rate: {metrics['demo']['win_rate']*100:.1f}%")
    log_info(f"Sim Win Rate:  {metrics['sim']['win_rate']*100:.1f}%")

    # Key differences
    diff_pnl = metrics['diff']['total_return']
    diff_wr = metrics['diff']['win_rate']

    log_info("-" * 60)
    log_info(f"PnL Difference:   ${diff_pnl:.2f} (Sim {'+' if diff_pnl > 0 else ''}{diff_pnl*100:.1f}% vs Demo)")
    log_info(f"Win Rate Diff:    {diff_wr*100:+.1f}%")
    log_info("=" * 60)


# ── Main Entry Point ────────────────────────────────────────────────────
def run_comparison(
    log_dir: str = "logs",
    demo_pattern: str = "trading_*.log",
    sim_pattern: str = "trades_sim_*.csv",
    output_html: str = "logs/comparison_report.html",
) -> Dict[str, Any]:
    """
    Main entry point for comparison mode.
    Parses logs, aligns trades, compares metrics, generates report.
    """
    log_info("=" * 60)
    log_info("Starting Sim vs Demo Comparison")
    log_info("=" * 60)

    # Parse logs
    demo_trades = parse_demo_logs(log_dir, demo_pattern)
    sim_trades = parse_sim_logs(log_dir, sim_pattern)

    if demo_trades.empty and sim_trades.empty:
        log_error("No trades found in either demo or sim logs")
        return {}

    # Align trades
    aligned = align_trades(demo_trades, sim_trades)

    # Compare metrics
    metrics = compare_metrics(demo_trades, sim_trades)

    # Print summary
    print_comparison_summary(demo_trades, sim_trades, metrics)

    # Generate report
    report_path = generate_comparison_report(
        demo_trades, sim_trades, aligned, metrics, output_html
    )

    # Save aligned trades to CSV
    aligned_csv = Path(output_html).with_name("aligned_trades.csv")
    if not aligned.empty:
        aligned.to_csv(aligned_csv, index=False)
        log_info(f"Aligned trades saved: {aligned_csv}")

    log_info("=" * 60)
    log_info("Comparison Complete")
    log_info(f"Report: {report_path}")
    log_info("=" * 60)

    return metrics