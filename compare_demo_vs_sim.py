#!/usr/bin/env python3
'''Compare demo and simulation trade outputs.

Reads demo/simulation trade CSVs plus daily summaries, prints a markdown
comparison report to stdout, and writes the same report to disk.
'''

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is optional.
    pd = None

NEWLINE = chr(10)
TS_FORMAT = '%Y-%m-%d %H:%M:%S'
DATE_FORMAT = '%Y-%m-%d'
TRADE_COLUMNS = (
    'entry_ts',
    'exit_ts',
    'side',
    'entry_price',
    'exit_price',
    'qty',
    'exit_reason',
    'pnl_raw',
    'regime',
)
DAILY_COLUMNS = (
    'date',
    'entries',
    'exits',
    'win_rate',
    'pnl_total',
    'vol_flt',
    'htf_trd',
    'adapt_thr',
    'riskguard',
    'chop',
    'veto',
    'regime_trend_pct',
    'regime_chop_pct',
    'regime_highvol_pct',
)
GATE_KEYS = (
    'vol_flt',
    'htf_trd',
    'adapt_thr',
    'riskguard',
    'chop',
    'veto',
)
REGIME_KEYS = (
    'regime_trend_pct',
    'regime_chop_pct',
    'regime_highvol_pct',
)
GATE_ALIASES = {
    'volume_filter': 'vol_flt',
    'htf_trend': 'htf_trd',
    'adaptive_threshold': 'adapt_thr',
    'riskguard': 'riskguard',
    'chop_regime': 'chop',
    'strategic_veto': 'veto',
}
GATE_LABELS = {
    'vol_flt': 'Volume filter',
    'htf_trd': 'HTF trend',
    'adapt_thr': 'Adaptive threshold',
    'riskguard': 'RiskGuard',
    'chop': 'Chop regime',
    'veto': 'Strategic veto',
}
REGIME_LABELS = {
    'regime_trend_pct': 'Trend',
    'regime_chop_pct': 'Chop',
    'regime_highvol_pct': 'High-vol',
}


@dataclass(frozen=True)
class TradeRecord:
    '''Normalized trade row from demo_trades.csv or sim_trades.csv.'''

    entry_ts: datetime
    exit_ts: Optional[datetime]
    side: str
    entry_price: float
    exit_price: float
    qty: float
    exit_reason: str
    pnl_raw: float
    pnl_missing: bool
    regime: str


@dataclass(frozen=True)
class DailyRow:
    '''Normalized daily summary row.'''

    date: str
    entries: int
    exits: int
    win_rate: float
    pnl_total: float
    gates: Dict[str, int]
    regimes: Dict[str, float]


@dataclass(frozen=True)
class TradeTotals:
    '''Aggregate trade-level metrics for the report summary.'''

    trades: int
    wins: int
    win_rate: float
    pnl_total: float
    pnl_missing_count: int


@dataclass(frozen=True)
class GateContribution:
    '''How strongly a gate aligns with the demo-vs-sim entry gap.'''

    gate_key: str
    contribution_score: int
    total_count: int
    active_gap_days: int


@dataclass(frozen=True)
class ReportArtifacts:
    '''Computed report text plus parsed structures for callers or tests.'''

    report_text: str
    demo_trade_totals: TradeTotals
    sim_trade_totals: TradeTotals
    demo_daily: Dict[str, DailyRow]
    sim_daily: Dict[str, DailyRow]


class SchemaError(ValueError):
    '''Raised when an input CSV does not match the expected schema.'''


def _date_sort_key(date_text: str) -> datetime:
    '''Parse a daily-summary date for stable sorting.'''
    return datetime.strptime(date_text, DATE_FORMAT)


def _gate_sort_key(item: GateContribution) -> tuple[int, int, int, int]:
    '''Sort gate contributions by explanatory strength, then by volume.'''
    return (
        item.contribution_score,
        item.total_count,
        item.active_gap_days,
        GATE_KEYS.index(item.gate_key),
    )


def _clean_cell(value: object) -> str:
    '''Convert a CSV cell into a stripped string.'''
    if value is None:
        return ''
    text = str(value).strip()
    if text.lower() == 'nan':
        return ''
    return text


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    '''Read CSV rows using pandas when available, else csv.DictReader.'''
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f'CSV file not found: {path}')

    if pd is not None:
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        records = frame.to_dict(orient='records')
        return [
            {str(key).strip(): _clean_cell(value) for key, value in record.items()}
            for record in records
        ]

    with csv_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SchemaError(f'CSV file has no header row: {path}')
        return [
            {str(key).strip(): _clean_cell(value) for key, value in row.items()}
            for row in reader
        ]


def _parse_float(value: str, field_name: str, path: str, row_index: int) -> float:
    '''Parse a float field, treating empty strings as zero.'''
    text = _clean_cell(value)
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError as exc:
        raise SchemaError(
            f'Invalid float for {field_name} in {path} row {row_index}: {value!r}'
        ) from exc


def _parse_int(value: str, field_name: str, path: str, row_index: int) -> int:
    '''Parse an int field, accepting integral floats like 5.0.'''
    text = _clean_cell(value)
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError as exc:
        raise SchemaError(
            f'Invalid int for {field_name} in {path} row {row_index}: {value!r}'
        ) from exc


def _normalize_fraction(
    value: str,
    field_name: str,
    path: str,
    row_index: int,
) -> float:
    '''Normalize a win-rate field into a 0..1 fraction.'''
    text = _clean_cell(value)
    if not text:
        return 0.0
    is_percent = text.endswith('%')
    numeric_text = text[:-1] if is_percent else text
    numeric_value = _parse_float(numeric_text, field_name, path, row_index)
    if is_percent or numeric_value > 1.0:
        return numeric_value / 100.0
    return numeric_value


def _normalize_percentage_points(
    value: str,
    field_name: str,
    path: str,
    row_index: int,
) -> float:
    '''Normalize a regime percentage into 0..100 percentage points.'''
    text = _clean_cell(value)
    if not text:
        return 0.0
    is_percent = text.endswith('%')
    numeric_text = text[:-1] if is_percent else text
    numeric_value = _parse_float(numeric_text, field_name, path, row_index)
    if not is_percent and 0.0 <= numeric_value <= 1.0:
        return numeric_value * 100.0
    return numeric_value


def _parse_timestamp(
    value: str,
    field_name: str,
    path: str,
    row_index: int,
) -> Optional[datetime]:
    '''Parse a UTC timestamp in the expected schema format.'''
    text = _clean_cell(value)
    if not text:
        return None
    try:
        return datetime.strptime(text, TS_FORMAT)
    except ValueError as exc:
        raise SchemaError(
            f'Invalid timestamp for {field_name} in {path} row {row_index}: {value!r}'
        ) from exc


def _require_columns(
    available_columns: Sequence[str],
    required_columns: Sequence[str],
    path: str,
) -> None:
    '''Validate that a CSV contains all required columns.'''
    missing = [column for column in required_columns if column not in available_columns]
    if missing:
        raise SchemaError(f'Missing columns in {path}: {", ".join(missing)}')


def _sorted_dates(
    demo_daily: Mapping[str, DailyRow],
    sim_daily: Mapping[str, DailyRow],
) -> List[str]:
    '''Return the union of daily-summary dates in chronological order.'''
    all_dates = set(demo_daily.keys()) | set(sim_daily.keys())
    return sorted(all_dates, key=_date_sort_key)


def _empty_gate_counts() -> Dict[str, int]:
    '''Build a zeroed gate counter mapping.'''
    return {gate_key: 0 for gate_key in GATE_KEYS}


def load_daily(path: str) -> List[Dict[str, str]]:
    '''Return daily-summary rows as header-driven dictionaries.

    The loader is schema-order agnostic and normalizes known live-gate aliases
    so callers always receive the canonical short names.
    '''
    rows = _read_csv_rows(path)
    if not rows:
        return []

    canonical_rows: List[Dict[str, str]] = []
    for row in rows:
        canonical_row: Dict[str, str] = {}
        for key, value in row.items():
            canonical_key = GATE_ALIASES.get(key, key)
            canonical_row[canonical_key] = value
        canonical_rows.append(canonical_row)

    _require_columns(tuple(canonical_rows[0].keys()), DAILY_COLUMNS, path)
    return canonical_rows


def load_trades(path: str) -> List[TradeRecord]:
    '''Load and normalize trade rows from the demo or sim trade schema.'''
    rows = _read_csv_rows(path)
    if not rows:
        return []

    _require_columns(tuple(rows[0].keys()), TRADE_COLUMNS, path)
    trades: List[TradeRecord] = []
    for row_index, row in enumerate(rows, start=2):
        entry_ts = _parse_timestamp(row.get('entry_ts', ''), 'entry_ts', path, row_index)
        if entry_ts is None:
            raise SchemaError(f'Missing entry_ts in {path} row {row_index}')
        exit_ts = _parse_timestamp(row.get('exit_ts', ''), 'exit_ts', path, row_index)
        pnl_text = _clean_cell(row.get('pnl_raw', ''))
        pnl_missing = not pnl_text
        trades.append(
            TradeRecord(
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                side=_clean_cell(row.get('side', '')).upper(),
                entry_price=_parse_float(
                    row.get('entry_price', ''), 'entry_price', path, row_index
                ),
                exit_price=_parse_float(
                    row.get('exit_price', ''), 'exit_price', path, row_index
                ),
                qty=_parse_float(row.get('qty', ''), 'qty', path, row_index),
                exit_reason=_clean_cell(row.get('exit_reason', '')),
                pnl_raw=(
                    0.0
                    if pnl_missing
                    else _parse_float(pnl_text, 'pnl_raw', path, row_index)
                ),
                pnl_missing=pnl_missing,
                regime=_clean_cell(row.get('regime', '')),
            )
        )
    return trades


def parse_daily_rows(path: str) -> Dict[str, DailyRow]:
    '''Parse daily summary rows into typed objects keyed by date.'''
    rows = load_daily(path)
    parsed_rows: Dict[str, DailyRow] = {}
    for row_index, row in enumerate(rows, start=2):
        date_text = _clean_cell(row.get('date', ''))
        if not date_text:
            raise SchemaError(f'Missing date in {path} row {row_index}')
        try:
            datetime.strptime(date_text, DATE_FORMAT)
        except ValueError as exc:
            raise SchemaError(
                f'Invalid date in {path} row {row_index}: {date_text!r}'
            ) from exc

        gates = {
            gate_key: _parse_int(row.get(gate_key, '0'), gate_key, path, row_index)
            for gate_key in GATE_KEYS
        }
        regimes = {
            regime_key: _normalize_percentage_points(
                row.get(regime_key, '0'), regime_key, path, row_index
            )
            for regime_key in REGIME_KEYS
        }
        parsed_rows[date_text] = DailyRow(
            date=date_text,
            entries=_parse_int(row.get('entries', '0'), 'entries', path, row_index),
            exits=_parse_int(row.get('exits', '0'), 'exits', path, row_index),
            win_rate=_normalize_fraction(
                row.get('win_rate', '0'), 'win_rate', path, row_index
            ),
            pnl_total=_parse_float(
                row.get('pnl_total', '0'), 'pnl_total', path, row_index
            ),
            gates=gates,
            regimes=regimes,
        )
    return parsed_rows


def summarize_trades(trades: Sequence[TradeRecord]) -> TradeTotals:
    '''Aggregate trade-level totals for the report summary.'''
    trade_count = len(trades)
    wins = sum(1 for trade in trades if trade.pnl_raw > 0.0)
    pnl_total = sum(trade.pnl_raw for trade in trades)
    missing_pnl_count = sum(1 for trade in trades if trade.pnl_missing)
    win_rate = (wins / trade_count) if trade_count else 0.0
    return TradeTotals(
        trades=trade_count,
        wins=wins,
        win_rate=win_rate,
        pnl_total=pnl_total,
        pnl_missing_count=missing_pnl_count,
    )


def _format_percent_from_fraction(value: float) -> str:
    '''Format a 0..1 fraction as a percentage string.'''
    percent_value = value * 100.0
    if abs(percent_value - round(percent_value)) < 1e-9:
        return f'{int(round(percent_value))}%'
    return f'{percent_value:.1f}%'


def _format_percent_points(value: float) -> str:
    '''Format 0..100 percentage points for display.'''
    if abs(value - round(value)) < 1e-9:
        return f'{int(round(value))}%'
    return f'{value:.1f}%'


def _format_signed_int(value: int) -> str:
    '''Format an integer with an explicit sign.'''
    return f'{value:+d}'


def _format_signed_float(value: float) -> str:
    '''Format a float with sign and three decimal places.'''
    return f'{value:+.3f}'


def _build_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    '''Build a markdown table with padded cells for console readability.'''
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(cells: Sequence[str]) -> str:
        padded = [cells[index].ljust(widths[index]) for index in range(len(cells))]
        return '| ' + ' | '.join(padded) + ' |'

    divider = ['-' * width for width in widths]
    lines = [format_row(headers), format_row(divider)]
    for row in rows:
        lines.append(format_row(row))
    return NEWLINE.join(lines)


def _table_row_for_day(
    date_text: str,
    demo_row: Optional[DailyRow],
    sim_row: Optional[DailyRow],
) -> List[str]:
    '''Build one per-day comparison row.'''
    demo_entries = demo_row.entries if demo_row is not None else 0
    sim_entries = sim_row.entries if sim_row is not None else 0
    delta = demo_entries - sim_entries
    demo_win = (
        _format_percent_from_fraction(demo_row.win_rate)
        if demo_row is not None
        else ''
    )
    sim_win = (
        _format_percent_from_fraction(sim_row.win_rate)
        if sim_row is not None
        else ''
    )
    demo_pnl = _format_signed_float(demo_row.pnl_total) if demo_row is not None else ''
    sim_pnl = _format_signed_float(sim_row.pnl_total) if sim_row is not None else ''
    return [
        date_text,
        str(demo_entries),
        str(sim_entries),
        _format_signed_int(delta),
        demo_win,
        sim_win,
        demo_pnl,
        sim_pnl,
    ]


def _build_per_day_section(
    demo_daily: Mapping[str, DailyRow],
    sim_daily: Mapping[str, DailyRow],
    demo_totals: TradeTotals,
    sim_totals: TradeTotals,
) -> str:
    '''Create the per-day side-by-side section.'''
    headers = (
        'Date',
        'Demo entries',
        'Sim entries',
        'Delta',
        'Demo win%',
        'Sim win%',
        'Demo PnL',
        'Sim PnL',
    )
    rows: List[List[str]] = []
    for date_text in _sorted_dates(demo_daily, sim_daily):
        rows.append(
            _table_row_for_day(
                date_text,
                demo_daily.get(date_text),
                sim_daily.get(date_text),
            )
        )

    demo_entry_total = sum(row.entries for row in demo_daily.values())
    sim_entry_total = sum(row.entries for row in sim_daily.values())
    demo_pnl_total = sum(row.pnl_total for row in demo_daily.values())
    sim_pnl_total = sum(row.pnl_total for row in sim_daily.values())
    if not rows:
        demo_entry_total = demo_totals.trades
        sim_entry_total = sim_totals.trades
        demo_pnl_total = demo_totals.pnl_total
        sim_pnl_total = sim_totals.pnl_total

    rows.append(
        [
            'TOTAL',
            str(demo_entry_total),
            str(sim_entry_total),
            _format_signed_int(demo_entry_total - sim_entry_total),
            _format_percent_from_fraction(demo_totals.win_rate),
            _format_percent_from_fraction(sim_totals.win_rate),
            _format_signed_float(demo_pnl_total),
            _format_signed_float(sim_pnl_total),
        ]
    )
    return '## Per-Day Comparison' + NEWLINE * 2 + _build_table(headers, rows)


def _build_gate_attribution_section(
    demo_daily: Mapping[str, DailyRow],
    sim_daily: Mapping[str, DailyRow],
) -> str:
    '''Create the gate-attribution section from demo daily summaries.'''
    headers = (
        'Date',
        'vol_flt',
        'htf_trd',
        'adapt_thr',
        'riskguard',
        'chop',
        'veto',
        'Live suppressed',
        'Sim-Demo gap',
    )
    table_rows: List[List[str]] = []
    note_lines: List[str] = []

    for date_text in _sorted_dates(demo_daily, sim_daily):
        demo_row = demo_daily.get(date_text)
        sim_row = sim_daily.get(date_text)
        demo_entries = demo_row.entries if demo_row is not None else 0
        sim_entries = sim_row.entries if sim_row is not None else 0
        sim_minus_demo = sim_entries - demo_entries
        gates = demo_row.gates if demo_row is not None else _empty_gate_counts()
        live_suppressed = (
            gates['vol_flt']
            + gates['htf_trd']
            + gates['riskguard']
            + gates['chop']
            + gates['veto']
        )
        table_rows.append(
            [
                date_text,
                str(gates['vol_flt']),
                str(gates['htf_trd']),
                str(gates['adapt_thr']),
                str(gates['riskguard']),
                str(gates['chop']),
                str(gates['veto']),
                str(live_suppressed),
                _format_signed_int(sim_minus_demo),
            ]
        )
        note_lines.append(
            f'- {date_text}: live gates suppressed {live_suppressed} '
            '(vol_flt + htf_trd + riskguard + chop + veto) signals that day, '
            f'vs sim {sim_entries} trades; the gap demonstrates why demo trades < '
            'sim trades. If these counters sum close to (sim_entries - '
            'demo_entries), live gates explain the gap.'
        )

    lines = [
        '## Gate Attribution',
        '',
        _build_table(headers, table_rows),
        '',
        *note_lines,
    ]
    return NEWLINE.join(lines)


def _largest_regime_disagreement(demo_row: DailyRow, sim_row: DailyRow) -> str:
    '''Describe the regime bucket with the largest demo-vs-sim disagreement.'''
    largest_key = REGIME_KEYS[0]
    largest_delta = -1.0
    for regime_key in REGIME_KEYS:
        delta = abs(demo_row.regimes[regime_key] - sim_row.regimes[regime_key])
        if delta > largest_delta:
            largest_delta = delta
            largest_key = regime_key
    label = REGIME_LABELS[largest_key]
    signed_delta = demo_row.regimes[largest_key] - sim_row.regimes[largest_key]
    return f'Largest disagreement: {label} ({signed_delta:+.1f} pts)'


def _build_regime_section(
    demo_daily: Mapping[str, DailyRow],
    sim_daily: Mapping[str, DailyRow],
) -> str:
    '''Create the regime distribution comparison section.'''
    headers = (
        'Date',
        'Demo trend',
        'Sim trend',
        'Demo chop',
        'Sim chop',
        'Demo high-vol',
        'Sim high-vol',
        'Note',
    )
    rows: List[List[str]] = []
    for date_text in _sorted_dates(demo_daily, sim_daily):
        demo_row = demo_daily.get(date_text)
        sim_row = sim_daily.get(date_text)
        if demo_row is None or sim_row is None:
            rows.append(
                [
                    date_text,
                    (
                        _format_percent_points(demo_row.regimes['regime_trend_pct'])
                        if demo_row
                        else ''
                    ),
                    (
                        _format_percent_points(sim_row.regimes['regime_trend_pct'])
                        if sim_row
                        else ''
                    ),
                    (
                        _format_percent_points(demo_row.regimes['regime_chop_pct'])
                        if demo_row
                        else ''
                    ),
                    (
                        _format_percent_points(sim_row.regimes['regime_chop_pct'])
                        if sim_row
                        else ''
                    ),
                    (
                        _format_percent_points(demo_row.regimes['regime_highvol_pct'])
                        if demo_row
                        else ''
                    ),
                    (
                        _format_percent_points(sim_row.regimes['regime_highvol_pct'])
                        if sim_row
                        else ''
                    ),
                    'Missing one side of daily regime data',
                ]
            )
            continue

        rows.append(
            [
                date_text,
                _format_percent_points(demo_row.regimes['regime_trend_pct']),
                _format_percent_points(sim_row.regimes['regime_trend_pct']),
                _format_percent_points(demo_row.regimes['regime_chop_pct']),
                _format_percent_points(sim_row.regimes['regime_chop_pct']),
                _format_percent_points(demo_row.regimes['regime_highvol_pct']),
                _format_percent_points(sim_row.regimes['regime_highvol_pct']),
                _largest_regime_disagreement(demo_row, sim_row),
            ]
        )

    lines = [
        '## Regime Distribution Comparison',
        '',
        'Chop-regime disagreement is especially important because live chop '
        'gating directly reduces demo trade count.',
        '',
        _build_table(headers, rows),
    ]
    return NEWLINE.join(lines)


def _rank_gate_contributions(
    demo_daily: Mapping[str, DailyRow],
    sim_daily: Mapping[str, DailyRow],
) -> List[GateContribution]:
    '''Rank gates by how well their counts align with positive sim-vs-demo gaps.'''
    aggregates: Dict[str, Dict[str, int]] = {
        gate_key: {'score': 0, 'total': 0, 'days': 0} for gate_key in GATE_KEYS
    }
    for date_text in _sorted_dates(demo_daily, sim_daily):
        demo_row = demo_daily.get(date_text)
        sim_row = sim_daily.get(date_text)
        if demo_row is None:
            continue
        positive_gap = max((sim_row.entries if sim_row is not None else 0) - demo_row.entries, 0)
        for gate_key in GATE_KEYS:
            gate_count = demo_row.gates[gate_key]
            aggregates[gate_key]['total'] += gate_count
            if gate_count > 0 and positive_gap > 0:
                aggregates[gate_key]['days'] += 1
            aggregates[gate_key]['score'] += min(gate_count, positive_gap)

    ranked = [
        GateContribution(
            gate_key=gate_key,
            contribution_score=values['score'],
            total_count=values['total'],
            active_gap_days=values['days'],
        )
        for gate_key, values in aggregates.items()
    ]
    ranked.sort(key=_gate_sort_key, reverse=True)
    return ranked


def _worst_date_gap(
    demo_daily: Mapping[str, DailyRow],
    sim_daily: Mapping[str, DailyRow],
) -> Optional[str]:
    '''Return the date with the largest sim-over-demo entry gap.'''
    worst_date: Optional[str] = None
    worst_gap: Optional[int] = None
    for date_text in _sorted_dates(demo_daily, sim_daily):
        demo_entries = demo_daily.get(date_text).entries if date_text in demo_daily else 0
        sim_entries = sim_daily.get(date_text).entries if date_text in sim_daily else 0
        gap = sim_entries - demo_entries
        if worst_gap is None or gap > worst_gap:
            worst_gap = gap
            worst_date = date_text
    return worst_date


def _build_summary_section(
    demo_totals: TradeTotals,
    sim_totals: TradeTotals,
    demo_daily: Mapping[str, DailyRow],
    sim_daily: Mapping[str, DailyRow],
) -> str:
    '''Create the summary section with totals and ranked gate drivers.'''
    worst_date = _worst_date_gap(demo_daily, sim_daily)
    if worst_date is None:
        worst_gap_line = '- Worst date gap: n/a'
    else:
        demo_entries = demo_daily.get(worst_date).entries if worst_date in demo_daily else 0
        sim_entries = sim_daily.get(worst_date).entries if worst_date in sim_daily else 0
        worst_gap_line = (
            f'- Worst date gap: {worst_date} (demo {demo_entries} vs sim '
            f'{sim_entries}, gap {sim_entries - demo_entries:+d})'
        )

    contribution_lines: List[str] = []
    for index, contribution in enumerate(_rank_gate_contributions(demo_daily, sim_daily), start=1):
        label = GATE_LABELS[contribution.gate_key]
        contribution_lines.append(
            f'{index}. {label}: score {contribution.contribution_score}, '
            f'total count {contribution.total_count}, active on '
            f'{contribution.active_gap_days} gap day(s)'
        )

    lines = [
        '## Summary',
        '',
        f'- Total trades: demo {demo_totals.trades} vs sim {sim_totals.trades} '
        f'(delta {demo_totals.trades - sim_totals.trades:+d})',
        f'- Win rate: demo {_format_percent_from_fraction(demo_totals.win_rate)} '
        f'vs sim {_format_percent_from_fraction(sim_totals.win_rate)}',
        f'- PnL total: demo {_format_signed_float(demo_totals.pnl_total)} vs sim '
        f'{_format_signed_float(sim_totals.pnl_total)}',
        worst_gap_line,
        f'- Missing pnl_raw rows treated as zero: demo '
        f'{demo_totals.pnl_missing_count}, sim {sim_totals.pnl_missing_count}',
        '',
        'Top contributing gate counters:',
        *contribution_lines,
    ]
    return NEWLINE.join(lines)


def build_report(
    demo_trades_csv: str,
    sim_trades_csv: str,
    demo_daily_csv: str,
    sim_daily_csv: str,
) -> ReportArtifacts:
    '''Load the four comparison CSVs and build the markdown report.'''
    demo_trades = load_trades(demo_trades_csv)
    sim_trades = load_trades(sim_trades_csv)
    demo_daily = parse_daily_rows(demo_daily_csv)
    sim_daily = parse_daily_rows(sim_daily_csv)
    demo_totals = summarize_trades(demo_trades)
    sim_totals = summarize_trades(sim_trades)

    sections = [
        '# Demo vs Sim Comparison Report',
        '',
        _build_per_day_section(demo_daily, sim_daily, demo_totals, sim_totals),
        '',
        _build_gate_attribution_section(demo_daily, sim_daily),
        '',
        _build_regime_section(demo_daily, sim_daily),
        '',
        _build_summary_section(demo_totals, sim_totals, demo_daily, sim_daily),
        '',
    ]
    report_text = NEWLINE.join(sections).rstrip() + NEWLINE
    return ReportArtifacts(
        report_text=report_text,
        demo_trade_totals=demo_totals,
        sim_trade_totals=sim_totals,
        demo_daily=demo_daily,
        sim_daily=sim_daily,
    )


def compare(
    demo_trades_csv: str,
    sim_trades_csv: str,
    demo_daily_csv: str,
    sim_daily_csv: str,
    out_path: str = 'comparison_report.md',
) -> None:
    '''Generate the report, print it to stdout, and write it to disk.'''
    artifacts = build_report(
        demo_trades_csv=demo_trades_csv,
        sim_trades_csv=sim_trades_csv,
        demo_daily_csv=demo_daily_csv,
        sim_daily_csv=sim_daily_csv,
    )
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(artifacts.report_text, encoding='utf-8')
    print(artifacts.report_text, end='')


def _build_arg_parser() -> argparse.ArgumentParser:
    '''Create the CLI argument parser.'''
    parser = argparse.ArgumentParser(
        description='Compare demo and simulation CSV outputs and write a markdown report.'
    )
    parser.add_argument('--demo', required=True, help='Path to demo_trades.csv')
    parser.add_argument('--sim', required=True, help='Path to sim_trades.csv')
    parser.add_argument(
        '--demo-daily',
        required=True,
        help='Path to demo_daily_summary.csv',
    )
    parser.add_argument(
        '--sim-daily',
        required=True,
        help='Path to sim_daily_summary.csv',
    )
    parser.add_argument(
        '--out',
        default='comparison_report.md',
        help='Output markdown path (default: comparison_report.md)',
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    '''CLI entrypoint.'''
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    compare(
        demo_trades_csv=args.demo,
        sim_trades_csv=args.sim,
        demo_daily_csv=args.demo_daily,
        sim_daily_csv=args.sim_daily,
        out_path=args.out,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
