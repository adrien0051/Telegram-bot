"""
Crypto Telegram Signal Bot v4
-----------------------------
Five-coin spot-market alert bot for BTC, ETH, SOL, BNB and XRP.

Signals use CLOSED Binance 5-minute candles only, with 15m trend confirmation.
This is an alert/paper-trading tool, not an execution bot and not financial advice.

Install:
    pip install "python-telegram-bot[job-queue]>=21,<23" pandas numpy requests

Environment:
    BOT_TOKEN=your_telegram_bot_token
    CHAT_ID=your_telegram_chat_id
Optional:
    INTERVAL_SECONDS=300
    SIGNAL_THRESHOLD=6
    MIN_SIGNAL_SCORE=6
    FEE_PER_SIDE=0.001
    DB_PATH=signals_v4.db
    BINANCE_API=https://data-api.binance.vision
    TIMEZONE=Asia/Dhaka

Run:
    python crypto_signal_bot_v4.py

Telegram commands:
    /start  - setup information
    /now    - analyze immediately
    /stats  - paper-trading performance
    /coins  - current watchlist
    /help   - command help

Important:
- Signals are not guaranteed predictions. The score is indicator agreement, NOT win probability.
- The bot never places trades or asks for exchange API keys.
- It records each symbol's signal at most once per 5-minute candle.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# ------------------------------- Configuration

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()
INTERVAL_SECONDS = max(60, int(os.getenv("INTERVAL_SECONDS", "300")))
BINANCE_API = os.getenv("BINANCE_API", "https://data-api.binance.vision").rstrip("/")
DB_PATH = os.getenv("DB_PATH", "signals_v4.db")
TIMEZONE_NAME = os.getenv("TIMEZONE", "Asia/Dhaka")
FEE_PER_SIDE = max(0.0, float(os.getenv("FEE_PER_SIDE", "0.001")))
MIN_SIGNAL_SCORE = int(os.getenv("MIN_SIGNAL_SCORE", "6"))
LOOKBACK_SIGNALS = 500

WATCHLIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("crypto-signal-bot")
logging.getLogger("httpx").setLevel(logging.WARNING)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CryptoSignalBot/4.0"})


# ------------------------------- Storage

def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                candle_open_ms INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                side TEXT NOT NULL CHECK(side IN ('BUY','SELL')),
                score INTEGER NOT NULL,
                entry REAL NOT NULL,
                exit REAL,
                resolved_at INTEGER,
                correct INTEGER,
                gross_return REAL,
                net_return REAL,
                UNIQUE(symbol, candle_open_ms)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_symbol_time
            ON signals(symbol, created_at)
        """)


# ------------------------------- Market data

def fetch_klines(symbol: str, interval: str = "5m", limit: int = 250) -> pd.DataFrame:
    """Fetch Binance candles and remove the currently-forming candle."""
    url = f"{BINANCE_API}/api/v3/klines"
    response = SESSION.get(
        url,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
    )
    response.raise_for_status()
    rows = response.json()
    if not rows:
        raise RuntimeError(f"No candle data returned for {symbol} ({interval})")

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce").astype("int64")

    # A candle is closed only after its close_time has passed.
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] < now_ms].copy()
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    if len(df) < 60:
        raise RuntimeError(f"Not enough closed {interval} candles for {symbol}")
    return df.reset_index(drop=True)


# ------------------------------- Indicators

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100)
    result = result.mask((avg_loss == 0) & (avg_gain == 0), 50)
    return result


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close = x["close"]
    x["ema20"] = close.ewm(span=20, adjust=False).mean()
    x["ema50"] = close.ewm(span=50, adjust=False).mean()
    x["ema200"] = close.ewm(span=200, adjust=False).mean()
    x["rsi"] = rsi(close, 14)

    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    x["macd"] = fast - slow
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()

    typical = (x["high"] + x["low"] + close) / 3
    # Rolling 24-candle VWAP (two hours); avoids treating a session-reset VWAP
    # as meaningful for a continuously traded crypto market.
    x["vwap"] = (
        (typical * x["volume"]).rolling(24).sum()
        / x["volume"].rolling(24).sum().replace(0, np.nan)
    )

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    x["vol_sma20"] = x["volume"].rolling(20).mean()
    x["mom15"] = close.pct_change(3) * 100
    return x


@dataclass
class Signal:
    symbol: str
    side: str
    score: int
    max_score: int
    entry: float
    stop: float
    target: float
    rsi_value: float
    mom15: float
    candle_open_ms: int
    reasons: list[str]


def score_symbol(symbol: str) -> tuple[Optional[Signal], str]:
    five = add_indicators(fetch_klines(symbol, "5m", 260))
    fifteen = add_indicators(fetch_klines(symbol, "15m", 220))
    a = five.iloc[-1]
    b = fifteen.iloc[-1]

    required = ["ema20", "ema50", "ema200", "rsi", "macd", "macd_signal",
                "vwap", "atr", "vol_sma20", "mom15"]
    if any(pd.isna(a[k]) for k in required) or any(
        pd.isna(b[k]) for k in ["ema20", "ema50", "ema200"]
    ):
        return None, "Indicators not ready"

    score = 0
    reasons: list[str] = []

    # Five-minute trend stack: up to 2 points.
    if a["ema20"] > a["ema50"]:
        score += 1
        reasons.append("5m EMA20 above EMA50")
    elif a["ema20"] < a["ema50"]:
        score -= 1
        reasons.append("5m EMA20 below EMA50")

    if a["close"] > a["ema200"]:
        score += 1
        reasons.append("5m price above EMA200")
    elif a["close"] < a["ema200"]:
        score -= 1
        reasons.append("5m price below EMA200")

    # Fifteen-minute trend confirmation: up to 2 points.
    if b["ema20"] > b["ema50"] and b["close"] > b["ema200"]:
        score += 2
        reasons.append("15m trend bullish")
    elif b["ema20"] < b["ema50"] and b["close"] < b["ema200"]:
        score -= 2
        reasons.append("15m trend bearish")

    # Momentum and MACD: up to 2 points.
    if a["macd"] > a["macd_signal"]:
        score += 1
        reasons.append("MACD bullish")
    elif a["macd"] < a["macd_signal"]:
        score -= 1
        reasons.append("MACD bearish")

    if a["mom15"] > 0:
        score += 1
        reasons.append("15m momentum positive")
    elif a["mom15"] < 0:
        score -= 1
        reasons.append("15m momentum negative")

    # RSI is used as a directional confirmation only outside neutral zone.
    if 50 < a["rsi"] < 70:
        score += 1
        reasons.append("RSI bullish")
    elif 30 < a["rsi"] < 50:
        score -= 1
        reasons.append("RSI bearish")

    # VWAP and volume confirmation: up to 2 points.
    if a["close"] > a["vwap"]:
        score += 1
        reasons.append("Price above rolling VWAP")
    elif a["close"] < a["vwap"]:
        score -= 1
        reasons.append("Price below rolling VWAP")

    if a["volume"] > a["vol_sma20"]:
        if a["close"] > a["open"]:
            score += 1
            reasons.append("Above-average bullish candle volume")
        elif a["close"] < a["open"]:
            score -= 1
            reasons.append("Above-average bearish candle volume")

    # Maximum absolute score is 10. Require a clear score and 15m alignment.
    if score >= MIN_SIGNAL_SCORE and b["ema20"] > b["ema50"]:
        side = "BUY"
    elif score <= -MIN_SIGNAL_SCORE and b["ema20"] < b["ema50"]:
        side = "SELL"
    else:
        return None, f"WAIT (score {score:+d}/10; 15m confirmation required)"

    entry = float(a["close"])
    atr_value = float(a["atr"])
    if not np.isfinite(atr_value) or atr_value <= 0:
        return None, "ATR unavailable"

    # ATR-based illustrative risk levels. No order is placed.
    stop_distance = 1.5 * atr_value
    target_distance = 2.0 * stop_distance
    if side == "BUY":
        stop, target = entry - stop_distance, entry + target_distance
    else:
        stop, target = entry + stop_distance, entry - target_distance

    return Signal(
        symbol=symbol,
        side=side,
        score=score,
        max_score=10,
        entry=entry,
        stop=stop,
        target=target,
        rsi_value=float(a["rsi"]),
        mom15=float(a["mom15"]),
        candle_open_ms=int(a["open_time"]),
        reasons=reasons,
    ), "Signal"


# ------------------------------- Signal tracking / outcomes

def save_signal(signal: Signal) -> bool:
    now = int(time.time())
    with db_connect() as conn:
        try:
            conn.execute(
                """INSERT INTO signals
                (symbol, candle_open_ms, created_at, side, score, entry)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    signal.symbol, signal.candle_open_ms, now,
                    signal.side, signal.score, signal.entry,
                ),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def resolve_due_signals() -> None:
    """Resolve at least one full 5m candle after signal candle has closed."""
    now = int(time.time())
    with db_connect() as conn:
        rows = conn.execute(
            """SELECT id, symbol, side, entry FROM signals
               WHERE resolved_at IS NULL AND created_at <= ?""",
            (now - 300,),
        ).fetchall()

    for row in rows:
        try:
            # One 5m candle after entry: use latest completed close only when
            # the candle timestamp is sufficiently later than the signal candle.
            candles = fetch_klines(row["symbol"], "5m", 5)
            latest = candles.iloc[-1]
            with db_connect() as conn:
                pending = conn.execute(
                    "SELECT candle_open_ms FROM signals WHERE id=?", (row["id"],)
                ).fetchone()
            if int(latest["open_time"]) <= int(pending["candle_open_ms"]):
                continue

            exit_price = float(latest["close"])
            entry = float(row["entry"])
            gross = (exit_price / entry - 1) if row["side"] == "BUY" else (entry / exit_price - 1)
            # Approximate round-trip fee, applied to entry notional.
            net = gross - (2 * FEE_PER_SIDE)
            correct = int(
                (exit_price > entry and row["side"] == "BUY")
                or (exit_price < entry and row["side"] == "SELL")
            )
            with db_connect() as conn:
                conn.execute(
                    """UPDATE signals SET exit=?, resolved_at=?, correct=?,
                       gross_return=?, net_return=? WHERE id=?""",
                    (exit_price, now, correct, gross, net, row["id"]),
                )
        except Exception:
            log.exception("Could not resolve signal id=%s", row["id"])


# ------------------------------- Formatting

def fmt_price(value: float) -> str:
    if value >= 1000:
        return f"${value:,.2f}"
    if value >= 1:
        return f"${value:,.4f}"
    return f"${value:,.6f}"


def local_time() -> str:
    try:
        return datetime.now(ZoneInfo(TIMEZONE_NAME)).strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def analyze_all() -> str:
    resolve_due_signals()
    lines = [
        "📊 CRYPTO SIGNAL SCAN",
        f"🕒 {local_time()}",
        "Timeframes: closed 5m candles + 15m confirmation",
        "",
    ]

    for symbol in WATCHLIST:
        try:
            signal, status = score_symbol(symbol)
            if signal is None:
                lines.append(f"⚪ {symbol}: {status}")
                continue

            saved = save_signal(signal)
            icon = "🟢" if signal.side == "BUY" else "🔴"
            state = "NEW ALERT" if saved else "Already alerted for this candle"
            lines.extend([
                f"{icon} {symbol} — {signal.side} ({signal.score:+d}/10)",
                f"Entry reference: {fmt_price(signal.entry)}",
                f"Stop reference: {fmt_price(signal.stop)}",
                f"Target reference: {fmt_price(signal.target)}",
                f"RSI: {signal.rsi_value:.1f} | 15m momentum: {signal.mom15:+.3f}%",
                f"Status: {state}",
                "Checks: " + "; ".join(signal.reasons[:6]),
                "",
            ])
        except requests.HTTPError as exc:
            lines.append(f"⚠️ {symbol}: market-data HTTP error ({exc})")
        except Exception as exc:
            log.exception("Analysis failed for %s", symbol)
            lines.append(f"⚠️ {symbol}: analysis unavailable ({type(exc).__name__})")

    lines.extend([
        "Score = indicator agreement, NOT win probability.",
        "Paper signals only. No trades are placed. Fees/slippage can erase small moves.",
        "Not financial advice.",
    ])
    return "\n".join(lines)


def performance_report() -> str:
    with db_connect() as conn:
        rows = conn.execute(
            """SELECT symbol, COUNT(*) AS n,
                      SUM(correct) AS wins,
                      AVG(gross_return) AS avg_gross,
                      AVG(net_return) AS avg_net,
                      SUM(net_return) AS sum_net
               FROM signals WHERE resolved_at IS NOT NULL
               GROUP BY symbol ORDER BY symbol"""
        ).fetchall()
        total = conn.execute(
            """SELECT COUNT(*) AS n, SUM(correct) AS wins,
                      AVG(gross_return) AS avg_gross,
                      AVG(net_return) AS avg_net,
                      SUM(net_return) AS sum_net
               FROM signals WHERE resolved_at IS NOT NULL"""
        ).fetchone()
        pending = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE resolved_at IS NULL"
        ).fetchone()[0]

    if not total["n"]:
        return (
            "📈 PAPER-TRADING STATS\n"
            "No resolved signals yet. Let the bot collect signals and wait for their outcomes."
        )

    lines = ["📈 PAPER-TRADING PERFORMANCE"]
    for row in rows:
        wins = row["wins"] or 0
        accuracy = wins / row["n"] * 100
        lines.append(
            f"{row['symbol']}: {wins}/{row['n']} directionally correct "
            f"({accuracy:.1f}%) | avg gross {row['avg_gross']*100:+.3f}% "
            f"| avg net {row['avg_net']*100:+.3f}%"
        )

    wins = total["wins"] or 0
    lines.extend([
        "",
        f"Combined: {wins}/{total['n']} correct ({wins/total['n']*100:.1f}%)",
        f"Average gross return: {total['avg_gross']*100:+.3f}% per signal",
        f"Average estimated net return: {total['avg_net']*100:+.3f}% per signal",
        f"Sum of per-signal net returns (not compounded): {total['sum_net']*100:+.2f}%",
        f"Unresolved signals: {pending}",
        f"Fee assumption: {FEE_PER_SIDE*100:.3f}% per side; spread/slippage excluded.",
        "Small samples are unreliable. Directional accuracy is not the same as profitability.",
        "Paper results only; not financial advice.",
    ])
    return "\n".join(lines)


# ------------------------------- Telegram handlers

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if CHAT_ID and str(update.effective_chat.id) != CHAT_ID:
        await update.message.reply_text("This bot is private.")
        return
    await update.message.reply_text(
        "Welcome to Crypto Signal Bot v4.\n\n"
        "Watchlist: BTC, ETH, SOL, BNB, XRP.\n"
        "Signals use closed 5m candles with 15m confirmation.\n\n"
        "/now — scan now\n"
        "/stats — paper performance\n"
        "/coins — watchlist\n"
        "/help — help\n\n"
        "Send me your chat ID only to configure CHAT_ID in your hosting environment. "
        "This bot does not execute trades."
    )


def authorized(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not CHAT_ID:
            await update.message.reply_text(
                "Set CHAT_ID in your hosting environment. Use /start to see your chat ID."
            )
            return
        if str(update.effective_chat.id) != CHAT_ID:
            await update.message.reply_text("This bot is private.")
            return
        await func(update, context)
    wrapper.__name__ = func.__name__
    return wrapper


@authorized
async def now_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Scanning five coins...")
    report = await asyncio.to_thread(analyze_all)
    await update.message.reply_text(report)


@authorized
async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    report = await asyncio.to_thread(performance_report)
    await update.message.reply_text(report)


@authorized
async def coins_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Watchlist: " + ", ".join(WATCHLIST))


async def scheduled_scan(context: ContextTypes.DEFAULT_TYPE):
    if not CHAT_ID:
        return
    try:
        report = await asyncio.to_thread(analyze_all)
        await context.bot.send_message(chat_id=CHAT_ID, text=report)
    except Exception:
        log.exception("Scheduled scan failed")


async def post_init(app: Application):
    if CHAT_ID:
        app.job_queue.run_repeating(
            scheduled_scan,
            interval=INTERVAL_SECONDS,
            first=10,
            name="five-minute-crypto-scan",
        )


def main():
    if not BOT_TOKEN:
        raise SystemExit("Missing BOT_TOKEN environment variable.")
    init_db()
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("now", now_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("coins", coins_command))
    log.info("Starting Crypto Signal Bot v4")
    app.run_polling()


if __name__ == "__main__":
    main()
