import os
import time
import sqlite3
import asyncio
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # keeps your token out of the logs

TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ.get("CHAT_ID")  # send /start to the bot to learn yours
INTERVAL_SECONDS = 600  # 10 minutes
TZ_HOURS = float(os.environ.get("TZ_HOURS", "6"))  # your time zone, e.g. 6 = GMT+6

# Yahoo Finance symbols: stocks, forex (=X), gold (GC=F)...
WATCHLIST = ["AAPL", "MSFT", "NVDA", "TSLA", "EURUSD=X", "GC=F"]

# Binance spot pairs (public market data, no API key needed)
BINANCE_WATCHLIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
# If your host is blocked from Binance (HTTP 451), set BINANCE_API to https://api.binance.us
BINANCE_API = os.environ.get("BINANCE_API", "https://data-api.binance.vision")

# --- learning settings ---
MIN_SAMPLES = 15      # need this many checked signals before judging a symbol
MUTE_BELOW = 0.45     # if it was right less often than this, show NEUTRAL instead
LOOKBACK = 50         # judge on the last 50 checked signals

# --- storage ---
# On Railway, add a Volume mounted at /data, otherwise the history is wiped on each redeploy.
DATA_DIR = os.environ.get("DATA_DIR", "/data" if os.path.isdir("/data") else ".")
DB_PATH = os.path.join(DATA_DIR, "signals.db")

DISCLAIMER = (
    "Not financial advice. Signals are not predictions.\n"
    "Ranges show where the price usually stays (about 2 out of 3 times); it can go outside."
)


# ---------------------------------------------------------------- database
def db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER, symbol TEXT, lean TEXT, price REAL,
                resolved INTEGER DEFAULT 0, correct INTEGER
            )"""
        )


def record_signals(results):
    now = int(time.time())
    with db() as c:
        for a in results:
            if a["lean"] != "NEUTRAL":
                c.execute(
                    "INSERT INTO signals (ts, symbol, lean, price) VALUES (?,?,?,?)",
                    (now, a["symbol"], a["lean"], a["price"]),
                )


def resolve_pending(results):
    """Check signals that are at least 1 hour old against the current price."""
    prices = {a["symbol"]: a["price"] for a in results}
    cutoff = int(time.time()) - 3600
    with db() as c:
        rows = c.execute(
            "SELECT id, symbol, lean, price FROM signals WHERE resolved=0 AND ts<=?", (cutoff,)
        ).fetchall()
        for sid, sym, lean, price in rows:
            now_price = prices.get(sym)
            if now_price is None:
                continue
            if now_price == price:  # market closed / no movement: don't count it
                correct = None
            elif lean == "UP":
                correct = int(now_price > price)
            else:
                correct = int(now_price < price)
            c.execute("UPDATE signals SET resolved=1, correct=? WHERE id=?", (correct, sid))


def symbol_record(symbol):
    with db() as c:
        rows = c.execute(
            "SELECT correct FROM signals WHERE symbol=? AND correct IS NOT NULL "
            "ORDER BY ts DESC LIMIT ?",
            (symbol, LOOKBACK),
        ).fetchall()
    total = len(rows)
    right = sum(r[0] for r in rows)
    return right, total


# ---------------------------------------------------------------- indicators
def rsi_series(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def score_series(symbol: str, close: pd.Series):
    if len(close) < 60:
        return None
    r = float(rsi_series(close).iloc[-1])
    sma_fast = close.rolling(20).mean().iloc[-1]
    sma_slow = close.rolling(50).mean().iloc[-1]
    change = (close.iloc[-1] / close.iloc[-12] - 1) * 100  # ~last hour (5m candles)
    vol = float(close.pct_change().std() * (12 ** 0.5) * 100)  # typical 1-hour move, in %

    score = 0
    score += 1 if sma_fast > sma_slow else -1
    score += 1 if change > 0 else -1
    if r < 30:
        score += 1   # oversold, possible bounce
    elif r > 70:
        score -= 1   # overbought, possible pullback

    if score >= 2:
        lean = "UP"
    elif score <= -2:
        lean = "DOWN"
    else:
        lean = "NEUTRAL"
    return {"symbol": symbol, "price": float(close.iloc[-1]), "rsi": r,
            "change": float(change), "lean": lean, "vol": vol}


# ---------------------------------------------------------------- data sources
def analyze_yahoo(symbol: str):
    df = yf.download(symbol, period="5d", interval="5m", progress=False, auto_adjust=True)
    if df.empty:
        return None
    return score_series(symbol, df["Close"].squeeze().dropna())


def fetch_binance_closes(symbol: str, total: int = 200) -> pd.Series:
    rows, end = [], None
    while len(rows) < total:
        params = {"symbol": symbol, "interval": "5m", "limit": min(1000, total - len(rows))}
        if end:
            params["endTime"] = end
        resp = requests.get(f"{BINANCE_API}/api/v3/klines", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        rows = data + rows
        end = data[0][0] - 1
        if len(data) < params["limit"]:
            break
    return pd.Series([float(k[4]) for k in rows])


def analyze_binance(symbol: str):
    return score_series(f"{symbol} (Binance)", fetch_binance_closes(symbol, 200))


# ---------------------------------------------------------------- report
def fmt(p: float) -> str:
    return f"${p:,.2f}" if p >= 10 else f"${p:,.4f}"


def clock(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0").lower()


def collect():
    jobs = [(x, analyze_yahoo) for x in WATCHLIST] + [(x, analyze_binance) for x in BINANCE_WATCHLIST]
    results = []
    for sym, fn in jobs:
        try:
            a = fn(sym)
        except Exception as e:
            logging.warning("Failed %s: %s", sym, e)
            a = None
        if a:
            results.append(a)
    return results


def build_report(record: bool = False) -> str:
    results = collect()
    if not results:
        return "No data right now (markets closed or data source down)."
    if record:
        resolve_pending(results)
        record_signals(results)

    icons = {"UP": "🟢", "DOWN": "🔴", "NEUTRAL": "⚪"}
    now = datetime.now(timezone(timedelta(hours=TZ_HOURS)))
    t30, t60 = clock(now + timedelta(minutes=30)), clock(now + timedelta(minutes=60))
    lines = []
    for a in results:
        right, total = symbol_record(a["symbol"])
        lean, note = a["lean"], ""
        if total >= MIN_SAMPLES and right / total < MUTE_BELOW and lean != "NEUTRAL":
            lean, note = "NEUTRAL", " (muted: weak record)"
        record_txt = f"{right}/{total} right ({right / total:.0%})" if total else "learning..."
        p, v60 = a["price"], a["vol"]
        v30 = v60 * (0.5 ** 0.5)
        lines.append(
            f"{icons[lean]} {a['symbol']}: {lean}{note}\n"
            f"   Now ({clock(now)}): {fmt(p)} | 1h {a['change']:+.2f}% | RSI {a['rsi']:.0f}\n"
            f"   By {t30}: up to {fmt(p * (1 + v30 / 100))} / down to {fmt(p * (1 - v30 / 100))}\n"
            f"   By {t60}: up to {fmt(p * (1 + v60 / 100))} / down to {fmt(p * (1 - v60 / 100))}\n"
            f"   Record: {record_txt}"
        )
    return "\n\n".join(lines) + f"\n\n{DISCLAIMER}"


# ---------------------------------------------------------------- stats + backtest
def build_stats() -> str:
    with db() as c:
        rows = c.execute(
            "SELECT symbol, COUNT(*), SUM(correct) FROM signals "
            "WHERE correct IS NOT NULL GROUP BY symbol"
        ).fetchall()
        pending = c.execute("SELECT COUNT(*) FROM signals WHERE resolved=0").fetchone()[0]
    if not rows:
        return f"No checked signals yet. {pending} waiting for their 1-hour check."
    lines, tot, right = [], 0, 0
    for sym, n, ok in rows:
        ok = ok or 0
        tot += n
        right += ok
        lines.append(f"{sym}: {ok}/{n} right ({ok / n:.0%})")
    lines.append(f"\nAll: {right}/{tot} right ({right / tot:.0%}). Waiting: {pending}")
    lines.append("Around 50% means no better than a coin flip.")
    return "\n".join(lines)


def backtest_symbol(symbol: str, candles: int = 3000) -> str:
    close = fetch_binance_closes(symbol, candles)
    rsi_s = rsi_series(close)
    fast = close.rolling(20).mean()
    slow = close.rolling(50).mean()
    chg = close / close.shift(12) - 1
    fwd = close.shift(-12) / close - 1  # what happened in the next hour

    score = (
        np.where(fast > slow, 1, -1)
        + np.where(chg > 0, 1, -1)
        + np.where(rsi_s < 30, 1, 0)
        - np.where(rsi_s > 70, 1, 0)
    )
    valid = (slow.notna() & rsi_s.notna() & chg.notna() & fwd.notna()).to_numpy()
    up = valid & (score >= 2)
    down = valid & (score <= -2)
    base_up = float((fwd[valid] > 0).mean())

    parts = [f"{symbol} ({len(close)} candles, ~{len(close) * 5 / 1440:.0f} days)"]
    for name, mask, hit in (("UP", up, fwd > 0), ("DOWN", down, fwd < 0)):
        n = int(mask.sum())
        if n == 0:
            parts.append(f"  {name}: no signals")
            continue
        rate = float(hit[mask].mean())
        baseline = base_up if name == "UP" else 1 - base_up
        parts.append(
            f"  {name}: {n} signals, right {rate:.0%} (market did this {baseline:.0%} of the time, "
            f"edge {100 * (rate - baseline):+.0f} pts)"
        )
    return "\n".join(parts)


def build_backtest() -> str:
    out = []
    for sym in BINANCE_WATCHLIST:
        try:
            out.append(backtest_symbol(sym))
        except Exception as e:
            out.append(f"{sym}: failed ({e})")
    out.append(
        "\nEdge = how much better the signal was than just guessing the market's usual direction. "
        "Near 0 or negative means the signal adds nothing. Past results do not guarantee future ones."
    )
    return "\n\n".join(out)


# ---------------------------------------------------------------- telegram handlers
async def scheduled(context: ContextTypes.DEFAULT_TYPE):
    text = await asyncio.to_thread(build_report, True)
    await context.bot.send_message(chat_id=CHAT_ID, text=text)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Your chat id: {update.effective_chat.id}\n"
        "Set it as the CHAT_ID environment variable to get reports every 10 min.\n\n"
        "/now - report right now\n"
        "/stats - how often the signals were right\n"
        "/backtest - test the signals on past Binance data"
    )


async def now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyzing...")
    await update.message.reply_text(await asyncio.to_thread(build_report, False))


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(await asyncio.to_thread(build_stats))


async def backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Running backtest on about 10 days of data, this takes a minute...")
    await update.message.reply_text(await asyncio.to_thread(build_backtest))


def main():
    init_db()
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("now", now))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("backtest", backtest))
    if CHAT_ID:
        app.job_queue.run_repeating(scheduled, interval=INTERVAL_SECONDS, first=10)
    app.run_polling()


if __name__ == "__main__":
    main()
