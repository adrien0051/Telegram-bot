import os
import logging

import pandas as pd
import requests
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)

TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ.get("CHAT_ID")  # send /start to the bot to learn yours
INTERVAL_SECONDS = 600  # 10 minutes

# Yahoo Finance symbols. Edit freely: stocks, crypto (-USD), forex (=X), gold (GC=F)...
WATCHLIST = ["AAPL", "MSFT", "NVDA", "TSLA", "EURUSD=X", "GC=F"]

# Binance spot pairs (public market data, no API key needed)
BINANCE_WATCHLIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
# If your host is blocked from Binance (HTTP 451), set BINANCE_API to https://api.binance.us
BINANCE_API = os.environ.get("BINANCE_API", "https://data-api.binance.vision")

DISCLAIMER = "Not financial advice. Indicator signals, not predictions."


def rsi(close: pd.Series, n: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, 1e-9)
    return float(100 - 100 / (1 + rs.iloc[-1]))


def score_series(symbol: str, close: pd.Series):
    if len(close) < 60:
        return None
    r = rsi(close)
    sma_fast = close.rolling(20).mean().iloc[-1]
    sma_slow = close.rolling(50).mean().iloc[-1]
    change = (close.iloc[-1] / close.iloc[-12] - 1) * 100  # ~last hour (5m candles)

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
    return {"symbol": symbol, "price": float(close.iloc[-1]), "rsi": r, "change": float(change), "lean": lean}


def analyze_yahoo(symbol: str):
    df = yf.download(symbol, period="5d", interval="5m", progress=False, auto_adjust=True)
    if df.empty:
        return None
    return score_series(symbol, df["Close"].squeeze().dropna())


def analyze_binance(symbol: str):
    resp = requests.get(
        f"{BINANCE_API}/api/v3/klines",
        params={"symbol": symbol, "interval": "5m", "limit": 200},
        timeout=15,
    )
    resp.raise_for_status()
    close = pd.Series([float(k[4]) for k in resp.json()])
    return score_series(f"{symbol} (Binance)", close)


def build_report() -> str:
    icons = {"UP": "🟢", "DOWN": "🔴", "NEUTRAL": "⚪"}
    lines = []
    jobs = [(x, analyze_yahoo) for x in WATCHLIST] + [(x, analyze_binance) for x in BINANCE_WATCHLIST]
    for sym, fn in jobs:
        try:
            a = fn(sym)
        except Exception as e:
            logging.warning("Failed %s: %s", sym, e)
            a = None
        if a:
            lines.append(
                f"{icons[a['lean']]} {a['symbol']}: {a['lean']}  "
                f"${a['price']:,.2f} | 1h {a['change']:+.2f}% | RSI {a['rsi']:.0f}"
            )
    if not lines:
        return "No data right now (markets closed or data source down)."
    return "\n".join(lines) + f"\n\n{DISCLAIMER}"


async def scheduled(context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=CHAT_ID, text=build_report())


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Your chat id: {update.effective_chat.id}\n"
        "Set it as the CHAT_ID environment variable to get reports every 10 min.\n"
        "/now - get a report right now"
    )


async def now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyzing...")
    await update.message.reply_text(build_report())


def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("now", now))
    if CHAT_ID:
        app.job_queue.run_repeating(scheduled, interval=INTERVAL_SECONDS, first=10)
    app.run_polling()


if __name__ == "__main__":
    main()
