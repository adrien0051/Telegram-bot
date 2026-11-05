# Telegram Bot

## 1. Get a token
In Telegram, chat with @BotFather -> /newbot -> copy the token.

## 2. Upload to GitHub
Create a new repo and upload these files. Never commit your token.

## 3. Run it 24/7 (no manual commands)
GitHub only stores code; it can't keep a bot running. Connect your repo to a host that does:

- Railway / Render / Fly.io: "New project" -> "Deploy from GitHub repo" -> pick this repo.
  It detects the Dockerfile automatically.
- Add an environment variable: BOT_TOKEN = your token
- Deploy once, send /start to your bot, copy the chat id it replies with, add it as env variable CHAT_ID, and redeploy.
- Edit WATCHLIST in bot.py to track other symbols (Yahoo Finance tickers).
- Deploy. It restarts automatically if it crashes and redeploys on every git push.

## Local test (optional)
    pip install -r requirements.txt
    BOT_TOKEN=your_token python bot.py

## Binance
Binance pairs (BINANCE_WATCHLIST in bot.py) use Binance's public market-data API, so no API key is needed.
If your host is blocked (HTTP 451 in logs), add env variable BINANCE_API = https://api.binance.us
