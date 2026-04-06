# Fabio Valentini — Auction Market Simulator

Backtest systemu tradingowego opartego na Auction Market Theory (NQ/MNQ futures).

## Uruchomienie w GitHub Codespaces

1. Otwórz repo na GitHub → zielony przycisk **Code** → zakładka **Codespaces** → **Create codespace on main**

2. Po otwarciu terminala w Codespace:

```bash
# Zainstaluj uv (menedżer pakietów Python)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Stwórz virtualenv Python 3.12 i zainstaluj zależności
cd backend
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt

# Uruchom serwer
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. Codespace automatycznie zaproponuje otwarcie portu 8000 w przeglądarce.

## Uruchomienie lokalne

```bash
cd backend
uv venv --python 3.12
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
uv pip install -r requirements.txt
cp .env.example .env             # uzupełnij ALPHA_VANTAGE_API_KEY
uvicorn app.main:app --reload
```

Aplikacja dostępna pod: http://localhost:8000

## Zmienne środowiskowe

Klucze API są skonfigurowane jako **Codespaces secrets** w ustawieniach repo — w Codespace są dostępne automatycznie jako zmienne środowiskowe, bez potrzeby tworzenia `.env`.

Do uruchomienia lokalnego skopiuj `backend/.env.example` → `backend/.env` i uzupełnij klucze.

## Stack

- **Backend**: FastAPI + uvicorn, pandas, numpy, pytz
- **Frontend**: HTML/CSS/JS + Chart.js (serwowany przez FastAPI)
- **Dane**: Alpha Vantage intraday API (QQQ/SPY jako proxy NQ)
- **Deploy**: VPS Rocky Linux 9.5, PM2, GitHub Actions CI/CD
