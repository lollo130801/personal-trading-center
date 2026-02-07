# Personal Trading Center

Piattaforma locale per visualizzare mercati e indicatori in maniera interattiva, con la possibilità di aggiungere indicatori Python personalizzati.

## Caratteristiche principali
- Titolo modificabile liberamente.
- Grafici candlestick, linee o barre.
- Volumi opzionali in un pannello dedicato.
- Indicatori sovrapposti, in pannelli sotto al grafico o in tab dedicate.
- Indicatori di default: SMA, EMA, WMA, RSI, filtro Fourier, filtro di Kalman.
- Onde principali di Fourier con suggerimenti ML per rialzo/ribasso.
- Supporto per indicatori personalizzati in Python.
- Dati in tempo reale (minimo 1 secondo, massimo 1 mese) tramite Binance o Yahoo Finance.
- Sessioni salvabili e news/report economici (Investing.com) opzionali.

## Avvio rapido

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Apri il browser su `http://localhost:8501`.

## Note sui dati
- Intervalli in secondi (`1s`, `5s`, `15s`) sono disponibili solo con la fonte **Binance** e vengono ricostruiti dai trade recenti.
- Yahoo Finance supporta solo intervalli da `1m` in su.

## Indicatori personalizzati
1. Crea un file Python in `custom_indicators/`.
2. Esporta una lista `INDICATORS` con istanze di `IndicatorSpec`.

Esempio:

```python
from indicators.defaults import IndicatorSpec
import pandas as pd

def my_indicator(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    return pd.DataFrame({"MyIndicator": df["close"].rolling(10).mean()})

INDICATORS = [
    IndicatorSpec(
        name="MyIndicator",
        description="Esempio personalizzato",
        params={},
        placement="overlay",
        compute=my_indicator,
    )
]
```

## Struttura
- `app.py` — UI Streamlit.
- `services/market_data.py` — accesso ai dati.
- `indicators/` — indicatori di default.
- `custom_indicators/` — indicatori custom.
- `sessions/` — sessioni salvate localmente.
