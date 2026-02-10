from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import feedparser
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots

from services.indicators import SelectedIndicator, compute_indicator, load_indicators
from services.market_data import MarketRequest, fetch_ohlcv


st.set_page_config(page_title="Personal Trading Center", layout="wide")

st.title("Personal Trading Center")

custom_indicator_dir = Path(__file__).parent / "custom_indicators"
session_dir = Path(__file__).parent / "sessions"
session_dir.mkdir(exist_ok=True)


def _default_symbols(source: str) -> List[str]:
    if source == "binance":
        return ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"]
    return ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL"]


def _load_sessions() -> List[str]:
    return sorted([path.stem for path in session_dir.glob("*.json")])


def _save_session(name: str, payload: Dict[str, object]) -> None:
    if not name:
        return
    with (session_dir / f"{name}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _load_session(name: str) -> Dict[str, object]:
    path = session_dir / f"{name}.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fetch_investing_rss(url: str) -> List[Tuple[str, str]]:
    response = requests.get(
        url,
        timeout=10,
        headers={"User-Agent": "Mozilla/5.0 (compatible; PersonalTradingCenter/1.0)"},
    )
    response.raise_for_status()
    feed = feedparser.parse(response.text)
    return [(entry.get("title", "Notizia"), entry.get("link", "#")) for entry in feed.entries]


def _fetch_economic_reports() -> List[Tuple[str, str]]:
    try:
        return _fetch_investing_rss("https://www.investing.com/rss/economic-calendar.rss")
    except Exception:  # noqa: BLE001
        return []


def _fetch_economic_news() -> List[Tuple[str, str]]:
    try:
        return _fetch_investing_rss("https://www.investing.com/rss/news_25.rss")
    except Exception:  # noqa: BLE001
        return []


def _fetch_fxstreet_events(
    start: datetime,
    end: datetime,
    min_importance: int = 3,
) -> List[Dict[str, object]]:
    params = {
        "from": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "to": end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "timezone": "UTC",
        "culture": "it-IT",
    }
    response = requests.get(
        "https://calendar-api.fxstreet.com/v1/events",
        params=params,
        timeout=10,
        headers={"User-Agent": "Mozilla/5.0 (compatible; PersonalTradingCenter/1.0)"},
    )
    response.raise_for_status()
    events = response.json()
    filtered = []
    for event in events:
        importance = event.get("importance", 0)
        if importance >= min_importance:
            filtered.append(event)
    return filtered


def _detect_divergences(
    price: pd.Series,
    indicator: pd.Series,
    window: int = 5,
) -> Tuple[List[int], List[int]]:
    lows = []
    highs = []
    for idx in range(window, len(price) - window):
        price_slice = price.iloc[idx - window : idx + window + 1]
        if price.iloc[idx] == price_slice.min():
            lows.append(idx)
        if price.iloc[idx] == price_slice.max():
            highs.append(idx)
    bullish = []
    bearish = []
    for i in range(1, len(lows)):
        prev_idx, curr_idx = lows[i - 1], lows[i]
        if price.iloc[curr_idx] < price.iloc[prev_idx] and indicator.iloc[curr_idx] > indicator.iloc[prev_idx]:
            bullish.append(curr_idx)
    for i in range(1, len(highs)):
        prev_idx, curr_idx = highs[i - 1], highs[i]
        if price.iloc[curr_idx] > price.iloc[prev_idx] and indicator.iloc[curr_idx] < indicator.iloc[prev_idx]:
            bearish.append(curr_idx)
    return bullish, bearish

DEFAULT_STATE = {
    "title": "Mercati e Indicatori",
    "source": "binance",
    "interval": "1d",
    "limit": 200,
    "chart_type": "Candlestick",
    "show_volume": True,
    "auto_refresh": False,
    "refresh_seconds": 5,
    "selected_indicators": ["SMA", "RSI", "Kalman", "FourierWaves", "SupportResistance"],
    "compare_symbols": [],
    "show_reports": False,
    "show_news": True,
}

for key, value in DEFAULT_STATE.items():
    st.session_state.setdefault(key, value)

with st.sidebar:
    st.header("Mercati")
    title_override = st.text_input("Titolo dashboard", value=st.session_state["title"], key="title")
    source = st.selectbox(
        "Fonte dati",
        ["binance", "yfinance"],
        index=["binance", "yfinance"].index(st.session_state["source"]),
        key="source",
    )
    base_symbols = _default_symbols(source)
    symbol_default = st.session_state.get("symbol", base_symbols[0])
    if symbol_default not in base_symbols:
        symbol_default = base_symbols[0]
    symbol = st.selectbox(
        "Simbolo principale",
        base_symbols,
        index=base_symbols.index(symbol_default),
        key="symbol",
    )
    compare_symbols = st.multiselect(
        "Confronta con",
        [item for item in base_symbols if item != symbol],
        default=st.session_state.get("compare_symbols", []),
        key="compare_symbols",
    )
    interval = st.selectbox(
        "Intervallo",
        ["1s", "5s", "15s", "1m", "5m", "15m", "1h", "1d", "1wk", "1mo"],
        index=["1s", "5s", "15s", "1m", "5m", "15m", "1h", "1d", "1wk", "1mo"].index(
            st.session_state["interval"]
        ),
        key="interval",
    )
    limit = st.slider(
        "Numero candele",
        min_value=50,
        max_value=500,
        value=st.session_state["limit"],
        step=10,
        key="limit",
    )
    st.divider()
    st.header("Grafico")
    chart_type = st.selectbox(
        "Tipo grafico",
        ["Candlestick", "Linea", "Barre"],
        index=["Candlestick", "Linea", "Barre"].index(st.session_state["chart_type"]),
        key="chart_type",
    )
    show_volume = st.toggle("Mostra volumi", value=st.session_state["show_volume"], key="show_volume")
    st.caption("Zoom verticale/orizzontale con scroll o box-select. Drag per pan.")
    auto_refresh = st.toggle(
        "Aggiornamento automatico", value=st.session_state["auto_refresh"], key="auto_refresh"
    )
    refresh_seconds = st.number_input(
        "Secondi tra refresh",
        min_value=1,
        max_value=60,
        value=st.session_state["refresh_seconds"],
        key="refresh_seconds",
    )

    st.divider()
    st.header("Indicatori")
    available_indicators = load_indicators(custom_indicator_dir)
    indicator_lookup = {indicator.name: indicator for indicator in available_indicators}
    selected_indicator_names = st.multiselect(
        "Seleziona indicatori",
        list(indicator_lookup.keys()),
        default=st.session_state["selected_indicators"],
        key="selected_indicators",
    )

    indicator_selections: List[SelectedIndicator] = []
    for name in selected_indicator_names:
        spec = indicator_lookup[name]
        with st.expander(f"{spec.name} · {spec.description}", expanded=True):
            placement = st.selectbox(
                "Posizione",
                ["overlay", "below", "tab"],
                index=["overlay", "below", "tab"].index(spec.placement),
                key=f"placement-{spec.name}",
            )
            params: Dict[str, float] = {}
            for param_name, default_value in spec.params.items():
                params[param_name] = st.number_input(
                    f"{param_name}",
                    value=float(default_value),
                    key=f"param-{spec.name}-{param_name}",
                )
            indicator_selections.append(
                SelectedIndicator(spec=spec, params=params, placement=placement)
            )

    st.divider()
    st.header("Sessione")
    session_name = st.text_input("Nome sessione", value="sessione-mercati")
    saved_sessions = _load_sessions()
    selected_session = st.selectbox("Carica sessione", [""] + saved_sessions)
    col_save, col_load = st.columns(2)
    with col_save:
        if st.button("Salva"):
            _save_session(
                session_name,
                {
                    "title": title_override,
                    "source": source,
                    "symbol": symbol,
                    "compare_symbols": compare_symbols,
                    "interval": interval,
                    "limit": limit,
                    "chart_type": chart_type,
                    "show_volume": show_volume,
                    "auto_refresh": auto_refresh,
                    "refresh_seconds": refresh_seconds,
                    "selected_indicators": selected_indicator_names,
                    "show_reports": show_reports,
                    "show_news": show_news,
                },
            )
    with col_load:
        if st.button("Carica") and selected_session:
            session_payload = _load_session(selected_session)
            st.session_state.update(session_payload)
            st.rerun()

    st.divider()
    st.header("Macro & News")
    show_reports = st.toggle(
        "Mostra report economici (alta importanza)",
        value=st.session_state["show_reports"],
        key="show_reports",
    )
    show_news = st.toggle(
        "Ticker news economiche",
        value=st.session_state["show_news"],
        key="show_news",
    )

st.subheader(title_override)

if auto_refresh:
    if hasattr(st, "autorefresh"):
        st.autorefresh(interval=refresh_seconds * 1000, key="auto-refresh")
    else:
        st.info("Auto refresh non disponibile in questa versione di Streamlit.")

if show_news and hasattr(st, "autorefresh"):
    st.autorefresh(interval=15000, key="news-refresh")

request = MarketRequest(
    symbol=symbol,
    interval=interval,
    limit=limit,
    source=source,
)

@st.cache_data(ttl=5)
def load_data(request: MarketRequest) -> pd.DataFrame:
    return fetch_ohlcv(request)


@st.cache_data(ttl=5)
def load_compare_data(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for item in symbols:
        compare_request = MarketRequest(
            symbol=item,
            interval=interval,
            limit=limit,
            source=source,
        )
        datasets[item] = fetch_ohlcv(compare_request)
    return datasets

try:
    data = load_data(request)
    compare_data = load_compare_data(compare_symbols) if compare_symbols else {}
except Exception as exc:  # noqa: BLE001
    st.error(f"Errore durante il fetch dei dati: {exc}")
    st.stop()

if data.empty:
    st.warning("Nessun dato disponibile per i parametri selezionati.")
    st.stop()

st.subheader("Filtro periodo")
min_date = data["timestamp"].min().date()
max_date = data["timestamp"].max().date()
date_range = st.date_input(
    "Seleziona intervallo",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = date_range, max_date

mask = (data["timestamp"].dt.date >= start_date) & (data["timestamp"].dt.date <= end_date)
data = data.loc[mask].reset_index(drop=True)
filtered_compare_data = {}
for key, compare_df in compare_data.items():
    if compare_df.empty:
        filtered_compare_data[key] = compare_df
        continue
    compare_mask = (compare_df["timestamp"].dt.date >= start_date) & (
        compare_df["timestamp"].dt.date <= end_date
    )
    filtered_compare_data[key] = compare_df.loc[compare_mask].reset_index(drop=True)
compare_data = filtered_compare_data

if data.empty:
    st.warning("Nessun dato disponibile per il periodo selezionato.")
    st.stop()

events = []
if show_reports or interval == "1d":
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time()) + timedelta(days=7)
    try:
        events = _fetch_fxstreet_events(start_dt, end_dt, min_importance=3)
    except Exception:  # noqa: BLE001
        events = []

below_indicators = [ind for ind in indicator_selections if ind.placement == "below"]
overlay_indicators = [ind for ind in indicator_selections if ind.placement == "overlay"]
tab_indicators = [ind for ind in indicator_selections if ind.placement == "tab"]

volume_row = 1 if show_volume else 0
rows = 1 + volume_row + len(below_indicators)
volume_space = 0.1 if show_volume else 0.0
price_space = 0.65 if rows > 1 else 0.8
remaining_space = max(0.0, 1.0 - price_space - volume_space)
row_heights = [price_space]
if show_volume:
    row_heights.append(volume_space)
if below_indicators:
    row_heights.extend([remaining_space / max(len(below_indicators), 1)] * len(below_indicators))
fig = make_subplots(
    rows=rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=row_heights,
)

if chart_type == "Candlestick":
    fig.add_trace(
        go.Candlestick(
            x=data["timestamp"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name=symbol,
            increasing_line_color="#16a34a",
            increasing_fillcolor="#22c55e",
            decreasing_line_color="#dc2626",
            decreasing_fillcolor="#ef4444",
        ),
        row=1,
        col=1,
    )
elif chart_type == "Linea":
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"],
            y=data["close"],
            mode="lines",
            name=symbol,
        ),
        row=1,
        col=1,
    )
else:
    fig.add_trace(
        go.Bar(
            x=data["timestamp"],
            y=data["close"],
            name=symbol,
        ),
        row=1,
        col=1,
    )

if compare_data:
    for compare_symbol, compare_df in compare_data.items():
        if compare_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=compare_df["timestamp"],
                y=compare_df["close"],
                mode="lines",
                name=compare_symbol,
                line=dict(width=1.5, dash="dot"),
            ),
            row=1,
            col=1,
        )

if interval == "1d" and events:
    event_x = []
    event_y = []
    event_text = []
    for event in events:
        date_raw = event.get("date")
        title = event.get("title", "Evento macro")
        if not date_raw:
            continue
        try:
            event_dt = datetime.fromisoformat(str(date_raw).replace("Z", "+00:00"))
        except ValueError:
            continue
        event_x.append(event_dt)
        if data["timestamp"].min() <= event_dt <= data["timestamp"].max():
            nearest_idx = (data["timestamp"] - event_dt).abs().idxmin()
            event_y.append(data.loc[nearest_idx, "close"])
        else:
            event_y.append(data["close"].iloc[-1])
        event_text.append(title)
    if event_x:
        fig.add_trace(
            go.Scatter(
                x=event_x,
                y=event_y,
                mode="markers",
                name="Eventi macro",
                marker=dict(color="#f59e0b", size=8, symbol="diamond"),
                text=event_text,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )

current_row = 2
if show_volume:
    fig.add_trace(
        go.Bar(
            x=data["timestamp"],
            y=data["volume"],
            name=f"Volume {symbol}",
            marker_color="rgba(148, 163, 184, 0.6)",
        ),
        row=current_row,
        col=1,
    )
    fig.update_yaxes(showticklabels=False, row=current_row, col=1)
    current_row += 1

for indicator in overlay_indicators:
    indicator_data = compute_indicator(indicator, data)
    for column in indicator_data.columns:
        if column == "timestamp":
            continue
        fig.add_trace(
            go.Scatter(
                x=indicator_data["timestamp"],
                y=indicator_data[column],
                mode="lines",
                name=column,
            ),
            row=1,
            col=1,
        )

for indicator in below_indicators:
    indicator_data = compute_indicator(indicator, data)
    for column in indicator_data.columns:
        if column == "timestamp":
            continue
        fig.add_trace(
            go.Scatter(
                x=indicator_data["timestamp"],
                y=indicator_data[column],
                mode="lines",
                name=column,
            ),
            row=current_row,
            col=1,
        )
        if indicator.spec.name == "RSI":
            bullish, bearish = _detect_divergences(data["close"], indicator_data[column])
            if bullish:
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["timestamp"].iloc[bullish],
                        y=indicator_data[column].iloc[bullish],
                        mode="markers",
                        name="Divergenza positiva",
                        marker=dict(symbol="triangle-up", color="#16a34a", size=10),
                    ),
                    row=current_row,
                    col=1,
                )
            if bearish:
                fig.add_trace(
                    go.Scatter(
                        x=indicator_data["timestamp"].iloc[bearish],
                        y=indicator_data[column].iloc[bearish],
                        mode="markers",
                        name="Divergenza negativa",
                        marker=dict(symbol="triangle-down", color="#dc2626", size=10),
                    ),
                    row=current_row,
                    col=1,
                )
    current_row += 1

fig.update_layout(
    height=720,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=40, r=40, t=40, b=30),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    hovermode="x unified",
    dragmode="pan",
)
fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(148, 163, 184, 0.2)",
    rangeslider=dict(
        visible=True,
        bgcolor="rgba(148, 163, 184, 0.12)",
        bordercolor="rgba(148, 163, 184, 0.4)",
        borderwidth=1,
        thickness=0.06,
    ),
)
fig.update_yaxes(
    showgrid=True,
    gridcolor="rgba(148, 163, 184, 0.2)",
    fixedrange=False,
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config=dict(scrollZoom=True, displaylogo=False, responsive=True),
)

if show_news:
    news_items = _fetch_economic_news()
    if news_items:
        news_index = st.session_state.get("news_index", 0) % len(news_items)
        title, link = news_items[news_index]
        st.markdown(f"**News**: [{title}]({link})")
        st.session_state["news_index"] = news_index + 1
    else:
        st.info("Nessuna news disponibile al momento.")

if show_reports:
    reports = _fetch_economic_reports()
    with st.expander("Report economici principali"):
        st.caption("Fonti: FXStreet (eventi con importanza alta) e Investing.com (feed RSS).")
        if events:
            for event in events[:12]:
                date_raw = event.get("date")
                title = event.get("title", "Evento macro")
                try:
                    event_dt = datetime.fromisoformat(str(date_raw).replace("Z", "+00:00"))
                    formatted = event_dt.strftime("%Y-%m-%d %H:%M UTC")
                except ValueError:
                    formatted = "Data non disponibile"
                st.markdown(f"- **{formatted}** · {title}")
        elif reports:
            for title, link in reports[:10]:
                st.markdown(f"- [{title}]({link})")
        else:
            st.info("Impossibile recuperare il calendario economico.")

st.divider()
st.subheader("Suggerimenti ML su onde Fourier")
try:
    waves_df = compute_indicator(
        SelectedIndicator(
            spec=indicator_lookup["FourierWaves"],
            params={"top_waves": 3},
            placement="below",
        ),
        data,
    ).drop(columns=["timestamp"])
    returns = data["close"].pct_change().shift(-1).dropna()
    features = waves_df.iloc[:-1].fillna(0)
    target = (returns > 0).astype(int)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)
    model = LogisticRegression(max_iter=500)
    model.fit(x_scaled, target)
    latest_features = scaler.transform(waves_df.tail(1).fillna(0))
    probability = model.predict_proba(latest_features)[0][1]
    direction = "rialzo" if probability >= 0.5 else "ribasso"
    st.metric("Probabilità rialzo", f"{probability:.2%}", direction)
    coeffs = pd.Series(model.coef_[0], index=features.columns).abs().sort_values(ascending=False)
    best_waves = ", ".join(coeffs.head(2).index)
    st.caption(f"Le onde più influenti: {best_waves}")
except Exception as exc:  # noqa: BLE001
    st.info(f"Suggerimenti ML non disponibili: {exc}")

if tab_indicators:
    st.subheader("Indicatori in tab")
    tabs = st.tabs([ind.spec.name for ind in tab_indicators])
    for tab, indicator in zip(tabs, tab_indicators, strict=False):
        with tab:
            indicator_data = compute_indicator(indicator, data)
            tab_fig = go.Figure()
            for column in indicator_data.columns:
                if column == "timestamp":
                    continue
                tab_fig.add_trace(
                    go.Scatter(
                        x=indicator_data["timestamp"],
                        y=indicator_data[column],
                        mode="lines",
                        name=column,
                    )
                )
            tab_fig.update_layout(height=300, margin=dict(l=40, r=40, t=20, b=20))
            st.plotly_chart(tab_fig, use_container_width=True)

st.divider()
with st.expander("Come aggiungere indicatori personalizzati"):
    st.markdown(
        """
        Puoi aggiungere indicatori Python creando un file nella cartella `custom_indicators/`.
        Ogni file deve esportare una lista `INDICATORS` con istanze di `IndicatorSpec`.

        Esempio minimo:
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
        """
    )
