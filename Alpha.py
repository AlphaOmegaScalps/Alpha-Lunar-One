import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import ephem
from polygon import RESTClient
from collections import defaultdict
import google.generativeai as genai
import io
from PIL import Image
import base64
import json
import re
import streamlit.components.v1 as components
import requests

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
    AGGRID_AVAILABLE = True
except ImportError:
    AGGRID_AVAILABLE = False

# Official Public.com Python SDK (optional at import time so Polygon/REST fallback
# still works if the SDK has not yet been installed locally).
try:
    from public_api_sdk import (
        PublicApiClient,
        PublicApiClientConfiguration,
        OrderInstrument,
        InstrumentType,
        OptionExpirationsRequest,
        OptionChainRequest,
    )
    from public_api_sdk.auth_config import ApiKeyAuthConfig
    PUBLIC_SDK_AVAILABLE = True
except ImportError:
    PUBLIC_SDK_AVAILABLE = False

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(page_title="Swell Labs: Alpha Tool Suite", layout="wide", page_icon="🚀")
st.markdown("""
<style>

/* Normal Streamlit footer */
footer {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

/* Streamlit toolbar */
[data-testid="stToolbar"] {
    display: none !important;
    visibility: hidden !important;
}

/* Streamlit decoration */
[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}

/* Status widget */
[data-testid="stStatusWidget"] {
    display: none !important;
    visibility: hidden !important;
}

/* Bottom container */
[data-testid="stBottomBlockContainer"] {
    display: none !important;
    visibility: hidden !important;
}

/* Header */
header {
    visibility: hidden !important;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# PUBLIC.COM API — DEV SETTINGS
# ==============================
PUBLIC_API_SECRET = "6H2L9p5dHqlQlLHZeXypOWoplfCikEYz"
# Public credentials are loaded from Streamlit Secrets inside main_app().
# --- MAIN APPLICATION (YOUR ORIGINAL CODE MOVED INTO THIS FUNCTION) ---
#_______________________________________________________________________#
# --- USER AUTHENTICATION (NEW CODE) ---
def check_login():
    """Checks if the user is logged in."""
    if not st.session_state.get("logged_in"):
        # If not logged in, show the login form
        show_login_form()
        return False
    return True

def show_login_form():
    """Displays a login form."""
    with st.form("login_form"):
        st.title("The Alpha Login")
        username = st.text_input("Username").lower()
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

        if submitted:
            # Check if the username exists and the password is correct
            if username in st.secrets["credentials"]["usernames"] and \
               password == st.secrets["credentials"]["usernames"][username]["password"]:
                
                # If login is successful, set session state
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["name"] = st.secrets["credentials"]["usernames"][username]["name"]
                st.rerun() # Rerun the app to show the main content
            else:
                st.error("Invalid username or password")


def get_eclipse_calendar_data():
    """NASA eclipse calendar data for the Eclipses page and future market studies."""
    eclipses = [
        {"date":"2026-08-28","kind":"Lunar","type":"Partial","saros":138,"visibility":"Eastern Pacific, Americas, Europe, Africa","solar_path":False},
        {"date":"2027-02-06","kind":"Solar","type":"Annular","saros":131,"visibility":"South America, Antarctica, western & southern Africa","solar_path":True},
        {"date":"2027-02-20","kind":"Lunar","type":"Penumbral","saros":143,"visibility":"Americas, Europe, Africa, Asia","solar_path":False},
        {"date":"2027-07-18","kind":"Lunar","type":"Penumbral","saros":110,"visibility":"Eastern Africa, Asia, Australia, Pacific","solar_path":False},
        {"date":"2027-08-02","kind":"Solar","type":"Total","saros":136,"visibility":"Africa, Europe, Middle East, western & southern Asia","solar_path":True},
        {"date":"2027-08-17","kind":"Lunar","type":"Penumbral","saros":148,"visibility":"Pacific, Americas","solar_path":False},
        {"date":"2028-01-12","kind":"Lunar","type":"Partial","saros":115,"visibility":"Americas, Europe, Africa","solar_path":False},
        {"date":"2028-01-26","kind":"Solar","type":"Annular","saros":141,"visibility":"Eastern North America, Central & South America, western Europe, northwest Africa","solar_path":True},
        {"date":"2028-07-06","kind":"Lunar","type":"Partial","saros":120,"visibility":"Europe, Africa, Asia, Australia","solar_path":False},
        {"date":"2028-07-22","kind":"Solar","type":"Total","saros":146,"visibility":"Southeast Asia, East Indies, Australia, New Zealand","solar_path":True},
        {"date":"2028-12-31","kind":"Lunar","type":"Total","saros":125,"visibility":"Europe, Africa, Asia, Australia, Pacific","solar_path":False},
        {"date":"2029-01-14","kind":"Solar","type":"Partial","saros":151,"visibility":"North America, Central America","solar_path":False},
        {"date":"2029-06-12","kind":"Solar","type":"Partial","saros":118,"visibility":"Arctic, Scandinavia, Alaska, northern Asia, northern Canada","solar_path":False},
        {"date":"2029-06-26","kind":"Lunar","type":"Total","saros":130,"visibility":"Americas, Europe, Africa, Middle East","solar_path":False},
        {"date":"2029-07-11","kind":"Solar","type":"Partial","saros":156,"visibility":"Southern Chile, southern Argentina","solar_path":False},
        {"date":"2029-12-05","kind":"Solar","type":"Partial","saros":123,"visibility":"Southern Argentina, southern Chile, Antarctica","solar_path":False},
        {"date":"2029-12-20","kind":"Lunar","type":"Total","saros":135,"visibility":"Americas, Europe, Africa, Asia","solar_path":False},
        {"date":"2030-06-01","kind":"Solar","type":"Annular","saros":128,"visibility":"Europe, northern Africa, Middle East, Asia, Arctic, Alaska","solar_path":True},
        {"date":"2030-06-15","kind":"Lunar","type":"Partial","saros":140,"visibility":"Europe, Africa, Asia, Australia","solar_path":False},
        {"date":"2030-11-25","kind":"Solar","type":"Total","saros":133,"visibility":"Southern Africa, southern Indian Ocean, East Indies, Australia, Antarctica","solar_path":True},
    ]
    df = pd.DataFrame(eclipses)
    df["date"] = pd.to_datetime(df["date"])
    df["label"] = df["date"].dt.strftime("%b %d, %Y") + " · " + df["type"] + " " + df["kind"]
    df["days_away"] = (df["date"] - pd.Timestamp(dt.datetime.utcnow().date())).dt.days
    return df.sort_values("date").reset_index(drop=True)


def get_historical_eclipse_study_data(start_date, end_date):
    """NASA-sourced eclipse dates for historical price-event studies."""
    eclipses = [
        # Solar eclipses, 2016-2026
        ("2016-09-01", "Solar", "Annular", 135),
        ("2017-02-26", "Solar", "Annular", 140),
        ("2017-08-21", "Solar", "Total", 145),
        ("2018-02-15", "Solar", "Partial", 150),
        ("2018-07-13", "Solar", "Partial", 117),
        ("2018-08-11", "Solar", "Partial", 155),
        ("2019-01-06", "Solar", "Partial", 122),
        ("2019-07-02", "Solar", "Total", 127),
        ("2019-12-26", "Solar", "Annular", 132),
        ("2020-06-21", "Solar", "Annular", 137),
        ("2020-12-14", "Solar", "Total", 142),
        ("2021-06-10", "Solar", "Annular", 147),
        ("2021-12-04", "Solar", "Total", 152),
        ("2022-04-30", "Solar", "Partial", 119),
        ("2022-10-25", "Solar", "Partial", 124),
        ("2023-04-20", "Solar", "Hybrid", 129),
        ("2023-10-14", "Solar", "Annular", 134),
        ("2024-04-08", "Solar", "Total", 139),
        ("2024-10-02", "Solar", "Annular", 144),
        ("2025-03-29", "Solar", "Partial", 149),
        ("2025-09-21", "Solar", "Partial", 154),
        ("2026-02-17", "Solar", "Annular", 121),
        ("2026-08-12", "Solar", "Total", 126),
        # Lunar eclipses, 2016-2026
        ("2016-09-16", "Lunar", "Penumbral", 147),
        ("2017-02-11", "Lunar", "Penumbral", 114),
        ("2017-08-07", "Lunar", "Partial", 119),
        ("2018-01-31", "Lunar", "Total", 124),
        ("2018-07-27", "Lunar", "Total", 129),
        ("2019-01-21", "Lunar", "Total", 134),
        ("2019-07-16", "Lunar", "Partial", 139),
        ("2020-01-10", "Lunar", "Penumbral", 144),
        ("2020-06-05", "Lunar", "Penumbral", 111),
        ("2020-07-05", "Lunar", "Penumbral", 149),
        ("2020-11-30", "Lunar", "Penumbral", 116),
        ("2021-05-26", "Lunar", "Total", 121),
        ("2021-11-19", "Lunar", "Partial", 126),
        ("2022-05-16", "Lunar", "Total", 131),
        ("2022-11-08", "Lunar", "Total", 136),
        ("2023-05-05", "Lunar", "Penumbral", 141),
        ("2023-10-28", "Lunar", "Partial", 146),
        ("2024-03-25", "Lunar", "Penumbral", 113),
        ("2024-09-18", "Lunar", "Partial", 118),
        ("2025-03-14", "Lunar", "Total", 123),
        ("2025-09-07", "Lunar", "Total", 128),
        ("2026-03-03", "Lunar", "Total", 133),
        ("2026-08-28", "Lunar", "Partial", 138),
    ]
    df = pd.DataFrame(eclipses, columns=["date", "kind", "type", "saros"])
    df["date"] = pd.to_datetime(df["date"])
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].sort_values("date").reset_index(drop=True)


def display_eclipse_price_study(get_price_data_func, ticker, end_date):
    """Price history with eclipse markers, same-body phase deltas, and future eclipse projections."""
    st.markdown("### Historical Eclipse Price Study")
    st.caption("Eclipse phases run from the first trading-day OPEN at one eclipse to the first trading-day OPEN at the next eclipse of the same body. Partial/total/annular subtype does not affect the phase calculation.")

    end_default = pd.Timestamp(end_date).date()
    start_default = (pd.Timestamp(end_date) - pd.DateOffset(years=10)).date()

    date_col1, date_col2, future_col = st.columns(3)
    with date_col1:
        study_start_date = st.date_input(
            "Study Start Date", value=start_default,
            max_value=end_default, key="eclipse_study_start"
        )
    with date_col2:
        study_end_date = st.date_input(
            "Study End Date", value=end_default,
            max_value=dt.date.today(), key="eclipse_study_end"
        )
    with future_col:
        future_eclipse_count = st.number_input(
            "Future Eclipses to Show", min_value=0, max_value=20, value=4, step=1,
            key="eclipse_future_count"
        )

    if study_start_date > study_end_date:
        st.error("Study Start Date must be on or before Study End Date.")
        return None, None

    start_ts = pd.Timestamp(study_start_date).normalize()
    end_ts = pd.Timestamp(study_end_date).normalize()

    # Keep the complete in-range eclipse set for the phase/database calculations.
    all_study_eclipses = get_historical_eclipse_study_data(start_ts, end_ts)
    chart_eclipses = all_study_eclipses.copy()

    filter_col1, filter_col2, filter_col3, filter_col4, filter_col5, filter_col6 = st.columns(6)
    with filter_col1:
        show_solar = st.checkbox("Show Solar Verticals", value=True, key="eclipse_show_solar")
    with filter_col2:
        show_lunar = st.checkbox("Show Lunar Verticals", value=True, key="eclipse_show_lunar")
    with filter_col3:
        show_phase_overlay = st.checkbox("Show % Delta Overlay", value=True, key="eclipse_show_delta_overlay")
    with filter_col4:
        show_fib_time_zones = st.checkbox(
            "Show 1/3 & 2/3 Time Zones", value=True, key="eclipse_show_fib_time_zones"
        )
    with filter_col5:
        show_future_fib_signals = st.checkbox(
            "Show Future Third Signals", value=True, key="eclipse_show_future_fib_signals"
        )
    with filter_col6:
        show_third_behavior = st.checkbox(
            "Measure 1/3 Price Behavior", value=True, key="eclipse_show_third_behavior"
        )

    num_eclipse_price_levels = st.number_input(
        "Show Price Levels for Last X Eclipses", min_value=0, max_value=20, value=2, step=1,
        key="eclipse_num_price_levels"
    )

    # Load only through the selected study end. Future eclipses are markers only;
    # no nonexistent future price data is requested.
    price_start = (start_ts - pd.Timedelta(days=7)).date()
    price_end = end_ts.date()
    with st.spinner(f"Loading {ticker} price history for the eclipse study..."):
        price_df = get_price_data_func(ticker, price_start, price_end)

    if price_df.empty:
        st.warning(f"No price history was returned for {ticker}.")
        return study_start_date, study_end_date

    price_df = price_df.copy()
    price_df["Date"] = pd.to_datetime(price_df["Date"]).dt.normalize()
    price_df = price_df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    price_indexed = price_df.set_index("Date").sort_index()

    # Resolve each eclipse to the first trading session on/after its calendar date.
    def first_trading_session(event_date):
        future = price_indexed[price_indexed.index >= pd.Timestamp(event_date).normalize()]
        if future.empty:
            return None, None
        return future.index[0], future.iloc[0]

    event_points = []
    for _, event in all_study_eclipses.iterrows():
        trade_date, trade_row = first_trading_session(event["date"])
        if trade_row is not None:
            event_points.append({
                "calendar_date": pd.Timestamp(event["date"]).normalize(),
                "trade_date": pd.Timestamp(trade_date).normalize(),
                "body": event["kind"],
                "type": event["type"],
                "saros": int(event["saros"]),
                "open": float(trade_row["Open"]),
            })

    # Build the same-body phase database. The phase ignores eclipse subtype.
    phase_rows = []
    for i, current in enumerate(event_points):
        next_same = next((x for x in event_points[i + 1:] if x["body"] == current["body"]), None)
        phase = {
            "Eclipse Date": current["calendar_date"],
            "Body": current["body"],
            "Type": current["type"],
            "Saros": current["saros"],
            "Trading Date": current["trade_date"],
            "Eclipse Open": current["open"],
            "Next Same-Class": next_same["calendar_date"] if next_same else pd.NaT,
            "Next Trading Date": next_same["trade_date"] if next_same else pd.NaT,
            "Next Open": next_same["open"] if next_same else None,
            "Delta ($)": None,
            "Delta (%)": None,
            "Max Profit": None,
            "Max Drawdown": None,
            "Status": "Open",
        }

        if next_same is not None:
            entry = current["open"]
            exit_open = next_same["open"]
            delta = exit_open - entry
            pct = (delta / entry) * 100.0 if entry else 0.0
            phase_df = price_df[
                (price_df["Date"] >= current["trade_date"]) &
                (price_df["Date"] <= next_same["trade_date"])
            ]
            phase["Delta ($)"] = delta
            phase["Delta (%)"] = pct
            phase["Max Profit"] = float(phase_df["High"].max() - entry) if not phase_df.empty else None
            phase["Max Drawdown"] = float(entry - phase_df["Low"].min()) if not phase_df.empty else None
            phase["Status"] = "Win" if delta > 0 else "Loss"

        phase_rows.append(phase)

    study_df = pd.DataFrame(phase_rows)

    # Chart starts with the selected historical price range.
    study_fig = go.Figure(data=[go.Candlestick(
        x=price_df["Date"].tolist(),
        open=price_df["Open"].astype(float).tolist(),
        high=price_df["High"].astype(float).tolist(),
        low=price_df["Low"].astype(float).tolist(),
        close=price_df["Close"].astype(float).tolist(),
        name=ticker
    )])

    colors = {"Solar": "#f59e0b", "Lunar": "#8b5cf6"}

    # Delta overlays: one translucent background block per completed same-body phase.
    # The overlay follows the SAME Solar/Lunar visibility toggles as the verticals,
    # and its boundaries use the eclipse calendar dates so the shading connects
    # directly to the corresponding vertical lines.
    if show_phase_overlay and not study_df.empty:
        for _, row in study_df.dropna(subset=["Delta (%)", "Next Same-Class"]).iterrows():
            body = row["Body"]
            if body == "Solar" and not show_solar:
                continue
            if body == "Lunar" and not show_lunar:
                continue

            start_x = pd.Timestamp(row["Eclipse Date"]).normalize()
            end_x = pd.Timestamp(row["Next Same-Class"]).normalize()
            pct = float(row["Delta (%)"])
            delta = float(row["Delta ($)"])
            fill = "rgba(34,197,94,0.10)" if pct >= 0 else "rgba(239,68,68,0.10)"
            text_color = "#28a745" if pct >= 0 else "#dc3545"

            study_fig.add_vrect(
                x0=start_x, x1=end_x, fillcolor=fill, line_width=0, layer="below"
            )
            mid_x = start_x + (end_x - start_x) / 2
            study_fig.add_annotation(
                x=mid_x, y=0.985, yref="paper",
                text=f"{'+' if delta >= 0 else ''}${delta:.2f} · {'+' if pct >= 0 else ''}{pct:.2f}%",
                showarrow=False, font=dict(size=11, color=text_color),
                bgcolor="rgba(13,18,25,0.55)", borderpad=3
            )

    # Fibonacci-style time zones: divide each SAME-BODY eclipse-to-eclipse
    # interval into thirds. The two internal time zones are exactly 1/3 and 2/3
    # of the elapsed calendar time from one eclipse to the next. This creates two
    # additional cycle markers inside every larger Solar-to-Solar and Lunar-to-Lunar
    # cycle without changing the underlying eclipse dates or price calculations.
    #
    # We use calendar time here because the eclipse cycle is defined by the actual
    # eclipse dates, not by trading-session count. The markers can therefore be
    # compared directly with the astronomical cycle boundaries.
    if show_fib_time_zones and event_points:
        for body in ("Solar", "Lunar"):
            if body == "Solar" and not show_solar:
                continue
            if body == "Lunar" and not show_lunar:
                continue

            body_events = [e for e in event_points if e["body"] == body]
            for start_event, end_event in zip(body_events[:-1], body_events[1:]):
                start_x = pd.Timestamp(start_event["calendar_date"]).normalize()
                end_x = pd.Timestamp(end_event["calendar_date"]).normalize()
                span = end_x - start_x
                if span <= pd.Timedelta(0):
                    continue

                color = colors[body]
                for fraction, label in ((1/3, "1/3"), (2/3, "2/3")):
                    zone_x = start_x + span * fraction
                    study_fig.add_vline(
                        x=zone_x, line_color=color, line_dash="dot", line_width=1.0,
                        opacity=0.65
                    )
                    study_fig.add_annotation(
                        x=zone_x, y=0.02, yref="paper", yshift=-2,
                        text=f"{body} {label}", showarrow=False, textangle=-90,
                        font=dict(size=8, color="#ffffff"),
                        bgcolor="rgba(13,18,25,0.55)", borderpad=2
                    )

    # Historical eclipse verticals. Solar/Lunar toggles control these independently.
    for event in event_points:
        if event["body"] == "Solar" and not show_solar:
            continue
        if event["body"] == "Lunar" and not show_lunar:
            continue
        color = colors[event["body"]]
        study_fig.add_vline(x=event["calendar_date"], line_color=color, line_dash="dash", line_width=1.5)
        study_fig.add_annotation(
            x=event["calendar_date"], y=1.0, yref="paper", yshift=8,
            text=event["body"], showarrow=False, textangle=-90,
            font=dict(size=9, color="#ffffff")
        )

    # Horizontal price levels use the OPEN of the most recent visible eclipses,
    # matching the lunar chart's price-level behavior. Solar/Lunar visibility
    # toggles also control their corresponding horizontal levels.
    if int(num_eclipse_price_levels) > 0 and event_points:
        visible_level_points = [
            event for event in event_points
            if (event["body"] == "Solar" and show_solar)
            or (event["body"] == "Lunar" and show_lunar)
        ]
        visible_level_points = visible_level_points[-int(num_eclipse_price_levels):]
        for event in visible_level_points:
            color = colors[event["body"]]
            study_fig.add_hline(
                y=event["open"],
                line_dash="dash",
                line_color=color,
                line_width=1.5,
                annotation_text=f'{event["body"]}: ${event["open"]:.2f}',
                annotation_position="bottom right",
                annotation_font=dict(color=color, size=11)
            )

    # Future verticals: use the NASA calendar and show only the next X eclipses
    # after the selected study end. They are visual markers only.
    future_calendar = get_eclipse_calendar_data()
    future_calendar = future_calendar[future_calendar["date"] > end_ts].sort_values("date")
    future_calendar = future_calendar.head(int(future_eclipse_count))
    # Future 1/3 and 2/3 signals: extend the same-body cycle from the last
    # available eclipse into the selected future eclipse calendar. These are
    # projections only; they do not use or imply future market prices.
    future_fib_points = []
    if show_future_fib_signals:
        combined_calendar = pd.concat([
            all_study_eclipses[["date", "kind", "type", "saros"]].copy(),
            future_calendar[["date", "kind", "type", "saros"]].copy()
        ], ignore_index=True).drop_duplicates(subset=["date", "kind"]).sort_values("date")

        for body in ("Solar", "Lunar"):
            if body == "Solar" and not show_solar:
                continue
            if body == "Lunar" and not show_lunar:
                continue
            body_events = combined_calendar[combined_calendar["kind"] == body].sort_values("date")
            for start_event, end_event in zip(body_events.iloc[:-1].itertuples(index=False), body_events.iloc[1:].itertuples(index=False)):
                start_x = pd.Timestamp(start_event.date).normalize()
                end_x = pd.Timestamp(end_event.date).normalize()
                if end_x <= end_ts or end_x <= start_x:
                    continue
                span = end_x - start_x
                for fraction, label in ((1/3, "1/3"), (2/3, "2/3")):
                    signal_x = start_x + span * fraction
                    if signal_x > end_ts:
                        future_fib_points.append({
                            "body": body, "label": label, "date": signal_x,
                            "start": start_x, "end": end_x
                        })

        for signal in future_fib_points:
            color = colors[signal["body"]]
            study_fig.add_vline(
                x=signal["date"], line_color=color, line_dash="dashdot",
                line_width=1.2, opacity=0.9
            )
            study_fig.add_annotation(
                x=signal["date"], y=0.02, yref="paper", yshift=-2,
                text=f'{signal["body"]} {signal["label"]} (FUTURE)',
                showarrow=False, textangle=-90,
                font=dict(size=8, color="#ffffff"),
                bgcolor="rgba(13,18,25,0.72)", borderpad=2
            )

    for _, event in future_calendar.iterrows():
        if event["kind"] == "Solar" and not show_solar:
            continue
        if event["kind"] == "Lunar" and not show_lunar:
            continue
        color = colors[event["kind"]]
        study_fig.add_vline(x=event["date"], line_color=color, line_dash="dot", line_width=1.4)
        study_fig.add_annotation(
            x=event["date"], y=1.0, yref="paper", yshift=8,
            text=f"{event['kind']} (FUTURE)", showarrow=False, textangle=-90,
            font=dict(size=9, color="#ffffff")
        )

    # Leave the x-axis wide enough to display future verticals without requesting
    # or plotting future OHLC prices.
    x_max = future_calendar["date"].max() if not future_calendar.empty else price_df["Date"].max()
    study_fig.update_layout(
        title=f"{ticker} — Eclipse Price History ({study_start_date} → {study_end_date})",
        xaxis_title="Date", yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False, height=700
    )
    study_fig.update_xaxes(
        range=[price_df["Date"].min(), x_max],
        rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider_visible=False
    )
    st.plotly_chart(study_fig, use_container_width=True)

    if show_third_behavior and not study_df.empty and not price_df.empty:
        # Measure actual market behavior inside each completed same-body cycle.
        # The cycle is split by calendar-time boundaries at 1/3 and 2/3, while
        # performance is measured only from available trading sessions.
        third_rows = []
        for _, cycle in study_df.dropna(subset=["Next Same-Class", "Next Trading Date"]).iterrows():
            cycle_start = pd.Timestamp(cycle["Eclipse Date"]).normalize()
            cycle_end = pd.Timestamp(cycle["Next Same-Class"]).normalize()
            span = cycle_end - cycle_start
            if span <= pd.Timedelta(0):
                continue
            boundaries = [cycle_start, cycle_start + span / 3, cycle_start + span * 2 / 3, cycle_end]
            body = cycle["Body"]

            for third_num in range(3):
                seg_start = boundaries[third_num]
                seg_end = boundaries[third_num + 1]
                # Use half-open intervals for thirds 1/2 so a boundary session
                # cannot be counted twice; include the final eclipse in third 3.
                if third_num < 2:
                    segment = price_df[(price_df["Date"] >= seg_start) & (price_df["Date"] < seg_end)].copy()
                else:
                    segment = price_df[(price_df["Date"] >= seg_start) & (price_df["Date"] <= seg_end)].copy()
                if segment.empty:
                    continue

                entry = float(segment.iloc[0]["Open"])
                exit_close = float(segment.iloc[-1]["Close"])
                delta = exit_close - entry
                pct = (delta / entry) * 100.0 if entry else 0.0
                max_profit = float(segment["High"].max() - entry)
                max_drawdown = float(entry - segment["Low"].min())

                third_rows.append({
                    "Cycle Start": cycle_start,
                    "Cycle End": cycle_end,
                    "Body": body,
                    "Third": f"Third {third_num + 1}",
                    "Start Boundary": seg_start,
                    "End Boundary": seg_end,
                    "Trading Days": int(len(segment)),
                    "Entry Open": entry,
                    "Exit Close": exit_close,
                    "Delta ($)": delta,
                    "Return (%)": pct,
                    "Max Profit": max_profit,
                    "Max Drawdown": max_drawdown,
                    "Result": "Win" if delta > 0 else "Loss"
                })

        third_df = pd.DataFrame(third_rows)
        if not third_df.empty:
            st.markdown("### 1/3 Cycle Price Behavior")
            st.caption("Each completed Solar-to-Solar and Lunar-to-Lunar cycle is divided into three equal calendar-time sections. Returns use the first available trading-day OPEN and the last available trading-day CLOSE inside each third; highs/lows show the excursion within that third.")

            behavior_col1, behavior_col2 = st.columns(2)
            with behavior_col1:
                behavior_body = st.multiselect(
                    "Behavior Body", ["Solar", "Lunar"], default=["Solar", "Lunar"],
                    key="eclipse_behavior_body"
                )
            with behavior_col2:
                behavior_third = st.multiselect(
                    "Cycle Third", ["Third 1", "Third 2", "Third 3"],
                    default=["Third 1", "Third 2", "Third 3"], key="eclipse_behavior_third"
                )

            behavior_display = third_df[
                third_df["Body"].isin(behavior_body) & third_df["Third"].isin(behavior_third)
            ].copy()
            behavior_display = behavior_display.sort_values(["Cycle Start", "Body", "Third"]).reset_index(drop=True)

            formatted_behavior = behavior_display.copy()
            for col in ["Cycle Start", "Cycle End", "Start Boundary", "End Boundary"]:
                formatted_behavior[col] = pd.to_datetime(formatted_behavior[col]).dt.strftime("%Y-%m-%d")
            for col in ["Entry Open", "Exit Close", "Delta ($)", "Max Profit", "Max Drawdown"]:
                formatted_behavior[col] = formatted_behavior[col].map(lambda x: f"${x:,.2f}")
            formatted_behavior["Return (%)"] = formatted_behavior["Return (%)"].map(lambda x: f"{x:+.2f}%")

            render_aggrid(formatted_behavior, height=360, key="eclipse_behavior_grid")

            summary = third_df[third_df["Body"].isin(behavior_body) & third_df["Third"].isin(behavior_third)].copy()
            if not summary.empty:
                summary_stats = summary.groupby(["Body", "Third"], as_index=False).agg(
                    Samples=("Return (%)", "count"),
                    Avg_Return=("Return (%)", "mean"),
                    Median_Return=("Return (%)", "median"),
                    Win_Rate=("Result", lambda x: (x == "Win").mean() * 100.0),
                    Avg_Max_Profit=("Max Profit", "mean"),
                    Avg_Max_Drawdown=("Max Drawdown", "mean")
                )
                st.markdown("#### Third-by-Third Summary")
                summary_display = summary_stats.copy()
                summary_display.columns = ["Body", "Third", "Samples", "Avg Return %", "Median Return %", "Win Rate %", "Avg Max Profit $", "Avg Max Drawdown $"]
                for col in ["Avg Return %", "Median Return %", "Win Rate %"]:
                    summary_display[col] = summary_display[col].map(lambda x: f"{x:+.2f}%")
                for col in ["Avg Max Profit $", "Avg Max Drawdown $"]:
                    summary_display[col] = summary_display[col].map(lambda x: f"${x:,.2f}")
                render_aggrid(summary_display, height=260, key="eclipse_summary_grid")

    if study_df.empty:
        st.info("No eclipse events overlap the selected price history.")
        return study_start_date, study_end_date

    # --- Eclipse Event Database ---
    st.markdown("### Eclipse Event Database")
    st.caption("Each row is one eclipse phase. Delta is the OPEN-to-OPEN move from that eclipse to the next eclipse of the same body. Positive rows are WINs and negative rows are LOSSes; the active phase remains OPEN until its next same-body eclipse occurs.")

    db1, db2, db3 = st.columns(3)
    with db1:
        db_body = st.multiselect("Filter Body", ["Solar", "Lunar"], default=["Solar", "Lunar"], key="eclipse_db_body")
    with db2:
        db_status = st.multiselect("Filter Result", ["Win", "Loss", "Open"], default=["Win", "Loss", "Open"], key="eclipse_db_result")
    with db3:
        sort_options = ["Eclipse Date", "Body", "Trading Date", "Eclipse Open", "Next Same-Class", "Delta ($)", "Delta (%)", "Max Profit", "Max Drawdown", "Status"]
        sort_by = st.selectbox("Sort Database By", sort_options, index=0, key="eclipse_db_sort")

    sort_direction = st.radio("Sort Direction", ["Newest / Highest", "Oldest / Lowest"], horizontal=True, key="eclipse_db_direction")

    filtered_df = study_df.copy()
    if db_body:
        filtered_df = filtered_df[filtered_df["Body"].isin(db_body)]
    else:
        filtered_df = filtered_df.iloc[0:0]
    if db_status:
        filtered_df = filtered_df[filtered_df["Status"].isin(db_status)]
    else:
        filtered_df = filtered_df.iloc[0:0]

    ascending = sort_direction == "Oldest / Lowest"
    filtered_df = filtered_df.sort_values(sort_by, ascending=ascending, na_position="last").reset_index(drop=True)

    display_df = filtered_df.copy()
    for col in ["Eclipse Date", "Trading Date", "Next Same-Class", "Next Trading Date"]:
        display_df[col] = pd.to_datetime(display_df[col], errors="coerce").apply(lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else "—")
    for col in ["Eclipse Open", "Next Open", "Delta ($)", "Max Profit", "Max Drawdown"]:
        display_df[col] = display_df[col].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    for col in ["Delta (%)"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")

    def color_eclipse_row(row):
        status = row.get("Status", "Open")
        bg = {"Win": "rgba(34,197,94,.16)", "Loss": "rgba(239,68,68,.16)", "Open": "rgba(148,163,184,.10)"}.get(status, "")
        return [f"background-color: {bg}" if bg else ""] * len(row)

    render_aggrid(display_df, height=420, key="eclipse_database_grid")

    completed = study_df[study_df["Status"].isin(["Win", "Loss"])].copy()
    if not completed.empty:
        wins = (completed["Delta ($)"] > 0).mean() * 100
        avg_delta = completed["Delta ($)"].mean()
        avg_pct = completed["Delta (%)"].mean()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Eclipses Tracked", len(study_df))
        c2.metric("Win Rate", f"{wins:.1f}%")
        c3.metric("Avg Delta", f"{avg_delta:+.2f}")
        c4.metric("Avg Delta %", f"{avg_pct:+.2f}%")

    return study_start_date, study_end_date


def display_eclipse_option_study(ticker, eclipse_df, get_all_contract_info_func, get_single_contract_details_func, get_option_chain_func=None, chart_type="Candlestick"):
    """Use the exact same option lookup/history workflow as Charts & Options, then overlay eclipse events."""
    st.markdown("### Eclipse Options Comparison")
    st.caption("This uses the same Public/Polygon contract lookup and historical-data logic as the main Charts & Options section. Select the exact expiration, strike, and call/put, then compare that contract's actual gains around the eclipse dates.")

    if not ticker:
        st.info("Select a stock ticker first.")
        return

    # Mirror the main Options Lookup: Public supplies expirations, the selected
    # expiration loads the complete chain, and the user can narrow the strike range.
    sorted_expirations, contract_data = get_all_contract_info_func(ticker)
    if not sorted_expirations:
        st.warning(f"Could not find any option expiration dates for {ticker}.")
        return

    exp_date_str = st.selectbox(
        "Select Expiration Date", options=sorted_expirations, key="eclipse_option_exp"
    )

    try:
        selected_chain = get_option_chain_func(ticker, exp_date_str) if get_option_chain_func is not None else None
        chain_rows = []
        if selected_chain is not None:
            # _chain_contract_rows() belongs to main_app(), so normalize the
            # already-fetched Public chain locally for the Eclipse selector.
            def _eclipse_strike(contract):
                if isinstance(contract, dict):
                    details = contract.get("optionDetails", {})
                    strike = contract.get("strikePrice", contract.get("strike"))
                else:
                    details = getattr(contract, "optionDetails", {})
                    strike = getattr(contract, "strikePrice", getattr(contract, "strike", None))
                if isinstance(details, dict):
                    strike = details.get("strikePrice", strike)
                else:
                    strike = getattr(details, "strikePrice", strike)
                try:
                    return float(strike) if strike not in (None, "") else None
                except (TypeError, ValueError):
                    return None

            if isinstance(selected_chain, dict):
                for contract in selected_chain.get("calls", []) or []:
                    strike = _eclipse_strike(contract)
                    if strike is not None:
                        chain_rows.append({"type": "call", "strike": strike})
                for contract in selected_chain.get("puts", []) or []:
                    strike = _eclipse_strike(contract)
                    if strike is not None:
                        chain_rows.append({"type": "put", "strike": strike})
            else:
                for contract in getattr(selected_chain, "calls", []) or []:
                    strike = _eclipse_strike(contract)
                    if strike is not None:
                        chain_rows.append({"type": "call", "strike": strike})
                for contract in getattr(selected_chain, "puts", []) or []:
                    strike = _eclipse_strike(contract)
                    if strike is not None:
                        chain_rows.append({"type": "put", "strike": strike})
        eclipse_chain_df = pd.DataFrame(chain_rows)
    except Exception as chain_error:
        st.warning(f"Could not load option chain for {exp_date_str}: {chain_error}")
        eclipse_chain_df = pd.DataFrame()

    if eclipse_chain_df.empty:
        strikes = contract_data.get(exp_date_str, [])
        eclipse_chain_df = pd.DataFrame([{"strike": s} for s in strikes])

    all_strikes = []
    if not eclipse_chain_df.empty and "strike" in eclipse_chain_df.columns:
        all_strikes = sorted(pd.to_numeric(eclipse_chain_df["strike"], errors="coerce").dropna().unique().tolist())
    if not all_strikes:
        st.warning("No strikes found for this expiration.")
        return

    st.markdown("**Strike Range Filter**")
    range_col1, range_col2 = st.columns(2)
    with range_col1:
        user_min_strike = st.number_input(
            "Min Strike", value=float(all_strikes[0]), step=1.0, key="eclipse_min_strike"
        )
    with range_col2:
        user_max_strike = st.number_input(
            "Max Strike", value=float(all_strikes[-1]), step=1.0, key="eclipse_max_strike"
        )

    strikes = [s for s in all_strikes if user_min_strike <= s <= user_max_strike]
    if not strikes:
        st.warning("No strikes found within your custom range. Resetting to full range.")
        strikes = all_strikes

    strike_price = st.selectbox("Select Strike Price", options=strikes, key="eclipse_option_strike")
    option_type = st.radio("Select Option Type", ["call", "put"], horizontal=True, key="eclipse_option_type")

    if not st.button("Fetch Contract Details", key="eclipse_option_fetch"):
        st.info("Choose an expiration, strike range, strike, and call/put, then click Fetch Contract Details.")
        return

    # EXACT SAME HISTORY FETCH AS THE MAIN OPTIONS SECTION.
    with st.spinner(f"Fetching {option_type.upper()} @ ${strike_price} expiring {exp_date_str}..."):
        details, history_df = get_single_contract_details_func(
            ticker, exp_date_str, strike_price, option_type
        )

    if details:
        st.subheader(f"Details for {details['symbol']} (as of yesterday's close)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Close", f"${details['close']:.2f}" if isinstance(details['close'], (int, float)) else "N/A")
        c2.metric("Open", f"${details['open']:.2f}" if isinstance(details['open'], (int, float)) else "N/A")
        c3.metric("High", f"${details['high']:.2f}" if isinstance(details['high'], (int, float)) else "N/A")
        c4.metric("Low", f"${details['low']:.2f}" if isinstance(details['low'], (int, float)) else "N/A")
        c5.metric("Volume", f"{details['volume']:,}" if isinstance(details['volume'], (int, float)) else "N/A")

    if history_df is None or history_df.empty:
        st.info("No price history found for this contract.")
        return

    history_df = history_df.copy().sort_values("Date").reset_index(drop=True)
    history_df["Date"] = pd.to_datetime(history_df["Date"]).dt.normalize()

    st.subheader("Contract Price History")
    y_min_hist, y_max_hist = history_df["Low"].min(), history_df["High"].max()

    # Same chart construction as Charts & Options.
    if chart_type == "Line":
        option_fig = go.Figure(data=[go.Scatter(
            x=history_df["Date"].tolist(),
            y=history_df["Close"].astype(float).tolist(),
            mode="lines",
            name="Close"
        )])
        option_fig.update_layout(title_text=f"Price History for {details['symbol']}")
    else:
        option_fig = go.Figure(data=[go.Candlestick(
            x=history_df["Date"].tolist(),
            open=history_df["Open"].astype(float).tolist(),
            high=history_df["High"].astype(float).tolist(),
            low=history_df["Low"].astype(float).tolist(),
            close=history_df["Close"].astype(float).tolist()
        )])
        option_fig.update_layout(title_text=f"Price History for {details['symbol']}", xaxis_rangeslider_visible=False)
        option_fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider_visible=False)

    # Only draw eclipse markers that actually fall inside this contract's available history.
    option_indexed = history_df.set_index("Date").sort_index()
    option_rows = []
    for _, event in eclipse_df.iterrows():
        event_date = pd.Timestamp(event["date"]).normalize()
        if event_date < history_df["Date"].min() or event_date > history_df["Date"].max():
            continue

        option_fig.add_vline(
            x=event_date,
            line_color="#f59e0b" if event["kind"] == "Solar" else "#8b5cf6",
            line_dash="dash",
            line_width=1.2
        )
        option_fig.add_annotation(
            x=event_date, y=1.0, yref="paper", yshift=8,
            text=f"{event['kind']} {event['type']}", showarrow=False,
            textangle=-90, font=dict(size=9, color="#ffffff")
        )

        future = option_indexed[option_indexed.index >= event_date]
        if future.empty:
            continue
        entry_date = future.index[0]
        entry_close = float(future.iloc[0]["Close"])
        entry_pos = history_df.index[history_df["Date"] == entry_date]
        if len(entry_pos) == 0:
            continue
        pos = int(entry_pos[0])

        # Add a horizontal price level at the contract price on the eclipse event.
        # This mirrors the price-level behavior used on the lunar/stock chart.
        event_color = "#f59e0b" if event["kind"] == "Solar" else "#8b5cf6"
        option_fig.add_hline(
            y=entry_close,
            line_dash="dash",
            line_color=event_color,
            line_width=1.5,
            annotation_text=f'{event["kind"]}: ${entry_close:.2f}',
            annotation_position="bottom right",
            annotation_font=dict(color=event_color, size=11)
        )

        row = {
            "Eclipse Date": event["date"],
            "Body": event["kind"],
            "Type": event["type"],
            "Option Entry Date": entry_date,
            "Option Entry": entry_close,
        }
        for days in (5, 10, 20):
            target = pos + days
            if target < len(history_df):
                exit_close = float(history_df.iloc[target]["Close"])
                row[f"{days}D Gain"] = (exit_close / entry_close - 1.0) * 100.0
            else:
                row[f"{days}D Gain"] = None
        option_rows.append(row)

    option_fig.update_xaxes(range=[history_df["Date"].min(), history_df["Date"].max()])
    option_fig.update_yaxes(range=[y_min_hist * 0.98, y_max_hist * 1.25] if y_min_hist >= 0 else None)
    st.plotly_chart(option_fig, use_container_width=True)

    option_study = pd.DataFrame(option_rows)
    if option_study.empty:
        st.info("No eclipse events overlap this contract's available historical price data. The contract itself was fetched using the same logic as Charts & Options.")
        return

    st.markdown("### Option Gains by Eclipse")
    option_display = option_study.copy()
    option_display["Eclipse Date"] = option_display["Eclipse Date"].dt.strftime("%Y-%m-%d")
    option_display["Option Entry Date"] = option_display["Option Entry Date"].dt.strftime("%Y-%m-%d")
    for col in ["Option Entry", "5D Gain", "10D Gain", "20D Gain"]:
        if "Gain" in col:
            option_display[col] = option_display[col].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
        else:
            option_display[col] = option_display[col].map(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")

    def color_option_row(row):
        val = row.get("20D Gain", "N/A")
        if isinstance(val, str) and val.startswith("+"):
            return ["background-color: rgba(34,197,94,.16)"] * len(row)
        if isinstance(val, str) and val.startswith("-"):
            return ["background-color: rgba(239,68,68,.16)"] * len(row)
        return [""] * len(row)

    render_aggrid(option_display, height=300, key="eclipse_option_gain_grid")

    valid20 = option_study["20D Gain"].dropna()
    if not valid20.empty:
        a, b, c = st.columns(3)
        a.metric("Avg 5D Option Gain", f"{option_study['5D Gain'].dropna().mean():+.2f}%")
        b.metric("Avg 10D Option Gain", f"{option_study['10D Gain'].dropna().mean():+.2f}%")
        c.metric("Avg 20D Option Gain", f"{valid20.mean():+.2f}%")

def render_aggrid(df, *, height=None, key=None, fit_columns=True):
    """Render an interactive, color-coded AG Grid table."""
    if df is None:
        return
    display = df.copy()
    if not AGGRID_AVAILABLE:
        st.warning("AG Grid is not installed. Add `streamlit-aggrid` to the app dependencies.")
        st.dataframe(display, use_container_width=True, hide_index=True)
        return
    gb = GridOptionsBuilder.from_dataframe(display)
    gb.configure_default_column(
        sortable=True, filter=True, resizable=True, minWidth=95,
        wrapText=False, autoHeight=False,
    )
    gb.configure_grid_options(
        animateRows=False,
        enableRangeSelection=True,
        suppressRowClickSelection=True,
        rowHeight=32,
        headerHeight=38,
    )
    if "Strike" in display.columns:
        gb.configure_column("Strike", pinned="left", width=105, lockPinned=True)
    elif "Expiration Date" in display.columns:
        gb.configure_column("Expiration Date", pinned="left", width=125, lockPinned=True)
    cell_style = JsCode(r"""
    function(params) {
        const c = params.colDef.field || '';
        const v = params.value;
        const style = {};
        if (c === 'Strike' || c === 'Expiration Date') {
            style.fontWeight = '700';
            style.color = '#f1c40f';
            style.backgroundColor = 'rgba(241,196,15,0.06)';
        }
        if (c.includes('Call')) {
            style.color = '#2ecc71';
            style.backgroundColor = 'rgba(46,204,113,0.06)';
        }
        if (c.includes('Put')) {
            style.color = '#e74c3c';
            style.backgroundColor = 'rgba(231,76,60,0.06)';
        }
        if (c.includes('Net') || c.includes('Delta') || c.includes('Return') || c.includes('Gain') || c.includes('P/L')) {
            const n = Number(v);
            if (!Number.isNaN(n)) style.color = n > 0 ? '#2ecc71' : (n < 0 ? '#e74c3c' : '#cbd5e1');
        }
        if (c === 'Status' || c === 'Result') {
            const t = String(v || '').toLowerCase();
            if (t === 'win') { style.color='#2ecc71'; style.fontWeight='700'; style.backgroundColor='rgba(46,204,113,0.12)'; }
            if (t === 'loss') { style.color='#e74c3c'; style.fontWeight='700'; style.backgroundColor='rgba(231,76,60,0.12)'; }
            if (t === 'open') { style.color='#94a3b8'; style.fontWeight='700'; }
        }
        return style;
    }
    """)
    for col in display.columns:
        gb.configure_column(col, cellStyle=cell_style)
    formatter = JsCode(r"""
    function(params) {
        if (params.value === null || params.value === undefined || params.value === '') return '—';
        const c = params.colDef.field || '';
        const n = Number(params.value);
        if (Number.isNaN(n)) return params.value;
        if (c.includes('%') || c.includes('Return') || c.includes('Gain') || c.includes('Rate'))
            return n.toFixed(2) + '%';
        if (c.includes('Strike') || c.includes('Price') || c.includes('Open') || c.includes('Close') ||
            c.includes('Bid') || c.includes('Ask') || c.includes('Last') || c.includes('Support') ||
            c.includes('Resistance') || c.includes('Level') || c.includes('Expected'))
            return '$' + n.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
        if (c.includes('DEX') || c.includes('GEX') || c.includes('Delta') || c.includes('Gamma') ||
            c.includes('Theta') || c.includes('Vega') || c.includes('Volume') || c.includes('OI') ||
            c.includes('Days') || c.includes('Samples'))
            return n.toLocaleString(undefined,{maximumFractionDigits:2});
        return n.toLocaleString(undefined,{maximumFractionDigits:2});
    }
    """)
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            gb.configure_column(col, valueFormatter=formatter)
    options = gb.build()
    kwargs = {
        "gridOptions": options,
        "update_mode": "NO_UPDATE",
        "allow_unsafe_jscode": True,
        "fit_columns_on_grid_load": fit_columns,
        "theme": "streamlit",
    }
    if height is not None:
        kwargs["height"] = height
    if key:
        kwargs["key"] = key
    return AgGrid(display, **kwargs)


def display_eclipse_page(get_price_data_func=None, ticker=None, end_date=None, get_all_contract_info_func=None, get_single_contract_details_func=None, get_option_chain_func=None, chart_type="Candlestick"):
    """Eclipse calendar, countdown, visibility feed, and NASA path map."""
    eclipse_df = get_eclipse_calendar_data()
    future_df = eclipse_df[eclipse_df["days_away"] >= 0].copy()
    st.markdown("<div class='panel-label'>Eclipse intelligence</div>", unsafe_allow_html=True)
    st.caption("Solar and lunar eclipse calendar based on NASA eclipse predictions, with an interactive path-map view for central solar eclipses.")
    if not future_df.empty:
        nxt = future_df.iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Next eclipse", nxt["date"].strftime("%b %d, %Y"))
        c2.metric("Type", f"{nxt['type']} {nxt['kind']}")
        c3.metric("Days away", f"{int(nxt['days_away']):,}")
        c4.metric("Saros", str(int(nxt["saros"])))
    a,b,c = st.columns(3)
    with a: view_mode = st.radio("Feed", ["Upcoming", "All 2026–2030"], horizontal=True)
    with b: selected_year = st.selectbox("Year", ["All"] + sorted(eclipse_df["date"].dt.year.unique().tolist()))
    with c: selected_kind = st.selectbox("Body", ["All", "Solar", "Lunar"])
    display_df = future_df.copy() if view_mode == "Upcoming" else eclipse_df.copy()
    if selected_year != "All": display_df = display_df[display_df["date"].dt.year == int(selected_year)]
    if selected_kind != "All": display_df = display_df[display_df["kind"] == selected_kind]
    table_df = display_df[["date","kind","type","saros","visibility"]].copy()
    table_df["date"] = table_df["date"].dt.strftime("%Y-%m-%d")
    table_df.columns = ["Date","Body","Eclipse","Saros","Primary visibility"]
    render_aggrid(table_df, height=420, key="eclipse_calendar_grid")
    st.markdown("### Eclipse map")
    map_candidates = display_df[display_df["solar_path"]].copy()
    if map_candidates.empty: map_candidates = future_df[future_df["solar_path"]].copy()
    if not map_candidates.empty:
        selected_label = st.selectbox("Select a central solar eclipse", map_candidates["label"].tolist())
        selected = map_candidates[map_candidates["label"] == selected_label].iloc[0]
        eclipse_id = selected["date"].strftime("%Y%m%d")
        nasa_map_url = f"https://eclipse.gsfc.nasa.gov/SEsearch/SEsearchmap.php?Ecl={eclipse_id}"
        st.info(f"{selected['label']} · Visibility: {selected['visibility']}")
        st.markdown(
    f"""
    <a href="{nasa_map_url}" target="_blank">
        <button style="
            padding: 0.6rem 1.2rem;
            border-radius: 6px;
            border: 1px solid #888;
            cursor: pointer;
        ">
            🌎 Open NASA Interactive Eclipse Map
        </button>
    </a>
    """,
    unsafe_allow_html=True
)
        st.info("Lunar eclipses are shown by global visibility region; there is no surface travel path like a solar eclipse's shadow track.")
    study_dates = (None, None)
    if get_price_data_func is not None and ticker and end_date is not None:
        study_dates = display_eclipse_price_study(get_price_data_func, ticker, end_date)

    if study_dates and study_dates[0] is not None and study_dates[1] is not None:
        option_eclipse_df = get_historical_eclipse_study_data(study_dates[0], study_dates[1])
        if get_all_contract_info_func is not None and get_single_contract_details_func is not None:
            display_eclipse_option_study(
                ticker, option_eclipse_df,
                get_all_contract_info_func, get_single_contract_details_func,
                get_option_chain_func, chart_type
            )
        else:
            st.warning("Options lookup functions are unavailable for the Eclipse Options Comparison.")

    st.markdown("### Eclipse research")
    st.caption("Historical eclipse price studies are exploratory and do not establish causation or a trading signal.")

def main_app():

    # --- API Key Configuration ---
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=GOOGLE_API_KEY)
    except (FileNotFoundError, KeyError):
        GOOGLE_API_KEY = ""

    # Public is now the primary market-data source. Keep the existing Polygon
    # key as a fallback so a Public/API configuration problem never destroys
    # the existing Alpha workflow. Public credentials belong in Streamlit Secrets.
    #try:
        #PUBLIC_API_SECRET = st.secrets["PUBLIC_API_SECRET"]
    #except (FileNotFoundError, KeyError):
    #    PUBLIC_API_SECRET = ""

    try:
        POLYGON_API_KEY = st.secrets["POLYGON_API_KEY"]
    except (FileNotFoundError, KeyError):
        # Backward-compatible fallback for the current local build.
        POLYGON_API_KEY = "EQYXN1ceqg4zbMsRpnIybN4AmkgtNwbW0"

    PUBLIC_API_BASE = "https://api.public.com"

    @st.cache_data(ttl=45 * 60, show_spinner=False)
    def get_public_access_token(secret):
        """Exchange the long-lived Public secret for a short-lived access token."""
        if not secret:
            return ""
        response = requests.post(
            f"{PUBLIC_API_BASE}/userapiauthservice/personal/access-tokens",
            json={"validityInMinutes": 45, "secret": secret},
            headers={"Content-Type": "application/json"},
            timeout=20,
        )
        response.raise_for_status()
        return response.json().get("accessToken", "")

    @st.cache_data(ttl=10 * 60, show_spinner=False)
    def get_public_account_id(secret):
        """Resolve the user's Public account ID for market-data endpoints."""
        token = get_public_access_token(secret)
        if not token:
            return ""
        response = requests.get(
            f"{PUBLIC_API_BASE}/userapigateway/trading/account",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=20,
        )
        response.raise_for_status()
        accounts = response.json().get("accounts", [])
        if not accounts:
            return ""
        return accounts[0].get("accountId", "")

    def public_request(method, path, *, json_body=None, params=None):
        """Small authenticated Public API wrapper used by Alpha's data layer."""
        token = get_public_access_token(PUBLIC_API_SECRET)
        if not token:
            raise RuntimeError("PUBLIC_API_SECRET is not configured in Streamlit secrets.")

        response = requests.request(
            method,
            f"{PUBLIC_API_BASE}{path}",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=json_body,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    # Official SDK client. The SDK handles access-token minting/refreshing and
    # gives us typed option-chain objects, while the REST wrapper above remains
    # available as a compatibility fallback.
    public_sdk_client = None
    if PUBLIC_SDK_AVAILABLE and PUBLIC_API_SECRET:
        try:
            public_account_id = get_public_account_id(PUBLIC_API_SECRET)
            if public_account_id:
                public_sdk_client = PublicApiClient(
                    ApiKeyAuthConfig(
                        api_secret_key=PUBLIC_API_SECRET,
                        validity_minutes=45,
                    ),
                    config=PublicApiClientConfiguration(
                        default_account_number=public_account_id
                    ),
                )
        except Exception as sdk_init_error:
            st.warning(f"Public Python SDK could not be initialized; using REST fallback. ({sdk_init_error})")
            public_sdk_client = None

    # --- AI-Generated Background ---
    def set_background():
        """Sets a custom AI-generated background for the app."""
        img_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAAXNSR0IArs4c6QAAAD5JREFUKFNjZGBgEGHAD97/p00MDOpaMGrAqAGjBowaMGoASAPgaMWgBowaMGrAqAGjBowaQGYg0QYAnzMKB+S2S2UAAAAASUVORK5CYII="
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{img_base64}");
            background-size: 30px 30px;
            background-repeat: repeat;
            background-attachment: fixed;
        }}
        .stDataFrame th {{
            background-color: #1a1a1a;
            color: white;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

    # Set the background after the page config
    set_background()

    # --- Helper Functions ---

    def exposure_millions(df, columns=("DEX", "GEX", "Call DEX", "Put DEX", "Net DEX", "Call GEX", "Put GEX", "Net GEX")):
        """Return a display copy with exposure fields expressed in $ millions."""
        result = df.copy()
        for col in columns:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce") / 1_000_000.0
        return result


    def display_event_countdown():
        """Calculates and displays the previous/next moon events and a countdown."""
        now = dt.datetime.utcnow()

        # Find next event
        fm = ephem.next_full_moon(now).datetime()
        nm = ephem.next_new_moon(now).datetime()
        fq = ephem.next_first_quarter_moon(now).datetime()
        lq = ephem.next_last_quarter_moon(now).datetime()
        
        events_after = {fm: "Full Moon", nm: "New Moon", fq: "First Quarter", lq: "Last Quarter"}
        next_event_dt = min(events_after.keys())
        next_event_name = events_after[next_event_dt]

        # Find previous event
        fm_prev = ephem.previous_full_moon(now).datetime()
        nm_prev = ephem.previous_new_moon(now).datetime()
        fq_prev = ephem.previous_first_quarter_moon(now).datetime()
        lq_prev = ephem.previous_last_quarter_moon(now).datetime()
        
        events_before = {fm_prev: "Full Moon", nm_prev: "New Moon", fq_prev: "First Quarter", lq_prev: "Last Quarter"}
        prev_event_dt = max(events_before.keys())
        prev_event_name = events_before[prev_event_dt]
        
        color_map = {'Full Moon': '#dc3545', 'New Moon': '#007bff', 'First Quarter': '#28a745', 'Last Quarter': '#28a745'}
        prev_color = color_map.get(prev_event_name, 'grey')
        next_color = color_map.get(next_event_name, 'grey')

        time_diff = next_event_dt - now
        days = time_diff.days
        hours, remainder = divmod(time_diff.seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        container = st.container()
        with container:
            col1, col2, col3 = st.columns([1.5, 2, 1.5])
            with col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px;">
                    <span style="color: grey; font-size: 0.9em;">PREVIOUS EVENT</span><br>
                    <strong style="color: {prev_color}; font-size: 1.1em;">{prev_event_name}</strong><br>
                    <span style="color: grey; font-size: 0.9em;">{prev_event_dt.strftime('%Y-%m-%d')}</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; border-left: 1px solid #444; border-right: 1px solid #444;">
                    <span style="color: {next_color}; font-size: 0.9em;">NEXT EVENT</span><br>
                    <strong style="color: {next_color}; font-size: 1.5em;">{next_event_name}</strong><br>
                    <span style="color: {next_color}; font-size: 0.9em;">{next_event_dt.strftime('%Y-%m-%d')}</span>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px;">
                    <span style="color: grey; font-size: 0.9em;">COUNTDOWN</span><br>
                    <strong style="color: {next_color}; font-size: 1.5em;">{days}d {hours}h {minutes}m</strong>
                </div>
                """, unsafe_allow_html=True)

    def _public_period_for_range(start_date, end_date):
        """Choose a Public historical-data period that covers the requested range."""
        span_days = (pd.Timestamp(end_date).date() - pd.Timestamp(start_date).date()).days
        if span_days <= 366:
            return "YEAR"
        if span_days <= 5 * 365:
            return "FIVE_YEARS"
        if span_days <= 10 * 365:
            return "TEN_YEARS"
        return "ALL"

    def _public_bars_to_ohlcv(payload, start_date=None, end_date=None):
        """Normalize Public bars v2 into Alpha's canonical OHLCV schema."""
        bars = payload.get("regularMarket", {}).get("bars", [])
        rows = []
        for bar in bars:
            rows.append({
                "Date": pd.to_datetime(bar.get("timestamp"), utc=True, errors="coerce").tz_convert(None).normalize()
                    if bar.get("timestamp") else pd.NaT,
                "Open": bar.get("open"),
                "High": bar.get("high"),
                "Low": bar.get("low"),
                "Close": bar.get("close"),
                "Volume": bar.get("volume", 0),
            })

        df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        if df.empty:
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"])

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = (
            df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
              .sort_values("Date")
              .drop_duplicates("Date", keep="last")
              .reset_index(drop=True)
        )

        if start_date is not None:
            df = df[df["Date"] >= pd.Timestamp(start_date).normalize()]
        if end_date is not None:
            df = df[df["Date"] <= pd.Timestamp(end_date).normalize()]

        df["Adj Close"] = df["Close"].astype(float)
        return df.reset_index(drop=True)

    def get_price_data(ticker, start_date, end_date):
        """Fetch one canonical OHLCV DataFrame for both charting and lunar analysis.

        Public is primary; Polygon remains an automatic fallback. Downstream Alpha
        code receives exactly the same normalized columns either way.
        """
        ticker = str(ticker).upper().strip()

        if PUBLIC_API_SECRET:
            try:
                period = _public_period_for_range(start_date, end_date)
                payload = public_request(
                    "GET",
                    f"/userapigateway/historicdata/EQUITY/{ticker}/{period}",
                    params={"tradingSessionToggle": "REGULAR_HOURS"},
                )
                df = _public_bars_to_ohlcv(payload, start_date, end_date)
                if not df.empty:
                    return df
            except Exception as public_error:
                st.warning(f"Public market data unavailable for {ticker}; using Polygon fallback. ({public_error})")

        # Existing Polygon implementation retained as a safety net.
        try:
            client = RESTClient(POLYGON_API_KEY)
            aggs = client.get_aggs(
                ticker=ticker, multiplier=1, timespan="day", from_=start_date, to=end_date,
                adjusted=True, sort="asc", limit=50000,
            )
            df = pd.DataFrame(aggs)
            if df.empty:
                return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
            df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
            df["Date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None).dt.normalize()
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = (df[["Date","Open","High","Low","Close","Volume"]]
                    .dropna(subset=["Date","Open","High","Low","Close"])
                    .sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True))
            df["Adj Close"] = df["Close"].astype(float)
            return df
        except Exception as e:
            st.error(f"Failed to retrieve price data for {ticker}: {e}")
            return pd.DataFrame()

    def add_moon_phases_to_fig(fig, visible_events):
        """Draws vertical lines on the chart for each moon event."""
        for date, event_type, color in visible_events:
            dash_style = 'dot' if event_type in ['Full Moon', 'New Moon'] else 'dash'
            fig.add_shape(type='line', x0=date, x1=date, y0=0, y1=1, yref='paper', line=dict(color=color, dash=dash_style, width=1))
        return fig

    @st.cache_data(ttl=600, show_spinner=False)
    def get_public_option_expirations(ticker):
        """Return Public option expirations using the official SDK first."""
        ticker = str(ticker).upper().strip()
        if public_sdk_client is not None:
            response = public_sdk_client.get_option_expirations(
                OptionExpirationsRequest(
                    instrument=OrderInstrument(symbol=ticker, type=InstrumentType.EQUITY)
                )
            )
            return sorted(list(response.expirations))

        account_id = get_public_account_id(PUBLIC_API_SECRET)
        if not account_id:
            return []
        payload = public_request(
            "POST",
            f"/userapigateway/marketdata/{account_id}/option-expirations",
            json_body={"instrument": {"symbol": ticker, "type": "EQUITY"}},
        )
        return sorted(payload.get("expirations", []))

    @st.cache_data(ttl=120, show_spinner=False)
    def get_public_option_chain(ticker, expiration):
        """Return a normalized dict-shaped option chain from Public SDK/REST."""
        ticker = str(ticker).upper().strip()
        expiration = str(expiration)

        if public_sdk_client is not None:
            response = public_sdk_client.get_option_chain(
                OptionChainRequest(
                    instrument=OrderInstrument(symbol=ticker, type=InstrumentType.EQUITY),
                    expiration_date=expiration,
                )
            )

            def sdk_contract_to_dict(contract):
                details = getattr(contract, "option_details", None)
                greeks = getattr(details, "greeks", None) if details else None
                instrument = getattr(contract, "instrument", None)
                return {
                    "instrument": {"symbol": getattr(instrument, "symbol", None)},
                    "last": getattr(contract, "last", None),
                    "bid": getattr(contract, "bid", None),
                    "ask": getattr(contract, "ask", None),
                    "volume": getattr(contract, "volume", 0),
                    "openInterest": getattr(contract, "open_interest", getattr(contract, "openInterest", 0)),
                    "optionDetails": {
                        "strikePrice": getattr(details, "strike_price", None) if details else None,
                        "midPrice": getattr(details, "mid_price", None) if details else None,
                        "greeks": {
                            "delta": getattr(greeks, "delta", None) if greeks else None,
                            "gamma": getattr(greeks, "gamma", None) if greeks else None,
                            "theta": getattr(greeks, "theta", None) if greeks else None,
                            "vega": getattr(greeks, "vega", None) if greeks else None,
                            "rho": getattr(greeks, "rho", None) if greeks else None,
                            "impliedVolatility": getattr(greeks, "implied_volatility", None) if greeks else None,
                        },
                    },
                }

            return {
                "baseSymbol": getattr(response, "base_symbol", ticker),
                "calls": [sdk_contract_to_dict(c) for c in response.calls],
                "puts": [sdk_contract_to_dict(p) for p in response.puts],
            }

        account_id = get_public_account_id(PUBLIC_API_SECRET)
        if not account_id:
            return {"baseSymbol": ticker, "calls": [], "puts": []}
        return public_request(
            "POST",
            f"/userapigateway/marketdata/{account_id}/option-chain",
            json_body={
                "instrument": {"symbol": ticker, "type": "EQUITY"},
                "expirationDate": expiration,
            },
        )

    def _chain_contract_rows(chain):
        """Normalize Public SDK/REST option chains across field-shape variants."""
        rows = []

        def dig(obj, *paths):
            for path in paths:
                cur = obj
                ok = True
                for key in path:
                    if isinstance(cur, dict):
                        cur = cur.get(key)
                    else:
                        cur = getattr(cur, key, None)
                    if cur is None:
                        ok = False
                        break
                if ok and cur not in (None, ""):
                    return cur
            return None

        def number(value):
            if isinstance(value, dict):
                value = value.get("value", value.get("raw", value.get("amount")))
            try:
                return float(value) if value not in (None, "") else None
            except (TypeError, ValueError):
                return None

        def integer(value):
            try:
                return int(float(value or 0))
            except (TypeError, ValueError):
                return 0

        for option_type, contracts in (("call", chain.get("calls", [])), ("put", chain.get("puts", []))):
            for item in contracts:
                details = dig(item, ("optionDetails",), ("option_details",)) or {}
                greeks = (
                    dig(details, ("greeks",), ("Greeks",))
                    or dig(item, ("greeks",), ("Greeks",))
                    or {}
                )

                strike = dig(
                    details, ("strikePrice",), ("strike_price",), ("strike",)
                )
                if strike is None:
                    strike = dig(
                        item, ("strikePrice",), ("strike_price",), ("strike",)
                    )

                oi = dig(
                    item,
                    ("openInterest",), ("open_interest",),
                    ("openInterestQuantity",),
                )
                if oi is None:
                    oi = dig(details, ("openInterest",), ("open_interest",))

                rows.append({
                    "type": option_type,
                    "symbol": dig(item, ("instrument", "symbol"), ("symbol",)),
                    "strike": number(strike),
                    "last": number(dig(item, ("last",), ("lastPrice",), ("last_price",), ("close",))),
                    "bid": number(dig(item, ("bid",), ("bidPrice",), ("bid_price",))),
                    "ask": number(dig(item, ("ask",), ("askPrice",), ("ask_price",))),
                    "volume": integer(dig(item, ("volume",), ("tradingVolume",), ("trading_volume",))),
                    "openInterest": integer(oi),
                    "midPrice": number(dig(details, ("midPrice",), ("mid_price",))),
                    "delta": number(dig(greeks, ("delta",), ("Delta",))),
                    "gamma": number(dig(greeks, ("gamma",), ("Gamma",))),
                    "theta": number(dig(greeks, ("theta",), ("Theta",))),
                    "vega": number(dig(greeks, ("vega",), ("Vega",))),
                    "rho": number(dig(greeks, ("rho",), ("Rho",))),
                    "iv": number(dig(
                        greeks,
                        ("impliedVolatility",), ("implied_volatility",), ("iv",)
                    )),
                })
        return rows


    @st.cache_data(ttl=600, show_spinner=False)
    def get_all_contract_info_free(ticker):
        """Return option expirations from Public.com only.

        The existing UI expects (expirations, strikes-by-expiration), so we preserve
        that shape. Public supplies the expirations first and the complete chain is
        loaded lazily after the user selects an expiration. Polygon is intentionally
        NOT used for option discovery.
        """
        if not PUBLIC_API_SECRET:
            st.error("Public API secret is not configured. Options lookup requires Public.com.")
            return [], {}

        try:
            return get_public_option_expirations(ticker), {}
        except Exception as public_error:
            st.error(f"Public option expiration lookup failed: {public_error}")
            return [], {}

    def get_single_contract_details_free(ticker, expiration, strike, type, history_start=None, history_end=None):
        """Fetch an option contract and its option history from Public.com only.

        Polygon remains available for stock/price-data fallback elsewhere in the app,
        but it is deliberately not used for option lookup or option contract history.
        """
        ticker = str(ticker).upper().strip()
        option_type = str(type).lower()

        if not PUBLIC_API_SECRET:
            st.error("Public API secret is not configured. Options lookup requires Public.com.")
            return None, None

        try:
            chain = get_public_option_chain(ticker, expiration)
            rows = _chain_contract_rows(chain)
            matches = [r for r in rows if r["type"] == option_type and r["strike"] is not None and abs(r["strike"] - float(strike)) < 1e-9]
            if not matches:
                st.warning("Contract not found in Public option chain.")
                return None, None

            selected = matches[0]
            option_ticker = selected["symbol"]
            today = dt.date.today()
            history_start = history_start or (today - dt.timedelta(days=365))
            # Public historical option bars are requested through today.  The
            # endpoint may not have a completed bar yet during the live session,
            # so the current chain snapshot below remains the source of truth for
            # Last/Bid/Ask/Volume/Open Interest/Greeks.
            history_end = history_end or today

            period = _public_period_for_range(history_start, history_end)
            payload = public_request(
                "GET",
                f"/userapigateway/historicdata/OPTION/{option_ticker}/{period}",
            )
            history_df = _public_bars_to_ohlcv(payload, history_start, history_end)

            last_day = history_df.iloc[-1] if not history_df.empty else None
            details = {
                "symbol": option_ticker,
                "open": float(last_day["Open"]) if last_day is not None else selected.get("last"),
                "close": float(last_day["Close"]) if last_day is not None else selected.get("last"),
                "high": float(last_day["High"]) if last_day is not None else None,
                "low": float(last_day["Low"]) if last_day is not None else None,
                "volume": selected.get("volume", int(last_day["Volume"]) if last_day is not None else 0),
                "open_interest": selected.get("openInterest", 0),
                "last": selected.get("last"),
                "bid": selected.get("bid"),
                "ask": selected.get("ask"),
                "delta": selected.get("delta"),
                "gamma": selected.get("gamma"),
                "theta": selected.get("theta"),
                "vega": selected.get("vega"),
                "rho": selected.get("rho"),
                "iv": selected.get("iv"),
            }
            return details, history_df
        except Exception as public_error:
            st.error(f"Public option data lookup failed: {public_error}")
            return None, None


    def calculate_option_exposure(df_chain, spot_price, contract_multiplier=100):
        """Calculate modeled dealer-style DEX/GEX exposure by strike.

        Public's option chain already provides delta/gamma, so no per-contract
        Greeks API calls are required. This is a modeled exposure convention,
        not a literal observation of dealer positions.
        """
        required = ["strike", "type", "openInterest", "delta", "gamma"]
        if df_chain is None or df_chain.empty:
            return pd.DataFrame(), None

        df = df_chain.copy()
        # Accept both Public's camelCase and common snake_case field names.
        aliases = {
            "open_interest": "openInterest",
            "option_type": "type",
            "strike_price": "strike",
        }
        for src, dst in aliases.items():
            if dst not in df.columns and src in df.columns:
                df[dst] = df[src]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return pd.DataFrame(), None

        for col in ["strike", "openInterest", "delta", "gamma"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df["type"] = df["type"].astype(str).str.lower().str.strip()
        df = df[df["strike"] > 0].copy()
        if df.empty:
            return pd.DataFrame(), None

        spot = float(spot_price)
        df["DEX"] = df["delta"] * df["openInterest"] * contract_multiplier * spot
        df["GEX"] = df["gamma"] * df["openInterest"] * contract_multiplier * (spot ** 2) * 0.01
        # Modeled put gamma exposure is shown as negative, per the app's
        # requested dealer-exposure convention.
        df.loc[df["type"].isin(["put", "p"]), "GEX"] *= -1

        df["Call OI"] = df["openInterest"].where(df["type"].isin(["call", "c"]), 0)
        df["Put OI"] = df["openInterest"].where(df["type"].isin(["put", "p"]), 0)
        df["Call DEX"] = df["DEX"].where(df["type"].isin(["call", "c"]), 0)
        df["Put DEX"] = df["DEX"].where(df["type"].isin(["put", "p"]), 0)
        df["Call GEX"] = df["GEX"].where(df["type"].isin(["call", "c"]), 0)
        df["Put GEX"] = df["GEX"].where(df["type"].isin(["put", "p"]), 0)

        exposure = df.groupby("strike", as_index=False).agg({
            "Call OI": "sum", "Put OI": "sum",
            "Call DEX": "sum", "Put DEX": "sum",
            "Call GEX": "sum", "Put GEX": "sum",
        }).rename(columns={"strike": "Strike"})
        exposure["Total OI"] = exposure["Call OI"] + exposure["Put OI"]
        exposure["Net DEX"] = exposure["Call DEX"] + exposure["Put DEX"]
        exposure["Net GEX"] = exposure["Call GEX"] + exposure["Put GEX"]
        exposure = exposure.sort_values("Strike").reset_index(drop=True)

        gamma_flip = None
        if len(exposure) >= 2:
            x = exposure["Strike"].to_numpy(dtype=float)
            y = exposure["Net GEX"].to_numpy(dtype=float)
            for i in range(len(y) - 1):
                if y[i] == 0:
                    gamma_flip = x[i]
                    break
                if y[i] * y[i + 1] < 0:
                    # Linear interpolation between the two strikes where net GEX
                    # changes sign.
                    gamma_flip = x[i] + (0 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
                    break
            if gamma_flip is None and y[-1] == 0:
                gamma_flip = x[-1]

        return exposure, gamma_flip

    @st.cache_data(ttl=300, show_spinner=False)
    def get_expiration_term_structure(ticker, expirations, spot_price):
        """Build current-session positioning for the FULL Public chain at every expiration.

        This intentionally does not use the selected strike-range filter from the
        single-expiration Options Lookup. Each expiration is fetched and calculated
        independently from its complete Public option chain.

        Historical 1D GEX/DEX changes are intentionally omitted until corresponding
        prior-day chain/snapshot data is available. No session cache is treated as a
        legitimate 1D change.
        """
        today = dt.date.today()
        rows = []
        chain_cache = {}
        for expiration in expirations:
            try:
                exp_date = pd.Timestamp(expiration).date()
            except Exception:
                continue
            if exp_date < today:
                continue

            try:
                chain = get_public_option_chain(ticker, str(expiration))
                chain_rows = _chain_contract_rows(chain)
                full_df = pd.DataFrame(chain_rows)
                chain_cache[str(expiration)] = full_df
                if full_df.empty:
                    continue

                exposure_df, _ = calculate_option_exposure(full_df, spot_price)
                if exposure_df.empty:
                    continue

                gex = float(exposure_df["Net GEX"].sum())
                dex = float(exposure_df["Net DEX"].sum())
                oi = float(exposure_df["Total OI"].sum())

                calls = full_df[full_df["type"].astype(str).str.lower().isin(["call", "c"])].copy()
                puts = full_df[full_df["type"].astype(str).str.lower().isin(["put", "p"])].copy()
                calls["gamma"] = pd.to_numeric(calls.get("gamma", 0), errors="coerce").fillna(0.0)
                puts["gamma"] = pd.to_numeric(puts.get("gamma", 0), errors="coerce").fillna(0.0)
                calls["openInterest"] = pd.to_numeric(calls.get("openInterest", 0), errors="coerce").fillna(0.0)
                puts["openInterest"] = pd.to_numeric(puts.get("openInterest", 0), errors="coerce").fillna(0.0)
                calls["volume"] = pd.to_numeric(calls.get("volume", 0), errors="coerce").fillna(0.0)
                puts["volume"] = pd.to_numeric(puts.get("volume", 0), errors="coerce").fillna(0.0)

                # Side-specific exposure identifies the strongest modeled resistance/support.
                calls["call_gex"] = calls["gamma"] * calls["openInterest"] * 100 * float(spot_price) ** 2 * 0.01
                puts["put_gex"] = -puts["gamma"] * puts["openInterest"] * 100 * float(spot_price) ** 2 * 0.01
                call_level = float(calls.loc[calls["call_gex"].idxmax(), "strike"]) if not calls.empty and calls["call_gex"].max() > 0 else None
                put_level = float(puts.loc[puts["put_gex"].idxmin(), "strike"]) if not puts.empty and puts["put_gex"].min() < 0 else None

                # Highest combined volume concentration.
                vol_by_strike = full_df.groupby("strike", as_index=False)["volume"].sum()
                vol_by_strike["volume"] = pd.to_numeric(vol_by_strike["volume"], errors="coerce").fillna(0.0)
                high_vol_level = float(vol_by_strike.loc[vol_by_strike["volume"].idxmax(), "strike"]) if not vol_by_strike.empty and vol_by_strike["volume"].max() > 0 else None

                # Expected move: ATM call + put mid/last through this expiration.
                work = full_df.copy()
                work["strike"] = pd.to_numeric(work["strike"], errors="coerce")
                work["bid"] = pd.to_numeric(work["bid"], errors="coerce")
                work["ask"] = pd.to_numeric(work["ask"], errors="coerce")
                work["last"] = pd.to_numeric(work["last"], errors="coerce")
                work["mid"] = ((work["bid"] + work["ask"]) / 2).where(
                    work["bid"].notna() & work["ask"].notna() & (work["ask"] > 0), work["last"]
                )
                work["distance"] = (work["strike"] - float(spot_price)).abs()
                atm_strike = work.sort_values("distance")["strike"].iloc[0] if not work.empty else None
                expected_move = None
                expected_low = None
                expected_high = None
                if atm_strike is not None:
                    atm = work[work["strike"] == atm_strike]
                    call_mid = pd.to_numeric(atm.loc[atm["type"].isin(["call", "c"]), "mid"], errors="coerce").dropna()
                    put_mid = pd.to_numeric(atm.loc[atm["type"].isin(["put", "p"]), "mid"], errors="coerce").dropna()
                    if not call_mid.empty and not put_mid.empty:
                        expected_move = float(call_mid.iloc[0] + put_mid.iloc[0])
                        expected_low = max(0.0, float(spot_price) - expected_move)
                        expected_high = float(spot_price) + expected_move

                dte = max(0, (exp_date - today).days)
                rows.append({
                    "Expiration Date": exp_date,
                    "DTE": dte,
                    "GEX": gex,
                    "DEX": dex,
                    "OI": oi,
                    "Call Resistance": call_level,
                    "Put Support": put_level,
                    "High Vol. Level": high_vol_level,
                    "Expiry Expected Move": expected_move,
                    "Expected Low": expected_low,
                    "Expected High": expected_high,
                })
            except Exception:
                # One bad expiration should not prevent the rest of the term structure.
                continue

        result = pd.DataFrame(rows).sort_values(["Expiration Date", "DTE"]).reset_index(drop=True) if rows else pd.DataFrame()
        if not result.empty:
            total_gex = float(result["GEX"].sum())
            total_dex = float(result["DEX"].sum())
            total_oi = float(result["OI"].sum())
            result["GEX Normalized"] = result["GEX"].apply(lambda x: (x / total_gex * 100) if total_gex else 0.0)
            result["DEX Normalized"] = result["DEX"].apply(lambda x: (x / total_dex * 100) if total_dex else 0.0)
            result["OI Normalized"] = result["OI"].apply(lambda x: (x / total_oi * 100) if total_oi else 0.0)
            # Explicit placeholders: legitimate 1D changes require historical/snapshot data.
            result["GEX Change 1D"] = None
            result["DEX Change 1D"] = None
        return result, chain_cache


    def load_data(ticker, start_date, end_date):
        with st.spinner("Fetching stock data..."):
            fresh_data = get_price_data(ticker, start_date, end_date)

        # Always replace the chart dataset as one atomic update.
        st.session_state.stock_data = fresh_data.copy()
        st.session_state.loaded_ticker = str(ticker).upper().strip()
        st.session_state.loaded_start_date = start_date
        st.session_state.loaded_end_date = end_date

        if 'stock_data' in st.session_state and not st.session_state.stock_data.empty:
            start_date_obj = start_date if isinstance(start_date, dt.date) else start_date.date()
            end_date_obj = end_date if isinstance(end_date, dt.date) else end_date.date()
            
            all_events = []
            current_date = start_date_obj - dt.timedelta(days=10)
            
            while current_date <= end_date_obj:
                events_after_current = {}
                try:
                    fm = ephem.next_full_moon(current_date).datetime().date()
                    nm = ephem.next_new_moon(current_date).datetime().date()
                    fq = ephem.next_first_quarter_moon(current_date).datetime().date()
                    lq = ephem.next_last_quarter_moon(current_date).datetime().date()
                    
                    events_after_current = {
                        fm: ('Full Moon', 'red'),
                        nm: ('New Moon', 'blue'),
                        fq: ('Quarter Moon', 'green'),
                        lq: ('Quarter Moon', 'green')
                    }
                except Exception:
                    break
                
                if not events_after_current:
                    break
                
                next_event_date = min(events_after_current.keys())
                
                if next_event_date > end_date_obj:
                    break
                
                if not any(e[0] == next_event_date for e in all_events):
                        event_type, event_color = events_after_current[next_event_date]
                        all_events.append((next_event_date, event_type, event_color))
                
                current_date = next_event_date + dt.timedelta(days=1)
            
            st.session_state.all_moon_events = sorted(all_events)

        st.session_state.data_loaded = True

    def analyze_images_with_gemini(images, prompt):
        model = genai.GenerativeModel('gemini-1.5-flash')
        try:
            content = [prompt] + images
            response = model.generate_content(content)
            return response.text
        except Exception as e:
            st.error(f"An error occurred with the Gemini API: {e}")
            return f"Error during API call: {e}"

    def add_lunar_analysis_annotations(fig, df, all_moon_events, open_col, high_col, low_col, close_col, y_max_total):
        analysis_results = []
        if not all_moon_events or df.empty or open_col not in df.columns:
            return fig, analysis_results

        # Normalize the index so moon-event dates and trading dates use
        # the same pandas Timestamp type.
        df_indexed = df.copy()
        df_indexed['Date'] = pd.to_datetime(df_indexed['Date']).dt.normalize()
        df_indexed = df_indexed.set_index('Date').sort_index()

        def find_next_trading_day(target_date, price_df):
            current_date = pd.Timestamp(target_date).normalize()
            last_date = pd.Timestamp(price_df.index.max()).normalize()

            while current_date <= last_date:
                if current_date in price_df.index:
                    return price_df.loc[current_date]
                current_date += pd.Timedelta(days=1)
            return None

        cycle_points = []
        for event_date, event_type, event_color in all_moon_events:
            trading_day_row = find_next_trading_day(event_date, df_indexed)
            if trading_day_row is not None:
                if not any(p['trading_row'].name == trading_day_row.name for p in cycle_points):
                    cycle_points.append({
                        "trading_row": trading_day_row,
                        "event_type": event_type,
                        "event_color": event_color,
                        "moon_date": event_date
                    })
        
        if len(cycle_points) < 2:
            return fig, []

        # Process completed cycles
        for i in range(len(cycle_points) - 1):
            start_day_data = cycle_points[i]['trading_row']
            end_day_data = cycle_points[i+1]['trading_row']
            
            entry_price = start_day_data[open_col]
            final_close_price = end_day_data[close_col]
            
            cycle_df = df_indexed.loc[start_day_data.name:end_day_data.name]
            
            max_high = cycle_df[high_col].max()
            min_low = cycle_df[low_col].min()

            end_to_end_pl = final_close_price - entry_price
            pct_change = (end_to_end_pl / entry_price) * 100 if entry_price != 0 else 0
            
            mfe = max_high - entry_price
            mae = entry_price - min_low

            analysis_results.append({
                "start_date": start_day_data.name, "end_date": end_day_data.name,
                "entry_price": entry_price,
                "end_price": final_close_price,
                "pl_delta": end_to_end_pl, 
                "pl_pct": pct_change,
                "max_profit": mfe,
                "max_drawdown": mae,
                "status": "Win" if end_to_end_pl > 0 else "Loss",
                "start_event_type": cycle_points[i]['event_type'], 
                "start_event_color": cycle_points[i]['event_color']
            })

            # RESTORED: Add the P/L annotation back to the chart for completed cycles
            text_color = '#28a745' if end_to_end_pl >= 0 else '#dc3545'
            text = f"{'+' if end_to_end_pl >= 0 else ''}${end_to_end_pl:.2f}<br>({pct_change:+.2f}%)"
            mid_date = start_day_data.name + (end_day_data.name - start_day_data.name) / 2
            
            # IMPROVED: Position annotation just above the cycle's high point
            y_pos = max_high * 1.05 

            fig.add_annotation(
                x=mid_date, y=y_pos, text=text, showarrow=False,
                font=dict(color=text_color, size=12), align="center"
            )
            
        # Process the in-progress cycle
        last_cycle_point = cycle_points[-1]
        start_day_data = last_cycle_point['trading_row']
        current_day_data = df_indexed.iloc[-1]
        
        if start_day_data.name < current_day_data.name:
            entry_price = start_day_data[open_col]
            
            cycle_df = df_indexed.loc[start_day_data.name:current_day_data.name]
            max_high = cycle_df[high_col].max()
            min_low = cycle_df[low_col].min()

            end_to_end_pl = current_day_data[close_col] - entry_price
            pct_change = (end_to_end_pl / entry_price) * 100 if entry_price != 0 else 0
            
            mfe = max_high - entry_price
            mae = entry_price - min_low
            
            projected_end_date = None
            try:
                last_moon_date = last_cycle_point["moon_date"]
                projected_end_date = min([d for d in [
                    ephem.next_full_moon(last_moon_date).datetime().date(),
                    ephem.next_new_moon(last_moon_date).datetime().date(),
                    ephem.next_first_quarter_moon(last_moon_date).datetime().date(),
                    ephem.next_last_quarter_moon(last_moon_date).datetime().date()
                ] if d > last_moon_date])
            except Exception: pass
            
            analysis_results.append({
                "start_date": start_day_data.name, "end_date": projected_end_date or current_day_data.name,
                "entry_price": entry_price,
                "end_price": current_day_data[close_col],
                "pl_delta": end_to_end_pl, 
                "pl_pct": pct_change,
                "max_profit": mfe,
                "max_drawdown": mae,
                "status": "In Progress",
                "start_event_type": last_cycle_point['event_type'], 
                "start_event_color": last_cycle_point['event_color']
            })

        return fig, analysis_results

    def add_price_level_lines(fig, analysis_results, num_levels):
        if num_levels == 0 or not analysis_results: return fig
        
        levels_to_plot = [r for r in analysis_results if r.get('entry_price')]
        if not levels_to_plot: return fig

        levels_to_plot = levels_to_plot[-num_levels:]
        
        color_map = {
            'red': 'rgba(255, 80, 80, 0.7)',
            'blue': 'rgba(80, 80, 255, 0.7)',
            'green': 'rgba(0, 200, 0, 0.6)'
        }

        for cycle in levels_to_plot:
            y_level = cycle.get('entry_price')
            event_type = cycle.get('start_event_type')
            event_color_name = cycle.get('start_event_color')
            status = cycle.get('status') 

            if y_level is None or event_type is None or event_color_name is None: continue
            
            color = color_map.get(event_color_name, 'grey')
            
            annotation_text = f"{event_type}: ${y_level:.2f}"
            if status == 'In Progress':
                annotation_text += " (Active)"

            fig.add_hline(
                y=y_level, line_dash="dash", line_color=color, line_width=1.5,
                annotation_text=annotation_text, annotation_position="bottom right",
                annotation_font=dict(color=color, size=12)
            )
        return fig

    def display_summary_and_active_cycle_stats(analysis_results):
        if not analysis_results: return

        completed_cycles_df = pd.DataFrame([r for r in analysis_results if r['status'] != 'In Progress'])
        in_progress_cycle = next((r for r in analysis_results if r['status'] == 'In Progress'), None)

        st.subheader("Historical Cycle Analysis")
        if completed_cycles_df.empty:
            st.info("Not enough historical data to calculate averages.")
            return
            
        win_rate = (completed_cycles_df['pl_delta'] > 0).mean() * 100
        winners = completed_cycles_df[completed_cycles_df['pl_delta'] > 0]['pl_delta']
        losers = completed_cycles_df[completed_cycles_df['pl_delta'] <= 0]['pl_delta']
        avg_win = winners.mean() if not winners.empty else 0
        avg_loss = losers.mean() if not losers.empty else 0

        def color_style(value, is_percent=False, is_dollar=True):
            color = "#28a745" if value >= 0 else "#dc3545"
            sign = "+" if value > 0 else ""
            if is_dollar:
                return f'<span style="color:{color}; font-size: 1.25em;">{sign}${abs(value):,.2f}</span>'
            elif is_percent:
                return f'<span style="color:{color}; font-size: 1.25em;">{sign}{value:,.2f}%</span>'
            else:
                win_color = "#28a745" if value >= 50 else "#dc3545"
                return f'<span style="color:{win_color}; font-size: 1.25em;">{value:,.1f}%</span>'

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Win Rate** <br> {color_style(win_rate, is_dollar=False, is_percent=False)}", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**Average Win** <br> {color_style(avg_win, is_dollar=True)}", unsafe_allow_html=True)
        with col3:
            st.markdown(f"**Average Loss** <br> {color_style(avg_loss, is_dollar=True)}", unsafe_allow_html=True)

        if not in_progress_cycle: return

        st.subheader("Active Cycle Analysis")
        current_pl = in_progress_cycle['pl_delta']
        avg_pl = completed_cycles_df['pl_delta'].mean()
        room_to_avg = avg_pl - current_pl
        
        col4, col5 = st.columns(2)
        with col4:
            st.markdown(f"**Current P/L** <br> {color_style(current_pl, is_dollar=True)}", unsafe_allow_html=True)
        with col5:
            room_label = "Room to Average" if room_to_avg >= 0 else "Beyond Average"
            st.markdown(f"**{room_label}** <br> {color_style(room_to_avg, is_dollar=True)}", unsafe_allow_html=True)

    def display_analysis_table(results_list, grid_key="lunar_cycle_analysis_grid"):
        if not results_list: return
        st.subheader("Lunar Cycle Analysis Results")
        df = pd.DataFrame(results_list)
        
        cols_to_display = ['start_date', 'end_date', 'entry_price', 'end_price', 'pl_delta', 'pl_pct', 'max_profit', 'max_drawdown', 'status']
        display_df = df[cols_to_display].copy()

        # Keep both date columns as pandas datetime64 values so Streamlit/Arrow
        # does not receive a mixture of datetime.date and pandas Timestamp.
        display_df["start_date"] = pd.to_datetime(display_df["start_date"], errors="coerce")
        display_df["end_date"] = pd.to_datetime(display_df["end_date"], errors="coerce")

        display_df.rename(columns={
            'start_date': 'Start Date', 'end_date': 'End Date', 'entry_price': 'Entry Price',
            'end_price': 'End Price', 'pl_delta': 'P/L ($)', 'pl_pct': 'Delta %',
            'max_profit': 'Max Profit', 'max_drawdown': 'Max Drawdown', 'status': 'Status'
        }, inplace=True)

        # Display dates explicitly as US month/day/year. The source values remain
        # true datetimes for all calculations; only the table presentation changes.
        display_df["Start Date"] = display_df["Start Date"].dt.strftime("%m/%d/%Y")
        display_df["End Date"] = display_df["End Date"].dt.strftime("%m/%d/%Y")

        def style_rows(row):
            color = 'background-color: rgba(0, 128, 0, 0.2)' if row['P/L ($)'] > 0 else 'background-color: rgba(128, 0, 0, 0.2)'
            return [color if col in ['P/L ($)', 'Delta %', 'Status'] else '' for col in display_df.columns]

        styled_df = display_df.style.apply(style_rows, axis=1).format({
            "Start Date": '{:%m/%d/%Y}', "End Date": '{:%m/%d/%Y}',
            "Entry Price": "${:,.2f}", "End Price": "${:,.2f}",
            "P/L ($)": "${:+.2f}", "Delta %": "{:+.2f}%",
            "Max Profit": "${:,.2f}", "Max Drawdown": "${:,.2f}"
        })
        
        render_aggrid(display_df, height=400, key=grid_key)


    # --- TradingView Top Ticker Tape ---
    def display_top_tradingview_ticker_tape():
        """Render the TradingView web-component ticker tape at the top of the app."""
        symbols = (
            "FOREXCOM:SPXUSD,FOREXCOM:NSXUSD,FOREXCOM:DJI,FX:EURUSD,"
            "BITSTAMP:BTCUSD,BITSTAMP:ETHUSD,CMCMARKETS:GOLD,"
            "NASDAQ:AAPL,NASDAQ:NVDA,NASDAQ:TSLA,NASDAQ:MSFT,NASDAQ:AMZN,"
            "NASDAQ:GOOGL,NASDAQ:META,NASDAQ:AVGO,NASDAQ:AMD,NASDAQ:NFLX,"
            "NASDAQ:PLTR,NASDAQ:MU,NASDAQ:ARM,NASDAQ:QCOM,NASDAQ:INTC,"
            "NASDAQ:ADBE,NASDAQ:AMAT,NASDAQ:CRWD,NASDAQ:CSCO,NASDAQ:SMCI,"
            "NASDAQ:MSTR,NASDAQ:MRVL,NASDAQ:TXN,NASDAQ:PEP,NASDAQ:COST,"
            "NYSE:JPM,NYSE:BAC,NYSE:GS,NYSE:V,NYSE:MA,NYSE:WMT,NYSE:COST,"
            "NYSE:XOM,NYSE:CVX,NYSE:LLY,NYSE:UNH,NYSE:CAT,NYSE:GE,NYSE:HD,"
            "NYSE:JNJ,NYSE:PG,NYSE:KO,NYSE:DIS,NYSE:CRM,NYSE:ORCL,NYSE:IBM,"
            "NYSE:BA,NYSE:MMM,NYSE:UPS,NYSE:RTX,NYSE:SPOT,NYSE:UBER"
        )
        ticker_html = f"""
        <!doctype html>
        <html lang=\"en\">
        <head>
            <meta charset=\"UTF-8\">
            <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
            <style>
                html, body {{ margin: 0; padding: 0; background: transparent; overflow: hidden; }}
                tv-ticker-tape {{ display: block; width: 100%; }}
            </style>
        </head>
        <body>
            <script type=\"module\" src=\"https://widgets.tradingview-widget.com/w/en/tv-ticker-tape.js\"></script>
            <tv-ticker-tape
                symbols=\"{symbols}\"
                hide-chart
                item-size=\"compact\"
                show-hover
                theme=\"light\"
            ></tv-ticker-tape>
        </body>
        </html>
        """
        components.html(ticker_html, height=48, scrolling=False)

    # --- TradingView Market Widgets ---
    def display_tradingview_market_widgets(ticker, exchange):
        """Render the TradingView market-data dashboard in ONE embedded frame.

        Keeping all TradingView widgets inside a single components.html call avoids
        creating a separate Streamlit iframe/window for every widget. The page can
        scroll normally, but the widgets themselves are given enough height so the
        user does not get a scrollbar for each individual Streamlit component.
        """
        ticker_clean = re.sub(r"[^A-Z0-9._-]", "", str(ticker).upper().strip())
        exchange_clean = re.sub(r"[^A-Z]", "", str(exchange).upper().strip())
        if not ticker_clean or not exchange_clean:
            return

        tv_symbol = f"{exchange_clean}:{ticker_clean}"

        symbol_json = json.dumps(tv_symbol)

        # One HTML document = one Streamlit component. The CSS grid below controls
        # the layout of all TradingView widgets inside that single component.
        market_dashboard_html = f"""
        <!doctype html>
        <html lang=\"en\">
        <head>
            <meta charset=\"UTF-8\">
            <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
            <style>
                * {{ box-sizing: border-box; }}
                html, body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    color: #ffffff;
                    font-family: Arial, sans-serif;
                    overflow-x: hidden;
                }}
                .dashboard {{
                    width: 100%;
                    padding: 0;
                }}
                .section-title {{
                    font-size: 16px;
                    font-weight: 600;
                    margin: 8px 0 8px 2px;
                    color: #e6e6e6;
                }}
                .tape {{
                    width: 100%;
                    min-height: 64px;
                    margin-bottom: 8px;
                    overflow: hidden;
                }}
                .snapshot {{
                    width: 100%;
                    min-height: 125px;
                    margin-bottom: 10px;
                    overflow: hidden;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
                    gap: 12px;
                    width: 100%;
                    align-items: start;
                }}
                .card {{
                    min-width: 0;
                    width: 100%;
                    overflow: hidden;
                }}
                .tech {{ height: 390px; }}
                .profile {{ height: 390px; }}
                .fundamentals {{ height: 510px; }}
                .news {{ height: 510px; }}
                .footer {{
                    height: 22px;
                }}
                .tradingview-widget-container,
                .tradingview-widget-container__widget {{
                    width: 100% !important;
                }}
                @media (max-width: 850px) {{
                    .grid {{ grid-template-columns: 1fr 1fr; gap: 8px; }}
                    .tech, .profile {{ height: 420px; }}
                    .fundamentals, .news {{ height: 540px; }}
                }}
            </style>
        </head>
        <body>
        <div class=\"dashboard\">

            <div class=\"section-title\">{ticker_clean} Market Snapshot</div>
            <div class=\"snapshot\">
                <div class=\"tradingview-widget-container\">
                    <div class=\"tradingview-widget-container__widget\"></div>
                    <script type=\"text/javascript\"
                        src=\"https://s3.tradingview.com/external-embedding/embed-widget-symbol-info.js\"
                        async>
                        {{
                            \"symbol\": {symbol_json},
                            \"width\": \"100%\",
                            \"locale\": \"en\",
                            \"colorTheme\": \"dark\",
                            \"isTransparent\": true
                        }}
                    </script>
                </div>
            </div>

            <div class=\"grid\">
                <div class=\"card tech\">
                    <div class=\"tradingview-widget-container\" style=\"height:100%; width:100%\">
                        <div class=\"tradingview-widget-container__widget\" style=\"height:100%; width:100%\"></div>
                        <script type=\"text/javascript\"
                            src=\"https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js\"
                            async>
                            {{
                                \"interval\": \"1D\",
                                \"width\": \"100%\",
                                \"height\": \"100%\",
                                \"isTransparent\": true,
                                \"symbol\": {symbol_json},
                                \"showIntervalTabs\": true,
                                \"displayMode\": \"single\",
                                \"locale\": \"en\",
                                \"colorTheme\": \"dark\"
                            }}
                        </script>
                    </div>
                </div>

                <div class=\"card profile\">
                    <div class=\"tradingview-widget-container\" style=\"height:100%; width:100%\">
                        <div class=\"tradingview-widget-container__widget\" style=\"height:100%; width:100%\"></div>
                        <script type=\"text/javascript\"
                            src=\"https://s3.tradingview.com/external-embedding/embed-widget-company-profile.js\"
                            async>
                            {{
                                \"width\": \"100%\",
                                \"height\": \"100%\",
                                \"colorTheme\": \"dark\",
                                \"isTransparent\": true,
                                \"symbol\": {symbol_json},
                                \"locale\": \"en\"
                            }}
                        </script>
                    </div>
                </div>

                <div class=\"card fundamentals\">
                    <div class=\"tradingview-widget-container\" style=\"height:100%; width:100%\">
                        <div class=\"tradingview-widget-container__widget\" style=\"height:100%; width:100%\"></div>
                        <script type=\"text/javascript\"
                            src=\"https://s3.tradingview.com/external-embedding/embed-widget-financials.js\"
                            async>
                            {{
                                \"colorTheme\": \"dark\",
                                \"isTransparent\": true,
                                \"largeChartUrl\": \"\",
                                \"displayMode\": \"adaptive\",
                                \"width\": \"100%\",
                                \"height\": \"100%\",
                                \"symbol\": {symbol_json},
                                \"locale\": \"en\"
                            }}
                        </script>
                    </div>
                </div>

                <div class=\"card news\">
                    <div class=\"tradingview-widget-container\" style=\"height:100%; width:100%\">
                        <div class=\"tradingview-widget-container__widget\" style=\"height:100%; width:100%\"></div>
                        <script type=\"text/javascript\"
                            src=\"https://s3.tradingview.com/external-embedding/embed-widget-timeline.js\"
                            async>
                            {{
                                \"feedMode\": \"symbol\",
                                \"symbol\": {symbol_json},
                                \"colorTheme\": \"dark\",
                                \"isTransparent\": true,
                                \"displayMode\": \"regular\",
                                \"width\": \"100%\",
                                \"height\": \"100%\",
                                \"locale\": \"en\"
                            }}
                        </script>
                    </div>
                </div>
            </div>

            <div class=\"footer\"></div>
        </div>
        </body>
        </html>
        """

        # Height is intentionally large enough for the full two-column dashboard so
        # the Streamlit component itself does not become a nested scrolling window.
        components.html(market_dashboard_html, height=1760, scrolling=False)

    def apply_professional_theme():
        """Apply a restrained, professional trading-terminal style without changing app logic."""
        st.markdown("""
        <style>
            :root {
                --alpha-bg: #0b0f14;
                --alpha-panel: rgba(18, 24, 32, 0.88);
                --alpha-panel-2: rgba(22, 29, 39, 0.92);
                --alpha-border: rgba(148, 163, 184, 0.14);
                --alpha-text: #eef2f7;
                --alpha-muted: #8f9baa;
                --alpha-accent: #5aa9ff;
            }

            .stApp {
                background: radial-gradient(circle at top right, rgba(58, 110, 170, 0.10), transparent 28%),
                            radial-gradient(circle at 15% 20%, rgba(70, 180, 150, 0.05), transparent 24%),
                            #0b0f14;
                color: var(--alpha-text);
            }

            [data-testid="stHeader"] {
                background: rgba(11, 15, 20, 0.78);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0b1016 0%, #0d131b 100%);
                border-right: 1px solid var(--alpha-border);
            }

            [data-testid="stSidebar"] .block-container {
                padding-top: 1.5rem;
                padding-left: 1.1rem;
                padding-right: 1.1rem;
            }

            .alpha-brand {
                padding: 6px 0 18px 0;
                border-bottom: 1px solid var(--alpha-border);
                margin-bottom: 18px;
            }
            .alpha-brand h1 {
                margin: 0;
                font-size: 1.62rem;
                letter-spacing: -0.03em;
                color: #f7fafc;
            }
            .alpha-brand p {
                margin: 5px 0 0 0;
                color: var(--alpha-muted);
                font-size: 0.84rem;
            }

            .alpha-section {
                margin: 18px 0 8px 0;
                color: #b5bfca;
                font-size: 0.74rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.12em;
            }

            .hero {
                display: flex;
                align-items: end;
                justify-content: space-between;
                gap: 16px;
                margin: 6px 0 18px 0;
                padding: 20px 22px;
                border: 1px solid var(--alpha-border);
                border-radius: 16px;
                background: linear-gradient(135deg, rgba(19, 27, 37, 0.94), rgba(13, 18, 25, 0.88));
                box-shadow: 0 12px 35px rgba(0,0,0,0.16);
            }
            .hero-title {
                margin: 0;
                font-size: 1.9rem;
                font-weight: 750;
                letter-spacing: -0.035em;
            }
            .hero-subtitle {
                margin-top: 6px;
                color: var(--alpha-muted);
                font-size: 0.92rem;
            }
            .ticker-pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 12px;
                border: 1px solid rgba(90,169,255,0.24);
                border-radius: 999px;
                background: rgba(90,169,255,0.08);
                color: #cfe5ff;
                font-weight: 700;
                font-size: 0.9rem;
                white-space: nowrap;
            }

            .panel-label {
                margin: 8px 0 10px 0;
                color: #cbd5df;
                font-size: 1.02rem;
                font-weight: 700;
                letter-spacing: -0.01em;
            }

            [data-testid="stTabs"] {
                margin-top: 6px;
            }
            [data-testid="stTabs"] button {
                font-weight: 650;
                color: #96a2af;
            }
            [data-testid="stTabs"] button[aria-selected="true"] {
                color: #f2f6fa;
            }

            div.stButton > button {
                border-radius: 9px;
                border: 1px solid rgba(148,163,184,0.18);
                background: linear-gradient(180deg, #1b2633 0%, #151d27 100%);
                color: #edf3f8;
                font-weight: 650;
                transition: all .15s ease;
            }
            div.stButton > button:hover {
                border-color: rgba(90,169,255,0.45);
                background: #1d2a39;
            }

            [data-testid="stMetric"] {
                background: rgba(17, 24, 32, 0.72);
                border: 1px solid var(--alpha-border);
                border-radius: 12px;
                padding: 12px 14px;
            }

            .stAlert {
                border-radius: 10px;
            }

            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div,
            textarea {
                border-radius: 9px !important;
            }

            hr {
                border-color: var(--alpha-border);
                margin: 22px 0;
            }

            [data-testid="stDataFrame"] {
                border: 1px solid var(--alpha-border);
                border-radius: 12px;
                overflow: hidden;
            }
        </style>
        """, unsafe_allow_html=True)

    apply_professional_theme()

    # --- Streamlit App UI ---
    # Define a safe current ticker before rendering the hero. The sidebar
    # input is created below, so referencing ticker_input here would raise
    # UnboundLocalError on the first run.
    current_ticker = st.session_state.get('ticker', 'PLTR')

    st.markdown(
        f"""
        <div class=\"hero\">
            <div>
                <div class=\"hero-title\">Alpha</div>
                <div class=\"hero-subtitle\">Cycle Trading with Lunar Phases · Market & Options Intelligence</div>
            </div>
            <div class=\"ticker-pill\">● {current_ticker}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Professional sidebar header while preserving the existing login/logout plugin.
    st.sidebar.markdown(
        f"""<div class=\"alpha-brand\"><h1>ALPHA</h1><p>Trading intelligence terminal · {st.session_state['name']}</p></div>""",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Logout", use_container_width=True):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    st.sidebar.markdown('<div class="alpha-section">Market</div>', unsafe_allow_html=True)
    st.sidebar.markdown('Choose the symbol and market source.', unsafe_allow_html=True)
    ticker_input = st.sidebar.text_input('Ticker', value='PLTR').upper()
    tv_exchange = st.sidebar.selectbox("TradingView Exchange", ["NASDAQ", "NYSE", "AMEX", "ARCA", "OTC"], index=0, help="Choose the exchange TradingView should use for the selected ticker.")
    start_date_input = st.sidebar.date_input('Start Date', value=pd.to_datetime('2025-07-01'))
    end_date_input = st.sidebar.date_input('End Date', value=dt.date.today())
    chart_type = st.sidebar.selectbox("Select Chart Type", ["Candlestick", "Line"])

    st.sidebar.markdown('<div class="alpha-section">Lunar overlay</div>', unsafe_allow_html=True)
    show_full_moon = st.sidebar.checkbox('Show Full Moon (Red)', value=True)
    show_new_moon = st.sidebar.checkbox('Show New Moon (Blue)', value=True)
    show_quarter_moon = st.sidebar.checkbox('Show Quarter Moon (Green)', value=True)

    st.sidebar.markdown('<div class="alpha-section">Analytics</div>', unsafe_allow_html=True)
    show_analysis = st.sidebar.checkbox('Show Lunar Analysis', value=True)
    num_price_levels = st.sidebar.number_input("Show Price Levels for Last X Cycles", min_value=0, max_value=20, value=2, step=1)

    if not GOOGLE_API_KEY:
        st.sidebar.markdown('<div class="alpha-section">Gemini AI</div>', unsafe_allow_html=True)
        user_google_key = st.sidebar.text_input("Enter your Google AI API Key", type="password")
        if user_google_key:
            GOOGLE_API_KEY = user_google_key
            genai.configure(api_key=GOOGLE_API_KEY)
            st.sidebar.success("API Key configured!")

    if st.sidebar.button("Update Chart"):
        st.session_state.ticker = ticker_input
        st.session_state.start_date = start_date_input
        st.session_state.end_date = end_date_input
        st.session_state.stock_data = pd.DataFrame()
        st.session_state.all_moon_events = []
        if 'options_loaded' in st.session_state:
            del st.session_state.options_loaded
        load_data(ticker_input, start_date_input, end_date_input)

    if 'data_loaded' not in st.session_state:
        st.session_state.ticker = ticker_input
        st.session_state.start_date = start_date_input
        st.session_state.end_date = end_date_input
        load_data(ticker_input, start_date_input, end_date_input)

    # TradingView market tape is global and sits at the very top of the page,
    # above all three app tabs.
    display_top_tradingview_ticker_tape()

    tab1, tab2, tab3, tab4 = st.tabs(["Charts & Options", "AI Analysis", "Market Heatmaps", "Eclipses"])

    fig = None
    stock_analysis_results = []

    with tab1:
        display_event_countdown()

        if 'stock_data' in st.session_state and not st.session_state.stock_data.empty:
            data = st.session_state.stock_data
            y_min, y_max = data['Low'].min(), data['High'].max()

            if chart_type == 'Line':
                fig = go.Figure(data=[go.Scatter(x=data['Date'].tolist(), y=data['Close'].astype(float).tolist(), mode='lines', name='Close')])
                fig.update_layout(title_text=f"{st.session_state.get('loaded_ticker', st.session_state.get('ticker', 'N/A'))} Stock Price")
            else: 
                fig = go.Figure(data=[go.Candlestick(x=data['Date'].tolist(), open=data['Open'].astype(float).tolist(), high=data['High'].astype(float).tolist(), low=data['Low'].astype(float).tolist(), close=data['Close'].astype(float).tolist())])
                fig.update_layout(title_text=f"{st.session_state.get('ticker', 'N/A')} Stock Price", xaxis_rangeslider_visible=False)
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider_visible=False)
            
            all_moon_events = st.session_state.get('all_moon_events', [])
            
            visible_moon_events = []
            if show_full_moon:
                visible_moon_events.extend([e for e in all_moon_events if e[1] == 'Full Moon'])
            if show_new_moon:
                visible_moon_events.extend([e for e in all_moon_events if e[1] == 'New Moon'])
            if show_quarter_moon:
                visible_moon_events.extend([e for e in all_moon_events if e[1] == 'Quarter Moon'])
            visible_moon_events.sort()

            fig = add_moon_phases_to_fig(fig, visible_moon_events)
            
            if show_analysis and not data.empty:
                fig, stock_analysis_results = add_lunar_analysis_annotations(fig, data, visible_moon_events, open_col='Open', high_col='High', low_col='Low', close_col='Close', y_max_total=y_max)
            
            # RE-ENABLED this call
            if num_price_levels > 0 and stock_analysis_results:
                fig = add_price_level_lines(fig, stock_analysis_results, num_price_levels)

            fig.update_xaxes(range=[data['Date'].min(), data['Date'].max()])
            fig.update_yaxes(range=[float(y_min) * 0.98, float(y_max) * 1.25])
            fig.update_layout(xaxis_title='Date', yaxis_title='Price (USD)')

        st.markdown(f"<div class='panel-label'>Price action · {st.session_state.get('ticker', 'N/A')}</div>", unsafe_allow_html=True)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enter a stock ticker and click 'Update Chart' in the sidebar to begin.")

        if show_analysis and stock_analysis_results:
            display_analysis_table(stock_analysis_results, grid_key="lunar_cycle_analysis_grid_stock")
            display_summary_and_active_cycle_stats(stock_analysis_results)

        st.markdown("---")

        st.markdown(f"<div class='panel-label'>Options research · {st.session_state.get('ticker', 'N/A')}</div>", unsafe_allow_html=True)
        st.caption("Options data is loaded from Public.com via the official Python SDK (with Public REST compatibility fallback). Polygon is not used for option lookup.")
        
        if 'options_loaded' not in st.session_state:
            st.session_state.options_loaded = False

        if not st.session_state.options_loaded:
            if st.button("Load Options Chain"):
                st.session_state.options_loaded = True
                

        if st.session_state.options_loaded:
            with st.spinner("Loading option expirations..."):
                sorted_expirations, contract_data = get_all_contract_info_free(st.session_state.get('ticker', 'N/A'))
            
            if not sorted_expirations:
                st.warning(f"Could not find any option expiration dates for {st.session_state.get('ticker', 'N/A')}.")
            else:
                exp_date_str = st.selectbox("Select Expiration Date", options=sorted_expirations, key="chain_exp_select")
                
                with st.spinner(f"Loading full option chain for {exp_date_str}..."):
                    try:
                        selected_chain = get_public_option_chain(st.session_state.get('ticker', 'N/A'), exp_date_str)
                        chain_rows = _chain_contract_rows(selected_chain)
                    except Exception as chain_error:
                        st.error(f"Could not load Public option chain: {chain_error}")
                        chain_rows = []

                if not chain_rows:
                    st.warning("No option chain data available for this expiration.")
                else:
                    df_chain = pd.DataFrame(chain_rows)
                    
                    type_col = next((col for col in ['option_type', 'type', 'contract_type'] if col in df_chain.columns), None)
                    strike_col = next((col for col in ['strike', 'strike_price'] if col in df_chain.columns), 'strike')
                    
                    if strike_col != 'strike' and strike_col in df_chain.columns:
                        df_chain['strike'] = df_chain[strike_col]

                    all_strikes = sorted(df_chain['strike'].unique()) if 'strike' in df_chain.columns else []
                    
                    if all_strikes:
                        default_min = float(all_strikes[0])
                        default_max = float(all_strikes[-1])
                        
                        st.markdown("**Strike Range Filter**")
                        rc1, rc2 = st.columns(2)
                        with rc1:
                            user_min_strike = st.number_input("Min Strike", value=default_min, step=1.0, key="chain_min_strike")
                        with rc2:
                            user_max_strike = st.number_input("Max Strike", value=default_max, step=1.0, key="chain_max_strike")
                        
                        strikes = [s for s in all_strikes if user_min_strike <= s <= user_max_strike]
                        if not strikes:
                            st.warning("No strikes found within your custom range. Resetting to full range.")
                            strikes = all_strikes
                    else:
                        strikes = []

                    # 1. Option Chain Matrix Table Data
                    chain_table_data = []
                    for s in strikes:
                        if type_col:
                            c_row = df_chain[(df_chain['strike'] == s) & (df_chain[type_col].astype(str).str.lower().isin(['call', 'c']))]
                            p_row = df_chain[(df_chain['strike'] == s) & (df_chain[type_col].astype(str).str.lower().isin(['put', 'p']))]
                        else:
                            c_row = pd.DataFrame()
                            p_row = pd.DataFrame()
                        
                        row_data = {}
                        row_data['Call Vol'] = int(c_row.iloc[0].get('volume', 0)) if not c_row.empty and pd.notna(c_row.iloc[0].get('volume')) else 0
                        row_data['Call Last'] = float(c_row.iloc[0].get('last', c_row.iloc[0].get('close', 0.0))) if not c_row.empty and pd.notna(c_row.iloc[0].get('last', c_row.iloc[0].get('close', 0.0))) else 0.0
                        row_data['Call Bid'] = float(c_row.iloc[0].get('bid', 0.0)) if not c_row.empty and pd.notna(c_row.iloc[0].get('bid')) else 0.0
                        row_data['Call Ask'] = float(c_row.iloc[0].get('ask', 0.0)) if not c_row.empty and pd.notna(c_row.iloc[0].get('ask')) else 0.0
                        
                        row_data['Strike'] = s
                        
                        row_data['Put Bid'] = float(p_row.iloc[0].get('bid', 0.0)) if not p_row.empty and pd.notna(p_row.iloc[0].get('bid')) else 0.0
                        row_data['Put Ask'] = float(p_row.iloc[0].get('ask', 0.0)) if not p_row.empty and pd.notna(p_row.iloc[0].get('ask')) else 0.0
                        row_data['Put Last'] = float(p_row.iloc[0].get('last', p_row.iloc[0].get('close', 0.0))) if not p_row.empty and pd.notna(p_row.iloc[0].get('last', p_row.iloc[0].get('close', 0.0))) else 0.0
                        row_data['Put Vol'] = int(p_row.iloc[0].get('volume', 0)) if not p_row.empty and pd.notna(p_row.iloc[0].get('volume')) else 0
                        
                        chain_table_data.append(row_data)
                    
                    chain_df = pd.DataFrame(chain_table_data)
                    
                    st.subheader(f"Option Chain Matrix — {exp_date_str}")
                    
                    def style_chain(df):
                        styles = pd.DataFrame('', index=df.index, columns=df.columns)
                        for col in df.columns:
                            if 'Call' in col:
                                styles[col] = 'color: #2ecc71;'
                            elif 'Put' in col:
                                styles[col] = 'color: #e74c3c;'
                            elif col == 'Strike':
                                styles[col] = 'font-weight: bold; color: #f1c40f; text-align: center;'
                        return styles

                    styled_chain_df = chain_df.style.apply(style_chain, axis=None).format({
                        'Call Last': '${:.2f}', 'Call Bid': '${:.2f}', 'Call Ask': '${:.2f}',
                        'Put Last': '${:.2f}', 'Put Bid': '${:.2f}', 'Put Ask': '${:.2f}',
                        'Strike': '{:.2f}'
                    }).bar(subset=['Call Vol'], color='#1b4d3e', vmin=0).bar(subset=['Put Vol'], color='#5c2121', vmin=0)

                    # Removed fixed height so it displays fully on the page
                    render_aggrid(chain_df, height=620, key="option_chain_matrix_grid")

                    # 2. Strike-Level Exposure & Net Flow Analysis Matrix
                    st.markdown("---")
                    st.subheader(f"Strike-Level Exposure & Volume Analysis — {exp_date_str}")
                    
                    strike_analysis_data = []
                    for s in strikes:
                        if type_col:
                            c_row = df_chain[(df_chain['strike'] == s) & (df_chain[type_col].astype(str).str.lower().isin(['call', 'c']))]
                            p_row = df_chain[(df_chain['strike'] == s) & (df_chain[type_col].astype(str).str.lower().isin(['put', 'p']))]
                        else:
                            c_row = pd.DataFrame()
                            p_row = pd.DataFrame()
                            
                        c_vol = int(c_row.iloc[0].get('volume', 0)) if not c_row.empty and pd.notna(c_row.iloc[0].get('volume')) else 0
                        p_vol = int(p_row.iloc[0].get('volume', 0)) if not p_row.empty and pd.notna(p_row.iloc[0].get('volume')) else 0
                        
                        c_oi = int(c_row.iloc[0].get('open_interest', c_row.iloc[0].get('openInterest', 0))) if not c_row.empty and pd.notna(c_row.iloc[0].get('open_interest', c_row.iloc[0].get('openInterest', 0))) else 0
                        p_oi = int(p_row.iloc[0].get('open_interest', p_row.iloc[0].get('openInterest', 0))) if not p_row.empty and pd.notna(p_row.iloc[0].get('open_interest', p_row.iloc[0].get('openInterest', 0))) else 0
                        
                        strike_analysis_data.append({
                            'Strike': s,
                            'Call Vol': c_vol,
                            'Put Vol': p_vol,
                            'Total Vol': c_vol + p_vol,
                            'Net Flow (C - P)': c_vol - p_vol,
                            'Call OI': c_oi,
                            'Put OI': p_oi
                        })
                        
                    strike_analysis_df = pd.DataFrame(strike_analysis_data)
                    
                    def style_strike_matrix(df):
                        styles = pd.DataFrame('', index=df.index, columns=df.columns)
                        for col in df.columns:
                            if col == 'Strike':
                                styles[col] = 'font-weight: bold; color: #f1c40f; text-align: center;'
                            elif 'Call' in col:
                                styles[col] = 'color: #2ecc71;'
                            elif 'Put' in col:
                                styles[col] = 'color: #e74c3c;'
                        return styles

                    styled_strike_df = strike_analysis_df.style.apply(style_strike_matrix, axis=None).format({
                        'Strike': '{:.2f}',
                        'Call Vol': '{:,}',
                        'Put Vol': '{:,}',
                        'Total Vol': '{:,}',
                        'Net Flow (C - P)': '{:,}',
                        'Call OI': '{:,}',
                        'Put OI': '{:,}'
                    }).bar(subset=['Net Flow (C - P)'], align='mid', color=['#e74c3c', '#2ecc71']).bar(subset=['Total Vol'], color='#3498db', vmin=0)

                    # Removed fixed height here as well
                    render_aggrid(strike_analysis_df, height=520, key="strike_flow_grid")

                    # 3. Greeks & modeled dealer exposure (DEX/GEX)
                    # IMPORTANT: exposure uses df_chain, i.e. the FULL Public chain for
                    # this expiration, not the user-selected strike range.
                    st.markdown("---")
                    st.subheader("Greeks & Dealer Exposure — Full Expiration Chain")

                    stock_data_for_spot = st.session_state.get("stock_data")
                    spot_price = None
                    if stock_data_for_spot is not None and not stock_data_for_spot.empty:
                        try:
                            spot_price = float(pd.to_numeric(stock_data_for_spot["Close"], errors="coerce").dropna().iloc[-1])
                        except Exception:
                            spot_price = None

                    if spot_price is None or spot_price <= 0:
                        st.warning("Unable to determine the latest underlying price, so DEX/GEX cannot be calculated.")
                    else:
                        exposure_df, gamma_flip = calculate_option_exposure(df_chain, spot_price)
                        raw_delta = pd.to_numeric(df_chain.get("delta"), errors="coerce").fillna(0.0) if "delta" in df_chain else pd.Series(dtype=float)
                        raw_gamma = pd.to_numeric(df_chain.get("gamma"), errors="coerce").fillna(0.0) if "gamma" in df_chain else pd.Series(dtype=float)
                        raw_oi = pd.to_numeric(df_chain.get("openInterest"), errors="coerce").fillna(0.0) if "openInterest" in df_chain else pd.Series(dtype=float)
                        if exposure_df.empty:
                            st.warning("Public option-chain data does not contain enough OI/Greek fields to calculate DEX/GEX.")
                        elif raw_delta.abs().sum() == 0 or raw_gamma.abs().sum() == 0 or raw_oi.sum() == 0:
                            st.warning("Public returned the chain, but Delta/Gamma/Open Interest are zero or missing. The chain parser was updated to accept SDK and REST field variants; inspect the raw Public response if this warning persists.")
                        else:
                            positive_levels = exposure_df[exposure_df["Net GEX"] > 0]
                            max_gex_row = positive_levels.loc[positive_levels["Net GEX"].idxmax()] if not positive_levels.empty else None
                            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                            mc1.metric("Underlying", f"${spot_price:,.2f}")
                            mc2.metric("Net DEX", f"${exposure_df['Net DEX'].sum()/1_000_000:,.2f}M")
                            mc3.metric("Net GEX", f"${exposure_df['Net GEX'].sum()/1_000_000:,.2f}M")
                            mc4.metric("Largest +GEX Strike", f"${max_gex_row['Strike']:,.2f}" if max_gex_row is not None else "N/A")
                            mc5.metric("Gamma Flip", f"${gamma_flip:,.2f}" if gamma_flip is not None else "N/A")

                            display_cols = ["Strike", "Call OI", "Put OI", "Total OI", "Call DEX", "Put DEX", "Net DEX", "Call GEX", "Put GEX", "Net GEX"]
                            exposure_display = exposure_millions(exposure_df[display_cols])
                            render_aggrid(exposure_display, height=520, key="strike_exposure_grid")

                            # Normalize plotted exposure fields explicitly. Public can return
                            # numeric values as nullable/object fields even when the AG Grid displays them.
                            # Grouped bars keep call/put exposure visible instead of visually canceling in a stack.
                            dex_plot = exposure_millions(exposure_df).copy()
                            gex_plot = exposure_millions(exposure_df).copy()
                            for col in ["Strike", "Call DEX", "Put DEX", "Net DEX"]:
                                dex_plot[col] = pd.to_numeric(dex_plot[col], errors="coerce")
                            for col in ["Strike", "Call GEX", "Put GEX", "Net GEX"]:
                                gex_plot[col] = pd.to_numeric(gex_plot[col], errors="coerce")
                            dex_plot = dex_plot.dropna(subset=["Strike", "Call DEX", "Put DEX", "Net DEX"]).sort_values("Strike").reset_index(drop=True)
                            gex_plot = gex_plot.dropna(subset=["Strike", "Call GEX", "Put GEX", "Net GEX"]).sort_values("Strike").reset_index(drop=True)

                            def _bar_width_from_strikes(plot_df):
                                if len(plot_df) < 2:
                                    return None
                                diffs = plot_df["Strike"].diff().dropna()
                                diffs = diffs[diffs > 0]
                                return float(diffs.median()) * 0.72 if not diffs.empty else None

                            dex_width = _bar_width_from_strikes(dex_plot)
                            gex_width = _bar_width_from_strikes(gex_plot)

                            st.markdown("#### DEX by Strike")
                            dex_fig = go.Figure()
                            dex_fig.add_trace(go.Bar(x=dex_plot["Strike"].tolist(), y=dex_plot["Call DEX"].tolist(), name="Call DEX", width=dex_width, marker_color=["#2ecc71" if (v or 0) >= 0 else "#e74c3c" for v in dex_plot["Call DEX"].tolist()]))
                            dex_fig.add_trace(go.Bar(x=dex_plot["Strike"].tolist(), y=dex_plot["Put DEX"].tolist(), name="Put DEX", width=dex_width, marker_color=["#2ecc71" if (v or 0) >= 0 else "#e74c3c" for v in dex_plot["Put DEX"].tolist()]))
                            dex_fig.add_trace(go.Scatter(x=dex_plot["Strike"].tolist(), y=dex_plot["Net DEX"].tolist(), name="Net DEX", mode="lines+markers"))
                            dex_fig.add_hline(y=0, line_width=1)
                            dex_fig.add_vline(x=spot_price, line_dash="dash", annotation_text="Spot")
                            dex_fig.update_layout(barmode="group", xaxis_title="Strike", yaxis_title="DEX ($M)", hovermode="x unified", height=520, xaxis_rangeslider_visible=False)
                            st.plotly_chart(dex_fig, use_container_width=True)

                            st.markdown("#### GEX by Strike")
                            gex_fig = go.Figure()
                            gex_fig.add_trace(go.Bar(x=gex_plot["Strike"].tolist(), y=gex_plot["Call GEX"].tolist(), name="Call GEX", width=gex_width, marker_color=["#2ecc71" if (v or 0) >= 0 else "#e74c3c" for v in gex_plot["Call GEX"].tolist()]))
                            gex_fig.add_trace(go.Bar(x=gex_plot["Strike"].tolist(), y=gex_plot["Put GEX"].tolist(), name="Put GEX", width=gex_width, marker_color=["#2ecc71" if (v or 0) >= 0 else "#e74c3c" for v in gex_plot["Put GEX"].tolist()]))
                            gex_fig.add_trace(go.Scatter(x=gex_plot["Strike"].tolist(), y=gex_plot["Net GEX"].tolist(), name="Net GEX", mode="lines+markers"))
                            gex_fig.add_hline(y=0, line_width=1)
                            gex_fig.add_vline(x=spot_price, line_dash="dash", annotation_text="Spot")
                            if gamma_flip is not None:
                                gex_fig.add_vline(x=gamma_flip, line_dash="dot", annotation_text="Gamma Flip")
                            gex_fig.update_layout(barmode="group", xaxis_title="Strike", yaxis_title="GEX ($M)", hovermode="x unified", height=520, xaxis_rangeslider_visible=False)
                            st.plotly_chart(gex_fig, use_container_width=True)

                            st.markdown("#### Major Gamma Levels")
                            positive_gex = exposure_df[exposure_df["Net GEX"] > 0].nlargest(5, "Net GEX")[['Strike', 'Net GEX']].copy()
                            negative_gex = exposure_df[exposure_df["Net GEX"] < 0].nsmallest(5, "Net GEX")[['Strike', 'Net GEX']].copy()
                            positive_gex = exposure_millions(positive_gex)
                            negative_gex = exposure_millions(negative_gex)
                            g1, g2 = st.columns(2)
                            with g1:
                                st.markdown("**Top 5 Positive Net GEX**")
                                render_aggrid(positive_gex, height=240, key="positive_gamma_grid")
                            with g2:
                                st.markdown("**Top 5 Negative Net GEX**")
                                render_aggrid(negative_gex, height=240, key="negative_gamma_grid")

                            # ------------------------------------------------------------
                            # Expiration Term Structure — every expiration uses its FULL
                            # Public option chain. The strike filter above never enters
                            # these calculations.
                            # ------------------------------------------------------------
                            st.markdown("---")
                            st.subheader("Options Expiration Term Structure")
                            st.caption("Current-session positioning is calculated independently from the full chain for every future expiration. GEX/DEX 1D changes are intentionally left blank until historical chain snapshots are available.")

                            exp_count_col, exp_info_col = st.columns([1, 3])
                            with exp_count_col:
                                expiration_count = st.number_input(
                                    "Expirations to Chart",
                                    min_value=1,
                                    max_value=min(30, len(sorted_expirations)),
                                    value=min(8, len(sorted_expirations)),
                                    step=1,
                                    key="expiration_term_count",
                                )
                            with exp_info_col:
                                st.caption(f"Showing the first {int(expiration_count)} available future expirations returned by Public.com.")

                            selected_expirations = tuple(sorted_expirations[:int(expiration_count)])
                            try:
                                term_df, _term_chain_cache = get_expiration_term_structure(
                                    st.session_state.get('ticker', 'N/A'),
                                    selected_expirations,
                                    spot_price,
                                )
                            except Exception as term_error:
                                term_df = pd.DataFrame()
                                st.error(f"Could not build expiration term structure: {term_error}")

                            if term_df.empty:
                                st.warning("No complete expiration-level Public chains were available for term-structure calculations.")
                            else:
                                total_gex = float(term_df["GEX"].sum())
                                total_dex = float(term_df["DEX"].sum())
                                total_oi = float(term_df["OI"].sum())
                                tx1, tx2, tx3, tx4 = st.columns(4)
                                tx1.metric("Tot. Exposure — GEX", f"{total_gex:,.0f}")
                                tx2.metric("Tot. Exposure — DEX", f"{total_dex:,.0f}")
                                tx3.metric("Tot. Exposure — OI", f"{total_oi:,.0f}")
                                tx4.metric("Expirations", f"{len(term_df):,}")

                                term_display = term_df[[
                                    "Expiration Date", "DTE", "GEX", "DEX", "GEX Normalized", "DEX Normalized", "OI Normalized",
                                    "GEX Change 1D", "DEX Change 1D", "Call Resistance", "Put Support", "High Vol. Level", "Expiry Expected Move",
                                    "Expected Low", "Expected High"
                                ]].copy()
                                term_display["Expiration Date"] = pd.to_datetime(term_display["Expiration Date"]).dt.strftime("%m/%d/%y")
                                term_display["Expiry Expected Move"] = term_display["Expiry Expected Move"].apply(lambda x: f"±${x:,.2f}" if pd.notna(x) else "N/A")
                                term_display["Expected Low"] = term_display["Expected Low"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
                                term_display["Expected High"] = term_display["Expected High"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
                                for col in ["GEX Change 1D", "DEX Change 1D"]:
                                    term_display[col] = "—"

                                term_display["GEX"] = pd.to_numeric(term_display["GEX"], errors="coerce") / 1_000_000.0
                                term_display["DEX"] = pd.to_numeric(term_display["DEX"], errors="coerce") / 1_000_000.0
                                term_display = term_display.rename(columns={
                                    "GEX Normalized": "GEX %", "DEX Normalized": "DEX %", "OI Normalized": "OI %",
                                })
                                render_aggrid(
                                    term_display,
                                    height=620,
                                    key="expiration_term_structure_grid",
                                )

                                # Normalize plotted fields explicitly so Public numeric/nullable
                                # values always reach Plotly as real numeric arrays.
                                term_plot = term_df.copy()
                                term_plot["Expiration Date"] = pd.to_datetime(term_plot["Expiration Date"], errors="coerce")
                                term_plot["GEX"] = pd.to_numeric(term_plot["GEX"], errors="coerce") / 1_000_000.0
                                term_plot["DEX"] = pd.to_numeric(term_plot["DEX"], errors="coerce") / 1_000_000.0
                                term_plot = term_plot.dropna(subset=["Expiration Date", "GEX", "DEX"]).sort_values("Expiration Date")
                                ts_fig = go.Figure()
                                ts_fig.add_trace(go.Bar(x=term_plot["Expiration Date"].tolist(), y=term_plot["GEX"].tolist(), name="GEX"))
                                ts_fig.add_trace(go.Scatter(x=term_plot["Expiration Date"].tolist(), y=term_plot["DEX"].tolist(), name="DEX", mode="lines+markers", yaxis="y2"))
                                ts_fig.update_layout(
                                    title="GEX / DEX by Expiration",
                                    xaxis_title="Expiration",
                                    yaxis_title="GEX ($M)",
                                    yaxis2=dict(title="DEX ($M)", overlaying="y", side="right"),
                                    height=480,
                                )
                                st.plotly_chart(ts_fig, use_container_width=True)

                    # 4. Contract Chart & Technical Deep Dive
                    # 4. Contract Chart & Technical Deep Dive
                    st.markdown("---")
                    st.subheader("Contract Chart & Technical Deep Dive")
                    
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        selected_strike = st.selectbox("Select Strike Price for Chart", options=strikes, key="chain_chart_strike")
                    with cc2:
                        selected_type = st.selectbox("Select Option Type", options=["call", "put"], key="chain_chart_type")

                    if st.button("Fetch Contract History & Chart", key="load_chain_chart"):
                        with st.spinner(f"Fetching history for {selected_type.upper()} @ ${selected_strike}..."):
                            details, history_df = get_single_contract_details_free(
                                st.session_state.get('ticker', 'N/A'), exp_date_str, selected_strike, selected_type
                            )
                        
                        if details:
                            st.subheader(f"Contract Snapshot — {details.get('symbol', 'Contract')} (Current Session)")
                            dc1, dc2, dc3, dc4, dc5 = st.columns(5)
                            dc1.metric("Last", f"${details['last']:.2f}" if isinstance(details.get('last'), (int, float)) else "N/A")
                            dc2.metric("Bid", f"${details['bid']:.2f}" if isinstance(details.get('bid'), (int, float)) else "N/A")
                            dc3.metric("Ask", f"${details['ask']:.2f}" if isinstance(details.get('ask'), (int, float)) else "N/A")
                            dc4.metric("Volume", f"{details['volume']:,}" if isinstance(details.get('volume'), (int, float)) else "N/A")
                            dc5.metric("Open Interest", f"{details.get('open_interest', 0):,}" if isinstance(details.get('open_interest'), (int, float)) else "N/A")
                            gc1, gc2, gc3, gc4 = st.columns(4)
                            gc1.metric("Delta", f"{details['delta']:.4f}" if isinstance(details.get('delta'), (int, float)) else "N/A")
                            gc2.metric("Gamma", f"{details['gamma']:.6f}" if isinstance(details.get('gamma'), (int, float)) else "N/A")
                            gc3.metric("IV", f"{details['iv']:.2%}" if isinstance(details.get('iv'), (int, float)) else "N/A")
                            gc4.metric("Theta", f"{details['theta']:.4f}" if isinstance(details.get('theta'), (int, float)) else "N/A")

                        if history_df is not None and not history_df.empty:
                            history_df = history_df.copy().sort_values("Date").reset_index(drop=True)
                            history_df["Date"] = pd.to_datetime(history_df["Date"]).dt.normalize()
                            y_min_hist, y_max_hist = history_df['Low'].min(), history_df['High'].max()

                            st.subheader("Contract Price History")
                            if chart_type == 'Line':
                                history_fig = go.Figure(data=[go.Scatter(
                                    x=history_df['Date'].tolist(),
                                    y=history_df['Close'].astype(float).tolist(),
                                    mode='lines',
                                    name='Close'
                                )])
                                history_fig.update_layout(title_text=f"Price History for {details.get('symbol', '')}")
                            else:
                                history_fig = go.Figure(data=[go.Candlestick(
                                    x=history_df['Date'].tolist(),
                                    open=history_df['Open'].astype(float).tolist(),
                                    high=history_df['High'].astype(float).tolist(),
                                    low=history_df['Low'].astype(float).tolist(),
                                    close=history_df['Close'].astype(float).tolist()
                                )])
                                history_fig.update_layout(title_text=f"Price History for {details.get('symbol', '')}", xaxis_rangeslider_visible=False)
                                history_fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider_visible=False)

                            all_moon_events_hist = st.session_state.get('all_moon_events', [])
                            visible_moon_events_hist = []
                            if show_full_moon: visible_moon_events_hist.extend([e for e in all_moon_events_hist if e[1] == 'Full Moon'])
                            if show_new_moon: visible_moon_events_hist.extend([e for e in all_moon_events_hist if e[1] == 'New Moon'])
                            if show_quarter_moon: visible_moon_events_hist.extend([e for e in all_moon_events_hist if e[1] == 'Quarter Moon'])
                            visible_moon_events_hist.sort()

                            history_fig = add_moon_phases_to_fig(history_fig, visible_moon_events_hist)
                            
                            options_analysis_results = []
                            if show_analysis and not history_df.empty:
                                history_fig, options_analysis_results = add_lunar_analysis_annotations(history_fig, history_df, visible_moon_events_hist, open_col='Open', high_col='High', low_col='Low', close_col='Close', y_max_total=y_max_hist)
                            
                            if num_price_levels > 0 and options_analysis_results:
                                history_fig = add_price_level_lines(history_fig, options_analysis_results, num_price_levels)

                            history_fig.update_xaxes(range=[history_df['Date'].min(), history_df['Date'].max()])
                            history_fig.update_yaxes(range=[y_min_hist * 0.98, y_max_hist * 1.25])
                            
                            st.plotly_chart(history_fig, use_container_width=True)

                            if show_analysis and options_analysis_results:
                                display_analysis_table(options_analysis_results, grid_key="lunar_cycle_analysis_grid_options")
                                display_summary_and_active_cycle_stats(options_analysis_results)
                        else:
                            st.info("No price history found for this contract.")

        # ------------------------------------------------------------
        # TradingView market widgets belong AFTER the chart + options
        # workflow so the user's proprietary Alpha analysis stays first.
        # ------------------------------------------------------------
        st.markdown("---")
        st.markdown("<div class='panel-label'>Market intelligence</div>", unsafe_allow_html=True)
        st.caption("Live TradingView context for the selected symbol — tape, snapshot, technicals, company profile, fundamentals, and news.")
        display_tradingview_market_widgets(st.session_state.get('ticker', current_ticker), tv_exchange)

    with tab2:
        st.markdown("<div class='panel-label'>AI chart analysis</div>", unsafe_allow_html=True)
        
        if not GOOGLE_API_KEY:
            st.warning("Please enter your Google AI API Key in the sidebar to enable AI analysis.")
        else:
            uploaded_files = st.file_uploader(
                "Upload one or more chart images for analysis",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True
            )

            if uploaded_files:
                st.image(uploaded_files, width=250)

            default_prompt = (
                "You are a financial analyst. Analyze the provided stock chart image(s). "
                "Identify key trends, support/resistance levels, and any notable chart patterns. "
                "Provide a concise, professional analysis."
            )
            user_prompt = st.text_area("Your Prompt:", value=default_prompt, height=150)

            if st.button("Generate AI Analysis"):
                if uploaded_files:
                    pil_images = [Image.open(file) for file in uploaded_files]
                    with st.spinner("Gemini is analyzing the image(s)..."):
                        analysis_result = analyze_images_with_gemini(pil_images, user_prompt)
                        st.markdown(analysis_result)
                else:
                    st.warning("Please upload at least one image to analyze.")



    with tab3:
        st.markdown("<div class='panel-label'>Market heatmaps</div>", unsafe_allow_html=True)
        st.caption("Turn individual TradingView heatmaps on or off. Disabled widgets are not loaded, which keeps the page lighter.")

        # Keep widget selection in Streamlit state so each toggle persists across reruns.
        # Use standard st.button controls instead of st.toggle so this page
        # also works with older Streamlit versions. Each button flips the
        # corresponding heatmap on/off and reruns the page.
        heatmap_defaults = {
            "show_stock_heatmap": True,
            "show_etf_heatmap": False,
            "show_crypto_heatmap": True,
            "show_forex_heatmap": False,
        }
        for heatmap_key, default_value in heatmap_defaults.items():
            if heatmap_key not in st.session_state:
                st.session_state[heatmap_key] = default_value

        def heatmap_toggle_button(label, key):
            state = st.session_state[key]
            status = "ON" if state else "OFF"
            if st.button(f"{label}: {status}", key=f"{key}_button", use_container_width=True):
                st.session_state[key] = not state
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
            return st.session_state[key]

        heatmap_col1, heatmap_col2, heatmap_col3, heatmap_col4 = st.columns(4)
        with heatmap_col1:
            show_stock_heatmap = heatmap_toggle_button("📈 Stocks", "show_stock_heatmap")
        with heatmap_col2:
            show_etf_heatmap = heatmap_toggle_button("📊 ETFs", "show_etf_heatmap")
        with heatmap_col3:
            show_crypto_heatmap = heatmap_toggle_button("₿ Crypto", "show_crypto_heatmap")
        with heatmap_col4:
            show_forex_heatmap = heatmap_toggle_button("💱 Forex", "show_forex_heatmap")

        if not any([show_stock_heatmap, show_etf_heatmap, show_crypto_heatmap, show_forex_heatmap]):
            st.info("Turn on at least one heatmap above to load TradingView market data.")

        def render_heatmap_widget(script_name, config, height=520):
            """Render one TradingView heatmap only when the user enables it."""
            config_json = json.dumps(config, separators=(",", ":"))
            html = f"""
            <!doctype html>
            <html lang=\"en\">
            <head>
                <meta charset=\"UTF-8\">
                <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
                <style>
                    html, body {{
                        margin: 0;
                        padding: 0;
                        width: 100%;
                        height: 100%;
                        background: transparent;
                        overflow: hidden;
                    }}
                    .tradingview-widget-container,
                    .tradingview-widget-container__widget {{
                        width: 100% !important;
                        height: 100% !important;
                    }}
                </style>
            </head>
            <body>
                <div class=\"tradingview-widget-container\" style=\"height:100%;width:100%\">
                    <div class=\"tradingview-widget-container__widget\" style=\"height:100%;width:100%\"></div>
                    <script type=\"text/javascript\"
                        src=\"https://s3.tradingview.com/external-embedding/{script_name}\"
                        async>
                        {config_json}
                    </script>
                </div>
            </body>
            </html>
            """
            components.html(html, height=height, scrolling=False)

        if show_stock_heatmap:
            st.markdown("### 📈 Stock Heatmap")
            st.caption("S&P 500 stocks grouped by sector, sized by market cap and colored by daily change.")
            render_heatmap_widget(
                "embed-widget-stock-heatmap.js",
                {
                    "dataSource": "SPX500",
                    "blockSize": "market_cap_basic",
                    "blockColor": "change",
                    "grouping": "sector",
                    "locale": "en",
                    "symbolUrl": "",
                    "colorTheme": "dark",
                    "exchanges": [],
                    "hasTopBar": True,
                    "isDataSetEnabled": True,
                    "isZoomEnabled": True,
                    "hasSymbolTooltip": True,
                    "isMonoSize": False,
                    "width": "100%",
                    "height": "100%"
                },
                height=560
            )

        if show_etf_heatmap:
            st.markdown("### 📊 ETF Heatmap")
            st.caption("U.S. ETFs grouped by asset class, sized by volume and colored by daily change.")
            render_heatmap_widget(
                "embed-widget-etf-heatmap.js",
                {
                    "dataSource": "AllUSEtf",
                    "blockSize": "volume",
                    "blockColor": "change",
                    "grouping": "asset_class",
                    "locale": "en",
                    "symbolUrl": "",
                    "colorTheme": "dark",
                    "hasTopBar": True,
                    "isDataSetEnabled": True,
                    "isZoomEnabled": True,
                    "hasSymbolTooltip": True,
                    "isMonoSize": False,
                    "width": "100%",
                    "height": "100%"
                },
                height=560
            )

        if show_crypto_heatmap:
            st.markdown("### ₿ Crypto Heatmap")
            st.caption("Crypto assets sized by market cap and colored by recent price change.")
            render_heatmap_widget(
                "embed-widget-crypto-coins-heatmap.js",
                {
                    "dataSource": "Crypto",
                    "blockSize": "market_cap_calc",
                    "blockColor": "24h_close_change|5",
                    "locale": "en",
                    "symbolUrl": "",
                    "colorTheme": "dark",
                    "hasTopBar": True,
                    "isDataSetEnabled": True,
                    "isZoomEnabled": True,
                    "hasSymbolTooltip": True,
                    "isMonoSize": False,
                    "width": "100%",
                    "height": "100%"
                },
                height=560
            )

        if show_forex_heatmap:
            st.markdown("### 💱 Forex Heatmap")
            st.caption("Major currencies compared in real time, with heatmap coloring for relative strength and weakness.")
            render_heatmap_widget(
                "embed-widget-forex-heat-map.js",
                {
                    "colorTheme": "dark",
                    "isTransparent": False,
                    "locale": "en",
                    "currencies": ["EUR", "USD", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD", "CNY"],
                    "backgroundColor": "#0F0F0F",
                    "width": "100%",
                    "height": "100%"
                },
                height=560
            )


    with tab4:
        display_eclipse_page(
            get_price_data,
            st.session_state.get('ticker', current_ticker),
            end_date_input,
            get_all_contract_info_free,
            get_single_contract_details_free,
            get_public_option_chain,
            chart_type
        )


# --- APP ROUTING (NEW CODE) ---
if check_login():
    main_app()
