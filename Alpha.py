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
import streamlit.components.v1 as components

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(page_title="The Alpha - Simplifying Your Trading", layout="wide", page_icon="📈")


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

# --- MAIN APPLICATION (YOUR ORIGINAL CODE MOVED INTO THIS FUNCTION) ---
def main_app():

    # --- API Key Configuration ---
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=GOOGLE_API_KEY)
    except (FileNotFoundError, KeyError):
        GOOGLE_API_KEY = ""

    # MODIFIED: Polygon key now loaded securely from secrets
    try:
        POLYGON_API_KEY = st.secrets["POLYGON_API_KEY"]
    except (FileNotFoundError, KeyError):
        st.error("Polygon API Key not found in secrets. Please add it to your secrets.toml file.")
        st.stop()


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

    def get_price_data(ticker, start_date, end_date):
        """Fetches historical price data for a stock from Polygon.io."""
        try:
            client = RESTClient(POLYGON_API_KEY)
            aggs = client.get_aggs(
                ticker=ticker, multiplier=1, timespan="day", from_=start_date, to=end_date,
                adjusted=True, sort="asc", limit=50000,
            )
            df = pd.DataFrame(aggs)
            if df.empty: return pd.DataFrame()
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Adj Close', 'volume': 'Volume'}, inplace=True)
            df['Close'] = df['Adj Close']
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

    @st.cache_data(ttl=600)
    def get_all_contract_info_free(ticker):
        try:
            client = RESTClient(POLYGON_API_KEY)
            contracts = client.list_options_contracts(underlying_ticker=ticker, limit=1000)
            expirations_with_strikes = defaultdict(set)
            for c in contracts:
                expirations_with_strikes[c.expiration_date].add(c.strike_price)
            sorted_expirations = sorted(expirations_with_strikes.keys())
            final_data = {exp: sorted(list(strikes)) for exp, strikes in expirations_with_strikes.items()}
            return sorted_expirations, final_data
        except Exception as e:
            st.error(f"Could not fetch contract info: {e}")
            return [], {}

    def get_single_contract_details_free(ticker, expiration, strike, type):
        try:
            client = RESTClient(POLYGON_API_KEY)
            contract_list = list(client.list_options_contracts(
                underlying_ticker=ticker, expiration_date=expiration, strike_price=strike,
                contract_type=type, limit=1
            ))
            if not contract_list:
                st.warning("Contract not found.")
                return None, None
            
            option_ticker = contract_list[0].ticker
            
            yesterday = dt.date.today() - dt.timedelta(days=1)
            one_year_ago = yesterday - dt.timedelta(days=365)
            
            history_aggs = client.get_aggs(option_ticker, 1, "day", one_year_ago.strftime('%Y-%m-%d'), yesterday.strftime('%Y-%m-%d'), limit=5000)
            
            df = pd.DataFrame(history_aggs)
            if not df.empty:
                df['Date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
                df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)

            last_day_details = None
            try:
                    last_day_details = client.get_daily_open_close_agg(option_ticker, yesterday.strftime('%Y-%m-%d'))
            except: 
                    pass

            details = {
                "symbol": option_ticker, "open": getattr(last_day_details, 'open', 'N/A'),
                "close": getattr(last_day_details, 'close', 'N/A'), "high": getattr(last_day_details, 'high', 'N/A'),
                "low": getattr(last_day_details, 'low', 'N/A'), "volume": getattr(last_day_details, 'volume', 'N/A')
            }
            return details, df
            
        except Exception as e:
            st.error(f"Could not fetch contract details: {e}")
            return None, None

    @st.cache_data(ttl=3600)
    def load_data(ticker, start_date, end_date):
        with st.spinner("Fetching stock data..."):
            st.session_state.stock_data = get_price_data(ticker, start_date, end_date)

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

        df_indexed = df.set_index('Date')

        def find_next_trading_day(target_date, price_df):
            current_date = pd.to_datetime(target_date).date()
            while current_date <= price_df.index.max():
                if current_date in price_df.index:
                    return price_df.loc[current_date]
                current_date += dt.timedelta(days=1)
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

    def display_analysis_table(results_list):
        if not results_list: return
        st.subheader("Lunar Cycle Analysis Results")
        df = pd.DataFrame(results_list)
        
        cols_to_display = ['start_date', 'end_date', 'entry_price', 'end_price', 'pl_delta', 'pl_pct', 'max_profit', 'max_drawdown', 'status']
        display_df = df[cols_to_display].copy()

        display_df.rename(columns={
            'start_date': 'Start Date', 'end_date': 'End Date', 'entry_price': 'Entry Price',
            'end_price': 'End Price', 'pl_delta': 'P/L ($)', 'pl_pct': 'P/L (%)',
            'max_profit': 'Max Profit', 'max_drawdown': 'Max Drawdown', 'status': 'Status'
        }, inplace=True)

        def style_rows(row):
            color = 'background-color: rgba(0, 128, 0, 0.2)' if row['P/L ($)'] > 0 else 'background-color: rgba(128, 0, 0, 0.2)'
            return [color if col in ['P/L ($)', 'P/L (%)', 'Status'] else '' for col in display_df.columns]

        styled_df = display_df.style.apply(style_rows, axis=1).format({
            "Start Date": '{:%Y-%m-%d}', "End Date": '{:%Y-%m-%d}',
            "Entry Price": "${:,.2f}", "End Price": "${:,.2f}",
            "P/L ($)": "${:+.2f}", "P/L (%)": "{:+.2f}%",
            "Max Profit": "${:,.2f}", "Max Drawdown": "${:,.2f}"
        })
        
        st.dataframe(styled_df, use_container_width=True)


    # --- Streamlit App UI ---
    st.title('Alpha - Cycle Trading with Lunar Phases')
    
    # NEW: Welcome message and logout button
    st.sidebar.write(f"Welcome, {st.session_state['name']}!")
    if st.sidebar.button("Logout"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    st.sidebar.title('Chart Controls')
    st.sidebar.header("Pick a Stock:")
    ticker_input = st.sidebar.text_input('Ticker', value='PLTR').upper()
    start_date_input = st.sidebar.date_input('Start Date', value=pd.to_datetime('2025-07-01'))
    end_date_input = st.sidebar.date_input('End Date', value=dt.date.today())
    chart_type = st.sidebar.selectbox("Select Chart Type", ["Candlestick", "Line"])

    st.sidebar.subheader("Moon Phases")
    show_full_moon = st.sidebar.checkbox('Show Full Moon (Red)', value=True)
    show_new_moon = st.sidebar.checkbox('Show New Moon (Blue)', value=True)
    show_quarter_moon = st.sidebar.checkbox('Show Quarter Moon (Green)', value=True)

    st.sidebar.subheader("Analytics")
    show_analysis = st.sidebar.checkbox('Show Lunar Analysis', value=True)
    num_price_levels = st.sidebar.number_input("Show Price Levels for Last X Cycles", min_value=0, max_value=20, value=2, step=1)

    if not GOOGLE_API_KEY:
        st.sidebar.subheader("Gemini AI Configuration")
        user_google_key = st.sidebar.text_input("Enter your Google AI API Key", type="password")
        if user_google_key:
            GOOGLE_API_KEY = user_google_key
            genai.configure(api_key=GOOGLE_API_KEY)
            st.sidebar.success("API Key configured!")

    if st.sidebar.button("Update Chart"):
        st.session_state.ticker = ticker_input
        st.session_state.start_date = start_date_input
        st.session_state.end_date = end_date_input
        if 'options_loaded' in st.session_state:
            del st.session_state.options_loaded
        load_data(ticker_input, start_date_input, end_date_input)

    if 'data_loaded' not in st.session_state:
        st.session_state.ticker = ticker_input
        st.session_state.start_date = start_date_input
        st.session_state.end_date = end_date_input
        load_data(ticker_input, start_date_input, end_date_input)

    tab1, tab2, tab3 = st.tabs(["Charts & Options", "AI Analysis", "TradingView"])

    fig = None
    stock_analysis_results = []

    with tab1:
        display_event_countdown()

        if 'stock_data' in st.session_state and not st.session_state.stock_data.empty:
            data = st.session_state.stock_data
            y_min, y_max = data['Low'].min(), data['High'].max()

            if chart_type == 'Line':
                fig = px.line(data, x='Date', y='Adj Close', title=f"{st.session_state.get('ticker', 'N/A')} Stock Price")
            else: 
                fig = go.Figure(data=[go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
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

            fig.update_yaxes(range=[y_min * 0.98, y_max * 1.25])
            fig.update_layout(xaxis_title='Date', yaxis_title='Price (USD)')

        st.header(f"Price Chart for {st.session_state.get('ticker', 'N/A')}")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enter a stock ticker and click 'Update Chart' in the sidebar to begin.")

        if show_analysis and stock_analysis_results:
            display_analysis_table(stock_analysis_results)
            display_summary_and_active_cycle_stats(stock_analysis_results)

        st.markdown("---")

        st.header(f"Option Contract Lookup for {st.session_state.get('ticker', 'N/A')}")
        st.info("This tool uses the free Polygon.io plan. Click below to load options and be mindful of the 5 API calls/minute limit.")
        
        if 'options_loaded' not in st.session_state:
            st.session_state.options_loaded = False

        if not st.session_state.options_loaded:
            if st.button("Load Options Chain"):
                st.session_state.options_loaded = True
                st.experimental_rerun()

        if st.session_state.options_loaded:
            sorted_expirations, contract_data = get_all_contract_info_free(st.session_state.get('ticker', 'N/A'))
            
            if not sorted_expirations:
                st.warning(f"Could not find any option expiration dates for {st.session_state.get('ticker', 'N/A')}.")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    exp_date_str = st.selectbox("Select Expiration Date", options=sorted_expirations)
                strikes = contract_data.get(exp_date_str, [])
                if not strikes:
                    st.warning("No strikes found for this expiration.")
                else:
                    with col2:
                        strike_price = st.selectbox("Select Strike Price", options=strikes)
                    with col3:
                        option_type = st.radio("Select Option Type", ["call", "put"], horizontal=True)
                    
                    if st.button("Fetch Contract Details"):
                        with st.spinner(f"Fetching {option_type.upper()} @ ${strike_price} expiring {exp_date_str}..."):
                            details, history_df = get_single_contract_details_free(st.session_state.get('ticker'), exp_date_str, strike_price, option_type)
                        
                        if details:
                            st.subheader(f"Details for {details['symbol']} (as of yesterday's close)")
                            # RESTORED these metrics
                            c1, c2, c3, c4, c5 = st.columns(5)
                            c1.metric("Close", f"${details['close']:.2f}" if isinstance(details['close'], (int, float)) else "N/A")
                            c2.metric("Open", f"${details['open']:.2f}" if isinstance(details['open'], (int, float)) else "N/A")
                            c3.metric("High", f"${details['high']:.2f}" if isinstance(details['high'], (int, float)) else "N/A")
                            c4.metric("Low", f"${details['low']:.2f}" if isinstance(details['low'], (int, float)) else "N/A")
                            c5.metric("Volume", f"{details['volume']:,}" if isinstance(details['volume'], (int, float)) else "N/A")

                        if history_df is not None and not history_df.empty:
                            st.subheader(f"Contract Price History")
                            y_min_hist, y_max_hist = history_df['Low'].min(), history_df['High'].max()
                            
                            if chart_type == 'Line':
                                history_fig = px.line(history_df, x='Date', y='Close', title=f"Price History for {details['symbol']}")
                            else:
                                history_fig = go.Figure(data=[go.Candlestick(x=history_df['Date'], open=history_df['Open'], high=history_df['High'], low=history_df['Low'], close=history_df['Close'])])
                                history_fig.update_layout(title_text=f"Price History for {details['symbol']}", xaxis_rangeslider_visible=False)
                                history_fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                            
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
                                display_analysis_table(options_analysis_results)
                                display_summary_and_active_cycle_stats(options_analysis_results)

                        elif history_df is not None:
                                st.info("No price history found for this contract.")

    with tab2:
        st.header("AI Chart Analysis")
        
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
        st.header(f"TradingView Chart for {st.session_state.get('ticker', 'N/A')}")
        st.info("You can view a live chart below, or paste a specific 'Share Link' from TradingView.com to see your saved layout.")
        
        tv_url = st.text_input("Paste your TradingView Chart URL here (optional):")
        
        if tv_url:
            try:
                components.iframe(tv_url, height=700, scrolling=True)
            except Exception as e:
                st.error(f"Could not load the URL. Please ensure it's a valid TradingView share link. Error: {e}")
        else:
            ticker_symbol = st.session_state.get('ticker', 'N/A')
            
            tradingview_widget_html = f"""
            <div class="tradingview-widget-container" style="height:100%;width:100%">
                <div id="tradingview_f24a1" style="height:calc(100% - 32px);width:100%"></div>
                <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                <script type="text/javascript">
                new TradingView.widget(
                {{
                "autosize": true,
                "symbol": "{ticker_symbol}",
                "interval": "D",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "enable_publishing": false,
                "allow_symbol_change": true,
                "container_id": "tradingview_f24a1"
            }}
                );
                </script>
            </div>
            """
            components.html(tradingview_widget_html, height=700)


# --- APP ROUTING (NEW CODE) ---
if check_login():
    main_app()