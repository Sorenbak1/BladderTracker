import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
from datetime import datetime
import json

# Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = dict(st.secrets["gcp_service_account"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)

# creds = ServiceAccountCredentials.from_json_keyfile_name("robust-zenith-469918-a5-2447c030e8ef.json", scope)
# client = gspread.authorize(creds)

# Replace with your private Google Sheet name
SHEET_NAME = "BladderTrackerData"
worksheet = client.open(SHEET_NAME).sheet1

# Fetch data
data = worksheet.get_all_records()
df_raw = pd.DataFrame(data)  # Keep the raw DataFrame for laying periods

# Combine Date and Time into a single datetime column
df_raw['DateTime'] = pd.to_datetime(
    df_raw['Date'].astype(str) + ' ' + df_raw['Time'].astype(str),
    format='%d/%m/%Y %H:%M',
    errors='coerce'  # <-- This will set invalid parses to NaT
)
df_raw = df_raw.dropna(subset=['DateTime'])

# Extract laying periods before filtering
laying_rows = df_raw[df_raw['Type'].astype(str).str.strip().str.lower() == 'laying down']

# Now continue with your filtered DataFrame for plotting
df = df_raw.loc[df_raw['Amount'].astype(str).str.isnumeric()].copy()
df['Amount'] = pd.to_numeric(df['Amount'])

# Sort by datetime
df = df.sort_values('DateTime')

# --- User selects custom start date/time ---
min_dt = df_raw['DateTime'].min()
max_dt = df_raw['DateTime'].max()
default_dt = max_dt - pd.Timedelta(hours=24) if (max_dt - min_dt).total_seconds() > 24*3600 else min_dt

# Find the latest 'First Of The Day' entry in the raw data
fod_mask = (
    (df_raw['Type'].astype(str).str.strip().str.lower() == 'emptying') &
    (df_raw['Emptying Type'].astype(str).str.strip().str.lower() == 'first of the day')
)
if fod_mask.any():
    fod_time = df_raw.loc[fod_mask, 'DateTime'].max()
    default_dt = fod_time + pd.Timedelta(minutes=1)
else:
    default_dt = min_dt

st.sidebar.header("Filter")

# Calculate default_end_dt before using it in the reset button
default_end_dt = min(default_dt + pd.Timedelta(hours=24), max_dt)

# Add a button to reset time inputs to defaults
if st.sidebar.button("Reset time inputs to defaults"):
    st.session_state['start_date'] = default_dt.date()
    st.session_state['start_time'] = default_dt.time()
    st.session_state['end_date'] = default_end_dt.date()
    st.session_state['end_time'] = default_end_dt.time()

selected_start_date = st.sidebar.date_input(
    "Select start date",
    value=st.session_state.get('start_date', default_dt.date()),
    min_value=min_dt.date(),
    max_value=max_dt.date(),
    key="start_date"
)
selected_start_time = st.sidebar.time_input(
    "Select start time",
    value=st.session_state.get('start_time', default_dt.time()),
    key="start_time"
)
start_dt = pd.to_datetime(f"{selected_start_date} {selected_start_time}")

# Default end time: 24 hours after start, but not after max_dt
default_end_dt = min(start_dt + pd.Timedelta(hours=24), max_dt)
selected_end_date = st.sidebar.date_input(
    "Select end date",
    value=st.session_state.get('end_date', default_end_dt.date()),
    min_value=start_dt.date(),
    max_value=max_dt.date(),
    key="end_date"
)
selected_end_time = st.sidebar.time_input(
    "Select end time",
    value=st.session_state.get('end_time', default_end_dt.time()),
    key="end_time"
)
end_dt = pd.to_datetime(f"{selected_end_date} {selected_end_time}")

st.sidebar.info(
    "Default start time is set to just after the latest 'First Of The Day' emptying event.\n"
    "Default end time is 24 hours after the start time."
)

# Filter all dataframes to only include entries between selected start and end datetime
df = df[(df['DateTime'] >= start_dt) & (df['DateTime'] <= end_dt)]
laying_rows = laying_rows[(laying_rows['DateTime'] >= start_dt) & (laying_rows['DateTime'] <= end_dt)]
earliest_time = start_dt
latest_time = end_dt


# Make Amount negative for 'Emptying' type
df['Signed Amount'] = df.apply(
    lambda row: -row['Amount'] if str(row['Type']).strip().lower() == 'emptying' else row['Amount'],
    axis=1
)

# Cumulative sum of Signed Amount
df['Cumulative Amount'] = df['Signed Amount'].cumsum()

# --- Summary Statistics ---
total_intake = df[df['Type'].str.strip().str.lower() == 'intake']['Amount'].sum()
total_emptying = df[df['Type'].str.strip().str.lower() == 'emptying']['Amount'].sum()
net_change = df['Signed Amount'].sum()

# Calculate time since last emptying (using Copenhagen time)
cph_tz = pytz.timezone('Europe/Copenhagen')
now_cph = datetime.now(cph_tz)
emptying_df = df[df['Type'].str.strip().str.lower() == 'emptying']
if not emptying_df.empty:
    last_emptying_time = emptying_df['DateTime'].max()
    # Localize last_emptying_time to Copenhagen if not already
    if last_emptying_time.tzinfo is None:
        last_emptying_time = cph_tz.localize(last_emptying_time)
    time_since_emptying = now_cph - last_emptying_time
    hours, remainder = divmod(time_since_emptying.total_seconds(), 3600)
    minutes = remainder // 60
    time_since_str = f"{int(hours)}h {int(minutes)}m"
else:
    time_since_str = "N/A"

# Streamlit UI
st.title("Bladder Volume Tracker")
st.markdown("#### Cumulative bladder volume (select time range in sidebar)")
st.write(
    "*Entries are shown from your selected date and time. "
    "Intake and emptying events are color-coded. Laying periods are highlighted in orange.*"
)

# Responsive columns for summary stats
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Summary")
        st.metric("Total Intake (ml)", f"{total_intake:.0f}")
        st.metric("Total Emptying (ml)", f"{total_emptying:.0f}")
        st.metric("Net Change (ml)", f"{net_change:.0f}")
        st.metric("Time since last emptying", time_since_str)
    with col2:
        st.subheader("Legend")
        st.markdown(
            """
            <span style='color:green'>● Emptying</span><br>
            <span style='color:blue'>● Intake (Water)</span><br>
            <span style='color:red'>● Intake (Coffee)</span><br>
            <span style='color:#0072B2'>● Other</span><br>
            <span style='background-color:orange; color:black'>Laying Down</span>
            """,
            unsafe_allow_html=True
        )

# Make the plot smaller and more readable for mobile
fig, ax = plt.subplots(figsize=(6, 3))

# Prepare colors for each event type for both line and dots
colors = []
for idx, row in df.iterrows():
    if str(row['Type']).strip().lower() == 'emptying':
        colors.append('green')
    elif str(row['Type']).strip().lower() == 'intake':
        intake_type = str(row['Intake Type']).strip().lower()
        if intake_type == 'water':
            colors.append('blue')
        elif intake_type == 'coffee':
            colors.append('red')
        else:
            colors.append('#0072B2')  # default for other intake types
    else:
        colors.append('#0072B2')  # default for other types

# Plot cumulative amount as a colored line and dots
for i in range(1, len(df)):
    ax.plot(df['DateTime'].iloc[i-1:i+1], df['Cumulative Amount'].iloc[i-1:i+1],
            color=colors[i], linewidth=2)
for i in range(len(df)):
    ax.plot(df['DateTime'].iloc[i], df['Cumulative Amount'].iloc[i],
            marker='o', color=colors[i], markersize=8)

# Add marks for laying periods
for idx, row in laying_rows.iterrows():
    start_time = row['DateTime']
    end_time_str = str(row['Laying - End']).strip()
    if end_time_str:
        # Use full date format
        end_time = pd.to_datetime(str(row['Date']) + ' ' + end_time_str, format='%d/%m/%Y %H:%M', errors='coerce')
        if pd.notnull(end_time):
            ax.axvspan(start_time, end_time, color='orange', alpha=0.2, label='Laying Down' if idx == laying_rows.index[0] else None)
            ax.text(start_time, ax.get_ylim()[1]*0.95, 'Laying', color='orange', fontsize=10, verticalalignment='top', rotation=90)

ax.set_xlabel('Time', fontsize=10)
ax.set_ylabel('Cumulative Amount (ml)', fontsize=10)
ax.set_title('Bladder Volume vs Time', fontsize=12, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Show only time
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
ax.grid(True, linestyle='--', alpha=0.6)
fig.tight_layout()
st.pyplot(fig, width='stretch')

# --- Event Table ---
st.subheader("Recent Events")
show_cols = ['DateTime', 'Type', 'Amount', 'Intake Type', 'Emptying Type', 'Laying - End']
df_display = df_raw[
    (df_raw['DateTime'] >= earliest_time) & (df_raw['DateTime'] <= latest_time)
][show_cols].sort_values('DateTime')
df_display['DateTime'] = df_display['DateTime'].dt.strftime('%d/%m/%Y %H:%M')
st.dataframe(
    df_display,
    width='stretch',
    height=300  # limit height for mobile scroll
)

# Add a note for mobile users
st.markdown(
    "<div style='font-size: 14px; color: #666; text-align: center;'>"
    "Tip: You can scroll the table sideways and pinch-to-zoom the plot on your phone."
    "</div>",
    unsafe_allow_html=True
)

bad_rows = df_raw[df_raw['DateTime'].isna()]
if not bad_rows.empty:
    st.warning("Some rows have invalid date/time format and were skipped.")
    st.write(bad_rows)