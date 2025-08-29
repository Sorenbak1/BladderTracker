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
    format='%d/%m/%Y %H:%M'
)

# Remove rows with missing or invalid DateTime
df_raw = df_raw.dropna(subset=['DateTime'])

# Extract laying periods before filtering
laying_rows = df_raw[df_raw['Type'].astype(str).str.strip().str.lower() == 'laying down']

# Now continue with your filtered DataFrame for plotting
df = df_raw[df_raw['Amount'].astype(str).str.isnumeric()]
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

# Streamlit UI
st.title("Bladder Volume Tracker")
st.markdown("#### Cumulative bladder volume since selected start time")
st.write("*Entries are shown starting from your selected date and time. Intake and emptying events are color-coded. Laying periods are highlighted in orange.*")

fig, ax = plt.subplots(figsize=(10, 5))

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

# Add legend manually
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='green', marker='o', linestyle='-', label='Emptying'),
    Line2D([0], [0], color='blue', marker='o', linestyle='-', label='Intake (Water)'),
    Line2D([0], [0], color='red', marker='o', linestyle='-', label='Intake (Coffee)'),
    Line2D([0], [0], color='#0072B2', marker='o', linestyle='-', label='Other')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

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

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Cumulative Amount (ml)', fontsize=12)
ax.set_title('Cumulative Bladder Volume vs Time', fontsize=14, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Show only time
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)
fig.tight_layout()
st.pyplot(fig)

# --- Summary Statistics ---
total_intake = df[df['Type'].str.strip().str.lower() == 'intake']['Amount'].sum()
total_emptying = df[df['Type'].str.strip().str.lower() == 'emptying']['Amount'].sum()
net_change = df['Signed Amount'].sum()

# Calculate time since last emptying (using Copenhagen time)
cph_tz = pytz.timezone('Europe/Copenhagen')
now_cph = datetime.now(cph_tz)
if not df[df['Type'].str.strip().str.lower() == 'emptying'].empty:
    last_emptying_time = df[df['Type'].str.strip().str.lower() == 'emptying']['DateTime'].max()
    # Localize last_emptying_time to Copenhagen if not already
    if last_emptying_time.tzinfo is None:
        last_emptying_time = cph_tz.localize(last_emptying_time)
    time_since_emptying = now_cph - last_emptying_time
    hours, remainder = divmod(time_since_emptying.total_seconds(), 3600)
    minutes = remainder // 60
    time_since_str = f"{int(hours)}h {int(minutes)}m"
else:
    time_since_str = "N/A"

st.subheader("Summary Statistics (since selected start and end time)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Intake (ml)", f"{total_intake:.0f}")
col2.metric("Total Emptying (ml)", f"{total_emptying:.0f}")
col3.metric("Net Change (ml)", f"{net_change:.0f}")
col4.metric("Time since last emptying", time_since_str)

# --- Event Table ---
st.subheader("Recent Events (since selected start and end time)")
show_cols = ['DateTime', 'Type', 'Amount', 'Intake Type', 'Emptying Type', 'Laying - End']
df_display = df_raw[(df_raw['DateTime'] >= earliest_time) & (df_raw['DateTime'] <= latest_time)][show_cols].sort_values('DateTime')
df_display['DateTime'] = df_display['DateTime'].dt.strftime('%d/%m %H:%M')
st.dataframe(df_display, use_container_width=True)