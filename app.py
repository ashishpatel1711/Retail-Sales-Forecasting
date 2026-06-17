import streamlit as st
import pandas as pd
import joblib
import datetime

# --- Load the Model and Columns ---
@st.cache_resource
def load_model():
    model = joblib.load('demand_model.pkl')
    cols = joblib.load('model_columns.pkl')
    return model, cols

model, model_columns = load_model()

# --- Extract Dropdown Options from Saved Columns ---
# This looks at your training columns (e.g., 'store_id_store_1') and extracts just the 'store_1' part
store_options = [col.replace('store_id_', '') for col in model_columns if col.startswith('store_id_')]
item_options = [col.replace('item_id_', '') for col in model_columns if col.startswith('item_id_')]

# Fallback just in case your columns were named differently
if not store_options: store_options = [f"store_{i}" for i in range(1, 51)] 
if not item_options: item_options = [f"item_{i}" for i in range(1, 51)]

# --- Initialize Memory (Session State) ---
# This keeps track of the rows you add without resetting when the page updates
if 'input_rows' not in st.session_state:
    st.session_state.input_rows = pd.DataFrame(columns=['store_id', 'item_id', 'sales_lag_1', 'sales_lag_7'])

# --- Webpage UI ---
st.title("📦 Dynamic Demand Forecasting")

# 1. Date Selection (Applies to the whole batch)
st.subheader("📅 Forecast Date")
selected_date = st.date_input("Select the date you are forecasting for", datetime.date.today())

month = selected_date.month
# week_of_year = selected_date.isocalendar()[1] # Uncomment if your model uses 'week'
day_of_week = selected_date.weekday() # Monday=0, Sunday=6
is_weekend = 1 if day_of_week in [5, 6] else 0

st.divider()

# 2. The Search & Add Section
st.subheader("🔍 Search and Add to Forecast Batch")
with st.form("add_row_form"):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Selectbox automatically allows typing to search!
        selected_store = st.selectbox("Search Store", options=store_options)
    with col2:
        selected_item = st.selectbox("Search Item", options=item_options)
    with col3:
        lag_1 = st.number_input("Yesterday's Sales", min_value=0, value=20)
    with col4:
        lag_7 = st.number_input("Last Week's Sales", min_value=0, value=25)
    
    add_button = st.form_submit_button("➕ Add to List")

    if add_button:
        # Append the new selection to our running list
        new_row = pd.DataFrame({
            'store_id': [selected_store],
            'item_id': [selected_item],
            'sales_lag_1': [lag_1],
            'sales_lag_7': [lag_7]
        })
        st.session_state.input_rows = pd.concat([st.session_state.input_rows, new_row], ignore_index=True)
        st.success(f"Added {selected_store} - {selected_item}!")

# 3. Display the Rows and Forecast
st.subheader("📋 Items to Forecast")

if not st.session_state.input_rows.empty:
    # Use data_editor so you can manually delete rows or edit lag values before predicting!
    edited_df = st.data_editor(st.session_state.input_rows, num_rows="dynamic", use_container_width=True)
    
    col_a, col_b = st.columns([1, 4])
    with col_a:
        run_forecast = st.button("🚀 Run Forecast", type="primary")
    with col_b:
        if st.button("🗑️ Clear List"):
            st.session_state.input_rows = pd.DataFrame(columns=['store_id', 'item_id', 'sales_lag_1', 'sales_lag_7'])
            st.rerun()

    if run_forecast:
        # Prepare the data matrix for the model
        forecast_df = edited_df.copy()
        forecast_df['month'] = month
        forecast_df['day_of_week'] = day_of_week
        forecast_df['is_weekend'] = is_weekend
        # forecast_df['week'] = week_of_year # Uncomment if your model uses 'week'

        # One-hot encode the whole table at once
        input_encoded = pd.get_dummies(forecast_df, columns=['store_id', 'item_id'])
        
        # Align with the 108 columns the model expects
        input_final = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        # 🛠️ THE FIX: Force all columns to be numeric so LightGBM doesn't crash
        input_final = input_final.astype(float)
        
        # Run bulk prediction
        predictions = model.predict(input_final)
        
        # Display Results
        st.success("✅ Forecast Complete!")
        
        # Create a clean results table
        results_df = edited_df[['store_id', 'item_id']].copy()
        results_df['Predicted Sales Units'] = predictions.round(2)
        
        st.dataframe(results_df, use_container_width=True)

else:
    st.info("Your list is empty. Use the search bars above to add stores and items!")