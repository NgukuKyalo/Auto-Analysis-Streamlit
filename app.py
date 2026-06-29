import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="Auto Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables to hold our data
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'filename' not in st.session_state:
    st.session_state['filename'] = ""

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
@st.cache_data
def load_csv_data(file):
    """Caches the dataset loading to improve performance."""
    return pd.read_csv(file)

def display_kpi_cards(dataframe):
    """Displays key performance indicators (KPIs) in a 4-column layout."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Rows", value=dataframe.shape[0])
    with col2:
        st.metric(label="Total Columns", value=dataframe.shape[1])
    with col3:
        missing_total = dataframe.isnull().sum().sum()
        st.metric(label="Missing Values", value=missing_total)
    with col4:
        duplicate_total = dataframe.duplicated().sum()
        st.metric(label="Duplicate Rows", value=duplicate_total)

def get_numeric_categorical_columns(dataframe):
    """Returns lists of numeric and categorical columns."""
    numeric_cols = dataframe.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = dataframe.select_dtypes(exclude=np.number).columns.tolist()
    return numeric_cols, categorical_cols

def aggregate_data(df, group_col, target_col, agg_method):
    """Helper function to aggregate data for charts."""
    if agg_method == "Count":
        # For count, we just size the groups
        return df.groupby(group_col).size().reset_index(name='Count')
    else:
        # For Sum, Mean, Min, Max
        agg_func = agg_method.lower()
        return df.groupby(group_col)[target_col].agg(agg_func).reset_index()

# ==========================================
# 3. SIDEBAR NAVIGATION & DATA UPLOAD
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
    ["Home", "Data Overview", "Data Cleaning", "Visualizations", "Correlation Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Handle file upload and reset data if a new file is uploaded
if uploaded_file is not None:
    if st.session_state['filename'] != uploaded_file.name:
        # Load new file using cached function
        df_loaded = load_csv_data(uploaded_file)
        st.session_state['raw_data'] = df_loaded.copy()
        st.session_state['df'] = df_loaded.copy()
        st.session_state['filename'] = uploaded_file.name
        st.sidebar.success("Dataset loaded successfully!")

# Ensure we have data before rendering the pages
if st.session_state['df'] is not None:
    df = st.session_state['df']
    
    # Download Cleaned Data Button in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("Export")
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Cleaned CSV",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )

    # ==========================================
    # PAGE: HOME
    # ==========================================
    if page == "Home":
        st.title("📊 Auto Analytics Dashboard")
        st.subheader("Upload a dataset and generate insights automatically")
        
        tab1, tab2, tab3 = st.tabs(["Quick Snapshot", "Dataset Health", "Data Preview"])
        
        with tab1:
            st.markdown("### Core Metrics")
            display_kpi_cards(df)
            
        with tab2:
            st.markdown("### Dataset Health Metrics")
            h_col1, h_col2, h_col3, h_col4 = st.columns(4)
            
            total_cells = df.shape[0] * df.shape[1]
            missing_pct = (df.isnull().sum().sum() / total_cells) * 100 if total_cells > 0 else 0
            dup_pct = (df.duplicated().sum() / df.shape[0]) * 100 if df.shape[0] > 0 else 0
            num_cols, cat_cols = get_numeric_categorical_columns(df)
            
            h_col1.metric("Missing Data %", f"{missing_pct:.2f}%")
            h_col2.metric("Duplicate Rows %", f"{dup_pct:.2f}%")
            h_col3.metric("Numeric Columns", len(num_cols))
            h_col4.metric("Categorical Columns", len(cat_cols))
            
        with tab3:
            st.markdown("### Dataset Preview")
            st.dataframe(df.head(15), use_container_width=True)

    # ==========================================
    # PAGE: DATA OVERVIEW
    # ==========================================
    elif page == "Data Overview":
        st.title("📋 Data Overview")
        display_kpi_cards(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Column Data Types")
            dtypes_df = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index()
            dtypes_df.rename(columns={'index': 'Column Name'}, inplace=True)
            dtypes_df['Data Type'] = dtypes_df['Data Type'].astype(str)
            st.dataframe(dtypes_df, use_container_width=True)
            
        with col2:
            st.markdown("### Missing Values per Column")
            missing_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Count']).reset_index()
            missing_df.rename(columns={'index': 'Column Name'}, inplace=True)
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            
        st.markdown("---")
        st.markdown("### Summary Statistics")
        stat_tab1, stat_tab2 = st.tabs(["Numeric Columns", "Categorical Columns"])
        
        with stat_tab1:
            num_cols, _ = get_numeric_categorical_columns(df)
            if num_cols:
                st.dataframe(df.describe(), use_container_width=True)
            else:
                st.info("No numeric columns found.")
                
        with stat_tab2:
            _, cat_cols = get_numeric_categorical_columns(df)
            if cat_cols:
                st.dataframe(df.describe(include=['object', 'category']), use_container_width=True)
            else:
                st.info("No categorical columns found.")

    # ==========================================
    # PAGE: DATA CLEANING
    # ==========================================
    elif page == "Data Cleaning":
        st.title("🧹 Data Cleaning")
        
        st.markdown("**Current Shape:** {} rows, {} columns | **Missing:** {} | **Duplicates:** {}".format(
            df.shape[0], df.shape[1], df.isnull().sum().sum(), df.duplicated().sum()
        ))
        
        clean_tab1, clean_tab2, clean_tab3 = st.tabs(["Row Operations", "Column Imputation", "Column Operations"])
        
        with clean_tab1:
            st.subheader("Row Operations")
            if st.button("Drop Duplicate Rows"):
                st.session_state['df'] = df.drop_duplicates()
                st.rerun()
            if st.button("Drop Rows with ANY Missing Values"):
                st.session_state['df'] = df.dropna()
                st.rerun()
                
        with clean_tab2:
            st.subheader("Impute Missing Values (Column Level)")
            target_col = st.selectbox("Select Column to Impute", df.columns.tolist())
            
            if target_col:
                is_numeric = target_col in get_numeric_categorical_columns(df)[0]
                missing_in_col = df[target_col].isnull().sum()
                st.write(f"Missing values in **{target_col}**: {missing_in_col}")
                
                if missing_in_col > 0:
                    if is_numeric:
                        method = st.selectbox("Imputation Method", ["Mean", "Median", "Mode", "Custom Value"])
                    else:
                        method = st.selectbox("Imputation Method", ["Mode", "Custom Value"])
                        
                    custom_val = None
                    if method == "Custom Value":
                        custom_val = st.text_input("Enter custom value")
                        
                    if st.button(f"Apply {method} to {target_col}"):
                        if method == "Mean":
                            fill_val = df[target_col].mean()
                        elif method == "Median":
                            fill_val = df[target_col].median()
                        elif method == "Mode":
                            fill_val = df[target_col].mode()[0]
                        else:
                            fill_val = float(custom_val) if is_numeric and custom_val.isnumeric() else custom_val
                            
                        # Avoid chained assignment
                        st.session_state['df'][target_col] = df[target_col].fillna(fill_val)
                        st.success(f"Filled missing values in {target_col} with {fill_val}")
                        st.rerun()
                else:
                    st.success("No missing values in this column!")

        with clean_tab3:
            st.subheader("Manage Columns")
            c_col1, c_col2 = st.columns(2)
            
            with c_col1:
                st.markdown("**Rename / Drop**")
                col_to_mod = st.selectbox("Select Column", df.columns.tolist(), key="mod_col")
                new_name = st.text_input("New Column Name")
                
                if st.button("Rename Column") and new_name:
                    st.session_state['df'].rename(columns={col_to_mod: new_name}, inplace=True)
                    st.rerun()
                    
                if st.button("Drop Column"):
                    st.session_state['df'].drop(columns=[col_to_mod], inplace=True)
                    st.rerun()
                    
            with c_col2:
                st.markdown("**Change Data Type**")
                col_to_type = st.selectbox("Select Column to Convert", df.columns.tolist(), key="type_col")
                new_type = st.selectbox("New Data Type", ["String", "Integer", "Float", "Datetime"])
                
                if st.button("Convert Type"):
                    try:
                        if new_type == "String":
                            st.session_state['df'][col_to_type] = df[col_to_type].astype(str)
                        elif new_type == "Integer":
                            st.session_state['df'][col_to_type] = df[col_to_type].astype(int)
                        elif new_type == "Float":
                            st.session_state['df'][col_to_type] = df[col_to_type].astype(float)
                        elif new_type == "Datetime":
                            st.session_state['df'][col_to_type] = pd.to_datetime(df[col_to_type])
                        st.success(f"Converted {col_to_type} to {new_type}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Conversion failed: {e}")

        st.markdown("---")
        if st.button("⚠️ Reset to Original Uploaded Data"):
            st.session_state['df'] = st.session_state['raw_data'].copy()
            st.rerun()

    # ==========================================
    # PAGE: VISUALIZATIONS
    # ==========================================
    elif page == "Visualizations":
        st.title("📈 Automatic Visualizations")
        
        num_cols, cat_cols = get_numeric_categorical_columns(df)
        filtered_df = df.copy()

        # --- MULTI-COLUMN INTERACTIVE FILTERS ---
        with st.expander("🔍 Interactive Data Filters", expanded=False):
            f_col1, f_col2 = st.columns(2)
            
            with f_col1:
                st.markdown("**Categorical Filters**")
                selected_cat_filters = st.multiselect("Add Categorical Filter", cat_cols)
                for cat_col in selected_cat_filters:
                    unique_vals = df[cat_col].dropna().unique()
                    selected_vals = st.multiselect(f"Values for {cat_col}", unique_vals, default=unique_vals)
                    filtered_df = filtered_df[filtered_df[cat_col].isin(selected_vals)]
            
            with f_col2:
                st.markdown("**Numeric Filters**")
                selected_num_filters = st.multiselect("Add Numeric Filter", num_cols)
                for num_col in selected_num_filters:
                    min_val, max_val = float(df[num_col].min()), float(df[num_col].max())
                    if min_val < max_val:
                        range_vals = st.slider(f"Range for {num_col}", min_val, max_val, (min_val, max_val))
                        filtered_df = filtered_df[
                            (filtered_df[num_col] >= range_vals[0]) & 
                            (filtered_df[num_col] <= range_vals[1])
                        ]
                    else:
                        st.info(f"{num_col} has a constant value.")
        
        st.markdown(f"*(Active Dataset: **{filtered_df.shape[0]}** rows)*")
        st.markdown("---")

        # --- BI-STYLE WORKFLOW: Settings on left, Chart on right ---
        viz_settings, viz_chart = st.columns([1, 3])
        
        with viz_settings:
            st.subheader("Chart Builder")
            chart_type = st.selectbox("Chart Type", 
                ["Bar Chart", "Line Chart", "Area Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot"]
            )
            
        with viz_chart:
            # We construct the chart inside this block based on settings in the left column
            if chart_type in ["Bar Chart", "Pie Chart"]:
                with viz_settings:
                    cat_col = st.selectbox("Category (X/Labels)", cat_cols if cat_cols else df.columns)
                    num_col = st.selectbox("Measure (Y/Values)", ["Count (No Measure)"] + num_cols)
                    agg = "Count"
                    if num_col != "Count (No Measure)":
                        agg = st.selectbox("Aggregation", ["Sum", "Mean", "Count", "Min", "Max"])
                
                if cat_col:
                    if num_col == "Count (No Measure)":
                        agg_df = aggregate_data(filtered_df, cat_col, cat_col, "Count")
                        plot_y = "Count"
                    else:
                        agg_df = aggregate_data(filtered_df, cat_col, num_col, agg)
                        plot_y = num_col

                    if chart_type == "Bar Chart":
                        fig = px.bar(agg_df, x=cat_col, y=plot_y, template="plotly_white", 
                                     title=f"{agg} of {plot_y} by {cat_col}")
                    else: # Pie Chart
                        fig = px.pie(agg_df, names=cat_col, values=plot_y, template="plotly_white", 
                                     title=f"{agg} of {plot_y} by {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type in ["Line Chart", "Area Chart"]:
                with viz_settings:
                    x_col = st.selectbox("X-Axis (Time/Sequence)", df.columns)
                    y_col = st.selectbox("Y-Axis (Measure)", num_cols)
                    agg = st.selectbox("Aggregation", ["Sum", "Mean", "Count", "Min", "Max"])
                
                if x_col and y_col:
                    agg_df = aggregate_data(filtered_df, x_col, y_col, agg).sort_values(by=x_col)
                    if chart_type == "Line Chart":
                        fig = px.line(agg_df, x=x_col, y=y_col, template="plotly_white", 
                                      title=f"{agg} of {y_col} over {x_col}")
                    else:
                        fig = px.area(agg_df, x=x_col, y=y_col, template="plotly_white", 
                                      title=f"{agg} of {y_col} over {x_col}")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Scatter Plot":
                with viz_settings:
                    x_col = st.selectbox("X-Axis", num_cols, index=0 if len(num_cols) > 0 else None)
                    y_col = st.selectbox("Y-Axis", num_cols, index=1 if len(num_cols) > 1 else 0)
                    color_col = st.selectbox("Color By (Optional)", ["None"] + df.columns.tolist())
                    size_col = st.selectbox("Size By (Optional)", ["None"] + num_cols)
                
                if x_col and y_col:
                    kwargs = {'x': x_col, 'y': y_col, 'template': "plotly_white", 'title': f"Scatter: {x_col} vs {y_col}"}
                    if color_col != "None": kwargs['color'] = color_col
                    if size_col != "None": kwargs['size'] = size_col
                    
                    fig = px.scatter(filtered_df, **kwargs)
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Histogram":
                with viz_settings:
                    num_col = st.selectbox("Numeric Column", num_cols)
                if num_col:
                    fig = px.histogram(filtered_df, x=num_col, template="plotly_white", title=f"Distribution of {num_col}")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Box Plot":
                with viz_settings:
                    num_col = st.selectbox("Numeric Column", num_cols)
                    cat_col = st.selectbox("Split by Category (Optional)", ["None"] + cat_cols)
                if num_col:
                    kwargs = {'y': num_col, 'template': "plotly_white", 'title': f"Box Plot of {num_col}"}
                    if cat_col != "None": kwargs['x'] = cat_col
                    fig = px.box(filtered_df, **kwargs)
                    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # PAGE: CORRELATION ANALYSIS
    # ==========================================
    elif page == "Correlation Analysis":
        st.title("🔗 Correlation Analysis")
        
        num_cols, _ = get_numeric_categorical_columns(df)
        
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns to calculate correlations.")
        else:
            with st.expander("📖 How to Interpret Correlation", expanded=False):
                st.markdown("""
                **Correlation** measures the strength and direction of the relationship between two numbers, ranging from **-1.0 to 1.0**.
                * **1.0**: Perfect positive relationship (As X goes up, Y goes up).
                * **-1.0**: Perfect negative relationship (As X goes up, Y goes down).
                * **0.0**: No linear relationship at all.
                
                **Strength Guide (Absolute Value):**
                * **0.80 to 1.00**: Very Strong
                * **0.60 to 0.79**: Strong
                * **0.40 to 0.59**: Moderate
                * **0.20 to 0.39**: Weak
                * **0.00 to 0.19**: Very Weak
                """)

            corr_matrix = df[num_cols].corr()
            
            st.markdown("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax, vmin=-1, vmax=1)
            st.pyplot(fig)
            
            st.markdown("---")
            st.markdown("### Strongest Correlations")
            
            # Flatten matrix, remove self-correlations and duplicates
            unstacked = corr_matrix.unstack().dropna()
            unstacked = unstacked[unstacked != 1.0]
            
            # Create a dataframe for easier dropping of reciprocal duplicates
            corr_df = unstacked.reset_index()
            corr_df.columns = ['Var 1', 'Var 2', 'Correlation']
            
            # Create a sorted tuple of the variables to identify duplicates like (A,B) and (B,A)
            corr_df['pair'] = corr_df.apply(lambda row: tuple(sorted([row['Var 1'], row['Var 2']])), axis=1)
            corr_df = corr_df.drop_duplicates(subset=['pair']).drop(columns=['pair'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Positive Correlations**")
                pos_corr = corr_df.sort_values(by='Correlation', ascending=False).head(5)
                st.dataframe(pos_corr, use_container_width=True, hide_index=True)
                
            with col2:
                st.markdown("**Top Negative Correlations**")
                neg_corr = corr_df.sort_values(by='Correlation', ascending=True).head(5)
                st.dataframe(neg_corr, use_container_width=True, hide_index=True)

else:
    st.title("📊 Auto Analytics Dashboard")
    st.info("👈 Please upload a CSV file from the sidebar to begin.")


