import io
import streamlit as st
import pandas as pd
import csv
import re
import plotly.express as px
from scipy.stats import skew
from prompt_models import *
from execution_programs import *
from llm_models import text_llm, code_llm

st.set_page_config(layout="wide")
    
# Custom style CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 0rem;
        padding-right: 0rem;
    }
    .element-container:has(.stColumn) {
        gap: 0rem !important;
    }
    div[data-testid="column"] {
        padding-left: 0rem !important;
        padding-right: 0rem !important;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
    }
    .cell {
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        background-color: #f9f9f9;
    }
    .output {
        border-left: 3px solid #4285f4;
        padding-left: 1rem;
        margin-top: 0.5rem;
    }
    .input-area {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .stButton > button {
        padding: 0rem 0.5rem; 
        border-radius: 5px; 
        font-size: 16px; 
        font-weight: bold; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); 
        transition: all 0.3s ease; 
    }

    .stButton > button:hover {
        background-color: #357ae8; 
        border-color: #357ae8; 
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); 
    }
    .history-item {
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #262626;
        align-self: flex-start;  
    }
    .chat-message.assistant {
        background-color: #262626;
        align-self: flex-end; 
    }
    .chat-message {
            font-size: 14px; 
    }
    .chat-message.user {
            font-size: 16px;  
    }
    .chat-message.assistant {
            font-size: 14px; 
    }
    .sidebar {
        width: 300px;
        padding-right: 2rem;
    }
    .delete-btn {
        background-color: #ff4b4b !important;
        color: white !important;
        border: none !important;
        padding: 0.25rem 0.5rem !important;
        font-size: 0.8rem !important;
        margin-top: 0.5rem;
    }
    .delete-btn:hover {
        background-color: #ff0000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'execution_history' not in st.session_state:
    st.session_state.execution_history = []
if 'df' not in st.session_state:
    st.session_state.df = None


# Main app layout
col1a, col2a = st.columns([4, 7], gap="small")
st.markdown("<div style='margin-top: 5px'></div>", unsafe_allow_html=True)
show_col1 = st.toggle("Ask chat bot", value=True)

if show_col1:
    col1a, col2a = st.columns([4,7]) 
else:
    _ , col2a = st.columns([0.0001, 2]) 
    
with col1a:
  if show_col1:
    with st.container(height=570):
        st.header("💬 Chatbot Assistant")
        
        # Initial state
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = {
                'chat_history': [],
                'prev_df': None,
                'prev_raw_dataframes': None
            }
        
        # button clear all chat
        if st.button("🗑️ Clear all chat", key="clear_all_chat"):
            st.session_state.chatbot['chat_history'] = []
        
        # show history chatbot
        for i in range(0, len(st.session_state.chatbot['chat_history']), 2):
            user_msg = st.session_state.chatbot['chat_history'][i]
            assistant_msg = st.session_state.chatbot['chat_history'][i+1] if i+1 < len(st.session_state.chatbot['chat_history']) else None

            with st.expander(f"Chat {i//2 + 1}", expanded=False):
                with st.chat_message("user"):
                    st.markdown(user_msg["content"])
                if assistant_msg:
                    with st.chat_message("assistant"):
                        st.markdown(assistant_msg["content"])

        
        # Input chat
        user_query = st.chat_input("Ask coding questions...", key="chatbot_input")

        if user_query:
            # save previous state
            st.session_state.chatbot['prev_df'] = st.session_state.get('df')
            st.session_state.chatbot['prev_raw_dataframes'] = st.session_state.get('raw_dataframes')
            
            # show chat user
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # add to history chat
            st.session_state.chatbot['chat_history'].append({"role": "user", "content": user_query})
            
            with st.spinner("Thinking..."):
                if "df" in st.session_state:
                    response = text_llm(prompt_chatbot(user_query,st.session_state.df.columns, st.session_state.df.shape, st.session_state.df.dtypes))
                    clean_response = re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>', '', response, flags=re.DOTALL | re.IGNORECASE)
                    
                    # show and save chatbot response
                    with st.chat_message("assistant"):
                        st.markdown(clean_response)
                    st.session_state.chatbot['chat_history'].append({"role": "assistant", "content": clean_response})
                else:
                    st.write('You need to upload the data first')

with col2a:  # Sidebar for chatbot
    with st.container(height=570):
        st.header("📊 Chatbot Data Analysis Coder")
        st.markdown("Analyze your CSV data with LLM Models")

        def reset_data_state():
            st.session_state.df = None
            st.session_state.raw_dataframes = None
            st.session_state.preview_df = None
            st.session_state.file_names = [f.name for f in uploaded_files] if uploaded_files else []

        # Initialize session state if not already present
        if 'df' not in st.session_state:
            st.session_state.df = None

        uploaded_files = st.file_uploader(
            "Upload CSV file(s)", 
            type=["csv"], 
            accept_multiple_files=True,
            help="Upload one or more CSV files to combine"
        )

        if uploaded_files:
            # Reset processing if files change
            if 'prev_uploaded_files' not in st.session_state or st.session_state.prev_uploaded_files != [f.name for f in uploaded_files]:
                st.session_state.df = None
                st.session_state.raw_dataframes = None
                st.session_state.preview_df = None
                st.session_state.prev_uploaded_files = [f.name for f in uploaded_files]
            
            # Process files but don't merge yet
            if st.session_state.raw_dataframes is None:
                dataframes = []
                processed_files = 0
                
                for uploaded_file in uploaded_files:
                    with st.expander(f"📄 File: {uploaded_file.name}", expanded=False):
                        try:
                            # Try to detect delimiter
                            raw_data = uploaded_file.getvalue().decode("utf-8")
                            sniffer = csv.Sniffer()
                            
                            try:
                                dialect = sniffer.sniff(raw_data.splitlines()[0])
                                delimiter = dialect.delimiter
                            except (csv.Error, IndexError):
                                # Try common delimiters if sniffing fails
                                for test_delim in [',', ';', '\t', '|']:
                                    if test_delim in raw_data:
                                        delimiter = test_delim
                                        break
                                else:
                                    delimiter = ','  # default
                            
                            uploaded_file.seek(0)  # Reset file pointer
                            
                            # Read CSV with error handling
                            try:
                                df = pd.read_csv(uploaded_file, delimiter=delimiter, on_bad_lines='warn')
                            except Exception as e:
                                st.warning(f"Using Python engine for {uploaded_file.name} due to parsing issues")
                                df = pd.read_csv(uploaded_file, delimiter=delimiter, engine='python')
                            
                            # Display file info
                            st.write(f"Shape: {df.shape}")
                            st.write(f"Detected delimiter: '{delimiter}'")
                            st.dataframe(df.head(5), use_container_width=True)
                            
                            dataframes.append(df)
                            processed_files += 1
                            
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                            continue
                
                if processed_files == 0:
                    st.error("No files were successfully processed")
                    st.stop()
                
                # Store raw dataframes in session state
                st.session_state.raw_dataframes = dataframes
            
            # Only proceed if we have raw dataframes
            if st.session_state.raw_dataframes:
                dataframes = st.session_state.raw_dataframes
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    merge_option = st.radio(
                        "Merge method:",
                        ("single data","Horizontal (concat columns)", "Vertical (concat rows)", "Horizontal (merge on common columns)"),
                        index=0,
                        horizontal=True,
                        key="merge_option"
                    )
                    
                    # Additional merge parameters
                    if merge_option == "Horizontal (merge on common columns)":
                        if len(dataframes) > 0:
                            common_cols = set(dataframes[0].columns)
                            for df in dataframes[1:]:
                                common_cols.intersection_update(df.columns)
                            
                            if common_cols:
                                selected_cols = st.multiselect(
                                    "Select columns to merge on:",
                                    options=list(common_cols),
                                    default=list(common_cols),
                                    key="merge_columns"
                                )
                                how_merge = st.selectbox(
                                    "Merge type:",
                                    ["inner", "outer", "left", "right"],
                                    index=1,
                                    key="how_merge"
                                )
                            else:
                                None
                
                with col2:
                    preview_button = st.button("🔄 Preview")
                    confirm_button = st.button("✅ Confirm")
                    reset_btn = st.button("🔄 Reset")
                
                # Reset logic
                if reset_btn:
                    reset_data_state()
                    st.rerun()
                
                # Preview logic
                if preview_button:
                    with st.spinner("Generating preview..."):
                        try:
                            if merge_option == "single data":
                                preview_df = dataframes[0]
                            elif merge_option == "Horizontal (concat columns)":
                                # Find common indices across ALL dataframes
                                common_indices = dataframes[0].index
                                for df in dataframes[1:]:
                                    common_indices = common_indices.intersection(df.index)
                                
                                if len(common_indices) == 0:
                                    st.error("No common indices found across all dataframes")
                                    st.stop()
                                
                                # Filter all dataframes to only common indices
                                filtered_dfs = [df.loc[common_indices] for df in dataframes]
                                
                                # Handle duplicate column names
                                all_columns = []
                                duplicate_counter = {}
                                
                                for i, df in enumerate(filtered_dfs):
                                    new_columns = []
                                    for col in df.columns:
                                        if col in all_columns:
                                            if col not in duplicate_counter:
                                                duplicate_counter[col] = 1
                                            duplicate_counter[col] += 1
                                            new_col = f"{col}_df{duplicate_counter[col]}"
                                            new_columns.append(new_col)
                                        else:
                                            new_columns.append(col)
                                    all_columns.extend(new_columns)
                                    filtered_dfs[i].columns = new_columns
                                
                                # Concatenate horizontally
                                preview_df = pd.concat(filtered_dfs, axis=1)
                                
                            elif merge_option == "Vertical (concat rows)":
                                preview_df = pd.concat(dataframes, axis=0, ignore_index=True)
                                
                            elif merge_option == "Horizontal (merge on common columns)":
                                def auto_merge_many(dfs):
                                    if not dfs:
                                        return None
                                    
                                    merged_df = dfs[0]
                                    
                                    for next_df in dfs[1:]:
                                        common_keys = list(set(merged_df.columns) & set(next_df.columns))
                                        if common_keys:
                                            merged_df = pd.merge(merged_df, next_df, on=common_keys)
                                        else:
                                            print(f"[WARNING] There is no matching key between:\n{merged_df.columns}\n&\n{next_df.columns}")
                                    
                                    return merged_df
                                
                                preview_df = auto_merge_many(dataframes)
                            
                            st.session_state.preview_df = preview_df
                            st.success("Preview generated!")
                            
                            # Show preview stats
                            st.subheader("Preview Results")
                            st.write(f"Shape: {preview_df.shape}")
                            
                            # Show sample data with tabs
                            tab1, tab2 = st.tabs(["First Rows", "Last Rows"])
                            with tab1:
                                st.dataframe(preview_df.head(20), use_container_width=True)
                            with tab2:
                                st.dataframe(preview_df.tail(20), use_container_width=True)
                            
                            # Show quick stats
                            with st.expander("Preview Statistics"):
                                st.write("Column types:")
                                st.dataframe(preview_df.dtypes.astype(str).reset_index().rename(
                                    columns={'index': 'Column', 0: 'DataType'}
                                ))
                                
                                st.write("Missing values:")
                                missing = preview_df.isnull().sum()
                                missing = missing[missing > 0].reset_index().rename(
                                    columns={'index': 'Column', 0: 'Missing Count'}
                                )
                                if len(missing) > 0:
                                    st.dataframe(missing)
                                else:
                                    st.success("No missing values found!")
                        
                        except Exception as e:
                            st.error(f"Preview failed: {str(e)}")
                            st.stop()
                
                # Confirm merge logic
                if confirm_button and st.session_state.preview_df is not None:
                    st.session_state.df = st.session_state.preview_df
                    st.session_state.preview_df = None  # Clear preview after confirmation
                    st.success("Data is now available for analysis.")
                    st.balloons()
                    st.rerun()
                
                # Show raw data info
                with st.expander("📦 Raw Data Summary", expanded=False):
                    st.write(f"Total files loaded: {len(dataframes)}")
                    for i, df in enumerate(dataframes, 1):
                        st.write(f"### Dataframe {i}")
                        st.write(f"- Shape: {df.shape}")
                        st.write("- Columns:")
                        st.dataframe(pd.DataFrame({
                            'Column': df.columns,
                            'Type': df.dtypes.values,
                            'Missing %': (df.isnull().mean() * 100).round(2)
                        }), hide_index=True)
                        
                        if st.checkbox(f"Show sample data for Dataframe {i}", key=f"show_raw_{i}"):
                            st.dataframe(df.head(5), use_container_width=True)
            
            # If merge is already confirmed, show the final data
            if st.session_state.df is not None:
                
                # Data cleaning options (only after merge is confirmed)
                with st.expander("**🧹 Data Cleaning**", expanded=True):
                    # Inisialisasi df_cleaned di session state jika belum ada

                    st.session_state.df_cleaned = st.session_state.df.copy()
                    
                    clean_options = st.columns(4)
                    
                    with clean_options[0]:
                        clean_nan = st.selectbox(
                            "Handle NaN values:",
                            ["Keep", "Remove rows", "Fill with zero", "Fill with mean"],
                            index=0,
                            key="clean_nan"
                        )
                        
                        if st.button("Apply NaN Handling"):
                            if clean_nan == "Fill with zero":
                                st.session_state.df_cleaned = st.session_state.df_cleaned.fillna(0)
                                st.success("Filled NaN with zeros!")
                            elif clean_nan == "Fill with mean":
                                st.session_state.df_cleaned = st.session_state.df_cleaned.fillna(st.session_state.df_cleaned.mean())
                                st.success("Filled NaN with mean values!")
                            elif clean_nan == "Remove rows":
                                initial_rows = len(st.session_state.df_cleaned)
                                st.session_state.df_cleaned = st.session_state.df_cleaned.dropna()
                                removed_nan = initial_rows - len(st.session_state.df_cleaned)
                                st.success(f"Removed {removed_nan} rows with NaN values!")
                    
                    with clean_options[1]:
                        clean_duplicates = st.selectbox(
                            "Handle duplicates:",
                            ["Keep", "Remove all", "Keep first", "Keep last"],
                            index=0,
                            key="clean_duplicates"
                        )
                        
                        if st.button("Apply Deduplication"):
                            if clean_duplicates != "Keep":
                                initial_rows = len(st.session_state.df_cleaned)
                                if clean_duplicates == "Remove all":
                                    st.session_state.df_cleaned = st.session_state.df_cleaned.drop_duplicates(keep=False)
                                elif clean_duplicates == "Keep first":
                                    st.session_state.df_cleaned = st.session_state.df_cleaned.drop_duplicates(keep='first')
                                elif clean_duplicates == "Keep last":
                                    st.session_state.df_cleaned = st.session_state.df_cleaned.drop_duplicates(keep='last')
                                
                                removed_dups = initial_rows - len(st.session_state.df_cleaned)
                                st.success(f"Removed {removed_dups} duplicate rows!")
                    
                    with clean_options[2]:
                        col_rename = st.text_input("Rename ('old:new')")
                        if st.button("Rename Column") and col_rename and ":" in col_rename:
                            old, new = col_rename.split(":", 1)
                            if old in st.session_state.df_cleaned.columns:
                                st.session_state.df_cleaned = st.session_state.df_cleaned.rename(columns={old: new})
                                st.success(f"Renamed column '{old}' to '{new}'")
                            else:
                                st.warning(f"Column '{old}' not found")
                                
                    with clean_options[3]:
                        # Fitur Drop Columns
                        cols_to_drop = st.text_input(
                            "Drop columns", 
                            placeholder="column1, column2, ...",
                            key="drop_cols_input"
                        )
                        
                        if st.button("Drop Columns"):
                            if cols_to_drop:
                                # Proses input: hilangkan spasi, split by comma, dan bersihkan
                                columns_list = [col.strip() for col in cols_to_drop.split(",") if col.strip()]
                                
                                # Fungsi untuk mencocokkan nama kolom dengan fleksibilitas
                                def find_matching_column(col_name, df_columns):
                                    col_name = col_name.lower().strip()
                                    for df_col in df_columns:
                                        if df_col.lower().strip() == col_name:
                                            return df_col
                                    return None
                                
                                # Cari kolom yang cocok
                                matched_cols = []
                                not_found = []
                                
                                for col in columns_list:
                                    matched = find_matching_column(col, st.session_state.df_cleaned.columns)
                                    if matched:
                                        matched_cols.append(matched)
                                    else:
                                        not_found.append(col)
                                
                                # Drop kolom yang ditemukan
                                if matched_cols:
                                    initial_cols = st.session_state.df_cleaned.columns.tolist()
                                    st.session_state.df_cleaned = st.session_state.df_cleaned.drop(columns=matched_cols)
                                    st.success(f"Successfully dropped columns: {', '.join(matched_cols)}")
                                    
                                    # Tampilkan warning untuk kolom yang tidak ditemukan
                                    if not_found:
                                        st.warning(f"Columns not found: {', '.join(not_found)}")
                                    
                                    # Tampilkan perubahan
                                    st.write(f"Remaining columns: {len(st.session_state.df_cleaned.columns)}")
                                else:
                                    st.error("No matching columns found to drop")
                            else:
                                st.warning("Please enter column names to drop")

                    # Tampilkan info missing values setelah cleaning
                    missing_values = st.session_state.df_cleaned.isnull().sum().sum()
                    duplicate_values = st.session_state.df_cleaned.duplicated().sum().sum()
                    if missing_values > 0 or duplicate_values > 0:
                        st.warning(f"⚠️ Warning: There are still {missing_values} missing values and {duplicate_values} ducplicate rows in the data")
                    else:
                        st.success("✅ No missing values and duplicate rows remaining!")
                
                # Final data display
                st.session_state.df = st.session_state.df_cleaned
                st.subheader("📊 Final Data")
                st.write(f"Shape: {st.session_state.df.shape}")
                
                # Interactive data explorer
                tab1, tab2, tab3, tab4 = st.tabs(["Data View", "Statistics", "Column Analysis", 'Type Data'])
                
                with tab1:
                    num_rows = st.slider(
                        "Number of rows to display:",
                        min_value=5,
                        max_value=100,
                        value=5,
                        key="num_rows_view"
                    )
                    st.dataframe(st.session_state.df.head(num_rows), use_container_width=True)
                
                with tab2:
                    st.write("### Descriptive Statistics")
                    st.dataframe(st.session_state.df.describe(include='all'), use_container_width=True)
                    
                
                with tab3:
                    selected_col = st.selectbox(
                        "Select column to analyze:",
                        st.session_state.df.columns,
                        key="col_analyze"
                    )
                    
                    col_data = st.session_state.df[selected_col]
                    st.write(f"**Type:** {col_data.dtype}")
                    st.write(f"**Unique values:** {len(col_data.unique())}")
                    st.write(f"**Missing values:** {col_data.isnull().sum()} ({col_data.isnull().mean()*100:.2f}%)")
                    

                    if col_data.nunique() < 20 and not pd.api.types.is_numeric_dtype(col_data):
                        ascending = st.checkbox("Ascending", value=False)
                        st.write("**Distribution**")
                        column_name = col_data.name

                        # sum and sort data
                        counts = col_data.value_counts(ascending=ascending)
                        ordered_categories = counts.index.tolist()

                        # create dataframe and category for data
                        df_temp = pd.DataFrame({column_name: col_data})
                        df_temp[column_name] = pd.Categorical(df_temp[column_name], categories=ordered_categories, ordered=True)

                        # create histogram
                        fig = px.histogram(
                            df_temp,
                            x=column_name,
                            opacity=0.7,
                            color_discrete_sequence=["#d06200"],
                        )

                        fig.update_traces(marker_line_color='gray', marker_line_width=1)

                        # Show x axis based on category sorting
                        fig.update_layout(
                            template='seaborn',
                            title=f"Histogram of {column_name}",
                            xaxis_title=column_name,
                            yaxis_title="Count",
                            bargap=0.1,
                            font=dict(family="Verdana", size=13),
                            xaxis=dict(
                                categoryorder="array",
                                categoryarray=ordered_categories,
                            )
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    elif pd.api.types.is_numeric_dtype(col_data):
                        st.write("**Distribution**")
                        column_name = col_data.name

                        df_temp = pd.DataFrame({column_name: col_data.dropna()})
                        
                        fig = px.histogram(
                            df_temp,
                            x=column_name,
                            nbins=30,
                            opacity=0.85,
                            color_discrete_sequence=["#d06200"],
                        )

                        fig.update_traces(marker_line_color='gray', marker_line_width=0.9)

                        fig.update_layout(
                            template='seaborn',
                            title=f"Histogram of {column_name}",
                            xaxis_title=column_name,
                            yaxis_title="Count",
                            bargap=0.1,
                            font=dict(family="Verdana", size=13),
                        )
                        fig.update_layout(
                            annotations=[
                                dict(
                                    text=f"Skewness: {skew(col_data):.2f}",
                                    showarrow=False,
                                    xref="paper", yref="paper",
                                    x=0.95, y=0.95,
                                    bordercolor='black',
                                    borderwidth=1
                                )
                            ]
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.write("**Value counts (top 20):**")
                        st.dataframe(col_data.value_counts().head(20))


                with tab4:
                    buffer = io.StringIO()
                    st.session_state.df.info(buf=buffer)
                    st.text(buffer.getvalue())

                    st.markdown("**Data Types:**")
                    st.json(st.session_state.df.dtypes.astype(str).to_dict())

                    st.markdown("**Missing Values:**")
                    st.json(st.session_state.df.isnull().sum().to_dict())

                    st.markdown("**Duplicate:**")
                    duplicate_count = st.session_state.df.duplicated().sum()
                    st.write(f"Total duplicate row: {duplicate_count}")

                    # Menampilkan jumlah duplikat dalam format JSON yang benar
                    st.json({"duplicate_count": duplicate_count})

                
                # Export options
                with st.expander("💾 Export Data", expanded=False):
                    export_format = st.radio(
                        "Export format:",
                        ["CSV", "Excel", "JSON"],
                        horizontal=True
                    )
                    
                    export_filename = st.text_input("Filename (without extension)", "clean_data")
                    
                    if st.button("Export Data"):
                        try:
                            if export_format == "CSV":
                                csv = st.session_state.df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download CSV",
                                    data=csv,
                                    file_name=f"{export_filename}.csv",
                                    mime='text/csv'
                                )
                            elif export_format == "Excel":
                                excel_buffer = io.BytesIO()
                                st.session_state.df.to_excel(excel_buffer, index=False)
                                st.download_button(
                                    label="Download Excel",
                                    data=excel_buffer,
                                    file_name=f"{export_filename}.xlsx",
                                    mime='application/vnd.ms-excel'
                                )
                            elif export_format == "JSON":
                                json_str = st.session_state.df.to_json(orient='records', indent=2)
                                st.download_button(
                                    label="Download JSON",
                                    data=json_str,
                                    file_name=f"{export_filename}.json",
                                    mime='application/json'
                                )
                            st.success("Export ready!")
                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")
                
        # Input area
        input_method = st.radio("Input method:", ["Auto run code","Manual run code"])

        user_input = st.text_area("Enter your Python code or question:", height=150,
                                placeholder="Write code here or ask a question about your data...",
                                key="user_input")
        
        

        # Mode handling
        if input_method == "Auto run code":
            col1, col2, col3 = st.columns(3)
            with col1: execute_btn = st.button("▶️ Execute")
            with col2: clear_btn = st.button("🗑️ Clear History")
            with col3: show_vars = st.button("📊 Show Variables")
            # Common button handling
            if clear_btn:
                st.session_state.execution_history = []
                
            if show_vars and st.session_state.execution_history:
                latest_vars = st.session_state.execution_history[-1].get('variables', {})
                st.json({k: str(type(v)) for k, v in latest_vars.items()})
            if execute_btn and user_input:
                with st.spinner("Execute code..."):
                    generated_code = code_llm(generate_prompt(user_input, st.session_state.df.columns.tolist()))
                    st.markdown("**Generated Code:**")
                    st.code(generated_code, language='python')
                    
                    result = execute_code(generated_code, st.session_state.df)
                    add_to_history(result)
                st.rerun()

        elif input_method == "Manual run code":
            col1, col2, col3, col4 = st.columns(4)
            with col1: code_btn = st.button("▶️ Code")
            with col2: execute_btn = st.button("▶️ Execute")
            with col3: clear_btn = st.button("🗑️ Clear History")
            with col4: show_vars = st.button("📊 Show Variables")

            if clear_btn:
                st.session_state.execution_history = []
                
            if show_vars and st.session_state.execution_history:
                latest_vars = st.session_state.execution_history[-1].get('variables', {})
                st.json({k: str(type(v)) for k, v in latest_vars.items()})

            if code_btn and user_input:
                with st.spinner("Generate Code"):
                    generated_code = code_llm(generate_prompt(user_input, st.session_state.df.columns.tolist()))
                    st.session_state.generated_code = generated_code
                    st.markdown("**Generated Code:**")
                    st.code(generated_code, language='python')
            
            if execute_btn:
                if 'generated_code' in st.session_state:
                    result = execute_code(st.session_state.generated_code, st.session_state.df)
                    add_to_history(result)
                elif user_input:
                    result = execute_code(user_input, st.session_state.df)
                    add_to_history(result)
                st.rerun()
                    
        # Display execution history
        display_history()