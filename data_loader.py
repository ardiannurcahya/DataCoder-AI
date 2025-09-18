import csv
import pandas as pd
import streamlit as st


def load_files():
    """Handle file upload and return list of raw dataframes."""
    if 'df' not in st.session_state:
        st.session_state.df = None

    uploaded_files = st.file_uploader(
        "Upload CSV file(s)",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or more CSV files to combine",
    )

    if not uploaded_files:
        return None

    if (
        'prev_uploaded_files' not in st.session_state
        or st.session_state.prev_uploaded_files != [f.name for f in uploaded_files]
    ):
        st.session_state.df = None
        st.session_state.raw_dataframes = None
        st.session_state.preview_df = None
        st.session_state.prev_uploaded_files = [f.name for f in uploaded_files]

    if st.session_state.raw_dataframes is None:
        dataframes = []
        processed_files = 0

        for uploaded_file in uploaded_files:
            with st.expander(f"📄 File: {uploaded_file.name}", expanded=False):
                try:
                    raw_data = uploaded_file.getvalue().decode("utf-8")
                    sniffer = csv.Sniffer()
                    try:
                        dialect = sniffer.sniff(raw_data.splitlines()[0])
                        delimiter = dialect.delimiter
                    except (csv.Error, IndexError):
                        for test_delim in [',', ';', '\t', '|']:
                            if test_delim in raw_data:
                                delimiter = test_delim
                                break
                        else:
                            delimiter = ','

                    uploaded_file.seek(0)
                    try:
                        df = pd.read_csv(uploaded_file, delimiter=delimiter, on_bad_lines='warn')
                    except Exception:
                        st.warning(
                            f"Using Python engine for {uploaded_file.name} due to parsing issues"
                        )
                        df = pd.read_csv(uploaded_file, delimiter=delimiter, engine='python')

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

        st.session_state.raw_dataframes = dataframes

    return st.session_state.raw_dataframes


def merge_dataframes(dataframes):
    """Provide merge options for uploaded dataframes."""
    if not dataframes:
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        merge_option = st.radio(
            "Merge method:",
            (
                "single data",
                "Horizontal (concat columns)",
                "Vertical (concat rows)",
                "Horizontal (merge on common columns)",
            ),
            index=0,
            horizontal=True,
            key="merge_option",
        )

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
                        key="merge_columns",
                    )
                    how_merge = st.selectbox(
                        "Merge type:",
                        ["inner", "outer", "left", "right"],
                        index=1,
                        key="how_merge",
                    )
                else:
                    selected_cols = []
                    how_merge = "inner"
            else:
                selected_cols = []
                how_merge = "inner"

    with col2:
        preview_button = st.button("🔄 Preview")
        confirm_button = st.button("✅ Confirm")
        reset_btn = st.button("🔄 Reset")

    if reset_btn:
        st.session_state.df = None
        st.session_state.raw_dataframes = None
        st.session_state.preview_df = None
        st.rerun()

    if preview_button:
        with st.spinner("Generating preview..."):
            try:
                if merge_option == "single data":
                    preview_df = dataframes[0]
                elif merge_option == "Horizontal (concat columns)":
                    common_indices = dataframes[0].index
                    for df in dataframes[1:]:
                        common_indices = common_indices.intersection(df.index)

                    if len(common_indices) == 0:
                        st.error("No common indices found across all dataframes")
                        st.stop()

                    filtered_dfs = [df.loc[common_indices] for df in dataframes]

                    all_columns = []
                    duplicate_counter = {}

                    for i, df in enumerate(filtered_dfs):
                        new_columns = []
                        for col in df.columns:
                            if col in all_columns:
                                duplicate_counter[col] = duplicate_counter.get(col, 0) + 1
                                new_col = f"{col}_df{duplicate_counter[col]}"
                                new_columns.append(new_col)
                            else:
                                new_columns.append(col)
                        all_columns.extend(new_columns)
                        filtered_dfs[i].columns = new_columns

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
                                print(
                                    f"[WARNING] There is no matching key between:\n{merged_df.columns}\n&\n{next_df.columns}"
                                )

                        return merged_df

                    preview_df = auto_merge_many(dataframes)

                st.session_state.preview_df = preview_df
                st.success("Preview generated!")

                st.subheader("Preview Results")
                st.write(f"Shape: {preview_df.shape}")

                tab1, tab2 = st.tabs(["First Rows", "Last Rows"])
                with tab1:
                    st.dataframe(preview_df.head(20), use_container_width=True)
                with tab2:
                    st.dataframe(preview_df.tail(20), use_container_width=True)

                with st.expander("Preview Statistics"):
                    st.write("Column types:")
                    st.dataframe(
                        preview_df.dtypes.astype(str)
                        .reset_index()
                        .rename(columns={"index": "Column", 0: "DataType"})
                    )

                    st.write("Missing values:")
                    missing = preview_df.isnull().sum()
                    missing = missing[missing > 0].reset_index().rename(
                        columns={"index": "Column", 0: "Missing Count"}
                    )
                    if len(missing) > 0:
                        st.dataframe(missing)
                    else:
                        st.success("No missing values found!")

            except Exception as e:
                st.error(f"Preview failed: {str(e)}")
                st.stop()

    if confirm_button and st.session_state.get("preview_df") is not None:
        st.session_state.df = st.session_state.preview_df
        st.session_state.preview_df = None
        st.success("Data is now available for analysis.")
        st.balloons()
        st.rerun()

    with st.expander("📦 Raw Data Summary", expanded=False):
        st.write(f"Total files loaded: {len(dataframes)}")
        for i, df in enumerate(dataframes, 1):
            st.write(f"### Dataframe {i}")
            st.write(f"- Shape: {df.shape}")
            st.write("- Columns:")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Column": df.columns,
                        "Type": df.dtypes.values,
                        "Missing %": (df.isnull().mean() * 100).round(2),
                    }
                ),
                hide_index=True,
            )

            if st.checkbox(
                f"Show sample data for Dataframe {i}", key=f"show_raw_{i}"
            ):
                st.dataframe(df.head(5), use_container_width=True)
