import io
import contextlib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime
from langchain.llms.base import LLM
import csv
import re
import pandas as pd
from io import StringIO
from typing import Optional, List
import plotly.express as px
import re

st.set_page_config(layout="wide")
# Custom CSS for Jupyter-like appearance
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
        background-color: #111016; /* Warna background */
        color: white; /* Warna teks */
        border: 1.5px solid #ffffff; /* Border dengan warna yang sama seperti background */
        padding: 0.1rem 0.5rem; /* Padding untuk tombol */
        border-radius: 5px; /* Membuat border bulat (round) */
        font-size: 16px; /* Ukuran font */
        font-weight: bold; /* Membuat teks menjadi tebal */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Tambahkan shadow agar tombol terlihat lebih modern */
        transition: all 0.3s ease; /* Transisi halus saat hover */
    }

    .stButton > button:hover {
        background-color: #357ae8; /* Ganti background saat hover */
        border-color: #357ae8; /* Ganti warna border saat hover */
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); /* Tambahkan efek bayangan saat hover */
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
        align-self: flex-start;  /* Pesan pengguna berada di sebelah kiri */
    }
    .chat-message.assistant {
        background-color: #262626;
        align-self: flex-end;  /* Pesan asisten berada di sebelah kanan */
    }
    .chat-message {
            font-size: 14px;  /* Ukuran huruf standar */
    }
    .chat-message.user {
            font-size: 16px;  /* Ukuran huruf untuk pesan pengguna */
    }
    .chat-message.assistant {
            font-size: 14px;  /* Ukuran huruf untuk pesan asisten */
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

# =======================
# Custom LLMs
# =======================
class GroqLLM(LLM):
    model: str = "qwen-2.5-coder-32b"
    temperature: float = 0.3
    api_key: str = "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"

    @property
    def _llm_type(self) -> str:
        return "groq-llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model,
            "temperature": self.temperature,
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"Error calling Groq API: {str(e)}")
            return "Sorry, there was an error processing your request."

class GroqTextLLM(LLM):
    model: str = "deepseek-r1-distill-llama-70b"
    temperature: float = 0.7
    api_key: str = "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"

    @property
    def _llm_type(self) -> str:
        return "groq-text-llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model,
            "temperature": self.temperature,
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"Error calling Groq API: {str(e)}")
            return "Sorry, there was an error processing your request."

# Initialize LLMs
code_llm = GroqLLM()
text_llm = GroqTextLLM()


def analyze_code_execution(code: str, execution_result: dict, df: pd.DataFrame) -> str:
    """
    Analyze only the execution results (output and variables) to provide focused insights.
    
    Args:
        code: The Python code that was executed
        execution_result: The dictionary containing execution results
        df: The DataFrame being analyzed (for context only)
        
    Returns:
        str: Natural language analysis focused on the execution results
    """
    # Extract relevant information from execution_result
    output = execution_result.get('stdout', 'Tidak ada output teks')
    variables = execution_result.get('variables', {})
    error = execution_result.get('error', None)
    
    # Prepare variable summaries
    var_summaries = []
    for var_name, var_value in variables.items():
        if isinstance(var_value, (pd.DataFrame, pd.Series)):
            var_summaries.append(f"- {var_name}: {type(var_value).__name__} dengan bentuk {var_value.shape}")
        elif isinstance(var_value, (plt.Figure, sns.axisgrid.Grid)):
            var_summaries.append(f"- {var_name}: Visualisasi plot")
        else:
            var_summaries.append(f"- {var_name}: {type(var_value).__name__}")

    # Prepare the prompt for the text model
    prompt = f"""
    Kamu adalah analis data yang menjelaskan hasil eksekusi kode Python kepada pemangku kepentingan non-teknis.

    Hasil eksekusi yang perlu dianalisis:
    1. Output teks:
    {output}
    
    2. Variabel yang dihasilkan:
    {chr(10).join(var_summaries) if var_summaries else 'Tidak ada variabel baru dibuat'}
    
    3. Error (jika ada):
    {error if error else 'Tidak ada error'}

    analisis juga data berikut:
    - Success: {execution_result['success']}
    - Output: {execution_result.get('stdout', 'No output')}
    - Figure: {execution_result.get('figure', 'None')}
    - Error: {execution_result.get('error', 'None')}
    - Variables created: {list(execution_result.get('variables', {}).keys())}
    - {code}
    
    Dataframe shape: {df.shape}
    
    Berikan insight lebih lanjut berdasarkan seluruh kolom data yang ada pada DataFrame. Fokus pada:
    - Fokuslah untuk membahas hasil eksekusi
    - Jangan menganalisis dari data yg tidak ada
    - Temuan atau pola menarik yang muncul dari data
    - Hubungan atau korelasi antara kolom-kolom dalam data
    - Faktor yang dapat mempengaruhi hasil analisis lebih lanjut
    - Saran untuk pengolahan atau analisis lebih lanjut yang dapat dilakukan pada data ini
    - Jangan membahas kode Python-nya
    - Berikan insight secara singkat dan padat algoritma yang perlu digunakan seperti pada data science dan AI beserta kegunaannya terhadap analisis
    - Gunakan bahasa Indonesia yang mudah dipahami
    - Hanya gunakan kata ganti orang ketiga
    """
    
    # Get analysis from the text model
    llm = GroqTextLLM()
    response = llm(prompt)

    # Bersihkan output untuk menghilangkan bagian yang tidak diinginkan
    clean_response = re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>', '', response, flags=re.DOTALL | re.IGNORECASE)
    
    return clean_response



# =======================
# Code Execution Functions
# =======================

# =======================
def execute_code(code: str, df: pd.DataFrame):
    """Execute Python code safely and return outputs"""
    # Extract Python code from markdown blocks
    if "```python" in code:
        clean_code = code.split("```python")[1].split("```")[0].strip()
    else:
        clean_code = code.strip()
    
    # Add required imports if missing
    required_imports = [
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns"
    ]
    
    for imp in required_imports:
        if imp not in clean_code:
            clean_code = imp + "\n" + clean_code
    
    # Prepare execution environment
    local_vars = {
        'df': df,
        'plt': plt,
        'sns': sns,
        'pd': pd
    }
    
    # Capture outputs
    stdout = io.StringIO()
    stderr = io.StringIO()
    fig = None
    
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(clean_code, local_vars)
            
            # Check for plots
            if plt.gcf().get_axes():
                fig = plt.gcf()
            
            return {
                'code': clean_code,
                'stdout': stdout.getvalue(),
                'stderr': stderr.getvalue(),
                'figure': fig,
                'success': True,
                'variables': {k: v for k, v in local_vars.items() 
                              if not k.startswith('_') and k not in ['df', 'plt', 'sns', 'pd']}
            }

    except Exception as e:
        return {
            'code': clean_code,
            'error': str(e),
            'stderr': stderr.getvalue(),
            'success': False
        }

def add_to_history(execution_result):
    """Add execution result to history with timestamp"""
    history_item = {
        **execution_result,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.execution_history.append(history_item)

    # Generate explanation for successful executions
    if execution_result['success']:
        with st.spinner("Menganalisis hasil..."):
            df_to_analyze = st.session_state.df_cleaned if 'df_cleaned' in st.session_state else st.session_state.df
            
            analysis = analyze_code_execution(
                code=execution_result['code'],
                execution_result=execution_result,
                df=df_to_analyze  # Only used for context, not analyzed
            )
            
            history_item['explanation'] = f"""
            **Analisis Hasil Eksekusi:**
            {analysis}
            """
            


# =======================
# Display Functions
# =======================

def display_code_with_highlight(code: str):
    """Display formatted Python code using Streamlit's built-in code block"""
    st.code(code, language='python')



def display_history():
    """Display all execution history items"""
    st.markdown("## Execution History")
    
    if not st.session_state.execution_history:
        st.info("No executions yet. Run some code to see results here.")
        return
    
    for i, item in enumerate(reversed(st.session_state.execution_history)):
        with st.container():
            st.markdown(f"### Execution #{len(st.session_state.execution_history)-i}")
            st.caption(f"Executed at {item['timestamp']}")
            
            # Display code
            with st.expander("View Code", expanded=False):
                display_code_with_highlight(item['code'])
            
            # Display outputs
            if item['success']:
                if 'explanation' in item:
                    with st.expander("See Explanation"):
                        st.markdown("**Explanation:**")
                        st.info(item['explanation'])

                if item['stdout']:
                    with st.expander("View Output", expanded=False):
                        st.markdown("**Output:**")
                        lines = item['stdout'].split('\n')
                        text_content = []
                        table_data = []
                        header_detected = False

                        # Iterasi melalui setiap baris untuk memisahkan teks dan tabel
                        for idx, line in enumerate(lines):
                            cleaned_line = line.strip()
                            if not cleaned_line:
                                continue
                            
                            # Deteksi header tabel
                            if not header_detected:
                                if re.match(r"^[\w\s_]+$", cleaned_line) and len(cleaned_line.split()) > 1:
                                    # Periksa baris berikutnya untuk memastikan ini adalah tabel
                                    if (idx + 1 < len(lines)) and re.match(r"^\d+\s+[\d\.e+-]+", lines[idx+1].strip()):
                                        header_detected = True
                                        table_data = lines[idx:]
                                        break
                                    else:
                                        text_content.append(cleaned_line)
                                else:
                                    text_content.append(cleaned_line)
                        
                        # Tampilkan konten teks
                        if text_content:
                            st.write("\n".join(text_content))
                            
                        # Coba parsing dan tampilkan tabel jika ada
                        if table_data:
                            try:
                                # Bersihkan data untuk DataFrame
                                clean_table = [line.strip() for line in table_data if line.strip()]
                                df = pd.read_csv(StringIO("\n".join(clean_table)), sep=r"\s+", engine="python")
                                st.dataframe(df)
                            except Exception as e:
                                st.write("\n".join(table_data))

                
                if item['figure']:
                    st.markdown("**Visualization:**")
                    fig = item['figure']
                    st.pyplot(fig)
                    plt.close()

                        
                if item['variables']:
                    with st.expander("Created Variables"):
                        st.json({k: str(type(v)) for k, v in item['variables'].items()})
            else:
                st.error("Execution failed")
                st.error(item['error'])
                if item['stderr']:
                    st.text("Error details:")
                    st.text(item['stderr'])
            
            st.markdown("---")

# =======================
# Main App
# =======================
# Main app layout
col1a, col2a = st.columns([4, 7], gap="small")
st.markdown("<div style='margin-top: 5px'></div>", unsafe_allow_html=True)
show_col1 = st.toggle("Ask chat bot", value=True)
# Atur rasio kolom berdasarkan kondisi
if show_col1:
    col1a, col2a = st.columns([4,7])  # Keduanya muncul
else:
    col2a, _ = st.columns([1,0.0001,])  # Kolom 2 disembunyikan, col1 melebar penuh
    
with col1a:
  if show_col1:
    with st.container(height=570):
        st.header("💬 Coding Assistant")
        
        # Inisialisasi state khusus untuk chatbot jika belum ada
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = {
                'chat_history': [],
                'prev_df': None,
                'prev_raw_dataframes': None
            }
        
        # Tombol hapus semua chat
        if st.button("🗑️ Hapus Semua Chat", key="clear_all_chat"):
            st.session_state.chatbot['chat_history'] = []
        
        # Tampilkan chat history
        # Tampilkan chat history per pasangan user-assistant
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
            # Simpan state sebelumnya khusus untuk chatbot
            st.session_state.chatbot['prev_df'] = st.session_state.get('df')
            st.session_state.chatbot['prev_raw_dataframes'] = st.session_state.get('raw_dataframes')
            
            # Tampilkan pesan user langsung
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # Tambahkan ke history
            st.session_state.chatbot['chat_history'].append({"role": "user", "content": user_query})
            
            with st.spinner("Thinking..."):
                prompt = f"""
                Anda adalah seorang analyst. Jawab pertanyaan berikut dengan jelas 

                Pertanyaan: {user_query}

                Ketentuan jawaban:
                
                1. Gunakan bahasa Indonesia yang formal
                2. Jangan sertakan contoh kode
                3. Format kode dalam blok code
                4. Jelaskan istilah teknis dengan analogi sederhana
                5. Selalu berikan tag untuk proses berpikir anda
                6. Langsung jawab pada intinya
                7. Hanya tuliskan poin-poinnya saja
                8 Jawablah sesingkat mungkin
                """
                response = text_llm(prompt)
                clean_response = re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>', '', response, flags=re.DOTALL | re.IGNORECASE)
                
                # Tampilkan dan simpan respon
                with st.chat_message("assistant"):
                    st.markdown(clean_response)
                st.session_state.chatbot['chat_history'].append({"role": "assistant", "content": clean_response})

with col2a:  # Sidebar untuk chatbot
    with st.container(height=570):
        st.title("📊 Data Analysis Powered by AI Assistance")
        st.markdown("Analyze your CSV data with LLM Models")

        def reset_data_state():
            st.session_state.df = None
            st.session_state.raw_dataframes = None
            st.session_state.preview_df = None
            st.session_state.file_names = [f.name for f in uploaded_files] if uploaded_files else []

        # Initialize session state if not already present
        if 'df' not in st.session_state:
            st.session_state.df = None

        st.header("CSV File Uploader")

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
                
                st.subheader("Merge Configuration")
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
                    st.write("")  # Spacer
                    st.write("")  # Spacer
                    preview_merge = st.button("🔄 Preview")
                    confirm_merge = st.button("✅ Confirm")
                    reset_btn = st.button("🔄 Reset")
                
                # Reset logic
                if reset_btn:
                    reset_data_state()
                    st.rerun()
                
                # Preview logic
                if preview_merge:
                    with st.spinner("Generating merge preview..."):
                        try:
                            if merge_option == "single data":
                                preview_df = dataframes
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
                                            print(f"[WARNING] Tidak ada key yang cocok antara:\n{merged_df.columns}\n&\n{next_df.columns}")
                                    
                                    return merged_df
                                
                                preview_df = auto_merge_many(dataframes)
                            
                            st.session_state.preview_df = preview_df
                            st.success("Merge preview generated!")
                            
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
                            st.error(f"Merge preview failed: {str(e)}")
                            st.stop()
                
                # Confirm merge logic
                if confirm_merge and st.session_state.preview_df is not None:
                    st.session_state.df = st.session_state.preview_df
                    st.session_state.preview_df = None  # Clear preview after confirmation
                    st.success("Merge confirmed! Data is now available for analysis.")
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
                st.subheader("🔍 Merged Data Analysis")
                
                # Data cleaning options (only after merge is confirmed)
                with st.expander("🧹 Data Cleaning", expanded=True):
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
                        col_rename = st.text_input("Rename (e.g., 'old:new')")
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
                    if missing_values > 0:
                        st.warning(f"⚠️ Warning: There are still {missing_values} missing values in the data")
                        if st.button("Show Missing Values Details"):
                            missing_df = st.session_state.df_cleaned.isnull().sum()
                            missing_df = missing_df[missing_df > 0].reset_index()
                            missing_df.columns = ['Column', 'Missing Count']
                            st.dataframe(missing_df)
                    else:
                        st.success("✅ No missing values remaining!")
                
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
                        value=20,
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
                    
                    if col_data.nunique() < 10:
                        st.write("**Distribution**")
                        column_name = col_data.name  # Nama kolom

                        # Masukkan col_data ke DataFrame agar px.histogram bisa membaca kolomnya
                        df_temp = pd.DataFrame({column_name: col_data})

                        fig = px.histogram(
                            df_temp,
                            x=column_name,
                            opacity=0.7,
                            color_discrete_sequence=["#636EFA"],  # Opsional: warna
                        )

                        fig.update_traces(marker_line_color='black', marker_line_width=1)

                        fig.update_layout(
                            template='seaborn',
                            title=f"Histogram of {column_name}",
                            xaxis_title=column_name,
                            yaxis_title="Count",  # Ganti dari "Density"
                            bargap=0.1,
                            font=dict(family="Verdana", size=13),
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
                    st.write(f"Jumlah duplikat: {duplicate_count}")

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
            
            # Reset all button
            if st.button("🔄 Reset All Data", type="secondary"):
                st.session_state.df = None
                st.session_state.df_cleaned = None
                st.session_state.raw_dataframes = None
                st.session_state.preview_df = None
                st.rerun()
                
        # Input area
        st.subheader("Code Input")
        input_method = st.radio("Input method:", ["Auto run code","Manual run code"])

        user_input = st.text_area("Enter your Python code or question:", height=150,
                                placeholder="Write code here or ask a question about your data...",
                                key="user_input")

        def generate_prompt(user_input, df_columns):
            """Generate optimized prompt for code generation"""
            return f"""
            Generate Python code to analyze pandas DataFrame 'df' according to request:
            {user_input}

            STRICT RULES (NUMBERED PRIORITY):
            1. CODE STRUCTURE:
                - Return complete executable code in ```python block
                - All new variables MUST be added as new columns to existing 'df' using df['new_column'] syntax
                - Never add a new variable (such as a list, Series, or array) to df if its length is less than the number of rows in df.
                - Never create new DataFrames or use merge/join operations
                - Print calculation result (such as a list, Series, or array) in console
                - Define all temporary variables explicitly

            2. DATA HANDLING:
                - Use only columns from: {list(df_columns)}
                - For new data storage:
                    if isinstance(result, (pd.Series, list, np.ndarray)):
                        df[new_col_name] = result
                    elif numerical/string value:
                        df[new_col_name] = [value]*len(df)
                - For column matching:
                    from fuzzywuzzy import process
                    def find_best_match(query, cols, threshold=70):
                        # [existing fuzzy match implementation]
                    
            3. COLUMN CREATION RULES:
                a. Untuk hasil operasi matematika:
                    df['hasil_kalkulasi'] = df['col1'] + df['col2']
                
                b. Untuk transformasi data:
                    df['kategori_baru'] = df['col_existing'].apply(lambda x: x*2)
                
                c. Untuk agregasi grup:
                    df['rata_grup'] = df.groupby('kolom_grup')['target'].transform('mean')
                
                d. Untuk hasil statistik:
                    df['zscore'] = (df['nilai'] - df['nilai'].mean())/df['nilai'].std()

            4. VISUALIZATION:
                - Visualisasi harus menggunakan kolom dari df yang telah dimodifikasi
                - Contoh: plt.plot(df['kolom_baru'], df['kolom_lama'])

            5. OUTPUT HANDLING:
                - Untuk menampilkan kolom baru:
                    print(f"Kolom baru yang dibuat:", list(df.columns[-N:]))  # N = jumlah kolom baru
                    print(df[['kolom_baru_1', 'kolom_baru_2']].head())
                
            6. ERROR PREVENTION:
                - Cek panjang data sebelum membuat kolom baru:
                    if len(variable) != len(df):
                        raise ValueError("Panjang variabel tidak sesuai dengan DataFrame")
                - Gunakan nama kolom unik:
                    if 'nama_kolom' in df.columns:
                        df['nama_kolom_rev'] = ...  # tambahkan suffix jika sudah ada

            FINAL CHECK:
            1. Pastikan tidak ada operasi merge/join/concat
            2. Verifikasi semua hasil kalkulasi ada di df sebagai kolom
            3. Cek konsistensi indeks untuk kolom baru
            4. Pastikan print terakhir menampilkan df dengan kolom baru
            5. Jangan menampilkan penjelasan apapun dari kode
            6. Jika user menyebut 'pivot' gunakan fungsi groupby
            """

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
                with st.spinner("Generating code with AI..."):
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
                with st.spinner("Membuat kode dengan AI..."):
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