import io
import contextlib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from io import StringIO
import plotly.graph_objects as go
from llm_models import text_llm
from prompt_models import prompt_analyze


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


    prompt = prompt_analyze(output, var_summaries, df, error)
    # Get analysis from the text model
    response = text_llm(prompt)

    # Clear thinking output
    clean_response = re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>', '', response, flags=re.DOTALL | re.IGNORECASE)
    
    return clean_response

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
        "import seaborn as sns",
        "import plotly.express as px",
        "import plotly.graph_objects as go"
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
    
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(clean_code, local_vars)
            
                # Deteksi plot Plotly
            plotly_figs = []
            for var_name, var_value in local_vars.items():
                if isinstance(var_value, go.Figure):
                    plotly_figs.append(var_value)
            
            return {
                'code': clean_code,
                'stdout': stdout.getvalue(),
                'stderr': stderr.getvalue(),
                'figure': plotly_figs,
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
        with st.spinner("Analyzing results..."):
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
    st.markdown("### Execution History")
    
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

                        for idx, line in enumerate(lines):
                            cleaned_line = line.strip()
                            if not cleaned_line:
                                continue
                            
                            if not header_detected:
                                if re.match(r"^[\w\s_]+$", cleaned_line) and len(cleaned_line.split()) > 1:
                                    if (idx + 1 < len(lines)) and re.match(r"^\d+\s+[\d\.e+-]+", lines[idx+1].strip()):
                                        header_detected = True
                                        table_data = lines[idx:]
                                        break
                                    else:
                                        text_content.append(cleaned_line)
                                else:
                                    text_content.append(cleaned_line)
                        
                        if text_content:
                            st.write("\n".join(text_content))
                            
                        if table_data:
                            try:
                                clean_table = [line.strip() for line in table_data if line.strip()]
                                df = pd.read_csv(StringIO("\n".join(clean_table)), sep=r"\s+", engine="python")
                                st.dataframe(df)
                            except Exception as e:
                                st.write("\n".join(table_data))

                
                if item['figure']:
                    st.markdown("**Visualisasi Interaktif:**")
                    for fig in item['figure']:
                        st.plotly_chart(fig, use_container_width=True)

                        
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