# **Documentation for DataCoder AI**  
**Last Updated**: 2025-04-12  

---

## **1. Introduction**  
**DataCoder AI** is a web-based interactive data analysis tool that combines:  
- **Data Analysis Python Coder**  
- **AI-powered code generation** (via Groq API)  
- **Data visualization**  
- **Chat-based coding assistance**  

Built with `Streamlit`, `Pandas`, and `Plotly`, designed for **data scientists** and **analysts**.  
Here are the **required libraries** for your DataCoder AI application, extracted from your code:

---

### **Core Dependencies**  
**Install via `pip install -r requirements.txt`:**
```text
streamlit==1.32.0        # Web UI framework
pandas==2.0.0           # Data manipulation
matplotlib==3.7.0       # Basic visualization
seaborn==0.12.2         # Advanced plots
plotly==5.18.0          # Interactive charts
requests==2.31.0        # API calls (Groq)
langchain==0.1.0        # LLM interface (if used)
python-dotenv==1.0.0    # Environment variables
```

### **Optional/Implicit Dependencies**  
```text
numpy==1.24.0           # (Auto-installed with pandas)
scipy==1.10.0           # (For stats operations)
fuzzywuzzy==0.18.0      # (For column name matching - if used)
```

---

### **Notes**:  
1. **Groq API**: No Python SDK required (uses raw `requests`).  
2. **Virtual Environment**: Recommended to avoid conflicts:  
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```

3. **Production Considerations**:  
   - Add `gunicorn` for deployment:  
     ```text
     gunicorn==20.1.0
     ```  
   - For Windows, use `waitress`:  
     ```text
     waitress==2.1.2
     ```
---

## **2. System Architecture**  

### **2.1 Component Diagram**  
```mermaid
flowchart TB
    A[Mulai] --> B[Tampilkan Antarmuka Utama]
    B --> C{Pilih Mode}
    C -->|Panel Kiri| D[Mode Chatbot]
    C -->|Panel Kanan| E[Mode Analisis Data]
    
    %% Flowchart Mode Chatbot
    D --> D1[Input Pertanyaan Pengguna]
    D1 --> D2[Proses dengan LLM (Groq)]
    D2 --> D3[Tampilkan Jawaban]
    D3 --> D4{Input Baru?}
    D4 -->|Ya| D1
    D4 -->|Tidak| D5[Simpan History Chat]
    
    %% Flowchart Mode Analisis Data
    E --> E1[Upload File CSV]
    E1 --> E2{File Valid?}
    E2 -->|Tidak| E3[Tampilkan Error]
    E2 -->|Ya| E4[Proses File]
    E4 --> E5{Tipe Merge?}
    E5 -->|Single| E6[Tampilkan Data]
    E5 -->|Horizontal| E7[Merge Columns]
    E5 -->|Vertical| E8[Concat Rows]
    E5 -->|Join| E9[Merge on Keys]
    E7 --> E10[Preview Data]
    E8 --> E10
    E9 --> E10
    E10 --> E11{Konfirmasi?}
    E11 -->|Tidak| E4
    E11 -->|Ya| E12[Simpan DataFrame]
    
    E12 --> E13[Pembersihan Data]
    E13 --> E14{Handle Missing Values?}
    E14 -->|Ya| E15[Apply NaN Handling]
    E14 -->|Tidak| E16{Lanjut}
    E13 --> E17{Handle Duplicates?}
    E17 -->|Ya| E18[Apply Deduplication]
    E17 -->|Tidak| E16
    E13 --> E19{Modifikasi Kolom?}
    E19 -->|Ya| E20[Rename/Drop Columns]
    E19 -->|Tidak| E16
    
    E16 --> E21[Input Kode Analisis]
    E21 --> E22{Metode Input?}
    E22 -->|Auto Run| E23[Generate Code dengan LLM]
    E22 -->|Manual Run| E24[Input Manual]
    E23 --> E25[Eksekusi Kode]
    E24 --> E25
    E25 --> E26{Error?}
    E26 -->|Ya| E27[Tampilkan Error]
    E26 -->|Tidak| E28[Simpan Hasil]
    E28 --> E29[Tampilkan Output]
    E29 --> E30[Analisis Hasil dengan LLM]
    E30 --> E31[Simpan ke History]
    
    E31 --> E32{Input Lagi?}
    E32 -->|Ya| E21
    E32 -->|Tidak| E33[Ekspor Data?]
    E33 -->|Ya| E34[Generate File Output]
    E33 -->|Tidak| F[Selesai]
```

### **2.2 Data Flow**  
1. **Input**: CSV files → Pandas DataFrame  
2. **Processing**: Merge/clean data → Modified DataFrame  
3. **Execution**: Python code → Results (text/plots/variables)  
4. **Output**: Visualizations + Export files  

---

## **3. Installation**  

### **3.1 Prerequisites**  
- Python 3.8+  
- `pip` package manager  

### **3.2 Setup**  
```bash
# Clone repository
git clone https://github.com/yourrepo/datacoder-ai.git
cd datacoder-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### **3.3 Configuration**  
Rename `.env.example` to `.env` and add your Groq API key:  
```env
GROQ_API_KEY=your_api_key_here
```

---

## **4. User Guide**  

### **4.1 Data Upload**  
1. Click **"Upload CSV file(s)"**  
2. Select 1+ CSV files  
3. Configure merge options:  
   - **Horizontal**: Combine columns (align by index)  
   - **Vertical**: Stack rows  

### **4.2 Data Cleaning**  
| Feature          | How to Use                              |
|------------------|----------------------------------------|
| Handle NaN       | Select: Keep/Remove/Fill with mean     |
| Remove Duplicates| Choose: Keep first/last/all            |
| Rename Columns   | Enter: `old_name:new_name`             |
| Drop Columns     | Enter comma-separated column names     |

### **4.3 Code Execution**  
#### **Auto-Run Mode**  
1. Type request (e.g., "Plot sales distribution")  
2. Click **▶️ Execute** → AI generates and runs code  

#### **Manual Mode**  
1. Write/edit Python code  
2. Click **▶️ Execute**  

### **4.4 Chat Assistant**  
Ask questions like:  
- _"How to normalize this data?"_  
- _"Explain the output of this code"_  

---

## **5. API Reference**  

### **5.1 Core Functions**  
#### `execute_code(code: str, df: pd.DataFrame) -> dict`  
**Parameters**:  
- `code`: Python code to execute  
- `df`: Target DataFrame  

**Returns**:  
```python
{
    "success": bool,
    "stdout": str,       # Console output
    "figure": Figure,     # Matplotlib plot
    "variables": dict     # New variables created
}
```

#### `analyze_code_execution(code: str, execution_result: dict, df: pd.DataFrame) -> str`  
Generates natural language analysis of results in **Indonesian**.  

### **5.2 LLM Classes**  
#### `GroqLLM()`  
- **Model**: `qwen-2.5-coder-32b`  
- **Temperature**: 0.3  
- **Purpose**: Code generation  

#### `GroqTextLLM()`  
- **Model**: `deepseek-r1-distill-llama-70b`  
- **Temperature**: 0.7  
- **Purpose**: Chat explanations  

---

## **6. Examples**  

### **6.1 Sample Workflow**  
```python
# AI-Generated Code Example (Auto-Run)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
df['sales'].plot(kind='hist', bins=20)
plt.title('Sales Distribution')
plt.show()
```

### **6.2 Expected Output**  
1. Histogram plot rendered  
2. Console output:  
   ```text
   Plot generated for column: sales
   ```

---

## **7. Troubleshooting**  

| Issue                          | Solution                               |
|--------------------------------|----------------------------------------|
| Groq API errors                | Check quota/network connection        |
| CSV parsing fails              | Verify delimiter (use `delimiter=';'`)|
| Plot not showing               | Ensure `plt.show()` is called         |

---

## **8. License & Contribution**  
- **License**: MIT  
- **Contribute**: Fork + PRs welcome  

---

**Notes for Production**:  
1. Replace hardcoded API keys with environment variables  
2. Add user authentication for multi-user support  
3. Implement rate limiting for Groq API calls  
