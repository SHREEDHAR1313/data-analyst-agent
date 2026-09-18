# AI Data Analyst Agent

A Streamlit data-analysis agent that lets you upload a CSV and ask questions in plain English. Gemini chooses from a small set of deterministic Python tools; pandas/numpy/matplotlib perform the actual analysis.



## Setup on Windows

Open the project folder in VS Code Terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and put your Google AI Studio key in it:

```env
GOOGLE_API_KEY=your-real-key-here
GEMINI_MODEL=gemini-3.6-flash
```

Then start the app:

```powershell
streamlit run app.py
```

If PowerShell blocks activation, you can skip activation and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## First test

The included sample dataset is in `sample_data/sales_data.csv`. Start the app and try:

- `Give me an overview of this dataset and flag any data quality issues.`
- `What's driving revenue the most?`
- `Are there any outliers in revenue?`
- `Show me revenue by region as a chart.`
- `Is there a relationship between unit price and units sold?`

## Project structure

```text
data-analyst-agent/
├── app.py                  # Streamlit UI
├── agent_gemini.py         # Gemini Interactions API + tool orchestration
├── tools.py                # Deterministic pandas/numpy/matplotlib tools
├── requirements.txt
├── .env.example
├── sample_data/
│   └── sales_data.csv
└── outputs/                # Generated charts appear here
```

