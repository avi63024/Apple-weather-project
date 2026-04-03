# Weather Data Project

## Project overview

This project is a clean Python weather data pipeline built for the Downtown LA weather assignment. It:

- parses 10 daily PDF weather reports with inconsistent formatting
- cleans a poorly formatted 3-month CSV weather file
- stores both sources in SQLite
- answers stakeholder questions from the database
- generates charts and a notebook for analysis
- includes a CLI chatbot grounded in the stored data

## Folder structure

```text
weather_project/
├── data/
│   └── raw/
│       ├── pdfs/
│       └── weather.csv
├── scripts/
│   ├── pdf_parser.py
│   ├── csv_cleaner.py
│   ├── db_loader.py
│   ├── analysis.py
│   ├── chatbot.py
│   └── run_pipeline.py
├── database/
│   └── weather.db
├── notebook/
│   ├── analysis.ipynb
│   ├── last_10_days_observed_vs_normal_high.png
│   ├── last_10_days_temperature_swing.png
│   ├── three_month_temperature_trend.png
│   └── three_month_monthly_precipitation.png
└── README.md
```

## Setup instructions

```bash
pip install pandas matplotlib pdfplumber pypdf
```

Notes:

- `sqlite3` is part of the Python standard library, so it does not need a separate install.
- `pdfplumber` is the preferred PDF parser.
- `pypdf` is used as a fallback when `pdfplumber` is unavailable.

## Where to place input data

- PDFs: `weather_project/data/raw/pdfs/`
- CSV: `weather_project/data/raw/weather.csv`

The provided source files are already copied into those paths.

## How to run

Run the full pipeline in one command:

```bash
cd "/Users/sangramthakur/Downloads/Data Science/Assignment/whether_app/weather_project"
python3 scripts/run_pipeline.py
```

Or run each part individually:

```bash
python3 scripts/pdf_parser.py
python3 scripts/csv_cleaner.py
python3 scripts/db_loader.py
python3 scripts/analysis.py
python3 scripts/chatbot.py
python3 scripts/chatbot.py --demo
streamlit run scripts/streamlit_chatbot.py
```

## Database schema overview

The project stores both PDF and CSV data in one scalable table named `weather_data`.

Columns:

- `id`: integer primary key
- `date`: observation date
- `max_temp`: observed daily high in Fahrenheit
- `min_temp`: observed daily low in Fahrenheit
- `avg_temp`: observed daily average in Fahrenheit
- `precipitation`: precipitation in inches
- `departure`: CSV departure from normal average temperature when available
- `normal_max_temp`: normal daily high when available from the PDFs
- `normal_min_temp`: normal daily low when available from the PDFs
- `normal_avg_temp`: normal daily average when available from the PDFs
- `record_high_temp`: all-time record high for that calendar day when available
- `record_low_temp`: all-time record low for that calendar day when available
- `record_high_year`: year of the record high
- `record_low_year`: year of the record low
- `qc_flag`: PDF quality-control flag when present
- `source_file`: originating file name
- `source`: `pdf` or `csv`

Design choice:

- A single weather fact table keeps ingestion simple while allowing nullable metadata fields for different source types.
- PDF rows hold richer daily context such as normals and records.
- CSV rows provide broader 3-month coverage and departure values.

## Assumptions and data handling

- Missing values such as `M`, blanks, or broken fields are stored as `NULL`.
- Temperatures remain in Fahrenheit because the source material is already in Fahrenheit.
- PDF layouts vary, so the parser uses multiple regex patterns and safe fallbacks.
- If a PDF is unreadable, it is skipped instead of failing the pipeline.
- For last-10-day analysis, the PDFs are treated as the primary source.
- For some “warmer than average” calculations, CSV departure values are used as a fallback when a PDF does not explicitly include a normal average.
- The CSV contains a few inconsistent rows and missing dates; the cleaner preserves valid daily observations and drops summary rows.

## Analysis deliverables

The notebook and scripts answer the assignment questions:

1. Hottest day in the last 10 days and how far above the daily normal it was
2. Biggest swing between high and low
3. Count of warmer-than-average days in the last 10 days
4. Record high/low checks and near-record events
5. Past 3-month temperature and precipitation trends

Charts are generated from the SQLite database, not from hardcoded values.

## Chatbot grounding

The chatbot is grounded directly in the SQLite-backed analysis logic:

- it loads the database
- it builds last-10-day and 3-month analysis views
- it answers rule-based questions from those views

For the assignment demo, run:

```bash
python3 scripts/chatbot.py --demo
```

That prints 5 grounded example Q&A pairs you can use for recording.

For an interactive chatbot UI, run:

```bash
streamlit run scripts/streamlit_chatbot.py
```

The Streamlit chatbot works in two modes:

- without an OpenAI API key, it answers using the local SQLite-backed weather analysis
- with an OpenAI API key pasted into the sidebar, it uses OpenAI with grounded database context


