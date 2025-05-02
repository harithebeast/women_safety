# Women's Safety Analysis Dashboard - India

## Setup Instructions

### 1. Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### 2. Installation Steps

1. **Clone or download the project**
   - Download and extract the project files to your computer

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required packages**
   Create a requirements.txt file in your project directory and run:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Required Files
Make sure you have these files in your project directory:
- `main.py` (The main dashboard application)
- `crimes_against_women_2001-2014.csv` (Crime data)
- `districts_shapefile/gadm41_IND_shp/gadm41_IND_3.shp` (India districts shapefile)

### 4. Running the Dashboard
1. Open terminal/command prompt
2. Navigate to the project directory
3. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```
4. The dashboard will open in your default web browser

### 5. Using the Dashboard
- Use the sidebar to select:
  - Year of analysis
  - Types of crimes to analyze
  - Specific state to focus on
- Explore different tabs:
  - Geographic Crime Analysis
  - Key Metrics
  - District Rankings
  - Safety Analysis
  - ML Prediction

### 6. Troubleshooting
- If you see "ModuleNotFoundError", make sure all required packages are installed
- If maps don't load, check if the shapefile is in the correct directory
- For any data loading issues, verify the CSV file is present and properly formatted

## Required Packages
```txt
streamlit
pandas
numpy
geopandas
matplotlib
seaborn
scikit-learn
folium
streamlit-folium
plotly
```

## Data Sources
- Crime data: National Crime Records Bureau (NCRB)
- Geographic data: GADM database of Global Administrative Areas

For any issues or questions, please contact the repository maintainer. 