import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.inspection import permutation_importance

# Set page configuration
st.set_page_config(
    page_title="Women's Safety Analysis - India",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    .st-emotion-cache-16idsys p {
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# 1. Load Data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("crimes_against_women_2001-2014.csv")
    return df

# 2. Load Shapefile with caching
@st.cache_data
def load_geodata():
    try:
        gdf = gpd.read_file("districts_shapefile/gadm41_IND_shp/gadm41_IND_3.shp")
        gdf = gdf.rename(columns={
            'NAME_1': 'State',
            'NAME_2': 'District',
            'NAME_3': 'SubDistrict'
        })
        return gdf
    except Exception as e:
        st.error(f"Error loading shapefile: {str(e)}")
        return None

# Load the data
data = load_data()
geo_df = load_geodata()

# Main title with styling
st.markdown("""
    <h1 style='text-align: center; color: #1E88E5;'>
        🛡️ Women's Safety Analysis Dashboard - India
    </h1>
    <p style='text-align: center; color: #666;'>
        Analyzing crime patterns and safety metrics across Indian districts (2001-2014)
    </p>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("📊 Analysis Controls")

# Year filter
years = sorted(data['Year'].unique())
selected_year = st.sidebar.selectbox("Select Year", years, index=len(years)-1)

# Filter data by year
data_filtered = data[data['Year'] == selected_year].copy()

# Crime type analysis
crime_types = ['Rape', 'Kidnapping and Abduction', 'Dowry Deaths', 
               'Assault on women with intent to outrage her modesty',
               'Insult to modesty of Women', 'Cruelty by Husband or his Relatives',
               'Importation of Girls']

selected_crimes = st.sidebar.multiselect(
    "Select Crime Types to Analyze",
    crime_types,
    default=['Rape', 'Dowry Deaths', 'Cruelty by Husband or his Relatives']
)

# Calculate total crimes for selected types
if selected_crimes:
    data_filtered.loc[:, 'Total_Crimes'] = data_filtered[selected_crimes].sum(axis=1)
else:
    data_filtered.loc[:, 'Total_Crimes'] = data_filtered[crime_types].sum(axis=1)

# Create safety label based on crime rate
median_crimes = data_filtered['Total_Crimes'].median()
data_filtered.loc[:, 'label'] = data_filtered['Total_Crimes'].apply(
    lambda x: 'Safe' if x < median_crimes else 'Unsafe'
)

# State selection
selected_state = st.sidebar.selectbox(
    "Select State to Focus",
    ["All States"] + sorted(data_filtered['STATE/UT'].unique().tolist())
)

# Merge geodata with filtered data
merged = geo_df.merge(
    data_filtered.rename(columns={'STATE/UT': 'State', 'DISTRICT': 'District'}),
    on=['State', 'District'],
    how='left'
)

# Main content area - using full width for map
st.subheader("🗺️ Geographic Crime Analysis")

# Filter and merge data
if selected_state != "All States":
    map_data = merged[merged["State"] == selected_state].copy()
else:
    map_data = merged.copy()

# Handle missing values for visualization
map_data['Total_Crimes'] = map_data['Total_Crimes'].fillna(0)

# Create choropleth map
fig = px.choropleth(
    map_data,
    geojson=map_data.geometry,
    locations=map_data.index,
    color="Total_Crimes",
    hover_name="District",
    hover_data={
        "State": True,
        "Total_Crimes": ":.0f",
        "label": True,
        **{crime: ":.0f" for crime in selected_crimes}
    },
    color_continuous_scale=[[0, "white"], [0.01, "lightpink"], [1, "darkred"]],
    range_color=[0, map_data["Total_Crimes"].quantile(0.95)]
)

fig.update_geos(
    fitbounds="locations",
    visible=False,
    projection_scale=1 if selected_state == "All States" else 2
)

st.plotly_chart(fig, use_container_width=True)

# Key Metrics Section below the map
st.subheader("📈 Key Metrics")

# Create three columns for metrics
col1, col2, col3 = st.columns(3)

with col1:
    total_crimes = int(map_data["Total_Crimes"].sum())
    st.metric("Total Crimes", f"{total_crimes:,}")
    
with col2:
    avg_crimes = map_data["Total_Crimes"].mean()
    st.metric("Average per District", f"{avg_crimes:,.0f}")

with col3:
    high_crime_districts = len(map_data[map_data["Total_Crimes"] > avg_crimes])
    st.metric("Districts Above Average", f"{high_crime_districts}")

# Trend Analysis in two columns
col_trend, col_dist = st.columns(2)

with col_trend:
    st.subheader("📊 Crime Trends")
    
    # Create time series for selected crimes
    if selected_state != "All States":
        trend_data = data[data['STATE/UT'] == selected_state]
    else:
        trend_data = data
        
    trend_fig = go.Figure()
    
    for crime in selected_crimes:
        yearly_data = trend_data.groupby('Year')[crime].sum()
        trend_fig.add_trace(go.Scatter(
            x=yearly_data.index,
            y=yearly_data.values,
            name=crime,
            mode='lines+markers'
        ))
        
    trend_fig.update_layout(
        title="Yearly Crime Trends",
        xaxis_title="Year",
        yaxis_title="Number of Cases",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(trend_fig, use_container_width=True)

with col_dist:
    st.subheader("🔍 Crime Distribution")
    
    # Create pie chart for crime distribution
    crime_totals = map_data[selected_crimes].sum()
    fig_pie = px.pie(
        values=crime_totals.values,
        names=crime_totals.index,
        title="Distribution of Crime Types"
    )
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

# Detailed Analysis Section
st.subheader("📋 Detailed Analysis")

tab1, tab2, tab3 = st.tabs(["District Rankings", "Safety Analysis", "ML Prediction"])

with tab1:
    # District rankings
    st.subheader("District-wise Crime Statistics")
    
    # Prepare data for display
    display_data = map_data[["District", "State", "Total_Crimes", "label"] + selected_crimes].copy()
    display_data = display_data.sort_values("Total_Crimes", ascending=False)
    
    # Convert numeric columns to appropriate format
    numeric_cols = ["Total_Crimes"] + selected_crimes
    for col in numeric_cols:
        display_data[col] = display_data[col].fillna(0).astype(int)
    
    st.dataframe(display_data, use_container_width=True)

with tab2:
    # Safety analysis
    col_safety1, col_safety2 = st.columns(2)
    
    with col_safety1:
        # Calculate safety metrics
        safe_districts = len(map_data[map_data["label"] == "Safe"])
        total_districts = len(map_data)
        
        # Add error handling for safety percentage calculation
        if total_districts > 0:
            safety_percentage = (safe_districts / total_districts) * 100
            st.metric("Safety Percentage", f"{safety_percentage:.1f}%")
        else:
            st.warning("No data available to calculate safety percentage.")
    
    with col_safety2:
        # Create safety distribution bar chart
        if total_districts > 0:
            safety_dist = map_data["label"].value_counts()
            fig_safety = px.bar(
                x=safety_dist.index,
                y=safety_dist.values,
                title="Distribution of Safe vs Unsafe Districts",
                labels={"x": "Safety Status", "y": "Number of Districts"}
            )
            st.plotly_chart(fig_safety, use_container_width=True)
        else:
            st.warning("No data available to display safety distribution.")

with tab3:
    st.subheader("🤖 Machine Learning Analysis")
    
    # Prepare data for ML
    ml_data = map_data[selected_crimes + ["label"]].copy()
    ml_data = ml_data.dropna()  # Remove rows with missing values
    
    if len(ml_data) > 0:
        # Prepare features and target
        X = ml_data[selected_crimes]
        y = ml_data["label"]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train SVM model
        model = SVC(kernel="rbf", C=1, gamma="scale")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Display results
        col_ml1, col_ml2 = st.columns(2)
        
        with col_ml1:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred)
            st.code(report)
            
            # Add explanation
            st.markdown("""
                **Metrics Explanation:**
                - Precision: Accuracy of positive predictions
                - Recall: Ability to find all positive instances
                - F1-score: Balance between Precision and Recall
                - Support: Number of samples
            """)
        
        with col_ml2:
            st.subheader("Confusion Matrix")
            fig_cm, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                confusion_matrix(y_test, y_pred),
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax,
                xticklabels=['Unsafe', 'Safe'],
                yticklabels=['Unsafe', 'Safe']
            )
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig_cm)
        
        # Feature Importance Analysis
        st.subheader("Feature Importance Analysis")
        
        # Use permutation importance for feature analysis
        perm_importance = permutation_importance(
            model, X_test, y_test,
            n_repeats=10,
            random_state=42
        )
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'Feature': selected_crimes,
            'Importance': perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        fig_imp = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Crime Type Importance in Safety Prediction'
        )
        fig_imp.update_layout(height=400)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # Add district safety prediction tool
        st.subheader("🎯 District Safety Predictor")
        st.write("Use the sliders below to predict safety for a district based on crime statistics:")
        
        # Create sliders for each crime type
        prediction_data = {}
        for crime in selected_crimes:
            max_value = float(map_data[crime].max())
            prediction_data[crime] = st.slider(
                f"{crime}",
                min_value=0.0,
                max_value=max_value,
                value=0.0,
                step=1.0
            )
        
        # Make prediction button
        if st.button("Predict Safety"):
            # Prepare input data
            input_data = np.array([prediction_data[crime] for crime in selected_crimes]).reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.decision_function(input_scaled)[0]
            
            # Display prediction with confidence
            confidence = abs(probability)
            if prediction == "Safe":
                st.success(f"Prediction: {prediction} (Confidence: {confidence:.2f})")
            else:
                st.error(f"Prediction: {prediction} (Confidence: {confidence:.2f})")
            
            # Add interpretation
            st.info("""
                💡 Interpretation:
                - A higher confidence value indicates a stronger prediction
                - The model uses standardized crime statistics for prediction
                - Consider multiple factors when interpreting the results
            """)
    else:
        st.warning("Insufficient data for ML analysis. Please select more crime types or ensure data availability.")

# Footer
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Data Source: National Crime Records Bureau (NCRB) | Last Updated: 2014</p>
    </div>
""", unsafe_allow_html=True)
