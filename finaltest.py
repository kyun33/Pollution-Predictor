import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler, LabelEncoder
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import requests
import re
# import ace_tools as tools

# Load the dataset
df = pd.read_csv('updated_pollution_dataset.csv')

# Function for removing outliers using IQR
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Apply outlier removal for pollutants and features
pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
features = ['Temperature', 'Humidity', 'Proximity_to_Industrial_Areas', 'Population_Density']

df_removed_outliers = remove_outliers(df, pollutants + features)




# Sidebar
st.sidebar.markdown('<span style="color: #fff; font-weight: bold; font-size: 1.1em;">Choose a section</span>', unsafe_allow_html=True)
section = st.sidebar.selectbox("Choose a section", ("EDA", "Prediction", "Live Air Quality"), label_visibility='collapsed')
#st.title("Air Quality Analysis")

st.markdown("""
    <style>
    /* Set background color for the main content */
    .stApp {
        background-color: #f4fbf4;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    /* Style the sidebar */
    section[data-testid="stSidebar"] {
        background-color: #22543d;
        color: #fff;
    }
    /* Style headers */
    h1, h2, h3, h4 {
        color: #22543d;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    /* Style dataframe headers */
    .css-1iyq8pb {
        background-color: #22543d !important;
        color: #fff !important;
    }
    /* Style buttons */
    .stButton>button {
        background-color: #38a169;
        color: #fff;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.5em;
        font-size: 1.1em;
        transition: background 0.3s;
    }
    .stButton>button:hover {
        background-color: #22543d;
        color: #fff;
    }
    /* Style selectbox */
    .stSelectbox>div>div>div {
        background-color: #e6f4ea;
        color: #22543d;
        border-radius: 8px;
    }
    /* Style slider */
    .stSlider>div>div>div {
        background-color: #e6f4ea;
    }
    /* Style subheaders */
    .stMarkdown h3 {
        color: #38a169;
    }
    /* Green slider track and thumb */
    input[type=range]::-webkit-slider-thumb {
        background: #38a169;
        border: 2px solid #22543d;
    }
    input[type=range]::-webkit-slider-runnable-track {
        background: #c6f6d5;
    }
    input[type=range]::-ms-fill-lower {
        background: #38a169;
    }
    input[type=range]::-ms-fill-upper {
        background: #c6f6d5;
    }
    /* For Firefox */
    input[type=range]::-moz-range-thumb {
        background: #38a169;
        border: 2px solid #22543d;
    }
    input[type=range]::-moz-range-track {
        background: #c6f6d5;
    }
    /* For IE */
    input[type=range]::-ms-thumb {
        background: #38a169;
        border: 2px solid #22543d;
    }
    input[type=range]::-ms-fill-lower {
        background: #38a169;
    }
    input[type=range]::-ms-fill-upper {
        background: #c6f6d5;
    }
    /* Style the prediction button */
    .stButton>button {
        background-color: #38a169 !important;
        color: #fff !important;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.5em;
        font-size: 1.1em;
        transition: background 0.3s;
    }
    .stButton>button:hover {
        background-color: #22543d !important;
        color: #fff !important;
    }
            
    
    </style>
""", unsafe_allow_html=True)



st.markdown("""
    <div style='background-color: #22543d; padding: 1.5em; border-radius: 10px; margin-bottom: 2em;'>
        <h1 style='color: #fff; text-align: center;'>Air Quality Analysis Dashboard</h1>
        <p style='color: #e6f4ea; text-align: center; font-size: 1.2em;'>
            Explore, predict, and monitor air quality across US cities.
        </p>
    </div>
""", unsafe_allow_html=True)

# Exploratory Data Analysis (EDA)
if section == "EDA":
    st.subheader('Exploratory Data Analysis (EDA)')
    st.write("### Dataset Overview")
    st.dataframe(df_removed_outliers)
    st.write("### Distribution of Pollutants")
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
    col1, col2 = st.columns(2)
    with col1:
        for pollutant in pollutants[:3]:
            st.subheader(f"Distribution of {pollutant}")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(df_removed_outliers[pollutant], kde=True, ax=ax)
            st.pyplot(fig)

    with col2:
        for pollutant in pollutants[3:]:
            st.subheader(f"Distribution of {pollutant}")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(df_removed_outliers[pollutant], kde=True, ax=ax)
            st.pyplot(fig)

    st.write("### Correlation Heatmap")
    df_numeric = df_removed_outliers.select_dtypes(include=['float64', 'int64'])
    corr_matrix = df_numeric.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    st.pyplot(fig)

    st.write("### Boxplot of Pollutants")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df[pollutants], ax=ax)
    st.pyplot(fig)

# Apply the outlier removal function to each column
    st.write("### Boxplot of Pollutants W/O Outliers")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df_removed_outliers[pollutants], ax=ax)
    st.pyplot(fig)



# Prediction: Pollutant & Air Quality Prediction
if section == "Prediction":
    st.subheader('Pollutant & Air Quality Prediction')

            # Log transformation for skewed pollutants (to stabilize variance)
    df_removed_outliers['PM2.5'] = np.log1p(df_removed_outliers['PM2.5'])
    df_removed_outliers['PM10'] = np.log1p(df_removed_outliers['PM10'])
    df_removed_outliers['NO2'] = np.log1p(df_removed_outliers['NO2'])
    df_removed_outliers['SO2'] = np.log1p(df_removed_outliers['SO2'])
    df_removed_outliers['CO'] = np.log1p(df_removed_outliers['CO'])

    # Separate numeric and categorical columns
    numeric_cols = df_removed_outliers.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df_removed_outliers.select_dtypes(include=['object']).columns

    # Fill missing values for numeric columns with median
    df_removed_outliers[numeric_cols] = df_removed_outliers[numeric_cols].fillna(df_removed_outliers[numeric_cols].median())

    # Fill missing values for categorical columns with mode (most frequent value)
    df_removed_outliers[categorical_cols] = df_removed_outliers[categorical_cols].fillna(df_removed_outliers[categorical_cols].mode().iloc[0])

    # Input sliders
    proximity_to_industry = st.slider("Proximity to Industrial Areas (km)", 2, 26, 13)
    population_density = st.slider("Population Density (people/km²)", 0, 1000, 500)
    temperature = st.slider("Temperature of Area (°C)", 0, 70, 30)
    humidity = st.slider("Humidity of Area (%)", 30, 150, 90)

    button_clicked = st.button("Predict")

    # Predict Air Quality using RandomForestClassifier
    y1 = df_removed_outliers['Air Quality']
    X1 = df_removed_outliers.drop(columns=['Air Quality'])  # Drop the target variable columns
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=420)

    model1 = RandomForestClassifier(n_estimators=200, random_state=42, oob_score=True)
    model1.fit(X1_train, y1_train)

    if button_clicked:
        accuracy = model1.score(X1_test, y1_test)
        st.write(f"#### Model Accuracy: {accuracy:.4f}")

        def predict_pollution_level(user_proximity, user_density):
            # Filter the dataset based on proximity and density ranges
            filtered_data = df_removed_outliers[
                (df_removed_outliers['Population_Density'] == user_density) & 
                (df_removed_outliers['Proximity_to_Industrial_Areas'] == user_proximity)
            ]

            if filtered_data.empty:
                # If no exact matches, filter within a range
                filtered_data = df_removed_outliers[
                    (df_removed_outliers['Population_Density'] >= user_density * 0.9) & 
                    (df_removed_outliers['Population_Density'] <= user_density * 1.1) & 
                    (df_removed_outliers['Proximity_to_Industrial_Areas'] >= user_proximity * 0.9) & 
                    (df_removed_outliers['Proximity_to_Industrial_Areas'] <= user_proximity * 1.1)
                ]
            
            if filtered_data.empty:
                filtered_data = df_removed_outliers

            mean_values = filtered_data[['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO']].mean()

            # Create a new data point with user input and the calculated mean values
            user_input = pd.DataFrame({
                'Temperature': [mean_values['Temperature']],  
                'Humidity': [mean_values['Humidity']],
                'PM2.5': [mean_values['PM2.5']],
                'PM10': [mean_values['PM10']],
                'NO2': [mean_values['NO2']],
                'SO2': [mean_values['SO2']],
                'CO': [mean_values['CO']],
                'Proximity_to_Industrial_Areas': [user_proximity],  
                'Population_Density': [user_density]  
            })

            predicted_pollution_level = model1.predict(user_input)
            return predicted_pollution_level[0]

        predicted_pollution_level = predict_pollution_level(proximity_to_industry, population_density)
        st.write(f"#### Predicted Pollution Level: {predicted_pollution_level}")

        # Evaluate classifier
        pred_train = model1.predict(X1_train)
        pred_test = model1.predict(X1_test)

        max_values = []
        # Regression models for pollutant prediction
        pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
        pollutant_predictions = {}

        for pollutant in pollutants:
            X = df_removed_outliers[['Temperature', 'Humidity', 'Proximity_to_Industrial_Areas', 'Population_Density']]
            y_pollutant = df_removed_outliers[pollutant]

            max_values.append(max(y_pollutant.tolist()))
            X_train_pollutant, X_test_pollutant, y_train_pollutant, y_test_pollutant = train_test_split(X, y_pollutant, test_size=0.25, random_state=42)
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_train_pollutant, y_train_pollutant)

            y_pred_pollutant = model.predict(X_test_pollutant)

            mae = mean_absolute_error(y_test_pollutant, y_pred_pollutant)
            rmse = mean_squared_error(y_test_pollutant, y_pred_pollutant, squared=False)
            #r2 = r2_score(y_test_pollutant, y_pred_pollutant)

            st.write(f"### Evaluation for {pollutant}")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"RMSE: {rmse:.2f}")
            #st.write(f"R² Score: {r2:.2f}")

            # Store prediction for user input
            user_input_df = pd.DataFrame([[temperature, humidity, proximity_to_industry, population_density]], 
                                          columns=['Temperature', 'Humidity', 'Proximity_to_Industrial_Areas', 'Population_Density'])

            pollutant_predictions[pollutant] = model.predict(user_input_df)[0]

        # Plot circular progress charts
        st.write("### Pollutant Predictions")

        def create_circular_chart(pollutant, predicted_value, percentage):
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=percentage,
                # delta={'reference': 0},
                title={'text': f'{pollutant} Prediction'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': 'black'},
                    'steps': [{'range': [0, 33], 'color': 'lightgreen'},
                              {'range': [33, 67], 'color': 'yellow'},
                              {'range': [67, 100], 'color': 'red'}],
                },
                number={'font': {'size': 30}, 'suffix': '%'},
            ))
            fig.update_layout(margin={'t': 0, 'b': 0, 'l': 0, 'r': 0}, height=120)
            return fig

        col1, col2, col3 = st.columns(3)
        for i, pollutant in enumerate(pollutants):
            predicted_value = pollutant_predictions[pollutant] 
            percentage = (predicted_value / max_values[i]) * 100
            chart = create_circular_chart(pollutant, predicted_value, percentage)
            if i < 2:
                with col1:
                    st.plotly_chart(chart)
                    st.write(f"Predicted {pollutant}: {predicted_value:.2f} µg/m³")
            elif 2 <= i < 4:
                with col2:
                    st.plotly_chart(chart)
                    st.write(f"Predicted {pollutant}: {predicted_value:.2f} µg/m³")
            else:
                with col3:
                    st.plotly_chart(chart)
                    st.write(f"Predicted {pollutant}: {predicted_value:.2f} µg/m³")
  # Confusion Matrix
        label_encoder = LabelEncoder()
        df_removed_outliers['Air Quality'] = label_encoder.fit_transform(df_removed_outliers['Air Quality']) + 1
        st.write("### Confusion Matrix")
        conf_matrix = confusion_matrix(y1_test, pred_test)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual Value')
        st.pyplot(fig)
if section == "Live Air Quality":

    API_KEY = '065322101517f6e49ac4efb1eb5a749b318c133fee1bac9cf18c219b97cf5225'

    filtered_cities = pd.read_csv('filtered_cities.csv')
    st.write(filtered_cities.shape)
    # Capitalize city names for display (use original city names)
    filtered_cities['city'] = filtered_cities['city'].str.title()
    filtered_cities = filtered_cities.sort_values(by="city")
    filtered_cities = filtered_cities.drop_duplicates(subset=["city"])

    # st.dataframe(filtered_cities)

    city_list = filtered_cities['city'].tolist()
    id_list = filtered_cities['Location ID'].tolist()
    city_choice = st.selectbox("Choose a city:", city_list)
  
    # Retrieve the corresponding Location ID for the selected city
    LOCATION_ID = filtered_cities[filtered_cities['city'] == city_choice]['Location ID'].values[0] if not filtered_cities[filtered_cities['city'] == city_choice].empty else None
    # st.write(filtered_cities.shape)

    def fetch_air_quality_data(api_key, location_id):
    # Fetch sensors for the given location
        location_url = f"https://api.openaq.org/v3/locations/{location_id}"
        headers = {"X-API-Key": api_key}
        response = requests.get(location_url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            #st.write(data)
            pm25_sensors = []
            sensor_id = []
            sensor_name = []
            sensor_units = []  # To store units of each pollutant

            for sensor in data["results"]:
                
                city_name = sensor['name']
                locality_name = sensor['locality']

            for sensor in data['results'][0]['sensors']:
                # Filter sensors for the desired pollutants
                if sensor["parameter"]["name"] in ["pm25", "pm10", "no2", "so2", "co"]:
                    pm25_sensors.append(sensor)
                    sensor_id.append(sensor['id'])
                    sensor_name.append(sensor["parameter"]["name"])
                    sensor_units.append(sensor["parameter"]["units"])  # Store the unit for each pollutant
                    
                pm25_df = pd.DataFrame(pm25_sensors)
            st.write(f"City: {city_name} Locality: {locality_name}")
            return sensor_id, sensor_name, sensor_units

    @st.cache_data(show_spinner=True)
    def fetch_pollutant_data(api_key, location_id):
        sensor_id, sensor_name, sensor_units = fetch_air_quality_data(api_key, location_id)
        pollutant_average_values = {name: [] for name in set(sensor_name)}  # Create a list for each pollutant
        pollutant_units = {name: None for name in set(sensor_name)}  # Store units for each pollutant

        # Loop through each sensor to fetch measurements
        for id, name, unit in zip(sensor_id, sensor_name, sensor_units):
            sensor_url = f"https://api.openaq.org/v3/sensors/{id}/measurements"
            headers = {"X-API-Key": api_key}
            response = requests.get(sensor_url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                values = [measurement['value'] for measurement in data['results']]
                
                # If there are values for the sensor, calculate the average and append it to the list for that pollutant
                if values:
                    pollutant_average_values[name].append(np.mean(values))
                    # Store the unit for the first sensor of this pollutant
                    if pollutant_units[name] is None:
                        pollutant_units[name] = unit

        final_pollutant_average = {}
        for pollutant, values in pollutant_average_values.items():
            if values:
                # Append the unit to the pollutant name in the final dictionary
                final_pollutant_average[pollutant] = {
                    "Average": np.mean(values),
                    "Unit": pollutant_units[pollutant]  # Include the unit for each pollutant
                }
            else:
                final_pollutant_average[pollutant] = {
                    "Average": None,
                    "Unit": None
                }

        return final_pollutant_average

    # Call the function to fetch and compute pollutant data
    pollutant_averages = fetch_pollutant_data(API_KEY, LOCATION_ID)

    st.markdown(f"### Air Quality Data for **{city_choice}**")

    if pollutant_averages:
        # Filter out pollutants with no data
        valid_pollutants = {k: v for k, v in pollutant_averages.items() if v['Average'] is not None}
        if valid_pollutants:
            cols = st.columns(len(valid_pollutants))
            for idx, (pollutant, data) in enumerate(valid_pollutants.items()):
                avg = data['Average']
                unit = data['Unit'] or ""
                with cols[idx]:
                    st.metric(
                        label=pollutant.upper(),
                        value=f"{avg:.2f} {unit}",
                        help=f"Average {pollutant.upper()} in {city_choice}"
                    )
        else:
            st.info("No pollutant data available for this city.")
    else:
        st.warning("No data returned from the API.")