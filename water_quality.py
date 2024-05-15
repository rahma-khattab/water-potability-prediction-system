import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu

######################## Functions & Definitions ########################

# Scale features and predict
def predict_potability(model, X):
    # Scale input data using the fitted scaler
    X_scaled = scaler.transform(X)
    # Predict potability
    y_pred = model.predict(X_scaled)
    return y_pred

# Display animation
def load_lottie(url): 
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Loading the scaler
scaler = joblib.load("D:\Water Potability System (AI)'24\water-potability-prediction-system\\scaler.joblib")

# Loading the models 
rf_model = joblib.load("D:\Water Potability System (AI)'24\water-potability-prediction-system\\rf.joblib")
GB_clf_model = joblib.load("D:\Water Potability System (AI)'24\water-potability-prediction-system\GB.joblib")
decision_tree_model = joblib.load("D:\Water Potability System (AI)'24\water-potability-prediction-system\decision_tree.joblib")
knn_model = joblib.load("D:\Water Potability System (AI)'24\water-potability-prediction-system\knn.joblib")
log_reg_model = joblib.load("D:\Water Potability System (AI)'24\water-potability-prediction-system\log_reg.joblib")
svm_model = joblib.load("D:\Water Potability System (AI)'24\water-potability-prediction-system\svm.joblib")

# Define models dictionary
models = {
    "GradientBoostingClassifier": GB_clf_model,
    "Random Forest": rf_model,
    "Decision Tree": decision_tree_model,
    "KNN": knn_model,
    "Logistic Regression": log_reg_model,
    "SVM": svm_model
}

# Define confusion matrices dictionary
confusionMatrices = {
    "GradientBoostingClassifier": "D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Matrix GB.png",
    "Random Forest": "D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Matrix RF.png",
    "Decision Tree": "D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Matix DT.png",
    "KNN": "D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Matrix KNN.png",
    "Logistic Regression": "D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Matrix LR.png",
    "SVM": "D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Matrix SVM.png"
}

# Define the columns
columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon',
            'Trihalomethanes', 'Turbidity']


######################## Main Display ########################

#Set page configuration
st.set_page_config(
    page_title='Water Potabiliy Prediction',
    page_icon='::magnifier::',
    initial_sidebar_state='expanded',
)

# Center align the header
st.markdown("<h1 style='text-align: center;'>Water Potability Prediction System</h1>", unsafe_allow_html=True)

# Display animation at the top
lottie_link = "https://lottie.host/12d4c21a-4d80-44bf-bee9-f12771920386/ho13u1DTVB.json"
animation = load_lottie(lottie_link)

# Error handling for animation
if animation is not None:
    st_lottie(animation, speed=1, height=300, key="initial")
else:
    st.write("Failed to load animation")


# Sidebar Design
with st.sidebar:
    choose = option_menu(None, ["About", "Predictions", "Graphs"],
                        icons=[ 'house','kanban', 'book'],
                        menu_icon="app-indicator", default_index=0,
                        styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": '#E0E0EF', "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#428DFF"},
    }
    )


# About Page
if choose == 'About':
    st.write('### Water Potability System About:')
    st.write('---')
    st.write("##### Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection.\n ##### This is important as a health and development issue at a national, regional, and local level.\n ##### In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit, since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the interventions.\n ##### This classification project involves building a model to evaluate water potability to help you access safe drinking water. 🚰💙")


# Predictions Page
elif choose == 'Predictions':
    st.write('### Water Potability System Predictions:')
    st.write('---')

    # Add input fields for all columns and take input from user
    inputs = {}
    for col in columns:
        inputs[col] = st.number_input(f'**Enter {col}**', min_value=0.0, format="%.9f")

    # Detect if Enter Values is pressed
    session_state = st.session_state
    if 'enter_values' not in session_state:
        session_state.enter_values = False
    
    # Toggles button so that when it's pressed, it stays pressed and doesn't refresh.
    if st.button('Enter Values'):
        session_state.enter_values = True

    if session_state.enter_values:       
        # Create list with columns values from inputs dictionary
        values_list = [] 
        for col, value in inputs.items():
            # Append each value to the list
            values_list.append(value)
        
        # Convert the list to a NumPy array
        X = np.array([values_list]).reshape(1, -1)
        
        # Select model using dropdown menu
        selected_model = st.selectbox("Select Model", list(models.keys()), key='model_selection')
        
        # Button to trigger prediction
        if st.button('Predict'):
            # Predict potability using the selected model
            y_pred = predict_potability(models[selected_model], X)

            # Display the predicted potability
            if y_pred == 1:
                st.write("#### Our Prediction Says: This Water Is Safe To Drink. 💙")
                st.balloons()
            elif y_pred == 0:
                st.write("#### Our Prediction Says: This Water Isn't Safe To Drink. ❌")
            st.write("#### This is the confussion matrix for the selected model:")
            st.write(" ")
            st.image(confusionMatrices[selected_model])

# Graphs Page
elif choose == 'Graphs':
    st.write('### Water Potability System Graphs :')
    st.write('---')
    
    
    # Pie chart
    st.write("### Distribution of potability class:")
    st.write("##### Before Oversampling:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\True and false biased.png")

    st.write("##### After Oversampling:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\True and false oversampled.png")
    

    # Box Plot
    st.write("### Box Plot Graph Before Scaling:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Boxplots of non Scaled Water Quality Parameters.png")
    st.write("### Box Plot Graph After Scaling:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Boxplots of Scaled Water Quality Parameters.png")


    # Correlation Map
    st.write("### Correlation Map Before Oversampling And Scaling:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Heat map.png")
    

    # Histograms
    st.write("### Histograms Of Unscaled Data Of Columns Containing Null Values:")
    st.write("##### Sulfate Histogram:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Sulfate Histogram.png")

    st.write("##### ph Histogram:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\ph Histogram.png")

    st.write("##### Trihalomethanes Histogram:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Trihalomethanes Histogram.png")
    
    st.write("### Histograms Of Scaled Data:")
    st.write("##### Chloromines Histogram:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\HistogramChloromines.png")
    
    st.write("##### Conductivity Histogram:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\HistogramConduct.png")
    
    st.write("##### Hardness Histogram:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\HistogramHardness.png")
    
    st.write("##### Organic Carbon Histogram:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\HistogramOrganicCO2.png")
    
    st.write("##### PH Histogram:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\HistogramPH.png")
    
    st.write("##### Solids Histogram:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\HistogramSolids.png")
    
    st.write("##### Sulphate Histogram:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\HistogramSulphate.png")
    
    st.write("##### Trihalomethanes Histogram:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\HistogramTrihalo.png")
    
    st.write("##### Turbidiry Histogram:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\HistogramTurbidity.png")
    
    # Confusion Matrices
    st.write("### Confusion Matrix Random Forest:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Matrix RF.png")

    st.write("### Confusion Matrix Decision Tree:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Matix DT.png")

    st.write("### Confusion Matrix Support Vector Machine:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Matrix SVM.png")

    st.write("### Confusion Matrix Logistic Regression:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Matrix LR.png")

    st.write("### Confusion Matrix KNN:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Matrix KNN.png")
    
    st.write("### Confusion Matrix Gradient Boosting:")
    st.image("D:\Water Potability System (AI)'24\water-potability-prediction-system\Graphs\Matrix GB.png")