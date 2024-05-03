import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu

#Set page configuration
st.set_page_config(
    page_title='Water Potabiliy Prediction',
    page_icon='::magnifier::',
    initial_sidebar_state='expanded',
)

def predict_potability(model, X):
    # Create a new StandardScaler instance
    scaler = StandardScaler()
    # Fit the scaler on the input data
    scaler.fit(X)
    # Scale input data using the fitted scaler
    X_scaled = scaler.transform(X)
    # Predict potability
    y_pred = model.predict(X_scaled)
    return y_pred


def load_lottie(url): 
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Center align the input fields
st.markdown("<h1 style='text-align: center;'>Water Potability Prediction System</h1>", unsafe_allow_html=True)


lottie_link = "https://lottie.host/12d4c21a-4d80-44bf-bee9-f12771920386/ho13u1DTVB.json"
animation = load_lottie(lottie_link)

if animation is not None:
    st_lottie(animation, speed=1, height=300, key="initial")
else:
    st.write("Failed to load animation")

# Loading the models 
clf_model = joblib.load("D:\Water Potability System (AI)'24\water-potability-prediction-system\clf")
decision_tree_model = joblib.load("D:\Water Potability System (AI)'24\water-potability-prediction-system\decision_tree")
knn_model = joblib.load("D:\Water Potability System (AI)'24\water-potability-prediction-system\knn")
log_reg_model = joblib.load("D:\Water Potability System (AI)'24\Water-Potability-Prediction-System\log_reg")
svm_model = joblib.load("D:\Water Potability System (AI)'24\Water-Potability-Prediction-System\svm")

# Define the columns
columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon',
            'Trihalomethanes', 'Turbidity', 'TDS', 'Chloramines_to_Solids_Ratio', 'Chloramines_per_Conductivity']


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

if choose=='About':
    st.write('### Water Potability System About:')
    st.write('---')
    st.write("##### Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection.\n ##### This is important as a health and development issue at a national, regional, and local level.\n ##### In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit, since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the interventions.\n ##### This classification project involves building a model to evaluate water potability to help you access safe drinking water. 🚰💙")

elif choose == 'Predictions':
    st.write('### Water Potability System Predictions:')
    st.write('---')
    # Add input fields for each column
    inputs = {}
    for col in columns:
        if col in ['TDS', 'Chloramines_to_Solids_Ratio', 'Chloramines_per_Conductivity']:
            inputs[col] = 0
        else:
            inputs[col] = st.number_input(f'**Enter {col}**', min_value=0.0, format="%.9f")

    session_state = st.session_state
    
    if 'display_derived_values' not in session_state:
        session_state.display_derived_values = False
    
    if st.button('Display Derived Values'):
        session_state.display_derived_values = True
        
    if session_state.display_derived_values:
        # Ensure all required keys are present in the inputs dictionary
        for col in ['Solids', 'Chloramines', 'Sulfate', 'Trihalomethanes', 'Conductivity']:
            if col not in inputs:
                inputs[col] = 0
        # Calculate and display derived columns
        inputs['TDS'] = inputs['Solids'] + inputs['Chloramines'] + inputs['Sulfate'] + inputs['Trihalomethanes']
        
        count =0
        
        while (inputs['Solids'] == 0 or inputs['Conductivity'] == 0):
            if ((inputs['Solids'] == 0 or inputs['Conductivity'] == 0) and count<1):
                st.write('Solids and Conductivity Value Cannot Equal Zero. ❌')
                count+=1
            elif ((inputs['Solids'] == 0 or inputs['Conductivity'] == 0) and count >= 1):
                continue
            
        inputs['Chloramines_to_Solids_Ratio'] = inputs['Chloramines'] / inputs['Solids']
        inputs['Chloramines_per_Conductivity'] = inputs['Chloramines'] / inputs['Conductivity']
        st.write('Derived Columns:')
        st.write('TDS:', inputs['TDS'])
        st.write('Chloramines to Solids Ratio:', inputs['Chloramines_to_Solids_Ratio'])
        st.write('Chloramines per Conductivity:', inputs['Chloramines_per_Conductivity'])
            
        values_list = [] 
        for col, value in inputs.items():
            # Append each value to the list
            values_list.append(value)
        # Convert the list to a NumPy array
        X = np.array([values_list])
        # Define models dictionary
        models = {
            "Random Forest": clf_model,
            "Decision Tree": decision_tree_model,
            "KNN": knn_model,
            "Logistic Regression": log_reg_model,
            "SVM": svm_model
        }
        # Select model using dropdown menu
        selected_model = st.selectbox("Select Model", list(models.keys()), key='model_selection')
        # Button to trigger prediction
        if st.button('Predict'):
            # Predict potability using the selected model
            y_pred = predict_potability(models[selected_model], X)
            # Display the predicted potability
            if y_pred[0] == 1:
                st.write("#### Our Prediction Says: This Water Is Safe To Drink. 💙")
                st.balloons()
            elif y_pred[0] == 0:
                st.write("#### Our Prediction Says: This Water Isn't Safe To Drink. ❌")

elif choose == 'Graphs':
    st.write('### Water Potability System Graphs :')
    st.write('---')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("### Confusion Matrix Random Forest:")
    st.image("D:\Water Potability System (AI)'24\Water-Potability-Prediction-System\Graphs\Matrix RF.png")
    st.write("### Confusion Matrix Decision Tree:")
    st.image("D:\Water Potability System (AI)'24\Water-Potability-Prediction-System\Graphs\Matix DT.png")
    st.write("### Confusion Matrix Support Vector Machine:")
    st.image("D:\Water Potability System (AI)'24\Water-Potability-Prediction-System\Graphs\Matrix SVM.png")
    st.write("### Confusion Matrix Logistic Regression:")
    st.image("D:\Water Potability System (AI)'24\Water-Potability-Prediction-System\Graphs\Matrix LR.png")
    st.write("### Confusion Matrix KNN:")
    st.image("D:\Water Potability System (AI)'24\Water-Potability-Prediction-System\Graphs\Matrix KNN.png")
    st.write("### Box Plot Graph Before Scaling:")
    st.image("D:\Water Potability System (AI)'24\Water-Potability-Prediction-System\Graphs\Original box plot.png")
    st.write("### Box Plot Graph After Scaling:")
    st.image("D:\Water Potability System (AI)'24\Water-Potability-Prediction-System\Graphs\Boxplots of Scaled Water Quality Parameters.png")
    st.write("### Correlation Map:")
    st.image("D:\Water Potability System (AI)'24\Water-Potability-Prediction-System\Graphs\Heat map4.png")