import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Titanic Passenger Survival Prediction System", page_icon="🚢")

st.title("🚢 Titanic Passenger Survival Prediction System")
st.write(
    "Enter the passenger details below and the trained "
    "Support Vector Classifier will predict whether the passenger survived."
)

# ---------------------------------------------------------------
# Load the trained model (saved in the notebook, Step 6.4)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("svc_trained_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------------------------------------------------------------
# Recreate the Label Encoders exactly as in the notebook (Step 5.1)
# ---------------------------------------------------------------
pclass_options = ["First", "Second", "Third"]
gender_options = ["Male", "Female"]
sibling_options = ["Zero", "One", "Two", "Three"]
embarked_options = ["Southampton", "Cherbourg", "Queenstown"]

pclass_label_encoder = LabelEncoder().fit(np.ravel(pclass_options))
gender_label_encoder = LabelEncoder().fit(np.ravel(gender_options))
sibling_label_encoder = LabelEncoder().fit(np.ravel(sibling_options))
embarked_label_encoder = LabelEncoder().fit(np.ravel(embarked_options))

# ---------------------------------------------------------------
# User input (Step 8.1)
# ---------------------------------------------------------------
st.subheader("Passenger Information")

pclass_input = st.selectbox("Passenger Class", pclass_options)
gender_input = st.selectbox("Gender", gender_options)
sibling_input = st.selectbox("Number of Siblings/Spouses Aboard", sibling_options)
embarked_input = st.selectbox("Port of Embarkation", embarked_options)

# ---------------------------------------------------------------
# Predict (Steps 8.2, 8.3, 8.5)
# ---------------------------------------------------------------
if st.button("Predict"):
    # Step 8.2: feature vector
    user_input = pd.DataFrame({
        "PClass": [pclass_input],
        "Gender": [gender_input],
        "Sibling": [sibling_input],
        "Embarked": [embarked_input],
    })

    # Step 8.3: label encoding
    unseen_data_features = user_input.copy()
    unseen_data_features["PClass"] = pclass_label_encoder.transform(user_input["PClass"])
    unseen_data_features["Gender"] = gender_label_encoder.transform(user_input["Gender"])
    unseen_data_features["Sibling"] = sibling_label_encoder.transform(user_input["Sibling"])
    unseen_data_features["Embarked"] = embarked_label_encoder.transform(user_input["Embarked"])

    # Step 8.5: prediction
    predicted_survival = model.predict(unseen_data_features)[0]

    if predicted_survival == 1:
        st.success("✅ SURVIVED")
    else:
        st.error("❌ NOT SURVIVED")

st.caption("This is a learning project (AIC354, COMSATS Lahore).")
