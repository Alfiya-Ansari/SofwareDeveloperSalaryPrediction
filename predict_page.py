import streamlit as st 
import pickle 
import numpy as np 

##command to run streamlit: python -m streamlit run predict_page.py

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_co = data["le_co"]
le_ed = data["le_ed"]


def show_predict_page():
    st.title('Software Developer Salary Prediction')

    st.write("""### Fill the Below Information to Predict Your Salary""")
    countries = (
        "United States of America",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )
    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)
    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        x = np.array([[country, education, experience]])
        x[:, 0] = le_co.transform(x[:,0])
        x[:, 1] = le_ed.transform(x[:,1])
        x = x.astype(float)

        salary = regressor.predict(x)
        st.subheader(f"The Estimated Salary is ${salary[0]:.2f}")
