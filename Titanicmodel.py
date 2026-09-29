from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Titanic-Dataset.csv"
MODEL_PATH = BASE_DIR / "titanic_svm_model.pkl"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
SEX_CODES = {"female": 0, "male": 1}
EMBARKED_CODES = {"C": 0, "Q": 1, "S": 2}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


st.set_page_config(page_title="Titanic Survival Explorer", page_icon="🚢", layout="wide")
st.title("Titanic Survival Explorer")
st.caption("Explore passenger outcomes and predict a passenger's survival with the trained SVM model.")

data = load_data()
model = load_model()

with st.sidebar.form("passenger_details"):
    st.subheader("Passenger details")
    passenger_class = st.selectbox("Passenger class", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["female", "male"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=22.0, step=1.0)
    siblings_spouses = st.number_input("Siblings / spouses aboard", min_value=0, max_value=10, value=0)
    parents_children = st.number_input("Parents / children aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    embarked = st.selectbox("Port of embarkation", ["C", "Q", "S"], index=2)
    submitted = st.form_submit_button("Predict survival", type="primary", use_container_width=True)

if submitted:
    passenger = pd.DataFrame(
        [[
            passenger_class,
            SEX_CODES[sex],
            age,
            siblings_spouses,
            parents_children,
            int(fare > 20),
            EMBARKED_CODES[embarked],
        ]],
        columns=FEATURES,
    )
    survived = int(model.predict(passenger)[0])
    if survived:
        st.success("The model predicts this passenger survived.")
    else:
        st.warning("The model predicts this passenger did not survive.")
    st.caption("This is a classification result, not a probability or guarantee.")
else:
    st.info("Enter passenger details in the sidebar to see a prediction.")

st.subheader("Dataset overview")
survival_rate = data["Survived"].mean() * 100
metric_columns = st.columns(3)
metric_columns[0].metric("Passengers", f"{len(data):,}")
metric_columns[1].metric("Observed survival rate", f"{survival_rate:.1f}%")
metric_columns[2].metric("Model", "RBF support vector machine")

chart_columns = st.columns(2)
survival_by_class = (
    data.groupby("Pclass", as_index=False)["Survived"].mean().assign(
        **{"Survival rate": lambda frame: frame["Survived"] * 100}
    )
)
survival_by_sex = (
    data.groupby("Sex", as_index=False)["Survived"].mean().assign(
        **{"Survival rate": lambda frame: frame["Survived"] * 100}
    )
)

with chart_columns[0]:
    st.markdown("**Survival rate by passenger class**")
    st.bar_chart(survival_by_class, x="Pclass", y="Survival rate", y_label="Survival rate (%)")

with chart_columns[1]:
    st.markdown("**Survival rate by sex**")
    st.bar_chart(survival_by_sex, x="Sex", y="Survival rate", y_label="Survival rate (%)")

st.caption("Training preprocessing is mirrored: sex and embarkation are label-encoded, and fares are split at 20.")

