
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from io import BytesIO

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Streamlit Page Setup ---
st.set_page_config(page_title="🌸 Iris Flower Prediction App", layout="centered")
st.title("🌸 Iris Flower Prediction with Machine Learning")
st.write("An interactive ML web app using **scikit-learn**, **pandas**, and **Streamlit**.")

# --- Load Dataset ---
iris = load_iris(as_frame=True)
df = iris.frame
X = df.drop(columns=["target"])
y = df["target"]
species = iris.target_names

st.subheader("📊 Dataset Overview")
st.dataframe(df.head())

# --- Visualization ---
st.subheader("📈 Visualize Dataset")

fig, ax = plt.subplots()
sns.scatterplot(x=df["sepal length (cm)"], y=df["petal length (cm)"],
                hue=[species[i] for i in y], palette="Set2", s=80)
plt.title("Sepal vs Petal Length by Species")
st.pyplot(fig)

# --- Sidebar: Model Selection ---
st.sidebar.header("🔧 Model Configuration")

model_name = st.sidebar.selectbox(
    "Select Model",
    ("Random Forest", "Logistic Regression", "K-Nearest Neighbors")
)

test_size = st.sidebar.slider("Test Split (%)", 10, 50, 25) / 100.0
scale_data = st.sidebar.checkbox("Scale Data", value=True)

# --- Model Parameters ---
if model_name == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 10, 200, 100)
    max_depth = st.sidebar.slider("max_depth (None=0)", 0, 20, 0)
    md = None if max_depth == 0 else max_depth
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=md, random_state=42)
elif model_name == "Logistic Regression":
    C = st.sidebar.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
    model = LogisticRegression(C=C, max_iter=1000)
else:
    n_neighbors = st.sidebar.slider("n_neighbors", 1, 20, 5)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

# --- Scale Data if Enabled ---
if scale_data:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
else:
    scaler = None

# --- Train Model ---
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# --- Show Metrics ---
st.subheader("✅ Model Performance")
st.write(f"**Accuracy:** {acc:.3f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred, target_names=species))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=species, yticklabels=species)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig_cm)

# --- Prediction UI ---
st.subheader("🌼 Predict Flower Species")
st.markdown("Enter flower measurements below to predict its species:")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.number_input("Sepal Width (cm)", 2.0, 5.0, 3.5)
with col2:
    petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.number_input("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("🔮 Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    if scale_data and scaler:
        input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    st.success(f"Predicted Species: **{species[prediction[0]].title()}** 🌺")

# --- Download Model ---
model_bytes = BytesIO()
joblib.dump({"model": model, "scaler": scaler}, model_bytes)
st.download_button(
    label="💾 Download Trained Model",
    data=model_bytes.getvalue(),
    file_name="iris_model.joblib",
    mime="application/octet-stream",
)

st.caption("Built with using scikit-learn, pandas, seaborn, and Streamlit.")
