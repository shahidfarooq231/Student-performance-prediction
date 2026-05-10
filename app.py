import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("student_data.csv")

X = data.drop("result", axis=1)
y = data["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.title("Student Performance Prediction System")

st.write("Predict whether a student will PASS or FAIL")

study_hours = st.slider("Study Hours", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 70)
previous_marks = st.slider("Previous Marks", 0, 100, 60)
sleep_hours = st.slider("Sleep Hours", 0, 10, 6)
assignments = st.slider("Assignments Completed", 0, 10, 5)

if st.button("Predict"):

    input_data = [[
        study_hours,
        attendance,
        previous_marks,
        sleep_hours,
        assignments
    ]]

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Student Will PASS")
    else:
        st.error("Student Will FAIL")

st.subheader("Model Accuracy")
st.info(f"{accuracy*100:.2f}%")