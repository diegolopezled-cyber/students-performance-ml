import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Rendimiento académico", page_icon="🎓")
st.title("🎓 Predicción experimental de GradeClass")
st.warning("Prototipo académico basado en datos sintéticos. No usar para decisiones reales sobre estudiantes.")

model = joblib.load("best_student_performance_model.joblib")
metadata = json.loads(Path("student_performance_model_metadata.json").read_text(encoding="utf-8"))
labels = {int(k): v for k, v in metadata["class_labels"].items()}

with st.form("student_form"):
    age = st.slider("Edad", 15, 18, 16)
    gender = st.selectbox("Género codificado", [0, 1])
    ethnicity = st.selectbox("Etnia codificada", [0, 1, 2, 3])
    parental_education = st.selectbox("Educación parental", [0, 1, 2, 3, 4])
    study_time = st.slider("Horas de estudio semanales", 0.0, 20.0, 10.0, 0.5)
    absences = st.slider("Ausencias", 0, 30, 5)
    tutoring = st.selectbox("Tutoría", [0, 1])
    parental_support = st.selectbox("Apoyo parental", [0, 1, 2, 3, 4])
    extracurricular = st.selectbox("Actividad extracurricular", [0, 1])
    sports = st.selectbox("Deportes", [0, 1])
    music = st.selectbox("Música", [0, 1])
    volunteering = st.selectbox("Voluntariado", [0, 1])
    submitted = st.form_submit_button("Predecir")

if submitted:
    row = pd.DataFrame([{
        "Age": age, "Gender": gender, "Ethnicity": ethnicity,
        "ParentalEducation": parental_education,
        "StudyTimeWeekly": study_time, "Absences": absences,
        "Tutoring": tutoring, "ParentalSupport": parental_support,
        "Extracurricular": extracurricular, "Sports": sports,
        "Music": music, "Volunteering": volunteering
    }])[metadata["features"]]
    pred = int(model.predict(row)[0])
    st.success(f"GradeClass estimada: {pred} ({labels[pred]})")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(row)[0]
        probability_table = pd.DataFrame({
            "GradeClass": [f"{int(c)} ({labels[int(c)]})" for c in model.classes_],
            "Probabilidad": probs
        })
        st.bar_chart(probability_table.set_index("GradeClass"))
