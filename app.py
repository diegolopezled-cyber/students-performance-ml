import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_student_performance_model.joblib"
METADATA_PATH = BASE_DIR / "student_performance_model_metadata.json"


def load_model_and_metadata():
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"No se encontró el modelo entrenado en: {MODEL_PATH}")
        st.stop()

    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        st.error(f"No se encontró el archivo de metadatos en: {METADATA_PATH}")
        st.stop()

    return model, metadata


def main():
    st.set_page_config(page_title="Predicción de rendimiento académico", page_icon="🎓")
    st.title("🎓 Predicción de rendimiento académico")
    st.caption("Sistema de soporte para estimación de clasificación académica basado en variables observables del estudiante.")
    st.warning("Este prototipo utiliza datos sintéticos y no debe emplearse como base para decisiones institucionales o académicas definitivas.")

    with st.sidebar:
        st.header("Información del prototipo")
        st.write("La aplicación estima la clase académica del estudiante a partir de indicadores demográficos, educativos y de hábitos de estudio.")
        st.markdown("- Edad\n- Género\n- Etnia\n- Educación de los padres\n- Horas de estudio\n- Ausencias\n- Tutoría\n- Apoyo familiar\n- Actividades extracurriculares")

    model, metadata = load_model_and_metadata()
    labels = {int(k): v for k, v in metadata["class_labels"].items()}

    st.subheader("Ingresar datos del estudiante")
    with st.form("student_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Edad", 15, 18, 16)
            gender = st.selectbox("Género codificado", [0, 1])
            ethnicity = st.selectbox("Etnia codificada", [0, 1, 2, 3])
            parental_education = st.selectbox("Educación parental", [0, 1, 2, 3, 4])
            study_time = st.slider("Horas de estudio semanales", 0.0, 20.0, 10.0, 0.5)
            absences = st.slider("Ausencias", 0, 30, 5)

        with col2:
            tutoring = st.selectbox("Tutoría", [0, 1])
            parental_support = st.selectbox("Apoyo parental", [0, 1, 2, 3, 4])
            extracurricular = st.selectbox("Actividad extracurricular", [0, 1])
            sports = st.selectbox("Deportes", [0, 1])
            music = st.selectbox("Música", [0, 1])
            volunteering = st.selectbox("Voluntariado", [0, 1])

        submitted = st.form_submit_button("Predecir clasificación", use_container_width=True)

    if submitted:
        row = pd.DataFrame([
            {
                "Age": age,
                "Gender": gender,
                "Ethnicity": ethnicity,
                "ParentalEducation": parental_education,
                "StudyTimeWeekly": study_time,
                "Absences": absences,
                "Tutoring": tutoring,
                "ParentalSupport": parental_support,
                "Extracurricular": extracurricular,
                "Sports": sports,
                "Music": music,
                "Volunteering": volunteering,
            }
        ])[metadata["features"]]

        pred = int(model.predict(row)[0])
        predicted_label = labels[pred]

        st.subheader("Resultado de la predicción")
        st.success(f"GradeClass estimada: {pred} ({predicted_label})")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(row)[0]
            probability_table = pd.DataFrame({
                "GradeClass": [f"{int(c)} ({labels[int(c)]})" for c in model.classes_],
                "Probabilidad": probs,
            })
            st.bar_chart(probability_table.set_index("GradeClass"))

            with st.expander("Detalle de probabilidades por clase"):
                st.dataframe(
                    probability_table.assign(Probabilidad=lambda df: df["Probabilidad"].map(lambda v: f"{v:.2%}")),
                    use_container_width=True,
                    hide_index=True,
                )


if __name__ == "__main__":
    main()
