from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Rendimiento académico', page_icon='🎓', layout='wide')

MODEL_PATH = Path('best_student_performance_model.joblib')
META_PATH = Path('student_performance_model_metadata.json')

st.title('🎓 Predicción experimental de rendimiento académico')
st.caption('Clasificación multiclase de GradeClass mediante Machine Learning')
st.warning('Prototipo académico basado en datos sintéticos. No debe utilizarse para tomar decisiones reales sobre estudiantes.')

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error('Faltan los artefactos del modelo. Añade best_student_performance_model.joblib y student_performance_model_metadata.json al repositorio.')
    st.stop()

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text(encoding='utf-8'))
    return model, metadata

model, metadata = load_assets()
labels = {int(k): v for k, v in metadata.get('class_labels', {'0':'A','1':'B','2':'C','3':'D','4':'F'}).items()}
features = metadata['features']

with st.sidebar:
    st.header('Modelo')
    st.write('Campeón:', metadata.get('champion_model', 'Modelo entrenado'))
    if 'test_metrics' in metadata:
        st.metric('F1-Macro Test', f"{metadata['test_metrics'].get('f1_macro', 0):.3f}")
        st.metric('Accuracy Test', f"{metadata['test_metrics'].get('accuracy', 0):.3f}")
    st.caption('GPA y StudentID no se utilizan como predictores.')

st.subheader('Datos del estudiante')
with st.form('student_form'):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider('Edad', 15, 18, 16)
        gender = st.selectbox('Género codificado', [0, 1], help='0 y 1 según la codificación original del dataset')
        ethnicity = st.selectbox('Etnia codificada', [0, 1, 2, 3])
        parental_education = st.selectbox('Educación parental', [0, 1, 2, 3, 4])
    with c2:
        study_time = st.slider('Horas de estudio semanales', 0.0, 20.0, 10.0, 0.5)
        absences = st.slider('Ausencias', 0, 30, 5)
        tutoring = st.selectbox('Tutoría', [0, 1])
        parental_support = st.selectbox('Apoyo parental', [0, 1, 2, 3, 4])
    with c3:
        extracurricular = st.selectbox('Actividad extracurricular', [0, 1])
        sports = st.selectbox('Deportes', [0, 1])
        music = st.selectbox('Música', [0, 1])
        volunteering = st.selectbox('Voluntariado', [0, 1])
    submitted = st.form_submit_button('Predecir GradeClass', type='primary', use_container_width=True)

if submitted:
    values = {
        'Age': age, 'Gender': gender, 'Ethnicity': ethnicity,
        'ParentalEducation': parental_education, 'StudyTimeWeekly': study_time,
        'Absences': absences, 'Tutoring': tutoring, 'ParentalSupport': parental_support,
        'Extracurricular': extracurricular, 'Sports': sports, 'Music': music,
        'Volunteering': volunteering
    }
    missing_features = [f for f in features if f not in values]
    if missing_features:
        st.error(f'La interfaz no tiene valores para estas variables requeridas por el modelo: {missing_features}')
        st.stop()

    row = pd.DataFrame([{f: values[f] for f in features}])
    pred = int(model.predict(row)[0])
    st.success(f'GradeClass estimada: {pred} ({labels.get(pred, pred)})')

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(row)[0]
        probability_table = pd.DataFrame({
            'GradeClass': [f'{int(c)} ({labels.get(int(c), c)})' for c in model.classes_],
            'Probabilidad': probs
        }).set_index('GradeClass')
        st.subheader('Probabilidad estimada por clase')
        st.bar_chart(probability_table)
        st.dataframe(probability_table.style.format({'Probabilidad': '{:.2%}'}), use_container_width=True)

st.divider()
st.caption('Proyecto educativo de Machine Learning · Dataset sintético · La importancia predictiva no implica causalidad.')
