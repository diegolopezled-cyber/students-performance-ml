# Predicción del rendimiento académico con Machine Learning

Aplicación interactiva en Streamlit para estimar `GradeClass` a partir de variables demográficas, hábitos de estudio, apoyo parental y actividades extracurriculares.

## Decisiones metodológicas

- `StudentID` se excluye por ser un identificador.
- `GPA` se excluye para prevenir **Data Leakage**, ya que `GradeClass` se deriva del GPA.
- División estratificada Train/Test 80/20.
- Validación cruzada estratificada de 5 folds.
- Comparación de Logistic Regression, Random Forest, SVM y KNN.
- Métrica principal: F1-Macro.

## Archivos necesarios para desplegar

```text
app.py
requirements.txt
best_student_performance_model.joblib
student_performance_model_metadata.json
```

Los dos últimos archivos se generan al ejecutar el Paso 10 del cuaderno de entrenamiento.

## Ejecución local

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Despliegue

1. Ejecutar completamente el cuaderno de entrenamiento y obtener el `.joblib` y el JSON de metadatos.
2. Colocar esos dos archivos junto a `app.py` y `requirements.txt`.
3. Subir los archivos al mismo repositorio de GitHub.
4. En Streamlit Community Cloud elegir el repositorio, rama y `app.py` como archivo principal.

## Dataset

Rabie El Kharoua, *Students Performance Dataset*, Kaggle, licencia CC BY 4.0.

## Limitación

El dataset es sintético. Esta aplicación es una demostración académica y no está validada para tomar decisiones reales sobre estudiantes.
