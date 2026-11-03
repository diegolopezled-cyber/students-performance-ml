# Predicción de rendimiento académico

Aplicación web desarrollada en Streamlit para estimar la clase académica de un estudiante a partir de variables observables relacionadas con su perfil sociodemográfico, hábitos de estudio y entorno familiar.

## Descripción general

Este proyecto implementa un prototipo de clasificación supervisada para la predicción de `GradeClass`. La interfaz permite ingresar información como:

- edad
- género
- etnia
- nivel educativo de los padres
- horas de estudio por semana
- ausencias
- tutoría
- apoyo parental
- actividad extracurricular

La aplicación devuelve la clase estimada y la distribución probabilística de las categorías disponibles.

## Metodología

Se entrenó un modelo de clasificación sobre un conjunto de datos sintético, seleccionando el mejor desempeño mediante validación cruzada con criterio de evaluación basado en F1-Macro.

### Variables excluidas

- `StudentID`: identificador sin valor predictivo para el problema.
- `GPA`: se excluye para evitar fuga de información, dado que `GradeClass` se deriva de esta variable.

### Procedimiento de evaluación

- partición estratificada Train/Test 80/20
- validación cruzada estratificada con 5 folds
- métrica principal: F1-Macro

## Estructura del repositorio

```text
app.py
requirements.txt
best_student_performance_model.joblib
student_performance_model_metadata.json
README.md
```

## Requisitos

- Python 3.11+
- pip
- Streamlit
- pandas
- scikit-learn
- joblib

## Instalación local

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Subir el repositorio a GitHub.
2. Mantener todos los archivos en la carpeta raíz del proyecto.
3. En Streamlit Community Cloud seleccionar el repositorio, la rama y `app.py` como archivo principal.
4. Confirmar que `requirements.txt` se use para instalar dependencias automáticamente.

## Fuente del dataset

Rabie El Kharoua, *Students Performance Dataset*, Kaggle, licencia CC BY 4.0.

## Limitaciones

El conjunto de datos es sintético y el prototipo tiene fines académicos y de demostración. No debe utilizarse como base para decisiones reales sobre estudiantes ni para evaluaciones institucionales.
