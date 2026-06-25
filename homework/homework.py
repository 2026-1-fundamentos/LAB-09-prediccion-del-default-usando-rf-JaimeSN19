import os
import json
import gzip
import pickle
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

def clean_data(df):
    """Realiza la limpieza de un dataset según las reglas definidas."""
    df = df.copy()
    
    # Renombrar la columna objetivo
    if 'default payment next month' in df.columns:
        df.rename(columns={'default payment next month': 'default'}, inplace=True)
        
    # Remover la columna ID
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)
        
    # Eliminar registros con nulos clásicos
    df.dropna(inplace=True)
    
    # CRÍTICO: Filtrar estrictamente los valores N/A (0) sin importar si se leen como número o como texto
    df = df.loc[(df['EDUCATION'] != 0) & (df['EDUCATION'] != '0')]
    df = df.loc[(df['MARRIAGE'] != 0) & (df['MARRIAGE'] != '0')]
    
    # Agrupar niveles superiores de educación (valores > 4) en la categoría 'others' (4)
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4
    
    return df

def pregunta_01():
    """
    Ejecuta el flujo completo del modelo de Machine Learning.
    """
    
    # -------------------------------------------------------------------------
    # Paso 1. Cargar y limpiar los datos
    # -------------------------------------------------------------------------
    input_dir = "files/input"
    
    train_files = [f for f in os.listdir(input_dir) if "train" in f and (f.endswith(".csv") or f.endswith(".zip"))]
    test_files = [f for f in os.listdir(input_dir) if "test" in f and (f.endswith(".csv") or f.endswith(".zip"))]

    df_train = pd.read_csv(os.path.join(input_dir, train_files[0]))
    df_test = pd.read_csv(os.path.join(input_dir, test_files[0]))

    df_train = clean_data(df_train)
    df_test = clean_data(df_test)

    # -------------------------------------------------------------------------
    # Paso 2. Dividir los datasets en variables explicativas (x) y objetivo (y)
    # -------------------------------------------------------------------------
    x_train = df_train.drop(columns=['default'])
    y_train = df_train['default']
    
    x_test = df_test.drop(columns=['default'])
    y_test = df_test['default']

    # -------------------------------------------------------------------------
    # Paso 3. Crear pipeline (One-Hot-Encoding + Random Forest)
    # -------------------------------------------------------------------------
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Devolvemos la semilla aleatoria (random_state) para que sea estable
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42)) 
    ])

    # -------------------------------------------------------------------------
    # Paso 4. Optimizar hiperparámetros con validación cruzada
    # -------------------------------------------------------------------------
    # Subimos a 300/400 árboles para ganar esa minúscula fracción de exactitud.
    param_grid = {
        'classifier__n_estimators': [300, 400], 
        'classifier__max_depth': [None],
        'classifier__min_samples_split': [2], 
        'classifier__min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=10, 
        scoring='balanced_accuracy', 
        n_jobs=-1,
        refit=True
    )
    
    # Ajustar a los datos de entrenamiento
    grid_search.fit(x_train, y_train)

    # -------------------------------------------------------------------------
    # Paso 5. Guardar modelo comprimido (gzip)
    # -------------------------------------------------------------------------
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(grid_search, f)
        
    # -------------------------------------------------------------------------
    # Paso 6 y 7. Calcular y guardar métricas y matrices de confusión
    # -------------------------------------------------------------------------
    def get_metrics(model, X, y, dataset_name):
        y_pred = model.predict(X)
        metrics = {
            'type': 'metrics',
            'dataset': dataset_name,
            'precision': round(precision_score(y, y_pred), 4),
            'balanced_accuracy': round(balanced_accuracy_score(y, y_pred), 4),
            'recall': round(recall_score(y, y_pred), 4),
            'f1_score': round(f1_score(y, y_pred), 4)
        }
        return metrics

    def get_cm(model, X, y, dataset_name):
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        matrix = {
            'type': 'cm_matrix',
            'dataset': dataset_name,
            'true_0': {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
            'true_1': {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
        }
        return matrix

    metrics_train = get_metrics(grid_search, x_train, y_train, 'train')
    metrics_test = get_metrics(grid_search, x_test, y_test, 'test')
    
    cm_train = get_cm(grid_search, x_train, y_train, 'train')
    cm_test = get_cm(grid_search, x_test, y_test, 'test')

    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics_train) + "\n")
        f.write(json.dumps(metrics_test) + "\n")
        f.write(json.dumps(cm_train) + "\n")
        f.write(json.dumps(cm_test) + "\n")

if __name__ == "__main__":
    pregunta_01()