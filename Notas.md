# Notas Tomadas del proceso de entrenamiento

## Ajustes para el modelo (xgb.XGBRegessor) con potenciacion con GPU

"Mejor RMSE": 1.0357838006016242

Mejores parámetros: {
'n_estimators': 1972,
'max_depth': 10,
'learning_rate':
0.01863442732172959,
'subsample': 0.6279084673487179,
'colsample_bytree': 0.747752473508877
}

## Para cargar el modelo en otra session

```py
import xgboost as xgb

# 1. Crear una instancia vacía del modelo (mismo tipo: Regressor)
nuevo_modelo = xgb.XGBRegressor()

# 2. Cargar los parámetros/pesos desde el archivo
nuevo_modelo.load_model('modelo_app_reviews.json')

print("Pesos cargados. El modelo está listo para predecir.")

```

### Para guardar el diccionario (TF-IDF)

```py
import joblib

# Guardar el vectorizador
joblib.dump(tfidf, 'vectorizador_tfidf.pkl')

# Para cargarlo después:
# tfidf_cargado = joblib.load('vectorizador_tfidf.pkl')
```
