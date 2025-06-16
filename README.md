# Clasificador de Condición de Producto - API con Streamlit

## Descripción del problema

En el contexto del Marketplace, se requiere un algoritmo de machine learning que prediga si un ítem listado en la plataforma es **nuevo** o **usado**.

El reto incluye:

- Análisis exploratorio de datos
- Selección y procesamiento de características relevantes
- Entrenamiento de un modelo de clasificación supervisada
- Evaluación rigurosa utilizando **accuracy** como métrica principal (mínimo requerido: 0.86)
- Elección y justificación de una **métrica secundaria apropiada**

Se proporciona un dataset en formato `.jsonlines` llamado `MLA_100k.jsonlines`, y una función `build_dataset` para cargarlo.

---

## Descripción del modelo y métricas obtenidas

Tras el análisis de datos y pruebas con distintos algoritmos, se optó por un modelo **XGBoostClassifier**, al cual se le aplicó preprocesamiento personalizado y búsqueda de hiperparámetros.

### Métricas obtenidas sobre test:

```
Accuracy: 0.8581
Precision: 0.8319
F1 Score: 0.8495
Recall: 0.8678
ROC AUC: 0.9364
```

### Conclusiones del rendimiento del modelo:

- **Accuracy ≈ 0.86**: Cumple con el mínimo exigido para el desafío.
- **Precision ≈ 0.84** y **Recall ≈ 0.87**: El modelo identifica correctamente la mayoría de los casos positivos y, cuando predice un positivo, suele acertar.
- **F1 Score ≈ 0.85**: Balance equilibrado entre precisión y sensibilidad.
- **ROC AUC ≈ 0.94**: Excelente capacidad de discriminación entre las clases.

### ¿Por qué usar *Recall* como segunda métrica?

En este contexto, es importante **no dejar pasar productos usados como si fueran nuevos**. Por lo tanto, maximizar el **Recall** (sensibilidad) es clave para capturar la mayor cantidad de productos usados (positivos).

---

## Estructura del proyecto

```
├── app/
├── models/
│   ├── xgb_model_v1.pkl           # Modelo entrenado
│   └── encoded_columns.npy        # Columnas resultantes del preprocesamiento
│
├── src/
│   ├── config/
│   │   └── config.py              # Rutas, constantes, paths globales
│
│   ├── data/
│   │   ├── raw/                   # Datos originales
│   │   └── processed/             # Datos ya transformados
│
│   ├── features/
│   │   └── preprocessing.py       # Clase `Preprocessing` reutilizable - Preparar datos para entrenamiento y prediccion
│
│   ├── models/
│   │   ├── base_model_xg_config.json   # Configuración base del modelo XGBoost
│   │   ├── train_model.py              # Script de entrenamiento principal
│   │   └── xgb_model.py                # Clase con lógica de entrenamiento, optimización, evaluación para el modelo XGBoost
│
│   ├── notebooks/
│   │   ├── model_xgboost.ipynb         # EDA + entrenamiento exploratorio
│   │   ├── preprocessing_eda.ipynb     # EDA del preprocesamiento
│   │   └── test_exec.ipynb             # Pruebas, testeo de api
│
│   ├── utils/
│   │   ├── utils.py                    # Funciones auxiliares: carga json, guardado, etc.
│   │   └── schemas.py                  # Esquemas Pydantic (usados por la API)
|
├── ui/
│   ├── streamlit_app.py                # Script para crear API con streamlit
├── requirements.txt                    # Librerías necesarias
├── README.md                           # Documentación del proyecto
├── train.py                            # Ejecuta entrenamiento (CLI)
└── logs/
    └── app.log                         # Logs de predicción y errores
└── README.md

```
Este proyecto proporciona una **interfaz visual amigable** para usuarios finales, construida con **Streamlit**, que permite cargar publicaciones de productos y obtener predicciones de forma accesible y rápida.

---

## Funcionalidades

Interfaz sin código, fácil de usar  
Permite tres formas de ingreso de datos:

- **Subir archivo CSV**  
  (Estructura preprocesada con columnas ya seleccionadas)

- **Subir archivo JSONLines (.jsonlines)**  
  (Formato anidado completo como el dataset original de MercadoLibre)

- **Ingreso manual de JSON**  
  (Pega el contenido de una publicación con estructura original)

Predicción en tiempo real  
Descarga del resultado (`CSV` o `JSON`)  
Traducción automática del resultado:  
`0 → new`, `1 → used`

---

## Ejecución del proyecto

### 1. Instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Entrenar el modelo

```bash
python train.py  # Para entrenamiento básico
# o
python train.py --optimize  # Para optimización de hiperparámetros
```

### 3. Ejecutar la aplicación

```bash
 PYTHONPATH=. streamlit run app/ui/streamlit_app.py
```

Se abrirá automáticamente en:

```
http://localhost:8501
```

---

## Ejemplo de JSON para ingreso manual

```json
{
  "inputs": [
    {
      "seller_address": {
        "country": {"name": "Argentina", "id": "AR"},
        "state": {"name": "Capital Federal", "id": "AR-C"},
        "city": {"name": "San Cristóbal", "id": "TUxBQlNBTjkwNTZa"}
      },
      "warranty": null,
      "condition": "new",
      "base_price": 80.0,
      "price": 80.0,
      "shipping": {
        "local_pick_up": true,
        "free_shipping": false,
        "mode": "not_specified",
        "tags": []
      },
      "non_mercado_pago_payment_methods": [
        {"description": "Transferencia bancaria", "id": "MLATB", "type": "G"},
        {"description": "Acordar con el comprador", "id": "MLAWC", "type": "G"},
        {"description": "Efectivo", "id": "MLAMO", "type": "G"}
      ],
      "seller_id": 74952096,
      "listing_type_id": "bronze",
      "available_quantity": 1,
      "sold_quantity": 0.0,
      "start_time": 1441485773000,
      "stop_time": 1446669773000,
      "accepts_mercadopago": true,
      "currency_id": "ARS",
      "tags": ["dragged_bids_and_visits"],
      "automatic_relist": false,
      "category_id": "MLA126406",
      "parent_item_id": "MLA568261029",
      "status": "active",
      "video_id": null,
      "original_price": null,
      "buying_mode": "buy_it_now",
      "initial_quantity": 1
    }
  ]
}
```
