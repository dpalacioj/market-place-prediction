# app/config.py
import os

# Ganratia

GARANTIA_APLICA = [
'garantia', 'cubre', 'respaldado', 'siempre que', 'proteccion',
'defectos de fabricacion', 'aplica', 'si', 'garantizado', 'por fallas',
'fabricacion', 'defecto'
]


GARANTIA_NO_APLICA = [
'no', 'desconocido', 'no aplica', 'sin garantia', 'no se aceptan devoluciones',
'sin', 'no cubre', 'no garantia', 'no respaldado', 'no proteccion',
'no defectos de fabricacion', 'no aplica', 'no garantizado', 'no por fallas',
'no fabricacion', 'no defecto'
]


# Columnas categóricas y numéricas a convertir
CATEGORY_COLS = [
    'condition', 'warranty', 'buying_mode', 'currency_id', 'seller_country',
    'seller_state', 'seller_city', 'shipping_mode', 'parent_item_id',
    'category_id', 'seller_id', 'official_store_id', 'video_id',
    'status', 'garantia_aplica', 'listing_type_id'
]

NUMERIC_COLS = [
    'initial_quantity', 'available_quantity', 'sold_quantity',
    'original_price', 'base_price', 'price'
]

BOOL_COLS = [
    'tarjeta_de_credito', 'transferencia_bancaria', 'shipping_local_pick_up',
    'efectivo', 'automatic_relist', 'acordar_con_el_comprador'
]

DROP_COLUMNS = [
    'seller_address', 'warranty', 'sub_status', 'seller_contact', 'deal_ids','shipping',
    'seller_id', 'variations','location', 'attributes','tags', 'parent_item_id', 'category_id',
    'descriptions','last_updated','international_delivery_mode','pictures', 'id','official_store_id',
    'original_price','thumbnail','title','date_created','secure_thumbnail','video_id','catalog_product_id',
    'start_time','stop_time', 'permalink','geolocation','shipping_tags','non_mercado_pago_payment_methods',
    'seller_country','site_id'
]


# Columnas finales seleccionadas tras feature engineering
SELECTED_COLUMNS = [
    'initial_quantity', 'listing_type_id', 'seller_city', 'price',
    'available_quantity', 'week_day', 'sold_quantity',
    'seller_state', 'garantia_aplica', 'shipping_mode', 'month_start',
    'month_stop', 'tarjeta_de_credito', 'dragged_bids_and_visits',
    'transferencia_bancaria', 'shipping_local_pick_up', 'efectivo',
    'automatic_relist', 'days_active', 'acordar_con_el_comprador',
    'condition'
]

# PATHS

# BASE_DIR
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

DATA_PATH_RAW = os.path.join(BASE_DIR, 'app', 'src', 'data', 'raw')
DATA_PATH_RAW_NAME = 'MLA_100k.jsonlines'
DATA_PATH_PROCESSED = os.path.join(BASE_DIR, 'app', 'src', 'data', 'processed')
DATA_PATH_PROCESSED_NAME = 'data_model'
MODEL_PATH = os.path.join(BASE_DIR, 'app', 'models')
MODEL_NAME = 'xgb_model_v1.pkl'

BASE_CONFIG_PATH = os.path.join(BASE_DIR, 'app', 'src', 'models')
BASE_CONFIG_XG_NAME = 'base_model_xg_config.json'
BASE_CONFIG_XG_OPTIMIZED_NAME = 'base_model_xg_optimized_config.json'


ENCODED_COLUMNS_PATH = os.path.join(BASE_DIR, 'app', 'models', 'encoded_columns.npy')



# # Paths
# DATA_PATH_RAW = 'app/src/data/raw'
# DATA_PATH_RAW_NAME = 'MLA_100k.jsonlines'
# DATA_PATH_PROCESSED = 'app/src/data/processed'
# DATA_PATH_PROCESSED_NAME = 'app/src/data/data_model'
# MODEL_PATH = 'app/models/xgb_model.pkl'
# ENCODED_COLUMNS_PATH = 'app/models/encoded_columns.npy'
