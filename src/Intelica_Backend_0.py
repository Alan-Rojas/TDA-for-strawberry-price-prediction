import numpy as np
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
from gtda.mapper import (
    make_mapper_pipeline,
    Projection,
    plot_static_mapper_graph
)
from sklearn.cluster import DBSCAN
from imblearn.over_sampling import SMOTE
import networkx as nx
import seaborn as sns
np.random.seed(42)
plt.style.use('ggplot')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score
from sklearn.linear_model import LogisticRegression

from gtda.diagrams import PersistenceEntropy, Scaler
from gtda.homology import VietorisRipsPersistence
from gtda.metaestimators import CollectionTransformer
from gtda.pipeline import Pipeline
from gtda.time_series import TakensEmbedding, SingleTakensEmbedding
from gtda.time_series import SlidingWindow
from gtda.plotting import plot_diagram, plot_point_cloud
from sklearn.decomposition import PCA
import sys
from persim import bottleneck
from sklearn.pipeline import Pipeline as sklearn_pipeline
import re


class Data:
    def __init__(self, file_path="berries_filtered.csv", date_col='report_begin_date', price_col='average_price'):
        """
        Inicializa la clase Data cargando el DataFrame y definiendo las columnas clave.
        """
        self.file_path = file_path
        self.date_col = date_col
        self.price_col = price_col
        self.df = self._load_and_clean_df() # Carga y limpia el DF al inicializar
        self.time_series = None # Atributo para almacenar la serie de tiempo preparada
        self.product = None
        self.freq = None

    def _load_and_clean_df(self) -> pd.DataFrame:
        """
        Método interno para cargar el CSV y realizar la limpieza inicial del DataFrame.
        Se llama automáticamente al inicializar la clase.
        Maneja valores no numéricos como '#DIV/0!' y asegura los tipos de datos correctos.
        """
        try:
            # Ampliamos na_values para incluir '#DIV/0!' y otros posibles strings problemáticos
            # Usamos 'latin-1' según tu indicación
            df = pd.read_csv(self.file_path, encoding='latin-1', 
                             na_values=['#¡DIV/0!', '#DIV/0!', 'NaN', '', ' ', None, 'null', 'N/A', '#N/A'])
            print(f"DataFrame cargado desde: {self.file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"El archivo '{self.file_path}' no fue encontrado. Asegúrate de que la ruta sea correcta.")
        except Exception as e:
            raise Exception(f"Error al cargar el DataFrame: {e}")
        
        # Validar que las columnas esperadas existan antes de proceder
        required_cols = ['commodity', self.date_col, self.price_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Faltan las siguientes columnas en el CSV: {', '.join(missing_cols)}")

        # Selección de columnas para trabajar (ya las tienes, pero es buena práctica asegurarlas)
        df = df[['commodity', self.date_col, self.price_col]].copy() # Usar .copy() para evitar SettingWithCopyWarning
        
        # Limpiar nombres de columnas (si tienen espacios, mayúsculas/minúsculas inconsistentes, etc.)
        # Ya que seleccionaste columnas por su nombre exacto, esto podría ser redundante si ya están limpias.
        # Pero es bueno si el CSV tiene inconsistencias.
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Convertir la columna de fecha a datetime
        # Usamos errors='coerce' para convertir fechas inválidas a NaT (Not a Time)
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors='coerce')
        
        # Limpieza y conversión de la columna de precio a float
        # Aquí consolidamos la lógica de limpieza más robusta
        
        # 1. Convertir a string, quitar espacios y reemplazar otros valores problemáticos.
        #    Los valores ya deberían ser NaN por na_values, pero esto es una doble verificación
        #    y manejo de strings que podrían no haber sido capturados.
        df[self.price_col] = df[self.price_col].astype(str).str.strip()

        # 2. Reemplazar cualquier cadena que esté vacía después del strip por NaN
        df[self.price_col] = df[self.price_col].replace('', np.nan)
        
        # 3. Eliminar caracteres no numéricos, EXCEPTO el punto decimal y el signo menos
        #    La expresión regular '[^0-9\.\-]' es correcta para esto.
        #    Aplicar esto SOLO si la columna NO es ya numérica (o NaN), para evitar errores.
        #    Esto es útil si hay símbolos de moneda, comas de miles (que no son separadores decimales), etc.
        df[self.price_col] = df[self.price_col].str.replace(r'[^0-9\.\-]', '', regex=True)

        # 4. Convertir la columna de precio a float. `errors='coerce'` es fundamental aquí:
        #    Cualquier valor que no pueda ser numérico (incluyendo NaNs o cadenas vacías restantes)
        #    será convertido a NaN.
        df[self.price_col] = pd.to_numeric(df[self.price_col], errors='coerce')
        
        # Opcional: Eliminar filas donde la fecha o el precio son NaN
        # Esto es importante antes de procesar series de tiempo
        initial_rows = len(df)
        df.dropna(subset=[self.date_col, self.price_col], inplace=True)
        if len(df) < initial_rows:
            print(f"Se eliminaron {initial_rows - len(df)} filas con valores NaN en fechas o precios.")

        # Ordenar por fecha (esto ya lo tenías, lo mantengo aquí)
        df.sort_values(self.date_col, inplace=True)
        
        print("DataFrame limpiado y listo.")
        print(df.info()) # Mostrar info después de la limpieza para verificación
        return df

    def set_df(self):
        """
        Devuelve el DataFrame procesado.
        """
        return self.df

    def prepare_series(self, product: str = "strawberries", freq: str = "W"):
        """
        Prepara una serie de tiempo a partir del DataFrame cargado, 
        filtrando por producto, remuestreando y promediando precios.
        
        Parámetros:
            product (str): Nombre del producto a filtrar (ej. "strawberries").
            freq (str): Frecuencia de remuestreo (ej. "W" para semanal, "M" para mensual).
        
        Retorna:
            pd.Series: La serie de tiempo preparada.
        """
        if self.df is None:
            raise ValueError("DataFrame no cargado. Llama a _load_and_clean_df() o asegúrate de que el archivo exista.")

        self.product = product
        self.freq = freq

        # Filtrar por producto (asegura el lower y strip para la columna 'commodity')
        df_filtered = self.df[self.df['commodity'].astype(str).str.strip().str.lower() == product.lower()].copy()
        
        if df_filtered.empty:
            raise ValueError(f"No se encontraron datos para el producto: {product} después de filtrar.")

        # Asegurarse de que las columnas clave existan y sean del tipo correcto
        # Estos cheques son redundantes si _load_and_clean_df() ya hizo su trabajo
        # pero sirven como validación final antes de construir la serie.
        if self.date_col not in df_filtered.columns or self.price_col not in df_filtered.columns:
            raise KeyError(f"Las columnas '{self.date_col}' o '{self.price_col}' no se encontraron en el DataFrame filtrado.")
        
        # Estos `is_datetime64_any_dtype` y `is_numeric_dtype` son excelentes para validar
        if not pd.api.types.is_datetime64_any_dtype(df_filtered[self.date_col]):
            print(f"Advertencia: La columna '{self.date_col}' no es datetime. Intentando convertir de nuevo.")
            df_filtered[self.date_col] = pd.to_datetime(df_filtered[self.date_col], errors='coerce')
        
        if not pd.api.types.is_numeric_dtype(df_filtered[self.price_col]):
            print(f"Advertencia: La columna '{self.price_col}' no es numérica. Intentando convertir de nuevo.")
            df_filtered[self.price_col] = pd.to_numeric(df_filtered[self.price_col], errors='coerce')
            
        # Eliminar cualquier NaN que pueda haber surgido de las conversiones coercitivas
        df_filtered.dropna(subset=[self.date_col, self.price_col], inplace=True)
        
        if df_filtered.empty:
            raise ValueError(f"El DataFrame para el producto {product} está vacío después de la limpieza de NaNs.")


        # Agrupar, promediar, remuestrear e interpolar
        # df_filtered debe tener ya el índice de fecha para .resample
        # Primero, establecer el índice de fecha y luego agrupar/promediar por índice para manejar duplicados
        df_filtered = df_filtered.set_index(self.date_col).sort_index()

        # Agrupar por el índice de fecha y tomar el promedio de 'average_price' para manejar duplicados
        series = df_filtered[self.price_col].groupby(df_filtered.index).mean()

        # Remuestrear la serie si se especifica una frecuencia
        series = series.resample(freq).mean() # Asumo que .mean() es el agregado deseado para la frecuencia
        series = series.interpolate('linear') # Interpolar valores NaN creados por el remuestreo (huecos)
        
        # Eliminar NaNs al inicio o al final que no pudieron ser interpolados
        series = series.dropna()

        if series.empty:
            raise ValueError(f"La serie de tiempo para el producto {product} está vacía después del remuestreo y la interpolación. "
                             "Revisa el rango de fechas, la frecuencia o la cantidad de NaNs.")

        series.name = self.price_col # Mantener el nombre original de la columna de precio para la serie
        self.time_series = series # Guardar la serie preparada como atributo
        print(f"Serie de tiempo preparada para '{product}' con frecuencia '{freq}' y {len(series)} puntos.")
        return series

    def get_time_series(self) -> pd.Series:
        """Retorna la serie de tiempo preparada, si existe."""
        if self.time_series is None:
            print("Advertencia: No se ha preparado ninguna serie de tiempo. Llama a prepare_series() primero.")
        return self.time_series
    
    def get_products(self):
        return self.df["commodity"].unique()

class AnalisisTDA:
    """
    Clase para realizar Análisis Topológico de Datos (TDA) en series de tiempo.
    Calcula features topológicos (entropía) y genera etiquetas de cambio de precio.
    """
    def __init__(self, embedding_type: str = "TK", only_compute_x: bool = True, 
                 univariate_mode: bool = False, univariate_option: int = None):
        """
        Inicializa el analizador TDA con parámetros de embedding y modos de operación.

        Parámetros:
            embedding_type (str): Tipo de embedding a usar ("TK", "STK", "SW").
            only_compute_x (bool): Si es True, solo computa features de x_datos.
            univariate_mode (bool): Si es True, activa el modo univariado.
            univariate_option (int): Opción específica para el modo univariado (1 o 2).
        """
        self.embedding_type = embedding_type
        self.only_compute_x = only_compute_x
        self.univariate_mode = univariate_mode
        self.univariate_option = univariate_option

        # Atributos para almacenar resultados y estado del último análisis
        self.labels_sequence = []
        self.last_window_size = None
        self.last_stride = None
        self.last_embedding_dimension = None
        self.last_embedding_time_delay = None
        self.last_x_datos = None
        self.num_windows_features = 0 # Inicializar a 0

        # Atributos para almacenar los resultados del último análisis
        self.last_diagrams = None
        self.last_trans_features = None
        self.last_plot_cloud = None
        
        # Atributos de los umbrales de get_labels para usar en get_last_label si fuera necesario
        self.last_k_h0 = None
        self.last_k_h1 = None
        self.last_k_h2 = None
        self.last_min_significant_dimensions = None
        
    def homologia_persistente(self, x_datos: np.ndarray, y_datos=None, 
                              embedding_dimension: int = 2, embedding_time_delay: int = 1, 
                              window_size: int = 100, stride: int = 4):
        """
        Calcula la homología persistente usando Vietoris-Rips para extraer características
        topológicas (entropías).

        Parámetros:
            x_datos (np.ndarray): Serie de tiempo de entrada (precios).
            y_datos: No utilizado directamente en esta función, considerar si es necesario.
            embedding_dimension (int): Dimensión del embedding.
            embedding_time_delay (int): Retardo de tiempo para el embedding.
            window_size (int): Tamaño de la ventana deslizante.
            stride (int): Paso de la ventana deslizante.

        Retorna:
            tuple: (diagramas, features_topologicos, plot_cloud - opcional)
                   diagramas: Diagramas de persistencia.
                   features_topologicos: Features de entropía.
                   plot_cloud: Nube de puntos proyectada (si univariate_option == 2).
        """
        # Guardar parámetros del último análisis
        self.last_window_size = window_size
        self.last_stride = stride
        self.last_embedding_dimension = embedding_dimension
        self.last_embedding_time_delay = embedding_time_delay
        self.last_x_datos = x_datos # Almacenar para get_predictive_targets

        # Selección del tipo de embedder
        embedder = None
        if self.embedding_type == "STK" or (self.univariate_mode and self.univariate_option == 1):
            embedder = TakensEmbedding(time_delay=embedding_time_delay, dimension=embedding_dimension, stride=stride)
        elif self.embedding_type == "TK" or (self.univariate_mode and self.univariate_option == 2):
            embedder = SingleTakensEmbedding(time_delay=embedding_time_delay, dimension=embedding_dimension, stride=stride)
        elif self.embedding_type == "SW":
            embedder = SlidingWindow(size=embedding_dimension, stride=stride)
        else:
            raise ValueError(f"Embedder '{self.embedding_type}' no disponible o configuración univariada inválida.")

        persistence = VietorisRipsPersistence(homology_dimensions=[0, 1, 2], n_jobs=-1)
        scaling = Scaler()
        entropy = PersistenceEntropy(normalize=False, nan_fill_value=-10)

        steps_1 = []
        if self.univariate_mode:
            sw = SlidingWindow(size=window_size, stride=stride)
            if self.univariate_option == 1:
                steps_1 = [("sw", sw), ("embedder", embedder), ("persistence", persistence)]
            elif self.univariate_option == 2:
                steps_1 = [("embedder", embedder), ("sw", sw), ("persistence", persistence)]
            else:
                raise ValueError("Opción univariada no válida. Debe ser 1 o 2.")
        else:
            pca = CollectionTransformer(PCA(n_components=3), n_jobs=-1)
            steps_1 = [("embedder", embedder), ("PCA", pca), ("persistence", persistence)]
        
        steps_2 = [("scaling", scaling), ("entropy", entropy)]

        topological_transformer_1 = Pipeline(steps_1)
        topological_transformer_2 = Pipeline(steps_2)

        plot_cloud = None
        if self.univariate_mode and self.univariate_option == 2:
            pca_plot = PCA(n_components=2) # Renombrar para evitar conflicto
            steps_v = [("embedder", embedder), ("PCA", pca_plot)]
            plot_cloud = Pipeline(steps_v).fit_transform(x_datos)
            
        diagrams = None
        trans_features = None

        if self.only_compute_x: # Usar el nombre de atributo más claro
            diagrams = topological_transformer_1.fit_transform(x_datos)
            trans_features = topological_transformer_2.fit_transform(diagrams)

        # Guardar la longitud de los features generados en self
        self.num_windows_features = len(trans_features) if trans_features is not None else 0
        self.last_diagrams = diagrams
        self.last_trans_features = trans_features
        self.last_plot_cloud = plot_cloud

        # Retornos consistentes o adaptados
        if self.univariate_mode and self.univariate_option == 1:
            return diagrams[0], trans_features # Asumiendo que diagrams[0] es lo que quieres
        elif self.univariate_mode and self.univariate_option == 2:
            return diagrams[0], trans_features, plot_cloud
        else: # Default case
            return diagrams, trans_features # Devuelve diagrams completo si no es opción univariada específica

    def get_predictive_targets(self) -> np.ndarray:
        """
        Genera los valores objetivo (y) para un modelo predictivo de ML.
        Cada y[i] es el precio al inicio de la ventana (i+1) de homología,
        correspondiente a los features X[i] de la ventana i.

        Requiere que homologia_persistente haya sido ejecutada previamente
        para establecer self.last_x_datos, self.last_window_size,
        self.last_stride y self.num_windows_features.

        Retorna:
            np.ndarray: Array con los precios futuros.
                        Tendrá una longitud de len(trans_features) - 1.
        """
        # Verificaciones para asegurar que homologia_persistente fue ejecutada
        if self.last_x_datos is None:
            raise ValueError("No se han generado los datos topológicos. Ejecuta homologia_persistente primero.")
        if self.last_window_size is None: # Si last_x_datos existe, estos deberían existir
            raise ValueError("Parámetros de ventana no definidos. Ejecuta homologia_persistente primero.")
        if self.last_stride is None:
            raise ValueError("Parámetros de stride no definidos. Ejecuta homologia_persistente primero.")
        if self.num_windows_features == 0 and self.last_trans_features is None: 
             raise ValueError("La longitud de los features topológicos no ha sido guardada o es cero. Asegúrate de ejecutar homologia_persistente y que guarde la longitud de trans_features.")

        original_price_series = np.asarray(self.last_x_datos).flatten()
        window_size = self.last_window_size
        stride = self.last_stride
        num_windows_features = self.num_windows_features

        predictive_targets = []
        
        # Itera sobre las ventanas de features, excluyendo la última
        for i in range(num_windows_features - 1):
            next_window_start_idx = (i + 1) * stride
            
            if next_window_start_idx >= len(original_price_series):
                break # Deja de agregar targets si no hay datos suficientes
            
            target_price = original_price_series[next_window_start_idx]
            predictive_targets.append(target_price)

        return np.array(predictive_targets)
    
    def mapper_algorithm(self, datos, n_cubiertas=15):
        """
        usar el algoritmo mapper. TODO: documentar 
        """
        # configurar algoritmo mapper
        mapper_algoritmo = make_mapper_pipeline(
            filter_func=Projection(columns=list(range(datos.shape[1]))),
            clusterer=DBSCAN(eps=0.3, min_samples=5),
            n_jobs=n_cubiertas,
            scaler=MinMaxScaler()
        )
        
        # aplicar mapper
        fig = plot_static_mapper_graph(
            pipeline=mapper_algoritmo,
            data=datos,
            node_color_statistic=np.mean
        )
        
        return fig

    def get_labels(self, features: np.ndarray, serie: np.ndarray, 
                   k_h0: float = 1.5, k_h1: float = 1.5, k_h2: float = 1.5, 
                   min_significant_dimensions: int = 1):
        """
        Genera etiquetas ('baja', 'estable', 'sube') basadas en cambios significativos
        en la entropía topológica y la dirección del cambio de precio.

        Parámetros:
            features (np.ndarray): Entropías topológicas (H0, H1, H2) con shape (n_windows, 3).
            serie (np.ndarray): Serie de precios original o promediada por ventana.
                                Debe tener una longitud que corresponda a las ventanas.
            k_h0, k_h1, k_h2 (float): Parámetros de sensibilidad para los umbrales de entropía.
            min_significant_dimensions (int): Número mínimo de dimensiones de entropía
                                              que deben cambiar significativamente para activar un "evento".

        Retorna:
            tuple: (features, list_of_labels)
                   features: El array de features de entrada sin modificar.
                   list_of_labels: Lista de etiquetas de estado ('baja', 'estable', 'sube').
        """
        # Guardar los parámetros para uso futuro si es necesario
        self.last_k_h0 = k_h0
        self.last_k_h1 = k_h1
        self.last_k_h2 = k_h2
        self.last_min_significant_dimensions = min_significant_dimensions

        if features.shape[0] == 0:
            return features, []

        H0_entropy = features[:, 0]
        H1_entropy = features[:, 1]
        H2_entropy = features[:, 2]

        # Asegurarse de que 'serie' tenga la longitud adecuada para los precios
        # correspondientes a las ventanas de features.
        # Si 'serie' es la serie de precios original, necesitarías mapear
        # los precios al inicio de cada ventana de los features.
        # Por ahora, asumo que `serie` tiene la misma longitud que `features`
        # y cada elemento de `serie` corresponde a la ventana `i`.
        if len(serie) < len(features):
            raise ValueError(f"La longitud de 'serie' ({len(serie)}) es menor que la de 'features' ({len(features)}). "
                             "Asegúrate de que 'serie' contenga los precios de cada ventana.")
        
        # La primera etiqueta siempre se determina por el precio inicial.
        # O, si no hay un "antes", se puede poner un valor por defecto.
        # Tu código original iniciaba con [1], que presumiblemente era "estable".
        # Si la primera ventana no tiene un cambio previo, puede ser "estable".
        # labels = ['estable'] 
        
        # Para mantener la misma lógica de "primer label", pero usando strings:
        # Asumiendo que tu 1 significa 'estable' o 'se mantiene'
        labels = [1] if len(features) > 0 else [] 

        if len(features) > 1:
            delta_H0 = np.abs(np.diff(H0_entropy))
            delta_H1 = np.abs(np.diff(H1_entropy))
            delta_H2 = np.abs(np.diff(H2_entropy))

            # Calcular umbrales dinámicos, manejando std=0
            threshold_H0 = np.mean(delta_H0) + k_h0 * np.std(delta_H0) if np.std(delta_H0) > 0 else (np.max(delta_H0) * k_h0 if delta_H0.size > 0 else 0)
            threshold_H1 = np.mean(delta_H1) + k_h1 * np.std(delta_H1) if np.std(delta_H1) > 0 else (np.max(delta_H1) * k_h1 if delta_H1.size > 0 else 0)
            threshold_H2 = np.mean(delta_H2) + k_h2 * np.std(delta_H2) if np.std(delta_H2) > 0 else (np.max(delta_H2) * k_h2 if delta_H2.size > 0 else 0)

            # Ajustar umbrales mínimos si todas las diferencias son cero
            if delta_H0.size > 0 and np.all(delta_H0 == 0): threshold_H0 = 0
            if delta_H1.size > 0 and np.all(delta_H1 == 0): threshold_H1 = 0
            if delta_H2.size > 0 and np.all(delta_H2 == 0): threshold_H2 = 0

            for i in range(1, len(features)):
                significant_H0 = delta_H0[i - 1] > threshold_H0
                significant_H1 = delta_H1[i - 1] > threshold_H1
                significant_H2 = delta_H2[i - 1] > threshold_H2

                num_significant_changes = sum([significant_H0, significant_H1, significant_H2])
                hay_evento = num_significant_changes >= min_significant_dimensions

                precio_antes = serie[i - 1]
                precio_despues = serie[i] 
                
                if hay_evento:
                    if precio_despues > precio_antes:
                        labels.append(2)
                    elif precio_despues < precio_antes:
                        labels.append(0)
                    else: # Precio se mantuvo a pesar del evento topológico
                        labels.append(1) 
                else: # No hay evento topológico significativo
                    labels.append(1)
                    
        self.labels_sequence = labels # Guardar la secuencia de labels

        return features, labels

    def get_last_label(self):
        """
        Retorna el último estado de la secuencia de etiquetas como un string.
        """
        if not self.labels_sequence:
            return None # O maneja el caso de secuencia vacía como prefieras

        # Asumiendo que self.labels_sequence es una lista de strings
        last_label = self.labels_sequence[-1]

        # Si por alguna razón last_label es una lista de un solo elemento (ej. ['sube']),
        # entonces extráelo:
        if isinstance(last_label, list) and len(last_label) == 1:
            return last_label[0]
        elif isinstance(last_label, np.ndarray) and last_label.size == 1:
            return str(last_label.item()) # Para manejar si todavía hay un ndarray aquí
        else:
            return str(last_label) # Asegurarse de que sea un string   

class MODELAJE:
    """
    Clase para el modelaje de series de tiempo y pronóstico,
    permitiendo elegir entre un modelo basado en TDA o un modelo ARIMA.
    """
    def __init__(self, tda_analyzer = None): # He quitado el tipo `AnalisisTDA` para no forzar la importación circular si no es necesario en el init
        self.tda_analyzer = tda_analyzer
        self.arima_model = None
        self.tda_ml_model = None

    def run_forecast_TDA(self, serie: pd.Series, test_size: int = 10, **kwargs) -> dict:
        """
        Evalúa el rendimiento de un RandomForestRegressor utilizando features topológicos (TDA)
        para predecir el precio del siguiente punto. Realiza una validación en un conjunto de prueba.

        Parámetros:
            serie (pd.Series): La serie de tiempo histórica de precios.
            test_size (int): El número de muestras a usar para el conjunto de prueba.
                             Estas serán las últimas 'test_size' muestras disponibles
                             para la evaluación del modelo.
            **kwargs: Parámetros adicionales para self.tda_analyzer.homologia_persistente,
                      como 'window_size', 'stride', 'embedding_dimension', 'embedding_time_delay',
                      y 'n_estimators_rf' para el RandomForestRegressor.

        Retorna:
            dict: Un diccionario con las predicciones del conjunto de prueba ('pred'),
                  la gráfica de resultados ('pred_plot'), las métricas de evaluación ('metrics'),
                  y los valores reales del conjunto de prueba ('actual_test').
        """
        if self.tda_analyzer is None:
            raise ValueError("Se seleccionó TDA pero no se proporcionó una instancia de AnalisisTDA al inicializar MODELAJE.")
        
        print("\n--- Ejecutando Evaluación de Modelo con Features TDA ---")
        
        # Generar features topológicos y targets
        # Los features se extraen de toda la serie 'x_datos=serie.values'
        # 'trans_features' contendrá los features para cada ventana.
        _, trans_features = self.tda_analyzer.homologia_persistente(
            x_datos=serie.values,
            window_size=kwargs.get('window_size', 100),
            stride=kwargs.get('stride', 4),
            embedding_dimension=kwargs.get('embedding_dimension', 2),
            embedding_time_delay=kwargs.get('embedding_time_delay', 1)
        )
        
        # 'predictive_targets' serán los precios futuros (el valor de la serie
        # en el inicio de la siguiente ventana) alineados con 'trans_features'.
        predictive_targets = self.tda_analyzer.get_predictive_targets()

        # Asegurarse de que X e Y estén alineados y tengan la misma longitud.
        # 'trans_features' tiene un elemento más que 'predictive_targets' porque
        # el último feature no tiene un 'siguiente precio' conocido.
        X_for_model = trans_features[:-1]
        y_for_model = predictive_targets

        if len(X_for_model) == 0 or len(y_for_model) == 0:
            raise ValueError("No se pudieron generar suficientes features o targets para el modelado TDA. Ajusta los parámetros de ventana/stride o proporciona más datos.")

        # Determinar el tamaño del conjunto de prueba.
        n_total_samples = len(X_for_model)
        if test_size >= n_total_samples:
            # Si el tamaño de prueba es mayor o igual al total, usar un porcentaje grande
            # para la prueba, asegurando que quede al menos 1 para el entrenamiento si n_total_samples > 1.
            test_size_ratio = 0.5 if n_total_samples > 1 else 0.0 # O ajusta según prefieras
            print(f"Advertencia: 'test_size' ({test_size}) es mayor o igual al número total de muestras disponibles ({n_total_samples}). "
                  f"Usando un test_size_ratio de {test_size_ratio:.2f}.")
        else:
            test_size_ratio = test_size / n_total_samples
            
        # Asegurar al menos un elemento en el conjunto de entrenamiento si es posible
        if n_total_samples > 1 and int(n_total_samples * (1 - test_size_ratio)) == 0:
            test_size_ratio = (n_total_samples - 1) / n_total_samples # Deja 1 para entrenamiento

        X_train, X_test, y_train, y_test = train_test_split(
            X_for_model, y_for_model, test_size=test_size_ratio, random_state=42
        )
        
        # Validar que los conjuntos no estén vacíos después del split
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("La división train/test resultó en un conjunto de entrenamiento o prueba vacío. Ajusta 'test_size' o proporciona más datos.")


        # Entrenamiento y predicción
        self.tda_ml_model = RandomForestRegressor(n_estimators=kwargs.get('n_estimators_rf', 100), random_state=42)
        self.tda_ml_model.fit(X_train, y_train)
        y_pred = self.tda_ml_model.predict(X_test)

        # Métricas de evaluación
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2 Score': r2
        }

        # Generación de gráfica
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.arange(len(y_test)), y_test, label='Valores Reales (Test)', color='orange')
        ax.plot(np.arange(len(y_pred)), y_pred, label='Predicciones TDA', color='green', linestyle='--')
        ax.set_title(f'Predicciones del Modelo TDA vs Valores Reales (Test)')
        ax.set_xlabel('Muestra del Conjunto de Prueba')
        ax.set_ylabel('Precio')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        return {"pred": y_pred, "pred_plot": fig, "metrics": metrics, "actual_test": y_test}

    def run_forecast_ARIMA(self, serie: pd.Series, forecast_steps: int, freq: str) -> dict:
        """
        Realiza un pronóstico ARIMA para una serie de tiempo dada, evaluando su rendimiento
        en los últimos 'forecast_steps' de la serie.

        Parámetros:
            serie (pd.Series): La serie de tiempo histórica.
            forecast_steps (int): Número de pasos a pronosticar/evaluar en el futuro inmediato de la serie.
                                  Estos serán los puntos utilizados como conjunto de prueba.
            freq (str): Frecuencia de la serie (ej. "W", "M") para indexación del pronóstico.

        Retorna:
            dict: Diccionario con los resultados del pronóstico. Incluye las predicciones ('pred'),
                  la gráfica ('pred_plot'), las métricas de evaluación ('metrics'),
                  y los valores reales del conjunto de prueba ('actual_test').
        """
        if not isinstance(serie, pd.Series):
            raise TypeError("La 'serie' debe ser un objeto pd.Series.")
        
        if not isinstance(serie.index, pd.DatetimeIndex):
            print("Advertencia: El índice de la serie no es DatetimeIndex. Intentando convertir...")
            try:
                serie.index = pd.to_datetime(serie.index) # Correcto: asignar el índice convertido
            except Exception as e:
                raise ValueError(f"No se pudo convertir el índice de la serie a DatetimeIndex: {e}")
        
        # Dividir la serie en entrenamiento y prueba
        # train_series: todos los datos excepto los últimos 'forecast_steps'
        # test_series: los últimos 'forecast_steps' de la serie original
        train_series = serie[:-forecast_steps]
        test_series = serie[-forecast_steps:]

        if len(train_series) == 0:
            raise ValueError("El conjunto de entrenamiento está vacío. Ajusta 'forecast_steps' o proporciona más datos.")
        if len(test_series) == 0:
            raise ValueError("El conjunto de prueba está vacío. Ajusta 'forecast_steps'.")

        # Entrenar el modelo ARIMA
        # Importante: Si tu serie tiene estacionalidad (ej. semanal, mensual, anual),
        # considera `seasonal=True` y el parámetro `m` (período estacional).
        # Por ejemplo: m=52 para datos semanales con ciclo anual, m=12 para mensual.
        arima_model = pm.auto_arima(train_series,
                                     seasonal=False, # Cambiar a True y añadir m=periodo si hay estacionalidad
                                     stepwise=True,
                                     suppress_warnings=True,
                                     error_action='ignore',
                                     n_jobs=-1)
        
        # Realizar predicciones para el período de prueba
        preds = arima_model.predict(n_periods=forecast_steps)
        
        # Crear índice de fecha para las predicciones
        last_train_date = train_series.index[-1]
        next_start_date = last_train_date + pd.tseries.frequencies.to_offset(freq)
        forecast_series = pd.Series(preds, index=pd.date_range(start=next_start_date, periods=forecast_steps, freq=freq), name='forecast')

        # Calcular métricas (comparando predicciones con datos reales de test)
        mae = mean_absolute_error(test_series, forecast_series)
        rmse = np.sqrt(mean_squared_error(test_series, forecast_series))
        
        # Cálculo de MAPE, asegurando robustez contra ceros
        mape = np.mean(np.abs((test_series - forecast_series) / test_series.replace(0, np.nan).fillna(np.finfo(float).eps))) * 100

        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE (%)': mape
        }

        # Generar gráfica
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_series.index, train_series, label='Datos de Entrenamiento')
        ax.plot(test_series.index, test_series, label='Datos Reales (Prueba)', color='orange')
        ax.plot(forecast_series.index, forecast_series, label='Pronóstico ARIMA', color='green', linestyle='--')
        ax.set_title(f'Pronóstico ARIMA para {serie.name or "Serie de Precios"} ({forecast_steps} pasos)')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Precio')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        return {"pred": forecast_series, "pred_plot": fig, "metrics": metrics, "actual_test": test_series}

class Markov:
    """
    Clase para modelar una serie de tiempo como una cadena de Markov,
    calcular su matriz de transición y estimar probabilidades a n pasos.
    """
    def __init__(self, series: np.ndarray, labels: np.ndarray = None, 
                 use_tda_labels: bool = False, threshold: float = 0.05):
        """
        Inicializa la clase Markov, procesando la serie para obtener estados
        y calculando la matriz de transición.

        Parámetros:
            series (np.ndarray): La serie de tiempo de precios original.
            labels (np.ndarray, opcional): Etiquetas de estado predefinidas (0, 1, 2)
                                            si se usa TDA. Por defecto es None.
            use_tda_labels (bool): Si es True, usa las 'labels' proporcionadas.
                                   Si es False, deriva los estados de la 'series'.
            threshold (float): Umbral para determinar los cambios de estado
                               cuando no se usan etiquetas TDA.
        """
        self.series = series
        self.labels = labels
        self.use_tda_labels = use_tda_labels # Renombrado para claridad
        self.threshold = threshold
        self.num_states = 3 # Asumiendo 3 estados: Baja (0), Mantiene (1), Sube (2)
        
        self.states = None
        self.transition_matrix = None

        self._process()

    def _process(self):
        """
        Método interno para determinar los estados y calcular la matriz de transición.
        """
        if self.use_tda_labels:
            if self.labels is None:
                raise ValueError("`labels` no puede ser None si `use_tda_labels` es True.")
            # Asegurar que labels sea un array de numpy
            
            self.states = np.asarray(self.labels)
            # Validación simple de etiquetas si es posible
            if not np.all(np.isin(self.states, [0, 1, 2])):
                print("Advertencia: Las etiquetas TDA contienen valores fuera de 0, 1, 2.")
        else:
            self.states = self._series_to_states(np.asarray(self.series), self.threshold)
        
        # Validación de estados generados
        if len(self.states) < 2:
            raise ValueError("No se generaron suficientes estados para calcular la matriz de transición (se requieren al menos 2).")

        self.transition_matrix = self._estimate_transition_matrix(self.states)

    def _series_to_states(self, series: np.ndarray, threshold: float) -> np.ndarray:
        """
        Convierte una serie de tiempo en una secuencia de estados (0:baja, 1:estable, 2:sube)
        basándose en cambios porcentuales y un umbral.

        Parámetros:
            series (np.ndarray): La serie de tiempo numérica.
            threshold (float): El umbral de cambio porcentual para clasificar los estados.

        Retorna:
            np.ndarray: Un array de enteros que representa la secuencia de estados.
        """
        states = []
        for i in range(len(series) - 1):
            # Evitar división por cero añadiendo un pequeño epsilon si series[i] es muy cercano a cero
            denominator = series[i] if series[i] != 0 else np.finfo(float).eps
            change = (series[i+1] - series[i]) / denominator
            
            if change < -threshold:
                states.append(0)  # Baja
            elif change > threshold:
                states.append(2)  # Sube
            else:
                states.append(1)  # Se Mantiene
        return np.array(states)

    def _estimate_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """
        Estima la matriz de transición de una cadena de Markov a partir de una secuencia de estados.

        Parámetros:
            states (np.ndarray): La secuencia de estados (enteros 0, 1, 2).

        Retorna:
            np.ndarray: La matriz de probabilidades de transición.
        """
        # Asegurarse de que states tenga al menos dos elementos para calcular transiciones
        if len(states) < 2:
            # Esto ya se valida en _process, pero es bueno tenerlo aquí también
            raise ValueError("Se requieren al menos 2 estados para estimar la matriz de transición.")

        transition_counts = np.zeros((self.num_states, self.num_states))
        for (i, j) in zip(states[:-1], states[1:]):
            transition_counts[i][j] += 1
        
        row_sums = transition_counts.sum(axis=1)
        
        # Evitar división por cero. Donde la suma de la fila sea 0, la fila de probabilidades será 0.
        transition_matrix = np.divide(transition_counts, row_sums[:, np.newaxis], 
                                     where=row_sums[:, np.newaxis]!=0)
        
        return transition_matrix

    def get_n_step_matrix(self, n: int) -> np.ndarray:
        """
        Calcula la matriz de transición a 'n' pasos.

        Parámetros:
            n (int): El número de pasos para calcular la matriz de transición.

        Retorna:
            np.ndarray: La matriz de probabilidades de transición a 'n' pasos.
        """
        if self.transition_matrix is None:
            raise ValueError("La matriz de transición no ha sido calculada. Asegúrate de que la clase se inicialice correctamente.")
        if not isinstance(n, int) or n < 1:
            raise ValueError("El número de pasos 'n' debe ser un entero positivo.")
        
        return np.linalg.matrix_power(self.transition_matrix, n)

    def get_highest_proba(self, n: int, last_state: int) -> tuple[float, int]:
        """
        Calcula la probabilidad más alta de transición a un estado futuro después de 'n' pasos,
        dado un estado inicial, y retorna ese estado.

        Parámetros:
            n (int): El número de pasos futuros.
            last_state (int): El estado actual (último estado observado) (0:baja, 1:estable, 2:sube).

        Retorna:
            tuple[float, int]: Una tupla que contiene:
                               - float: La probabilidad más alta de alcanzar un estado futuro.
                               - int: El estado futuro (0, 1, 2) con la probabilidad más alta.
        """
        if not isinstance(last_state, int) or last_state not in [0, 1, 2]:
            raise ValueError("`last_state` debe ser 0 (baja), 1 (se mantiene) o 2 (sube).")

        P_n = self.get_n_step_matrix(n)
        
        # Si P_n[last_state] es una fila de ceros (ej. si el estado inicial nunca fue visitado),
        # max() puede causar un error o devolver -inf.
        # Es mejor manejar esto explícitamente.
        if np.all(P_n[last_state] == 0):
            return (0.0, None) # No hay transiciones posibles desde este estado, o no se observó.
        
        highest_proba = np.max(P_n[last_state])
        next_state = np.argmax(P_n[last_state])

        return (highest_proba, next_state)

    # Opcional: Un método para obtener la representación en string del estado
    def get_state_name(self, state_code: int) -> str:
        """
        Retorna el nombre descriptivo de un código de estado.
        """
        state_map = {0: 'baja', 1: 'se mantiene', 2: 'sube'}
        return state_map.get(state_code, "Desconocido")

    # Si quieres una función que devuelva el nombre del estado directamente
    def get_highest_proba_named(self, n: int, last_state: int) -> tuple[float, str]:
        """
        Calcula la probabilidad más alta de transición a un estado futuro después de 'n' pasos,
        dado un estado inicial, y retorna el nombre descriptivo de ese estado.
        """
        proba, state_code = self.get_highest_proba(n, last_state)
        if state_code is None:
            return (proba, "N/A")
        return (proba, self.get_state_name(state_code))

class Results:
    """
    Clase para orquestar y compilar los resultados de los análisis
    de series de tiempo utilizando modelos ARIMA y TDA con Random Forest,
    así como el análisis de cadenas de Markov.
    """
    def __init__(self, use_tda: bool = True, product: str = "strawberries", steps: int = 10):
        """
        Inicializa la clase Results con la configuración deseada para el análisis.

        Parámetros:
            use_tda (bool): Si es True, el análisis se centrará en el modelo TDA.
                            Si es False, se centrará en el modelo ARIMA.
            product (str): El nombre del producto para el cual se realizará el análisis.
            steps (int): Número de pasos/períodos a pronosticar o usar para la evaluación.
        """
        self.use_tda = use_tda # Renombrado TDA a use_tda para evitar confusión con el objeto TDA_analyzer
        self.product = product
        self.steps = steps

    def _calculate_gain_percentage(self, prices) -> float:
        """
        Calcula la ganancia porcentual del primer al último valor de una serie de precios.

        Parámetros:
            prices (Union[pd.Series, np.ndarray]): Una serie o array de precios.

        Retorna:
            float: La ganancia porcentual. Retorna 0.0 si no se puede calcular.
        """
        if not isinstance(prices, (pd.Series, np.ndarray)) or len(prices) < 2:
            # No se puede calcular la ganancia con menos de 2 precios
            # O si no es un tipo de dato esperad
            return 0.0 
        
        first_price = prices.iloc[0] if isinstance(prices, pd.Series) else prices[0]
        last_price = prices.iloc[-1] if isinstance(prices, pd.Series) else prices[-1]
        
        # Usar un epsilon para evitar división por cero en casos donde el precio inicial es muy bajo
        if abs(first_price) < np.finfo(float).eps: # np.finfo(float).eps es un número muy pequeño
            # Si el precio inicial es cero o muy cercano a cero, y el final es positivo,
            # la "ganancia" es esencialmente infinita o muy grande.
            # Podrías decidir retornar np.inf, 100% si el final es > 0, o simplemente 0.0
            # Retornar 0.0 es seguro para evitar errores, pero puede ser engañoso.
            # Para este contexto, si el precio inicial es cero, una ganancia porcentual no tiene sentido.
            # Si es exactamente cero y last_price es > 0, podríamos interpretarlo como 100%.
            return 100.0 if last_price > 0 else 0.0
            
        return ((last_price - first_price) / first_price) * 100

    def get_last_state(self, prices, threshold: float = 0.05) -> int:
        """
        Determina el último estado (0:baja, 1:estable, 2:sube) de una serie de precios
        basado en el cambio porcentual de los dos últimos puntos.

        Parámetros:
            prices (Union[pd.Series, np.ndarray]): La serie de precios.
            threshold (float): Umbral para determinar los cambios de estado.

        Retorna:
            int: El código del último estado (0, 1, 2).
        """
        if not isinstance(prices, (pd.Series, np.ndarray)) or len(prices) < 2:
            # Si no hay suficientes datos para calcular un cambio, podrías:
            # - Levantar un ValueError
            # - Retornar un estado por defecto (ej., 1 para 'se mantiene')
            # - Retornar None y manejarlo en la llamada.
            # Para la robustez, levantar un error es más informativo.
            raise ValueError("Se requieren al menos 2 precios para determinar el último estado.")
        
        # Asegurarse de acceder a los valores correctamente si es un pd.Series
        p_last = prices.iloc[-1] if isinstance(prices, pd.Series) else prices[-1]
        p_second_last = prices.iloc[-2] if isinstance(prices, pd.Series) else prices[-2]

        denominator = p_second_last if p_second_last != 0 else np.finfo(float).eps
        delta = (p_last - p_second_last) / denominator
        
        if delta > threshold:
            return 2  # Sube
        elif delta < -threshold:
            return 0  # Baja
        else: 
            return 1  # Se Mantiene
        
    def arima_results(self, freq: str = "W") -> dict:
        """
        Realiza el análisis y pronóstico utilizando el modelo ARIMA y la cadena de Markov.

        Parámetros:
            freq (str): Frecuencia de la serie (ej. "W" para semanal).

        Retorna:
            dict: Un diccionario con el pronóstico ARIMA, la gráfica, la ganancia esperada
                  y las probabilidades de estado de Markov.
        """
        print(f"\n--- Ejecutando Análisis ARIMA para {self.product} ---")
        data_loader = Data() # Crear una instancia de Data
        df = data_loader.set_df()
        s = data_loader.prepare_series(self.product, freq)

        modeler_arima = MODELAJE(tda_analyzer=None) # No se usa TDA para ARIMA
        arima_forecast_results = modeler_arima.run_forecast_ARIMA(serie=s, forecast_steps=self.steps, freq=freq)
        
        # Calcular ganancia porcentual sobre las predicciones ARIMA
        percentage_gain = self._calculate_gain_percentage(arima_forecast_results["pred"])

        # Análisis de Markov basado en la serie original (para aprender la dinámica general)
        arima_markov = Markov(s.values, use_tda_labels=False, threshold=0.05) # Usar s.values para pasar np.ndarray

        # Obtener el último estado de la *serie de predicciones* para proyectar el futuro
        try:
            last_forecast_state = self.get_last_state(arima_forecast_results["pred"])
        except ValueError as e:
            print(f"Error al obtener el último estado de las predicciones ARIMA: {e}. No se calcularán probabilidades de Markov.")
            highest_proba_info = (0.0, 'N/A') # Valor por defecto
        else:
            # Obtener la probabilidad más alta para el estado siguiente desde el último estado pronosticado
            highest_proba_info = arima_markov.get_highest_proba_named(n=self.steps, last_state=last_forecast_state)


        final_results = {
            "Forecast": arima_forecast_results["pred"],
            "Forecast_Plot": arima_forecast_results["pred_plot"],
            "Metrics": arima_forecast_results["metrics"], # Incluir las métricas del modelo
            "Expected_Profit_Pct": percentage_gain,
            "Markov_Next_State_Proba": highest_proba_info
        }

        return final_results

    def TDA_results(self) -> dict:
        """
        Realiza el análisis y la evaluación del modelo Random Forest con features TDA,
        e integra el análisis de cadena de Markov basado en las etiquetas TDA.

        Retorna:
            dict: Un diccionario con las predicciones del modelo TDA, la gráfica,
                  las métricas de evaluación y las probabilidades de estado de Markov.
        """
        print(f"\n--- Ejecutando Análisis TDA para {self.product} ---")
        data_loader = Data()
        df = data_loader.set_df()
        # Asegúrate de pasar 'freq' si es necesario para prepare_series aquí
        s = data_loader.prepare_series(self.product, freq="W") # Ejemplo: asumo frecuencia semanal

        # Instanciar AnalisisTDA y MODELAJE
        tda_analyser_instance = AnalisisTDA(embedding_type="SW", univariate_mode=True, univariate_option=1)
        modeler_tda = MODELAJE(tda_analyzer=tda_analyser_instance)

        tda_params = {
            'window_size': 100, # Aumentar window_size si es posible para features más robustos
            'stride': 4,
            'embedding_dimension': 2,
            'embedding_time_delay': 1,
            'n_estimators_rf': 100 # Puedes añadir parámetros para el Random Forest aquí
        }

        tda_evaluation_results = None
        try:
            tda_evaluation_results = modeler_tda.run_forecast_TDA(
                serie=s,
                test_size=self.steps, # Renombrado a test_size en MODELAJE
                **tda_params
            )
        except ValueError as e:
            print(f"\nError al ejecutar modelo TDA: {e}")
            print("Asegúrate de que la serie de tiempo sea lo suficientemente larga para los parámetros de TDA.")
            # tda_evaluation_results se mantiene None
        
        # Si la evaluación del TDA falló, no podemos continuar con el resto
        if tda_evaluation_results is None:
            return {
                "Forecast": None,
                "Forecast_Plot": None,
                "Metrics": {},
                "Expected_Profit_Pct": 0.0,
                "Markov_Next_State_Proba": (0.0, 'N/A')
            }

        # Generar las etiquetas TDA desde el AnalisisTDA para pasarlas a Markov
        # Necesitamos volver a ejecutar homologia_persistente para obtener las etiquetas de la serie COMPLETA
        # o asegurarnos de que AnalisisTDA haya guardado esas etiquetas internamente.
        # Si AnalisisTDA.get_labels() se basa en la última ejecución de homologia_persistente,
        # entonces necesitamos ejecutar homologia_persistente de nuevo para toda la serie.
        
        # Una forma más directa: si AnalisisTDA ya tiene un método para generar etiquetas
        # para toda la serie que se pueda llamar sin ejecutar todo el forecast_TDA de nuevo.
        # Asumo que AnalisisTDA ya tiene los estados calculados internamente desde la última llamada
        # a homologia_persistente, o que 's' es la misma serie que se usó.
        
        # Aquí, vamos a generar las etiquetas directamente para la serie 's' para la clase Markov.
        _, features = tda_analyser_instance.homologia_persistente(
             x_datos=s.values,
             window_size=tda_params.get('window_size'),
             stride=tda_params.get('stride'),
             embedding_dimension=tda_params.get('embedding_dimension'),
             embedding_time_delay=tda_params.get('embedding_time_delay')
        )
        tda_features, tda_labels = tda_analyser_instance.get_labels(features= features, serie = s)

        # Usar las etiquetas TDA para la cadena de Markov
        # Aquí es donde pasas `labels` y `use_tda_labels=True
        # 
        # print("TDA Labels tipo:", type(tda_labels))
   

        tda_markov = Markov(series=s.values, labels=tda_labels, use_tda_labels=True)
        # _process() se llama automáticamente en el init, no necesitas llamarlo de nuevo.

        # Calcular ganancia porcentual sobre las *predicciones del conjunto de prueba* del modelo TDA
        percentage_gain = self._calculate_gain_percentage(tda_evaluation_results["pred"])
        
        # Obtener el último estado de las *predicciones del conjunto de prueba* para proyectar el futuro
        try:
            last_forecast_state = self.get_last_state(tda_evaluation_results["pred"])
        except ValueError as e:
            print(f"Error al obtener el último estado de las predicciones TDA: {e}. No se calcularán probabilidades de Markov.")
            highest_proba_info = (0.0, 'N/A')
        else:
            highest_proba_info = tda_markov.get_highest_proba_named(n=self.steps, last_state=last_forecast_state)


        final_results = {
            "Forecast": tda_evaluation_results["pred"], # Estas son predicciones del test set
            "Forecast_Plot": tda_evaluation_results["pred_plot"],
            "Metrics": tda_evaluation_results["metrics"], # Incluir las métricas del modelo
            "Expected_Profit_Pct": percentage_gain,
            "Markov_Next_State_Proba": highest_proba_info
        }

        return final_results

def trial(product, freq, steps):
    TDA = Results(use_tda = True, product=product, steps=steps)
    ARIMA = Results(use_tda=False, product= product, steps=steps)
    arima_results = ARIMA.arima_results(freq=freq)
    tda_results = TDA.TDA_results()

    return arima_results, tda_results

arima_results, tda_results = trial(product = "strawberries", freq = "W", steps = 10)


# FUNCIONAAAAAAA