import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Energía Renovable",
    page_icon="⚡",
    layout="wide"
)

# Título principal
st.title("📊 Análisis de Datos - Energía Renovable")
st.markdown("---")

# Sidebar
st.sidebar.header("📁 Carga de Datos")

# Opción para cargar datos
opcion_carga = st.sidebar.radio(
    "¿Cómo deseas cargar los datos?",
    ["Usar archivo predeterminado", "Cargar mi propio archivo"]
)

df = None

if opcion_carga == "Cargar mi propio archivo":
    uploaded_file = st.sidebar.file_uploader(
        "Sube tu archivo CSV",
        type=["csv"],
        help="Selecciona un archivo CSV para analizar"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ Archivo cargado exitosamente")
        except Exception as e:
            st.sidebar.error(f"❌ Error al cargar el archivo: {e}")
    else:
        st.sidebar.info("👆 Por favor, sube un archivo CSV para comenzar")
else:
    # Cargar datos predeterminados
    @st.cache_data
    def cargar_datos():
        try:
            df = pd.read_csv("energia_renovable.csv")
            return df
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            return None
    
    df = cargar_datos()
    if df is not None:
        st.sidebar.success("✅ Usando archivo predeterminado")

st.sidebar.markdown("---")
st.sidebar.header("Opciones de Análisis")
opcion = st.sidebar.selectbox(
    "Selecciona una sección:",
    ["Exploración de Datos", "Visualización", "Análisis Estadístico", "Modelado"]
)

if df is not None:
    # Sección: Exploración de Datos
    if opcion == "Exploración de Datos":
        st.header("🔍 Exploración de Datos")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filas", df.shape[0])
        with col2:
            st.metric("Columnas", df.shape[1])
        with col3:
            st.metric("Valores Nulos", df.isnull().sum().sum())
        
        st.subheader("Vista previa de los datos")
        st.dataframe(df.head(10))
        
        st.subheader("Información del Dataset")
        st.write(df.describe())
        
        st.subheader("Tipos de Datos")
        st.write(df.dtypes)
    
    # Sección: Visualización
    elif opcion == "Visualización":
        st.header("📈 Visualización de Datos")
        
        # Seleccionar columnas numéricas
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if columnas_numericas:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribución de Variables")
                columna_seleccionada = st.selectbox("Selecciona una variable:", columnas_numericas)
                fig, ax = plt.subplots()
                ax.hist(df[columna_seleccionada].dropna(), bins=30, edgecolor='black')
                ax.set_xlabel(columna_seleccionada)
                ax.set_ylabel("Frecuencia")
                st.pyplot(fig)
            
            with col2:
                st.subheader("Boxplot")
                fig, ax = plt.subplots()
                ax.boxplot(df[columna_seleccionada].dropna())
                ax.set_ylabel(columna_seleccionada)
                st.pyplot(fig)
            
            # Matriz de correlación
            if len(columnas_numericas) > 1:
                st.subheader("Matriz de Correlación")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[columnas_numericas].corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
    
    # Sección: Análisis Estadístico
    elif opcion == "Análisis Estadístico":
        st.header("📊 Análisis Estadístico")
        
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if columnas_numericas:
            columna = st.selectbox("Selecciona una variable para análisis:", columnas_numericas)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Media", f"{df[columna].mean():.2f}")
            with col2:
                st.metric("Mediana", f"{df[columna].median():.2f}")
            with col3:
                st.metric("Desv. Estándar", f"{df[columna].std():.2f}")
            with col4:
                st.metric("Rango", f"{df[columna].max() - df[columna].min():.2f}")
            
            st.subheader("Estadísticas Detalladas")
            st.write(df[columna].describe())
    
    # Sección: Modelado
    elif opcion == "Modelado":
        st.header("🤖 Modelado de Datos")
        
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columnas_numericas) >= 2:
            st.subheader("Configuración del Modelo")
            
            target = st.selectbox("Variable objetivo (Y):", columnas_numericas)
            features = st.multiselect(
                "Variables predictoras (X):",
                [col for col in columnas_numericas if col != target]
            )
            
            if features:
                # Preparar datos
                X = df[features].dropna()
                y = df.loc[X.index, target]
                
                test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.5, 0.2)
                
                if st.button("Entrenar Modelo"):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    st.success("Datos preparados para el modelo")
                    st.write(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
                    st.write(f"Conjunto de prueba: {X_test.shape[0]} muestras")
        else:
            st.warning("Se necesitan al menos 2 columnas numéricas para modelado")

else:
    st.error("No se pudo cargar el archivo de datos. Asegúrate de que 'energia_renovable.csv' existe en el directorio.")

# Footer
st.markdown("---")
st.markdown("Desarrollado con Streamlit 🎈")