import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

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

# Función para cargar datos predeterminados
@st.cache_data
def cargar_datos_predeterminados():
    try:
        # Intenta cargar el archivo desde diferentes ubicaciones
        if os.path.exists("energia_renovable.csv"):
            return pd.read_csv("energia_renovable.csv")
        elif os.path.exists("tallerfundamentosDC/energia_renovable.csv"):
            return pd.read_csv("tallerfundamentosDC/energia_renovable.csv")
        else:
            return None
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        return None

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
            df = None
    else:
        st.info("👆 Por favor, sube un archivo CSV en el sidebar para comenzar el análisis")
else:
    df = cargar_datos_predeterminados()
    if df is not None:
        st.sidebar.success("✅ Usando archivo predeterminado")
    else:
        st.sidebar.warning("⚠️ No se encontró el archivo predeterminado")
        st.info("💡 Cambia a 'Cargar mi propio archivo' para subir tu dataset")

st.sidebar.markdown("---")

# Solo mostrar opciones si hay datos cargados
if df is not None:
    st.sidebar.header("Opciones de Análisis")
    opcion = st.sidebar.selectbox(
        "Selecciona una sección:",
        ["Exploración de Datos", "Visualización", "Análisis Estadístico", "Modelado"]
    )
else:
    opcion = None

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
        tipo_datos = pd.DataFrame({
            'Columna': df.columns,
            'Tipo': df.dtypes.values
        })
        st.dataframe(tipo_datos)
    
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
                if columna_seleccionada:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(df[columna_seleccionada].dropna(), bins=30, edgecolor='black', color='steelblue')
                    ax.set_xlabel(columna_seleccionada)
                    ax.set_ylabel("Frecuencia")
                    ax.set_title(f"Distribución de {columna_seleccionada}")
                    st.pyplot(fig)
                    plt.close()
            
            with col2:
                st.subheader("Boxplot")
                if columna_seleccionada:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.boxplot(df[columna_seleccionada].dropna())
                    ax.set_ylabel(columna_seleccionada)
                    ax.set_title(f"Boxplot de {columna_seleccionada}")
                    st.pyplot(fig)
                    plt.close()
            
            # Matriz de correlación
            if len(columnas_numericas) > 1:
                st.subheader("Matriz de Correlación")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df[columnas_numericas].corr(), annot=True, cmap='coolwarm', ax=ax, center=0)
                ax.set_title("Matriz de Correlación")
                st.pyplot(fig)
                plt.close()
        else:
            st.warning("No se encontraron columnas numéricas en el dataset")
    
    # Sección: Análisis Estadístico
    elif opcion == "Análisis Estadístico":
        st.header("📊 Análisis Estadístico")
        
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if columnas_numericas:
            columna = st.selectbox("Selecciona una variable para análisis:", columnas_numericas)
            
            if columna:
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
        else:
            st.warning("No se encontraron columnas numéricas en el dataset")
    
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
                    with st.spinner("Preparando datos..."):
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                        
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        st.success("✅ Datos preparados para el modelo")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Conjunto de Entrenamiento", f"{X_train.shape[0]} muestras")
                        with col2:
                            st.metric("Conjunto de Prueba", f"{X_test.shape[0]} muestras")
        else:
            st.warning("Se necesitan al menos 2 columnas numéricas para modelado")

# Footer
st.markdown("---")
st.markdown("💻 Desarrollado con Streamlit 🎈")