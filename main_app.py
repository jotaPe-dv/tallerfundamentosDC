import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Datos EDA",
    page_icon="📊",
    layout="wide"
)

# Estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Título principal
st.title("📊 Análisis Exploratorio de Datos (EDA)")
st.markdown("### Aplicación multimodal para análisis de cualquier dataset")
st.markdown("---")

# Sidebar
st.sidebar.header("📁 Carga de Datos")

# Función para cargar datos predeterminados
@st.cache_data
def cargar_datos_predeterminados():
    try:
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
    ["Cargar mi propio archivo", "Usar archivo predeterminado"]
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
            st.sidebar.success(f"✅ Archivo cargado: {uploaded_file.name}")
            st.sidebar.info(f"📊 {df.shape[0]} filas × {df.shape[1]} columnas")
        except Exception as e:
            st.sidebar.error(f"❌ Error al cargar el archivo: {e}")
            df = None
    else:
        st.info("👆 Por favor, sube un archivo CSV en el sidebar para comenzar el análisis")
else:
    df = cargar_datos_predeterminados()
    if df is not None:
        st.sidebar.success("✅ Usando archivo predeterminado")
        st.sidebar.info(f"📊 {df.shape[0]} filas × {df.shape[1]} columnas")
    else:
        st.sidebar.warning("⚠️ No se encontró el archivo predeterminado")
        st.info("💡 Cambia a 'Cargar mi propio archivo' para subir tu dataset")

st.sidebar.markdown("---")

# Solo mostrar opciones si hay datos cargados
if df is not None:
    st.sidebar.header("⚙️ Opciones de Análisis")
    opcion = st.sidebar.selectbox(
        "Selecciona una sección:",
        ["📋 Resumen General", "🔍 Exploración Detallada", "📈 Visualizaciones Avanzadas", 
         "🔗 Análisis de Relaciones", "📊 Distribuciones", "🤖 Preparación para Modelado"]
    )
    
    # Identificar tipos de columnas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    columnas_datetime = df.select_dtypes(include=['datetime64']).columns.tolist()
else:
    opcion = None

if df is not None:
    
    # ============= RESUMEN GENERAL =============
    if opcion == "📋 Resumen General":
        st.header("📋 Resumen General del Dataset")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Filas", f"{df.shape[0]:,}")
        with col2:
            st.metric("📋 Total Columnas", f"{df.shape[1]}")
        with col3:
            st.metric("❌ Valores Nulos", f"{df.isnull().sum().sum():,}")
        with col4:
            memoria = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("💾 Tamaño", f"{memoria:.2f} MB")
        
        st.markdown("---")
        
        # Información de columnas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Tipos de Columnas")
            tipo_info = pd.DataFrame({
                'Tipo': ['Numéricas', 'Categóricas', 'Fechas'],
                'Cantidad': [len(columnas_numericas), len(columnas_categoricas), len(columnas_datetime)]
            })
            fig = px.pie(tipo_info, values='Cantidad', names='Tipo', 
                        title='Distribución de Tipos de Columnas',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Calidad de Datos")
            nulos_por_col = df.isnull().sum()
            if nulos_por_col.sum() > 0:
                nulos_df = pd.DataFrame({
                    'Columna': nulos_por_col[nulos_por_col > 0].index,
                    'Nulos': nulos_por_col[nulos_por_col > 0].values,
                    'Porcentaje': (nulos_por_col[nulos_por_col > 0] / len(df) * 100).values
                }).sort_values('Nulos', ascending=False)
                st.dataframe(nulos_df, use_container_width=True)
            else:
                st.success("✅ ¡No hay valores nulos en el dataset!")
        
        st.markdown("---")
        st.subheader("👀 Vista Previa de los Datos")
        num_filas = st.slider("Número de filas a mostrar:", 5, 50, 10)
        st.dataframe(df.head(num_filas), use_container_width=True)
        
        st.markdown("---")
        st.subheader("📑 Información Detallada de Columnas")
        info_cols = pd.DataFrame({
            'Columna': df.columns,
            'Tipo': df.dtypes.astype(str),
            'No Nulos': df.count(),
            'Nulos': df.isnull().sum(),
            '% Nulos': (df.isnull().sum() / len(df) * 100).round(2),
            'Únicos': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(info_cols, use_container_width=True)
    
    # ============= EXPLORACIÓN DETALLADA =============
    elif opcion == "🔍 Exploración Detallada":
        st.header("🔍 Exploración Detallada")
        
        tabs = st.tabs(["🔢 Variables Numéricas", "📝 Variables Categóricas", "🔎 Valores Únicos"])
        
        with tabs[0]:
            if columnas_numericas:
                st.subheader("📊 Estadísticas Descriptivas - Variables Numéricas")
                st.dataframe(df[columnas_numericas].describe().T, use_container_width=True)
                
                st.markdown("---")
                st.subheader("📉 Análisis por Variable")
                col_seleccionada = st.selectbox("Selecciona una variable numérica:", columnas_numericas)
                
                if col_seleccionada:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Media", f"{df[col_seleccionada].mean():.2f}")
                    with col2:
                        st.metric("Mediana", f"{df[col_seleccionada].median():.2f}")
                    with col3:
                        st.metric("Desv. Est.", f"{df[col_seleccionada].std():.2f}")
                    with col4:
                        st.metric("Mínimo", f"{df[col_seleccionada].min():.2f}")
                    with col5:
                        st.metric("Máximo", f"{df[col_seleccionada].max():.2f}")
            else:
                st.warning("No hay variables numéricas en el dataset")
        
        with tabs[1]:
            if columnas_categoricas:
                st.subheader("📝 Variables Categóricas")
                col_cat = st.selectbox("Selecciona una variable categórica:", columnas_categoricas)
                
                if col_cat:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Valores Únicos", df[col_cat].nunique())
                        st.metric("Valor Más Frecuente", df[col_cat].mode()[0] if len(df[col_cat].mode()) > 0 else "N/A")
                    
                    with col2:
                        frecuencias = df[col_cat].value_counts().head(10)
                        st.write("**Top 10 Valores Más Frecuentes:**")
                        st.dataframe(frecuencias, use_container_width=True)
            else:
                st.info("No hay variables categóricas en el dataset")
        
        with tabs[2]:
            st.subheader("🔎 Análisis de Valores Únicos")
            unicos_df = pd.DataFrame({
                'Columna': df.columns,
                'Valores Únicos': [df[col].nunique() for col in df.columns],
                'Porcentaje': [round(df[col].nunique() / len(df) * 100, 2) for col in df.columns]
            }).sort_values('Valores Únicos', ascending=False)
            
            fig = px.bar(unicos_df, x='Columna', y='Valores Únicos',
                        title='Cantidad de Valores Únicos por Columna',
                        color='Porcentaje',
                        color_continuous_scale='Viridis')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(unicos_df, use_container_width=True)
    
    # ============= VISUALIZACIONES AVANZADAS =============
    elif opcion == "📈 Visualizaciones Avanzadas":
        st.header("📈 Visualizaciones Avanzadas")
        
        if columnas_numericas:
            tipo_viz = st.radio("Selecciona el tipo de visualización:",
                               ["Histogramas", "Boxplots", "Violin Plots", "Gráficos de Densidad"])
            
            if tipo_viz == "Histogramas":
                st.subheader("📊 Distribución - Histogramas")
                num_cols = min(len(columnas_numericas), 4)
                cols_seleccionadas = st.multiselect(
                    "Selecciona variables (máx. 4):",
                    columnas_numericas,
                    default=columnas_numericas[:num_cols]
                )
                
                if cols_seleccionadas:
                    n_cols = min(len(cols_seleccionadas), 2)
                    n_rows = (len(cols_seleccionadas) + 1) // 2
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
                    if n_rows == 1 and n_cols == 1:
                        axes = [axes]
                    else:
                        axes = axes.flatten() if len(cols_seleccionadas) > 1 else [axes]
                    
                    for idx, col in enumerate(cols_seleccionadas):
                        axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', color='steelblue', alpha=0.7)
                        axes[idx].set_xlabel(col)
                        axes[idx].set_ylabel('Frecuencia')
                        axes[idx].set_title(f'Distribución de {col}')
                        axes[idx].grid(True, alpha=0.3)
                    
                    for idx in range(len(cols_seleccionadas), len(axes)):
                        fig.delaxes(axes[idx])
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            
            elif tipo_viz == "Boxplots":
                st.subheader("📦 Detección de Outliers - Boxplots")
                cols_seleccionadas = st.multiselect(
                    "Selecciona variables:",
                    columnas_numericas,
                    default=columnas_numericas[:4]
                )
                
                if cols_seleccionadas:
                    fig = go.Figure()
                    for col in cols_seleccionadas:
                        fig.add_trace(go.Box(y=df[col].dropna(), name=col))
                    
                    fig.update_layout(
                        title="Boxplots de Variables Seleccionadas",
                        yaxis_title="Valores",
                        showlegend=True,
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif tipo_viz == "Violin Plots":
                st.subheader("🎻 Violin Plots")
                col_violin = st.selectbox("Selecciona una variable:", columnas_numericas)
                
                if col_violin:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.violinplot(y=df[col_violin].dropna(), ax=ax, color='lightblue')
                    ax.set_ylabel(col_violin)
                    ax.set_title(f'Violin Plot de {col_violin}')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
            
            elif tipo_viz == "Gráficos de Densidad":
                st.subheader("📈 Gráficos de Densidad (KDE)")
                cols_seleccionadas = st.multiselect(
                    "Selecciona variables:",
                    columnas_numericas,
                    default=columnas_numericas[:3]
                )
                
                if cols_seleccionadas:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    for col in cols_seleccionadas:
                        df[col].dropna().plot(kind='kde', ax=ax, label=col, linewidth=2)
                    ax.set_xlabel('Valor')
                    ax.set_ylabel('Densidad')
                    ax.set_title('Gráficos de Densidad')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
        else:
            st.warning("No hay variables numéricas para visualizar")
    
    # ============= ANÁLISIS DE RELACIONES =============
    elif opcion == "🔗 Análisis de Relaciones":
        st.header("🔗 Análisis de Relaciones entre Variables")
        
        if len(columnas_numericas) >= 2:
            tabs = st.tabs(["🔄 Matriz de Correlación", "📊 Scatter Plots", "📈 Pairplot"])
            
            with tabs[0]:
                st.subheader("🔄 Matriz de Correlación")
                metodo_corr = st.selectbox("Método de correlación:", ["pearson", "spearman", "kendall"])
                
                corr_matrix = df[columnas_numericas].corr(method=metodo_corr)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                               fmt='.2f', square=True, linewidths=1, ax=ax,
                               cbar_kws={"shrink": 0.8})
                    ax.set_title(f'Matriz de Correlación ({metodo_corr.title()})')
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.write("**Correlaciones Más Fuertes:**")
                    corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_pairs.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlación': corr_matrix.iloc[i, j]
                            })
                    corr_df = pd.DataFrame(corr_pairs).sort_values('Correlación', 
                                                                   key=abs, 
                                                                   ascending=False).head(10)
                    st.dataframe(corr_df, use_container_width=True)
            
            with tabs[1]:
                st.subheader("📊 Gráficos de Dispersión (Scatter Plots)")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_var = st.selectbox("Variable X:", columnas_numericas, key='scatter_x')
                with col2:
                    y_var = st.selectbox("Variable Y:", 
                                        [c for c in columnas_numericas if c != x_var],
                                        key='scatter_y')
                
                if columnas_categoricas:
                    color_var = st.selectbox("Color por categoría (opcional):", 
                                           ['Ninguno'] + columnas_categoricas)
                else:
                    color_var = 'Ninguno'
                
                if x_var and y_var:
                    if color_var != 'Ninguno':
                        fig = px.scatter(df, x=x_var, y=y_var, color=color_var,
                                       title=f'{y_var} vs {x_var}',
                                       trendline="ols")
                    else:
                        fig = px.scatter(df, x=x_var, y=y_var,
                                       title=f'{y_var} vs {x_var}',
                                       trendline="ols")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calcular correlación
                    corr = df[[x_var, y_var]].corr().iloc[0, 1]
                    st.info(f"📊 Correlación de Pearson: **{corr:.3f}**")
            
            with tabs[2]:
                st.subheader("📈 Pairplot (Análisis Multivariado)")
                st.info("⚠️ El Pairplot puede tardar con muchas variables. Selecciona máximo 5.")
                
                vars_pairplot = st.multiselect(
                    "Selecciona variables (máx. 5):",
                    columnas_numericas,
                    default=columnas_numericas[:min(3, len(columnas_numericas))]
                )
                
                if len(vars_pairplot) > 5:
                    st.warning("Por favor selecciona máximo 5 variables")
                elif len(vars_pairplot) >= 2:
                    if st.button("Generar Pairplot"):
                        with st.spinner("Generando gráfico..."):
                            fig = sns.pairplot(df[vars_pairplot].dropna(), 
                                             diag_kind='kde',
                                             plot_kws={'alpha': 0.6})
                            st.pyplot(fig)
                            plt.close()
        else:
            st.warning("Se necesitan al menos 2 variables numéricas para análisis de relaciones")
    
    # ============= DISTRIBUCIONES =============
    elif opcion == "📊 Distribuciones":
        st.header("📊 Análisis de Distribuciones")
        
        if columnas_numericas:
            st.subheader("📉 Comparación de Distribuciones")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Histogramas Comparativos**")
                vars_comp = st.multiselect("Selecciona variables:", 
                                          columnas_numericas,
                                          default=columnas_numericas[:2])
                
                if vars_comp:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for var in vars_comp:
                        ax.hist(df[var].dropna(), bins=30, alpha=0.5, label=var, edgecolor='black')
                    ax.set_xlabel('Valor')
                    ax.set_ylabel('Frecuencia')
                    ax.set_title('Comparación de Distribuciones')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
            
            with col2:
                st.write("**Q-Q Plot (Normalidad)**")
                var_qq = st.selectbox("Variable para Q-Q Plot:", columnas_numericas)
                
                if var_qq:
                    from scipy import stats
                    fig, ax = plt.subplots(figsize=(10, 6))
                    stats.probplot(df[var_qq].dropna(), dist="norm", plot=ax)
                    ax.set_title(f'Q-Q Plot de {var_qq}')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
            
            # Gráficos para categóricas
            if columnas_categoricas:
                st.markdown("---")
                st.subheader("📊 Distribuciones de Variables Categóricas")
                
                var_cat = st.selectbox("Selecciona variable categórica:", columnas_categoricas)
                
                if var_cat:
                    top_n = st.slider("Mostrar top N categorías:", 5, 20, 10)
                    frecuencias = df[var_cat].value_counts().head(top_n)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(x=frecuencias.index, y=frecuencias.values,
                                   labels={'x': var_cat, 'y': 'Frecuencia'},
                                   title=f'Top {top_n} Categorías de {var_cat}')
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.pie(values=frecuencias.values, names=frecuencias.index,
                                   title=f'Distribución de {var_cat}')
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay variables numéricas en el dataset")
    
    # ============= PREPARACIÓN PARA MODELADO =============
    elif opcion == "🤖 Preparación para Modelado":
        st.header("🤖 Preparación para Modelado")
        
        if len(columnas_numericas) >= 2:
            tabs = st.tabs(["⚙️ Configuración", "🔍 PCA", "📊 Análisis de Componentes"])
            
            with tabs[0]:
                st.subheader("⚙️ Configuración del Modelo")
                
                target = st.selectbox("Variable objetivo (Y):", columnas_numericas)
                features = st.multiselect(
                    "Variables predictoras (X):",
                    [col for col in columnas_numericas if col != target],
                    default=[col for col in columnas_numericas if col != target][:3]
                )
                
                if features and target:
                    test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.5, 0.2, 0.05)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        escalar = st.checkbox("Escalar datos", value=True)
                    with col2:
                        random_state = st.number_input("Random State:", 0, 100, 42)
                    
                    if st.button("Preparar Datos"):
                        with st.spinner("Preparando datos..."):
                            # Eliminar filas con valores nulos
                            df_clean = df[features + [target]].dropna()
                            
                            X = df_clean[features]
                            y = df_clean[target]
                            
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=random_state
                            )
                            
                            if escalar:
                                scaler = StandardScaler()
                                X_train_scaled = scaler.fit_transform(X_train)
                                X_test_scaled = scaler.transform(X_test)
                            
                            st.success("✅ Datos preparados exitosamente")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Muestras", len(df_clean))
                            with col2:
                                st.metric("Entrenamiento", len(X_train))
                            with col3:
                                st.metric("Prueba", len(X_test))
                            
                            st.write("**Primeras filas de X_train:**")
                            st.dataframe(pd.DataFrame(X_train).head(), use_container_width=True)
            
            with tabs[1]:
                st.subheader("🔍 Análisis de Componentes Principales (PCA)")
                
                if len(columnas_numericas) >= 2:
                    n_components = st.slider("Número de componentes:", 2, 
                                            min(len(columnas_numericas), 10), 2)
                    
                    if st.button("Aplicar PCA"):
                        with st.spinner("Aplicando PCA..."):
                            # Preparar datos
                            df_pca = df[columnas_numericas].dropna()
                            
                            # Escalar
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(df_pca)
                            
                            # PCA
                            pca = PCA(n_components=n_components)
                            X_pca = pca.fit_transform(X_scaled)
                            
                            # Varianza explicada
                            var_exp = pca.explained_variance_ratio_
                            var_exp_cum = np.cumsum(var_exp)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.bar(range(1, n_components+1), var_exp, alpha=0.7, label='Individual')
                                ax.plot(range(1, n_components+1), var_exp_cum, 'ro-', label='Acumulada')
                                ax.set_xlabel('Componente Principal')
                                ax.set_ylabel('Varianza Explicada')
                                ax.set_title('Varianza Explicada por Componente')
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                plt.close()
                            
                            with col2:
                                st.write("**Varianza Explicada:**")
                                var_df = pd.DataFrame({
                                    'Componente': [f'PC{i+1}' for i in range(n_components)],
                                    'Varianza (%)': (var_exp * 100).round(2),
                                    'Acumulada (%)': (var_exp_cum * 100).round(2)
                                })
                                st.dataframe(var_df, use_container_width=True)
                            
                            if n_components >= 2:
                                st.subheader("Visualización de Componentes Principales")
                                pca_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
                                fig = px.scatter(pca_df, x='PC1', y='PC2',
                                               title='Proyección en PC1 y PC2')
                                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[2]:
                st.subheader("📊 Análisis de Importancia de Features")
                st.info("Análisis de correlación con variable objetivo")
                
                if columnas_numericas:
                    target_imp = st.selectbox("Variable objetivo:", columnas_numericas, key='imp_target')
                    
                    if target_imp:
                        correlaciones = df[columnas_numericas].corr()[target_imp].drop(target_imp).sort_values(key=abs, ascending=False)
                        
                        fig = px.bar(x=correlaciones.values, y=correlaciones.index,
                                   orientation='h',
                                   labels={'x': 'Correlación', 'y': 'Variable'},
                                   title=f'Correlación de Variables con {target_imp}',
                                   color=correlaciones.values,
                                   color_continuous_scale='RdBu_r')
                        fig.update_layout(height=max(400, len(correlaciones) * 25))
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Se necesitan al menos 2 variables numéricas para preparación de modelado")

# Footer
st.markdown("---")
st.markdown("### 💻 Desarrollado con Streamlit 🎈")
st.markdown("*Aplicación multimodal para análisis exploratorio de datos*")