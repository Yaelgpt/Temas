import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config.constants import CARRERAS, SEMESTRES_INGRESO

def mostrar_herramientas_estadisticas(analytics):
    """Mostrar herramientas estadísticas avanzadas"""
    st.markdown('<div class="sub-header">📈 Herramientas Estadísticas Avanzadas</div>', unsafe_allow_html=True)
    

    datos_visual = analytics.obtener_datos_para_analisis_visual()
    

    st.subheader("📊 Análisis Visual - Diagrama de Dispersión e Histograma")
    
    if datos_visual.empty:
        st.warning("No hay datos numéricos disponibles para análisis visual")
    else:

        variables_disponibles = datos_visual.columns.tolist()
        st.info(f"**Variables disponibles para análisis:** {', '.join(variables_disponibles)}")
        

        col1, col2 = st.columns(2)
        
        with col1:
            variable_x = st.selectbox(
                "Variable Eje X:",
                options=variables_disponibles,
                index=0
            )
        
        with col2:

            opciones_y = [v for v in variables_disponibles if v != variable_x]
            variable_y = st.selectbox(
                "Variable Eje Y:",
                options=opciones_y,
                index=0 if len(opciones_y) > 0 else 0
            )
        

        if st.button("🔄 Generar Diagrama de Dispersión"):
            if len(opciones_y) > 0:
                fig_dispersion = crear_diagrama_dispersion(datos_visual, variable_x, variable_y)
                st.pyplot(fig_dispersion)
                
  
                correlacion = datos_visual[variable_x].corr(datos_visual[variable_y])
                st.info(f"**Coeficiente de correlación:** {correlacion:.3f}")
                

                if abs(correlacion) > 0.7:
                    fuerza = "fuerte"
                elif abs(correlacion) > 0.4:
                    fuerza = "moderada"
                else:
                    fuerza = "débil"
                    
                if correlacion > 0:
                    direccion = "positiva"
                else:
                    direccion = "negativa"
                    
                st.write(f"**Interpretación:** Correlación {fuerza} {direccion} entre {variable_x} y {variable_y}")
            else:
                st.warning("Se necesitan al menos 2 variables diferentes para el diagrama de dispersión")


        st.subheader("📈 Histograma de Distribución")
        
        col3, col4 = st.columns(2)
        
        with col3:
            variable_hist = st.selectbox(
                "Variable para histograma:",
                options=variables_disponibles,
                index=len(variables_disponibles)-1 if len(variables_disponibles) > 0 else 0
            )
        
        with col4:
            bins_hist = st.slider(
                "Número de intervalos:",
                min_value=5,
                max_value=30,
                value=15
            )
        
        if st.button("📊 Generar Histograma"):
            fig_histograma = crear_histograma(datos_visual, variable_hist, bins_hist)
            st.pyplot(fig_histograma)
            

            datos = datos_visual[variable_hist].dropna()
            st.write(f"**Estadísticas de {variable_hist}:**")
            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
            with col_stats1:
                st.metric("Media", f"{datos.mean():.2f}")
            with col_stats2:
                st.metric("Mediana", f"{datos.median():.2f}")
            with col_stats3:
                st.metric("Desviación", f"{datos.std():.2f}")
            with col_stats4:
                st.metric("Mín-Máx", f"{datos.min():.1f}-{datos.max():.1f}")

  
    st.subheader("🎯 Estratificación de Datos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        carreras_keys = list(CARRERAS.keys())
        carreras_nombres = list(CARRERAS.values())
        
        carrera_filtro = st.multiselect(
            "Filtrar por Carrera:",
            options=carreras_nombres,
            default=carreras_nombres[:3] if len(carreras_nombres) > 3 else carreras_nombres
        )
    
    with col2:
        semestre_filtro = st.multiselect(
            "Filtrar por Semestre:",
            options=SEMESTRES_INGRESO,
            default=SEMESTRES_INGRESO
        )
    
    with col3:
        rango_calificaciones = st.slider(
            "Rango de Calificaciones:",
            min_value=0.0, max_value=100.0, value=(0.0, 100.0)
        )
    

    carreras_filtro_ids = []
    for nombre_carrera in carrera_filtro:
        for carrera_id, nombre in CARRERAS.items():
            if nombre == nombre_carrera:
                carreras_filtro_ids.append(carrera_id)
                break
    
    estudiantes_filtrados = analytics.df_estudiantes[
        (analytics.df_estudiantes['carrera_id'].isin(carreras_filtro_ids)) &
        (analytics.df_estudiantes['ingreso_semestre'].isin(semestre_filtro))
    ]
    
    calificaciones_filtradas = analytics.df_calificaciones[
        (analytics.df_calificaciones['calificacion_final'] >= rango_calificaciones[0]) &
        (analytics.df_calificaciones['calificacion_final'] <= rango_calificaciones[1])
    ]
    

    st.subheader("Estadísticas Filtradas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Estudiantes Filtrados", len(estudiantes_filtrados))
    
    with col2:
        st.metric("Calificaciones Filtradas", len(calificaciones_filtradas))
    
    with col3:
        if len(calificaciones_filtradas) > 0:
            if 'reprobado' in calificaciones_filtradas.columns:
                tasa_rep_filtrada = calificaciones_filtradas['reprobado'].mean() * 100
            else:
                reprobados = (calificaciones_filtradas['calificacion_final'] < 70).sum()
                tasa_rep_filtrada = (reprobados / len(calificaciones_filtradas)) * 100
            st.metric("Tasa Reprobación Filtrada", f"{tasa_rep_filtrada:.1f}%")
        else:
            st.metric("Tasa Reprobación Filtrada", "0%")
    
    with col4:
        if len(calificaciones_filtradas) > 0:
            promedio_calif = calificaciones_filtradas['calificacion_final'].mean()
            st.metric("Promedio Calificación", f"{promedio_calif:.1f}")
        else:
            st.metric("Promedio Calificación", "0.0")


    st.subheader("🔗 Análisis de Correlaciones")
    
    if not datos_visual.empty:
        variables_disponibles_corr = datos_visual.columns.tolist()
        variables_correlacion = st.multiselect(
            "Selecciona variables para análisis de correlación:",
            variables_disponibles_corr,
            default=variables_disponibles_corr[:2] if len(variables_disponibles_corr) >= 2 else variables_disponibles_corr
        )
        
        if len(variables_correlacion) >= 2:
            fig_correlacion = analytics.generar_matriz_correlacion(variables_correlacion)
            st.pyplot(fig_correlacion)
            
            st.markdown("""
            **Interpretación de Correlación:**
            - **+1.0**: Correlación positiva perfecta
            - **+0.7 a +0.9**: Correlación positiva fuerte
            - **+0.4 a +0.6**: Correlación positiva moderada
            - **-0.3 a +0.3**: Correlación débil o nula
            - **-0.4 a -0.6**: Correlación negativa moderada
            - **-0.7 a -0.9**: Correlación negativa fuerte
            - **-1.0**: Correlación negativa perfecta
            """)
    else:
        st.info("No hay datos disponibles para análisis de correlación")


    st.subheader("🏫 Análisis por Carrera")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Distribución por Carrera**")
        fig_barras = analytics.generar_grafico_barras_carreras()
        st.pyplot(fig_barras)
    
    with col2:
        st.write("**Tasas por Carrera**")
        fig_tasas = analytics.generar_grafico_tasas_por_carrera()
        st.pyplot(fig_tasas)

def crear_diagrama_dispersion(df, variable_x, variable_y):
    """Crear diagrama de dispersión profesional"""
    fig, ax = plt.subplots(figsize=(10, 6))
    

    datos_validos = df[[variable_x, variable_y]].dropna()
    
    if len(datos_validos) == 0:
        st.warning("No hay datos válidos para crear el diagrama de dispersión")
        return fig
    

    scatter = ax.scatter(
        datos_validos[variable_x], 
        datos_validos[variable_y],
        alpha=0.6, 
        s=60,
        c='blue',
        edgecolors='white',
        linewidth=0.5
    )
    
 
    z = np.polyfit(datos_validos[variable_x], datos_validos[variable_y], 1)
    p = np.poly1d(z)
    ax.plot(datos_validos[variable_x], p(datos_validos[variable_x]), "r--", alpha=0.8, linewidth=2)
    
    ax.set_xlabel(variable_x.replace('_', ' ').title())
    ax.set_ylabel(variable_y.replace('_', ' ').title())
    ax.set_title(f'Diagrama de Dispersión: {variable_x} vs {variable_y}')
    ax.grid(True, alpha=0.3)
    

    correlacion = datos_validos[variable_x].corr(datos_validos[variable_y])
    ax.text(0.05, 0.95, f'Correlación: {correlacion:.3f}', 
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    return fig

def crear_histograma(df, variable, bins=15):
    """Crear histograma profesional"""
    fig, ax = plt.subplots(figsize=(10, 6))
    

    datos_validos = df[variable].dropna()
    
    if len(datos_validos) == 0:
        st.warning("No hay datos válidos para crear el histograma")
        return fig
    
  
    n, bins, patches = ax.hist(
        datos_validos, 
        bins=bins, 
        alpha=0.7, 
        color='skyblue', 
        edgecolor='black', 
        linewidth=0.5
    )
    
    media = datos_validos.mean()
    mediana = datos_validos.median()
    
    ax.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.2f}')
    ax.axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
    
    ax.set_xlabel(variable.replace('_', ' ').title())
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Distribución de {variable.replace("_", " ").title()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
