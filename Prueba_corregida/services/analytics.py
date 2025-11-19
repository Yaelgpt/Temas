import pandas as pd
import numpy as np
import streamlit as st

class AnalyticsService:
    def __init__(self, database_service):
        self.db = database_service
        self._datos_cache = None
    
    def cargar_datos(self):
        if self._datos_cache is None:
            self._datos_cache = self._cargar_datos_actualizados()
        return self._datos_cache
    
    def _cargar_datos_actualizados(self):
        return {
            "estudiantes": self.db.cargar_estudiantes(),
            "calificaciones": self.db.cargar_calificaciones(),
            "factores": self.db.cargar_factores(),
            "materias": self.db.cargar_materias(),
            "grupos": self.db.cargar_grupos()
        }
    
    def actualizar_datos(self):
        """
        Limpia todo cache y recarga instantáneamente.
        """
        try:
            st.cache_data.clear()
        except Exception:
            pass
        self._datos_cache = None
        self.db.limpiar_cache()
        # opcional precarga inmediata
        self._datos_cache = self._cargar_datos_actualizados()
    
    @property
    def df_estudiantes(self):
        datos = self.cargar_datos()
        return datos["estudiantes"]
    
    @property
    def df_calificaciones(self):
        datos = self.cargar_datos()
        return datos["calificaciones"]
    
    @property
    def df_factores(self):
        datos = self.cargar_datos()
        return datos["factores"]
    
    @property
    def df_materias(self):
        datos = self.cargar_datos()
        return datos["materias"]
    
    @property
    def df_grupos(self):
        datos = self.cargar_datos()
        return datos["grupos"]
    
    @st.cache_data(ttl=300)
    def calcular_metricas_principales(_self):
        try:
            datos = _self.cargar_datos()
            df_calificaciones = datos["calificaciones"]
            df_estudiantes = datos["estudiantes"]
            
            if df_calificaciones.empty:
                return {
                    "total_estudiantes": len(df_estudiantes),
                    "total_calificaciones": 0,
                    "tasa_aprobacion": 0.0,
                    "tasa_reprobacion": 0.0,
                    "tasa_desercion": float((df_estudiantes["desercion"].mean() * 100) if "desercion" in df_estudiantes.columns else 0)
                }
            
            total_estudiantes = len(df_estudiantes)
            total_calificaciones = len(df_calificaciones)
            
            if "reprobado" in df_calificaciones.columns:
                tasa_reprobacion = df_calificaciones["reprobado"].mean() * 100
                tasa_aprobacion = 100 - tasa_reprobacion
            else:
                if "calificacion_final" in df_calificaciones.columns:
                    reprobados = (df_calificaciones["calificacion_final"] < 70).sum()
                    tasa_reprobacion = (reprobados / len(df_calificaciones)) * 100
                    tasa_aprobacion = 100 - tasa_reprobacion
                else:
                    tasa_reprobacion = 0.0
                    tasa_aprobacion = 0.0
            
            if "desercion" in df_estudiantes.columns:
                tasa_desercion = df_estudiantes["desercion"].mean() * 100
            else:
                tasa_desercion = 0.0
            
            return {
                "total_estudiantes": total_estudiantes,
                "total_calificaciones": total_calificaciones,
                "tasa_aprobacion": round(tasa_aprobacion, 2),
                "tasa_reprobacion": round(tasa_reprobacion, 2),
                "tasa_desercion": round(tasa_desercion, 2)
            }
        except Exception as e:
            st.error(f"Error calculando métricas: {e}")
            return {
                "total_estudiantes": 0,
                "total_calificaciones": 0,
                "tasa_aprobacion": 0.0,
                "tasa_reprobacion": 0.0,
                "tasa_desercion": 0.0
            }
    
    @st.cache_data(ttl=300)
    def generar_analisis_rendimiento(_self):
        try:
            df_calificaciones = _self.df_calificaciones
            if df_calificaciones.empty:
                return pd.DataFrame()
            columnas_necesarias = ["calificacion_final", "asistencia", "u1", "u2", "u3"]
            columnas_existentes = [col for col in columnas_necesarias if col in df_calificaciones.columns]
            if not columnas_existentes:
                return pd.DataFrame()
            return df_calificaciones[columnas_existentes].describe()
        except Exception as e:
            st.error(f"Error en análisis de rendimiento: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=300)
    def analizar_factores_riesgo(_self):
        try:
            df_factores = _self.df_factores
            if df_factores.empty:
                return pd.DataFrame()
            if "categoria" in df_factores.columns and "gravedad" in df_factores.columns:
                return df_factores.groupby("categoria").agg({
                    "gravedad": ["count", "mean"]
                }).round(2)
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error analizando factores: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=300)
    def obtener_tendencia_calificaciones(_self):
        try:
            df_calificaciones = _self.df_calificaciones
            if df_calificaciones.empty:
                return pd.DataFrame()
            unidades = ["u1", "u2", "u3"]
            unidades_existentes = [u for u in unidades if u in df_calificaciones.columns]
            if not unidades_existentes:
                return pd.DataFrame()
            tendencia = {}
            for unidad in unidades_existentes:
                tendencia[unidad] = df_calificaciones[unidad].mean()
            return pd.DataFrame(list(tendencia.items()), columns=["Unidad", "Promedio"])
        except Exception as e:
            st.error(f"Error obteniendo tendencia: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=300)
    def obtener_estadisticas_avanzadas(_self):
        try:
            df_calificaciones = _self.df_calificaciones
            if df_calificaciones.empty:
                return {}
            stats = {}
            if "calificacion_final" in df_calificaciones.columns:
                calif_final = df_calificaciones["calificacion_final"]
                stats["calificacion_final"] = {
                    "media": calif_final.mean(),
                    "mediana": calif_final.median(),
                    "desviacion_estandar": calif_final.std(),
                    "minimo": calif_final.min(),
                    "maximo": calif_final.max()
                }
            if "asistencia" in df_calificaciones.columns:
                asistencia = df_calificaciones["asistencia"]
                stats["asistencia"] = {
                    "media": asistencia.mean(),
                    "mediana": asistencia.median(),
                    "desviacion_estandar": asistencia.std(),
                    "minimo": asistencia.min(),
                    "maximo": asistencia.max()
                }
            return stats
        except Exception as e:
            st.error(f"Error calculando estadísticas avanzadas: {e}")
            return {}
    
    @st.cache_data(ttl=300)
    def generar_matriz_correlacion(_self, variables):
        import matplotlib.pyplot as plt
        import seaborn as sns
        try:
            df_calificaciones = _self.df_calificaciones
            if df_calificaciones.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, "No hay datos disponibles", ha="center", va="center", transform=ax.transAxes, fontsize=14)
                return fig
            variables_existentes = [v for v in variables if v in df_calificaciones.columns]
            if len(variables_existentes) < 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, "Se necesitan al menos 2 variables", ha="center", va="center", transform=ax.transAxes, fontsize=14)
                return fig
            correlacion = df_calificaciones[variables_existentes].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlacion, annot=True, cmap="coolwarm", center=0, square=True, ax=ax, fmt=".2f")
            ax.set_title("Matriz de Correlación")
            return fig
        except Exception as e:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            return fig

    @st.cache_data(ttl=300)
    def generar_grafico_barras_carreras(_self):
        import matplotlib.pyplot as plt
        try:
            df_estudiantes = _self.df_estudiantes
            if df_estudiantes.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No hay datos de estudiantes", ha="center", va="center", transform=ax.transAxes, fontsize=14)
                return fig
            from config.constants import CARRERAS
            conteo_carreras = df_estudiantes["carrera_id"].value_counts()
            nombres_carreras = [CARRERAS.get(cid, f"Carrera {cid}") for cid in conteo_carreras.index]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(nombres_carreras, conteo_carreras.values, color="skyblue", edgecolor="black")
            ax.set_xlabel("Carreras")
            ax.set_ylabel("Número de Estudiantes")
            ax.set_title("Distribución de Estudiantes por Carrera")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            return fig
        except Exception as e:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            return fig

    @st.cache_data(ttl=300)
    def generar_grafico_tasas_por_carrera(_self):
        import matplotlib.pyplot as plt
        try:
            df_calificaciones = _self.df_calificaciones
            df_estudiantes = _self.df_estudiantes
            if df_calificaciones.empty or df_estudiantes.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, "No hay datos suficientes", ha="center", va="center", transform=ax.transAxes, fontsize=14)
                return fig
            from config.constants import CARRERAS
            datos_combinados = df_calificaciones.merge(
                df_estudiantes[["id", "carrera_id"]],
                left_on="estudiante_id",
                right_on="id",
                how="left"
            )
            tasas_por_carrera = []
            for carrera_id in datos_combinados["carrera_id"].unique():
                datos_carrera = datos_combinados[datos_combinados["carrera_id"] == carrera_id]
                if len(datos_carrera) > 0:
                    tasa_reprobacion = datos_carrera["reprobado"].mean() * 100 if "reprobado" in datos_carrera.columns else ((datos_carrera["calificacion_final"] < 70).mean() * 100)
                    nombre_carrera = CARRERAS.get(carrera_id, f"Carrera {carrera_id}")
                    tasas_por_carrera.append((nombre_carrera, tasa_reprobacion))
            if not tasas_por_carrera:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, "No hay datos para calcular tasas", ha="center", va="center", transform=ax.transAxes, fontsize=14)
                return fig
            tasas_por_carrera.sort(key=lambda x: x[1])
            carreras, tasas = zip(*tasas_por_carrera)
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(carreras, tasas, color="lightcoral", edgecolor="black")
            ax.set_xlabel("Carreras")
            ax.set_ylabel("Tasa de Reprobación (%)")
            ax.set_title("Tasa de Reprobación por Carrera")
            for bar, tasa in zip(bars, tasas):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{tasa:.1f}%", ha="center", va="bottom")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            return fig
        except Exception as e:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha="center", va="center", transform=ax.transAxes, fontsize=12)
            return fig

    @st.cache_data(ttl=300)
    def generar_grafico_pareto(_self):
        try:
            df_factores = _self.df_factores
            if df_factores.empty:
                return pd.DataFrame()
            if "categoria" in df_factores.columns and "gravedad" in df_factores.columns:
                pareto_data = df_factores.groupby("categoria").agg({
                    "gravedad": "count"
                }).rename(columns={"gravedad": "frecuencia"}).reset_index()
                pareto_data = pareto_data.sort_values("frecuencia", ascending=False)
                pareto_data["porcentaje_acumulado"] = (pareto_data["frecuencia"].cumsum() / pareto_data["frecuencia"].sum()) * 100
                return pareto_data
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error generando Pareto: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=300)
    def obtener_datos_para_analisis_visual(_self):
        try:
            df_calificaciones = _self.df_calificaciones
            if df_calificaciones.empty:
                return pd.DataFrame()
            columnas_numericas = df_calificaciones.select_dtypes(include=[np.number]).columns.tolist()
            columnas_prioritarias = ["calificacion_final", "asistencia", "u1", "u2", "u3"]
            columnas_finales = [col for col in columnas_prioritarias if col in columnas_numericas]
            if len(columnas_finales) == 0:
                columnas_finales = columnas_numericas[:5] if len(columnas_numericas) > 0 else []
            if len(columnas_finales) == 0:
                return pd.DataFrame()
            return df_calificaciones[columnas_finales].copy()
        except Exception as e:
            st.error(f"Error obteniendo datos para análisis visual: {e}")
            return pd.DataFrame()