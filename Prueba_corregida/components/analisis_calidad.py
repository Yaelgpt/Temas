import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config.constants import CARRERAS
try:
    from services.rbac import es_docente, usuario_id
except Exception:
    def es_docente(): return False
    def usuario_id(): return None

# ========================
# Vista completa para admin
# ========================

def mostrar_analisis_calidad(analytics):
    st.markdown('<div class="sub-header">🔍 Herramientas de Análisis de Calidad</div>', unsafe_allow_html=True)

    st.subheader("Selecciona una herramienta de calidad:")
    herramienta = st.radio(
        "Herramienta",
        ["Diagrama de Pareto", "Diagrama de Dispersión", "Histograma", "Gráfico de Control"],
        horizontal=True,
        key="admin_tool_radio"
    )

    if herramienta == "Diagrama de Pareto":
        mostrar_pareto(analytics)
    elif herramienta == "Diagrama de Dispersión":
        mostrar_diagrama_dispersion(analytics)
    elif herramienta == "Histograma":
        grafica_histograma_admin(analytics)
    elif herramienta == "Gráfico de Control":
        grafica_control_admin(analytics)

def mostrar_pareto(analytics):
    st.subheader("Diagrama de Pareto   Factores de Riesgo")
    try:
        pareto_data = analytics.generar_grafico_pareto()
        if pareto_data is None or pareto_data.empty:
            st.info("No hay factores de riesgo registrados.")
            return

        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.bar(pareto_data['categoria'], pareto_data['frecuencia'], edgecolor='black', alpha=0.7, label='Frecuencia')
        ax1.set_xlabel('Categorías de Factores de Riesgo')
        ax1.set_ylabel('Frecuencia')
        ax1.tick_params(axis='x', rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(pareto_data['categoria'], pareto_data['porcentaje_acumulado'], marker='o', linewidth=2, label='% Acumulado')
        ax2.set_ylabel('Porcentaje Acumulado (%)')
        ax2.set_ylim(0, 100)
        ax2.axhline(y=80, linestyle='--', alpha=0.7, label='80%')

        plt.title('Diagrama de Pareto   Factores de Riesgo')
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Análisis de Resultados")
        factores_80 = pareto_data[pareto_data['porcentaje_acumulado'] <= 80]
        if not factores_80.empty:
            st.success(f"Factores críticos: {len(factores_80)} categorías")
            for _, f in factores_80.iterrows():
                st.write(f"• {f['categoria']}: {f['frecuencia']} ocurrencias ({f['porcentaje_acumulado']:.1f}% acumulado)")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Factores", pareto_data['frecuencia'].sum())
        with col2:
            st.metric("Categorías Críticas", len(factores_80))
        with col3:
            st.metric("Categorías Total", len(pareto_data))

    except Exception as e:
        st.error(f"Error generando Pareto: {e}")

def mostrar_diagrama_dispersion(analytics):
    st.info("Diagrama de Dispersión: relación entre dos variables numéricas")
    try:
        datos_visual = analytics.obtener_datos_para_analisis_visual()
    except Exception:
        st.error("No fue posible obtener datos para el diagrama")
        return

    if datos_visual.empty:
        st.warning("No hay datos disponibles")
        return

    variables_disponibles = datos_visual.columns.tolist()
    if len(variables_disponibles) < 2:
        st.warning("Se necesitan al menos 2 variables")
        return

    col1, col2 = st.columns(2)
    with col1:
        variable_x = st.selectbox("Variable Eje X", options=variables_disponibles, index=0, key="disp_x")
    with col2:
        opciones_y = [v for v in variables_disponibles if v != variable_x]
        variable_y = st.selectbox("Variable Eje Y", options=opciones_y, index=0, key="disp_y")

    fig, ax = plt.subplots(figsize=(10, 6))
    datos_validos = datos_visual[[variable_x, variable_y]].dropna()

    if len(datos_validos) > 0:
        ax.scatter(datos_validos[variable_x], datos_validos[variable_y], alpha=0.7, s=80, edgecolors='white', linewidth=0.5)
        z = np.polyfit(datos_validos[variable_x], datos_validos[variable_y], 1)
        p = np.poly1d(z)
        ax.plot(datos_validos[variable_x], p(datos_validos[variable_x]), linestyle="--", alpha=0.8, linewidth=2, label="Línea de tendencia")

        corr = datos_validos[variable_x].corr(datos_validos[variable_y])
        ax.set_xlabel(variable_x.replace('_', ' ').title())
        ax.set_ylabel(variable_y.replace('_', ' ').title())
        ax.set_title(f'Dispersión: {variable_x} vs {variable_y}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        st.subheader("Análisis de Correlación")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Coeficiente", f"{corr:.3f}")
        with col2:
            fuerza = "Fuerte" if abs(corr) > 0.7 else "Moderada" if abs(corr) > 0.4 else "Débil"
            st.metric("Fuerza", fuerza)
        with col3:
            st.metric("Dirección", "Positiva" if corr > 0 else "Negativa")
    else:
        st.warning("No hay datos válidos para el diagrama")

def mostrar_histograma_calidad(analytics):
    st.subheader("Histograma")

    mats = analytics.df_materias.sort_values("nombre") if not analytics.df_materias.empty else pd.DataFrame()
    if mats.empty:
        st.info("No hay materias registradas todavía.")
        return

    opciones = {f"ID {int(m['id'])}: {m['nombre']}": int(m["id"]) for _, m in mats.iterrows()}
    etiqueta = st.selectbox("Materia", list(opciones.keys()))
    materia_id = opciones[etiqueta]

    df = analytics.df_calificaciones.copy()
    if df.empty:
        st.info("No hay calificaciones registradas.")
        return

    for col in ["u1", "u2", "u3", "asistencia", "calificacion_final"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    vals = df.loc[df["materia_id"] == materia_id, "calificacion_final"].dropna()

    if vals.empty:
        st.warning("No hay calificaciones finales para la materia seleccionada.")
        return

    n = len(vals)
    bins_defecto = max(3, min(12, int(np.sqrt(n))))
    bins = st.slider("Número de bins", 3, 25, bins_defecto)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals, bins=bins, range=(0, 100), density=False, edgecolor="white", linewidth=0.6,
            histtype="bar", rwidth=0.9, color="#4c78a8", alpha=0.95)
    ax.set_xlabel("Calificación final")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución - {etiqueta}")

    ax.axvline(70, linestyle="--", linewidth=1.5, label="Límite 70", color="#d62728")
    media = float(np.mean(vals))
    mediana = float(np.median(vals))
    ax.axvline(media, linestyle=":", linewidth=1.5, label=f"Media {media:.1f}", color="#2ca02c")
    ax.axvline(mediana, linestyle="-.", linewidth=1.5, label=f"Mediana {mediana:.1f}", color="#ff7f0e")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.2)

    st.pyplot(fig)

    if n < 6:
        st.caption("Hay pocas observaciones, interpreta el histograma con cautela.")

def mostrar_grafico_control(analytics):
    st.info("Gráfico de Control: estabilidad del proceso")
    try:
        datos_visual = analytics.obtener_datos_para_analisis_visual()
    except Exception:
        st.error("No fue posible obtener datos")
        return

    if datos_visual.empty:
        st.warning("No hay datos disponibles")
        return

    variables_disponibles = datos_visual.columns.tolist()
    variable = st.selectbox("Variable", options=variables_disponibles, index=len(variables_disponibles)-1, key="ctrl_var")
    datos_validos = datos_visual[variable].dropna()

    if len(datos_validos) > 0:
        media = datos_validos.mean()
        desviacion = datos_validos.std()
        lcs = media + 3 * desviacion
        lci = media - 3 * desviacion

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(datos_validos)), datos_validos.values, marker='o', linewidth=2, markersize=4, label=variable)
        ax.axhline(media, linestyle='-', linewidth=2, label=f'Media ({media:.2f})')
        ax.axhline(lcs, linestyle='--', linewidth=1.5, label=f'LCS ({lcs:.2f})')
        ax.axhline(lci, linestyle='--', linewidth=1.5, label=f'LCI ({lci:.2f})')

        ax.set_xlabel('Observaciones')
        ax.set_ylabel(variable.replace('_', ' ').title())
        ax.set_title(f'Gráfico de Control   {variable.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        puntos_fuera = datos_validos[np.logical_or(datos_validos > lcs, datos_validos < lci)]
        st.subheader("Análisis de Control")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Observaciones", len(datos_validos))
        with col2: st.metric("Fuera Control", len(puntos_fuera))
        with col3:
            porcentaje = (len(puntos_fuera) / len(datos_validos) * 100) if len(datos_validos) > 0 else 0
            st.metric("% Fuera", f"{porcentaje:.1f}%")
    else:
        st.warning("No hay datos válidos para el gráfico de control")

# =========================================
# Utilidades y sección por materia y grupo
# =========================================

def _selector_materia_y_grupo(analytics, key_prefix="ac_", preselect_materia_id=None):
    mats = analytics.db.obtener_materias()
    if not mats:
        st.info("No hay materias")
        return None, None, None

    if preselect_materia_id:
        materia_id = preselect_materia_id
    else:
        opciones = {m['nombre']: m['id'] for m in mats}
        mat_nombre = st.selectbox("Materia", list(opciones.keys()), key=f"{key_prefix}materia_sel")
        materia_id = opciones[mat_nombre]

    grupos = analytics.db.obtener_grupos(materia_id=materia_id)
    if not grupos:
        st.info("No hay grupos para la materia seleccionada")
        return materia_id, None, None

    grp_labels = [f"{g['periodo']}   {g['grupo']}" for g in grupos]
    sel = st.selectbox("Grupo", grp_labels, key=f"{key_prefix}grupo_sel")
    gid = grupos[grp_labels.index(sel)]
    return materia_id, gid['periodo'], gid['grupo']

def _nombre_alumno_para_tabla(row: pd.Series) -> str:
    partes = []
    for k in ["nombres", "apellido_paterno", "apellido_materno"]:
        v = row.get(k)
        if pd.notna(v) and str(v).strip():
            partes.append(str(v).strip())
    if partes:
        return " ".join(partes)
    # fallback si solo hay 'nombre'
    if pd.notna(row.get("nombre")) and str(row.get("nombre")).strip():
        return str(row.get("nombre")).strip()
    return f"ID {int(row.get('estudiante_id', row.get('id', 0)))}"

def analitica_histograma_y_control(analytics, key_prefix="ac_", preselect_materia_id=None):
    st.subheader("Análisis por materia")

    # --- selector de materia y grupo (reutiliza tu helper si ya lo tienes)
    mats = analytics.db.obtener_materias() or []
    if not mats:
        st.info("No hay materias")
        return
    opciones_mats = {m["nombre"]: m["id"] for m in mats}
    if preselect_materia_id and preselect_materia_id in [m["id"] for m in mats]:
        materia_id = preselect_materia_id
        mat_label = next(k for k, v in opciones_mats.items() if v == materia_id)
        st.selectbox("Materia", [mat_label], key=f"{key_prefix}mat_fixed", disabled=True)
    else:
        mat_label = st.selectbox("Materia", list(opciones_mats.keys()), key=f"{key_prefix}materia_sel")
        materia_id = opciones_mats[mat_label]

    grupos = analytics.db.obtener_grupos(materia_id=materia_id) or []
    if not grupos:
        st.info("No hay grupos para esta materia")
        return
    grp_labels = [f"{g['periodo']} - {g['grupo']}" for g in grupos]
    grp_sel = st.selectbox("Grupo", grp_labels, key=f"{key_prefix}grupo_sel")
    g = grupos[grp_labels.index(grp_sel)]
    periodo, grupo = g["periodo"], g["grupo"]

    # --- datos del grupo
    datos = analytics.db.obtener_calificaciones_por(materia_id, periodo, grupo)
    if not datos:
        st.info("Sin calificaciones para este grupo")
        return

    df = pd.DataFrame(datos)
    # métricas de reprobados si existe columna 'reprobado'
    if "reprobado" in df.columns:
        reprob = int(df["reprobado"].sum())
        total = len(df)
        pct = round(100 * reprob / max(total, 1), 1)
        st.metric("Reprobados", f"{reprob}/{total}", delta=f"{pct} %")

    # ========= Histograma (calificacion_final)
    if "calificacion_final" in df.columns:
        vals = pd.to_numeric(df["calificacion_final"], errors="coerce").dropna()
        if not vals.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(vals, bins=max(6, min(20, int(np.sqrt(len(vals))))), edgecolor="black")
            ax.axvline(70, linestyle="--", color="red", linewidth=2, label="Límite 70")
            ax.set_xlabel("Calificación final")
            ax.set_ylabel("Frecuencia")
            ax.set_title("Histograma de calificación final")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("No hay calificaciones finales numéricas para graficar.")

    # ========= Gráfico de control simple con U1/U2/U3 si existen
    cols_ctrl = [c for c in ["u1", "u2", "u3"] if c in df.columns]
    if cols_ctrl:
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        for c in cols_ctrl:
            y = pd.to_numeric(df[c], errors="coerce")
            ax2.plot(y.index, y, marker="o", label=c.upper())
        ax2.axhline(70, linestyle="--", color="gray", linewidth=1)
        ax2.set_ylabel("Calificación")
        ax2.set_title("Gráfico de control por unidades")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)

    # ========= TABLA de alumnos del grupo con calificaciones
    st.markdown("### Alumnos y calificaciones del grupo")

    # merge con estudiantes para nombre
    cols_est = ["id", "nombres", "apellido_paterno", "apellido_materno", "nombre"]
    dfe = analytics.df_estudiantes[cols_est].copy() if not analytics.df_estudiantes.empty else pd.DataFrame(columns=cols_est)

    tabla = df.merge(dfe, left_on="estudiante_id", right_on="id", how="left")
    tabla["Alumno"] = tabla.apply(_nombre_alumno_para_tabla, axis=1)

    cols = ["estudiante_id", "Alumno"]
    for c in ["u1", "u2", "u3", "asistencia", "calificacion_final", "reprobado"]:
        if c in tabla.columns:
            cols.append(c)

    out = tabla[cols].rename(columns={
        "estudiante_id": "ID",
        "asistencia": "Asistencia",
        "calificacion_final": "Final",
        "reprobado": "Reprobado"
    }).sort_values(by=("Final" if "Final" in cols else "ID"), ascending=False)

    st.dataframe(out, use_container_width=True, hide_index=True)

# =========================================
# Vista reducida para DOCENTE con tabla
# =========================================

def _nombre_completo(row):
    piezas = []
    if pd.notna(row.get('nombres')) and str(row.get('nombres')).strip():
        piezas.append(str(row.get('nombres')).strip())
        if pd.notna(row.get('apellido_paterno')) and str(row.get('apellido_paterno')).strip():
            piezas.append(str(row.get('apellido_paterno')).strip())
        if pd.notna(row.get('apellido_materno')) and str(row.get('apellido_materno')).strip():
            piezas.append(str(row.get('apellido_materno')).strip())
        return " ".join(piezas)
    if pd.notna(row.get('nombre')) and str(row.get('nombre')).strip():
        return str(row.get('nombre')).strip()
    return f"ID {row.get('id','')}"

def _selector_grupo_para_tabla(analytics, materia_id: int, permitir_todos: bool = True):
    grupos = analytics.db.obtener_grupos(materia_id=materia_id)
    if not grupos:
        return None, None, False
    labels = [f"{g['periodo']}   {g['grupo']}" for g in grupos]
    if permitir_todos:
        labels = ["Todos"] + labels
    sel = st.selectbox("Grupo", labels, key="doc_tbl_grupo")
    if sel == "Todos":
        return None, None, True
    idx = labels.index(sel) - (1 if permitir_todos else 0)
    g = grupos[idx]
    return g['periodo'], g['grupo'], False

def mostrar_analisis_calidad_docente(analytics):
    st.markdown('<div class="sub-header">Análisis de Calidad</div>', unsafe_allow_html=True)

    materias = analytics.db.obtener_materias() or []
    if not materias:
        st.info("No tienes materias asignadas.")
        return

    opciones = {m['nombre']: m['id'] for m in materias}
    mat_nombre = st.selectbox("Materia", list(opciones.keys()), key="doc_materia_sel")
    materia_id = opciones[mat_nombre]

    periodo_sel, grupo_sel, todos = _selector_grupo_para_tabla(analytics, materia_id, permitir_todos=True)

    dfc = analytics.df_calificaciones
    n_total_materia = 0
    n_en_grupo = 0
    if not dfc.empty and {'materia_id','estudiante_id'}.issubset(dfc.columns):
        n_total_materia = int(dfc[dfc['materia_id'] == materia_id]['estudiante_id'].nunique())
        if todos:
            n_en_grupo = n_total_materia
        elif periodo_sel and grupo_sel:
            n_en_grupo = int(
                dfc[
                    (dfc['materia_id'] == materia_id) &
                    (dfc['periodo'] == periodo_sel) &
                    (dfc['grupo'] == grupo_sel)
                ]['estudiante_id'].nunique()
            )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Estudiantes en la materia", n_total_materia)
    with col2:
        st.metric("Estudiantes en la selección", n_en_grupo)

    st.divider()
    st.subheader("Alumnos de la materia")

    if todos or (periodo_sel and grupo_sel):
        dfe = analytics.df_estudiantes

        if todos:
            registros = dfc[dfc['materia_id'] == materia_id].copy()
        else:
            registros = dfc[
                (dfc['materia_id'] == materia_id) &
                (dfc['periodo'] == periodo_sel) &
                (dfc['grupo'] == grupo_sel)
            ].copy()

        if not registros.empty:
            cols_est = ['id', 'nombres', 'apellido_paterno', 'apellido_materno', 'nombre']
            cols_est = [c for c in cols_est if c in dfe.columns]
            df_merge = registros.merge(
                dfe[cols_est],
                left_on='estudiante_id',
                right_on='id',
                how='left'
            )
            df_merge['alumno'] = df_merge.apply(_nombre_completo, axis=1)

            cols = ['estudiante_id', 'alumno']
            if todos:
                if 'periodo' in df_merge.columns: cols.append('periodo')
                if 'grupo' in df_merge.columns: cols.append('grupo')
            for c in ['u1', 'u2', 'u3', 'asistencia', 'calificacion_final', 'reprobado']:
                if c in df_merge.columns:
                    cols.append(c)

            tabla = df_merge[cols].sort_values(
                by='calificacion_final' if 'calificacion_final' in df_merge.columns else cols[0],
                ascending=False,
                na_position='last'
            )

            if 'calificacion_final' in tabla.columns and tabla['calificacion_final'].notna().any():
                idx_max = tabla['calificacion_final'].idxmax()
                idx_min = tabla['calificacion_final'].idxmin()
                mejor = tabla.loc[idx_max]
                peor = tabla.loc[idx_min]
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Mejor calificación", f"{float(mejor['calificacion_final']):.2f}",
                              help=f"{mejor['alumno']}  id {int(mejor['estudiante_id'])}")
                with c2:
                    st.metric("Menor calificación", f"{float(peor['calificacion_final']):.2f}",
                              help=f"{peor['alumno']}  id {int(peor['estudiante_id'])}")

            renames = {
                'estudiante_id': 'ID',
                'alumno': 'Alumno',
                'calificacion_final': 'Calificación final',
                'asistencia': 'Asistencia',
                'reprobado': 'Reprobado'
            }
            if 'periodo' in tabla.columns: renames['periodo'] = 'Periodo'
            if 'grupo' in tabla.columns: renames['grupo'] = 'Grupo'

            st.dataframe(
                tabla.rename(columns=renames),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No hay registros para la selección actual.")

    st.divider()
    st.subheader("Análisis por materia y grupo")
    if todos:
        st.info("Selecciona un grupo específico para ver las gráficas por grupo.")
    else:
        analitica_histograma_y_control(analytics, key_prefix="doc_ac_", preselect_materia_id=materia_id)

def _select_materia(df_mats: pd.DataFrame, key: str):
    opciones = {"Todas": None}
    if not df_mats.empty:
        for _, m in df_mats.iterrows():
            opciones[f"ID {int(m['id'])}: {m['nombre']}"] = int(m["id"])
    elegido = st.selectbox("Materia", list(opciones.keys()), key=key)
    return opciones[elegido]

# ===== Metricas

def grafica_control_admin(analytics):
    df = analytics.df_calificaciones
    if df.empty:
        st.info("No hay calificaciones para mostrar")
        return

    st.subheader("Gráfico de Control: estabilidad del proceso")

    materia_id = _select_materia_admin(analytics, key="ctl_materia_filtro")
    datos = df.copy()
    if materia_id is not None:
        datos = datos[datos["materia_id"] == materia_id]

    if datos.empty:
        st.info("No hay datos con el filtro actual")
        return

    variables_disponibles = [c for c in ["u1", "u2", "u3", "asistencia", "calificacion_final"] if c in datos.columns]
    if not variables_disponibles:
        st.info("No existen columnas u1, u2, u3, asistencia o calificacion_final")
        return

    var = st.selectbox("Variable", variables_disponibles, key="ctl_var")

    y = pd.to_numeric(datos[var], errors="coerce").dropna().reset_index(drop=True)
    if len(y) < 2:
        st.info("Se requieren al menos 2 observaciones para el gráfico de control")
        return

    media = y.mean()
    sigma = y.std(ddof=1)
    ucl = media + 3 * sigma
    lcl = media - 3 * sigma

    fig, ax = plt.subplots(figsize=(12, 6))
    idx = np.arange(len(y))
    ax.plot(idx, y, marker="o")
    ax.axhline(media, color="blue", linestyle="--", linewidth=1.5, label="Media")
    ax.axhline(ucl, color="orange", linestyle="--", linewidth=1.5, label="UCL")
    ax.axhline(lcl, color="orange", linestyle="--", linewidth=1.5, label="LCL")

    reprobados = None
    if var in ["u1", "u2", "u3", "calificacion_final"]:
        malos = y < 70
        if malos.any():
            ax.scatter(idx[malos], y[malos], s=80, color="red", edgecolors="white", zorder=3, label="Reprobados")
        reprobados = int(malos.sum())

    ax.set_xlabel("Observaciones")
    ax.set_ylabel(var)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    st.pyplot(fig)

    col1, col2, col3 = st.columns(3)
    col1.metric("Observaciones", len(y))
    col2.metric("Media", f"{media:.2f}")
    col3.metric("Sigma", f"{sigma:.2f}")

    if reprobados is not None:
        p = 0 if len(y) == 0 else 100.0 * reprobados / len(y)
        c1, c2 = st.columns(2)
        c1.metric("Reprobados", reprobados)
        c2.metric("% Reprobados", f"{p:.1f}%")

def _select_materia_admin(analytics, key):
    mats = analytics.df_materias.copy()
    opciones = {"Todas": None}
    if not mats.empty:
        for _, m in mats.iterrows():
            opciones[f"ID {int(m['id'])}: {m['nombre']}"] = int(m["id"])
    elegido = st.selectbox("Materia", list(opciones.keys()), key=key)
    return opciones[elegido]

def grafica_histograma_admin(analytics):
    """
    Histograma con barras sólidas y sin esas líneas negras.
    Incluye filtro de materia, slider de bins y líneas de referencia.
    """
    df = analytics.df_calificaciones
    if df.empty or "calificacion_final" not in df.columns:
        st.info("No hay calificaciones para mostrar")
        return

    st.subheader("Histograma")
    materia_id = _select_materia_admin(analytics, key="hist_materia_filtro")

    datos = df.copy()
    if materia_id is not None:
        datos = datos[datos["materia_id"] == materia_id]

    if datos.empty:
        st.info("No hay datos con el filtro actual")
        return

    vals = (
        pd.to_numeric(datos["calificacion_final"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .clip(0, 100)
        .to_numpy()
    )
    if vals.size == 0:
        st.info("No hay calificaciones válidas para graficar")
        return

    bins_default = max(6, min(20, int(np.sqrt(vals.size))))
    bins = st.slider("Número de bins", 3, 30, bins_default, key="hist_bins_slider")

    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=140)
    ax.hist(
        vals,
        bins=bins,
        range=(0, 100),
        density=False,
        histtype="bar",
        rwidth=0.9,
        color="#4c78a8",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.95,
    )

    ax.axvline(70, color="#d62728", linestyle="--", linewidth=1.6, label="Límite 70")
    media = float(np.mean(vals))
    mediana = float(np.median(vals))
    ax.axvline(media, color="#2ca02c", linestyle=":", linewidth=1.6, label=f"Media {media:.1f}")
    ax.axvline(mediana, color="#ff7f0e", linestyle="-.", linewidth=1.6, label=f"Mediana {mediana:.1f}")

    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 5))
    ax.set_xlabel("Calificación final")
    ax.set_ylabel("Frecuencia")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left")

    st.pyplot(fig)
# -----------------------------------------------------------
# Helper: obtener materias visibles para el usuario actual
# prioriza catalogo por docente si existe
# -----------------------------------------------------------
def _materias_para_usuario(analytics):
    # 1) si existe un metodo especifico por docente, usarlo
    try:
        mats = analytics.db.obtener_materias_docente(usuario_id()) or []
        if mats:
            return mats
    except Exception:
        pass

    # 2) intentar catalogos amplios y filtrar si es docente
    mats = []
    for fn in ("obtener_materias_admin", "obtener_materias"):
        try:
            m = getattr(analytics.db, fn)() or []
            if m:
                mats = m
                break
        except Exception:
            continue

    if es_docente() and mats:
        uid = str(usuario_id())
        # filtrar por docente_user_id si existe esa columna
        try:
            mats = [m for m in mats if str(m.get("docente_user_id")) == uid]
        except Exception:
            # si no existe esa llave, dejamos la lista como esta
            pass
    return mats

# -----------------------------------------------------------
# Selector con filtro de carrera -> materia -> grupo
# -----------------------------------------------------------
def _selector_materia_y_grupo(analytics, key_prefix="ac_", preselect_materia_id=None):
    mats = _materias_para_usuario(analytics) or []
    if not mats:
        st.info("No hay materias para seleccionar")
        return None, None, None

    # 1) carrera disponible segun materias visibles
    carreras_ids = sorted({int(m.get("carrera_id")) for m in mats if m.get("carrera_id") is not None})
    etiquetas_carr = ["Todas"] + [f"ID {cid}: {CARRERAS.get(cid, f'Carrera {cid}')}" for cid in carreras_ids]

    carr_sel = st.selectbox("Carrera", etiquetas_carr, key=f"{key_prefix}carr_sel")
    carrera_id_elegida = None
    if carr_sel != "Todas":
        # parsear el id que viene en "ID X: nombre"
        try:
            carrera_id_elegida = int(carr_sel.split(":")[0].split()[-1])
        except Exception:
            carrera_id_elegida = None

    # 2) filtrar materias por carrera si se eligio una
    mats_filtradas = mats
    if carrera_id_elegida is not None:
        mats_filtradas = [m for m in mats if int(m.get("carrera_id", -1)) == carrera_id_elegida]

    if not mats_filtradas:
        st.info("No hay materias en la carrera seleccionada")
        return None, None, None

    # 3) seleccionar materia
    if preselect_materia_id:
        materia_id = preselect_materia_id
    else:
        opciones = {f"ID {m['id']}: {m['nombre']}": m['id'] for m in mats_filtradas}
        mat_nombre = st.selectbox("Materia", list(opciones.keys()), key=f"{key_prefix}materia_sel")
        materia_id = opciones[mat_nombre]

    # 4) seleccionar grupo
    grupos = analytics.db.obtener_grupos(materia_id=materia_id) or []
    if not grupos:
        st.info("No hay grupos para la materia seleccionada")
        return materia_id, None, None

    grp_labels = [f"{g['periodo']} - {g['grupo']}" for g in grupos]
    sel = st.selectbox("Grupo", grp_labels, key=f"{key_prefix}grupo_sel")
    gid = grupos[grp_labels.index(sel)]
    return materia_id, gid['periodo'], gid['grupo']


# -----------------------------------------------------------
# Usa el selector con carrera dentro del analisis por materia
# -----------------------------------------------------------
def analitica_histograma_y_control(analytics, key_prefix="ac_", preselect_materia_id=None):
    st.subheader("Análisis por materia")

    # usa el selector con Carrera -> Materia -> Grupo
    materia_id, periodo, grupo = _selector_materia_y_grupo(
        analytics,
        key_prefix=key_prefix,
        preselect_materia_id=preselect_materia_id
    )
    if not (materia_id and periodo and grupo):
        return

    # datos del grupo
    datos = analytics.db.obtener_calificaciones_por(materia_id, periodo, grupo)
    if not datos:
        st.info("Sin calificaciones")
        return

    df = pd.DataFrame(datos)

    # métrica de reprobados si existe
    if "reprobado" in df.columns:
        reprob = int(df["reprobado"].sum())
        total = int(len(df))
        pct = round(100 * reprob / max(total, 1), 1)
        st.metric("Reprobados", f"{reprob}/{total}", delta=f"{pct} %")

    # Histograma
    st.markdown("**Histograma de calificación final**")
    if "calificacion_final" in df.columns:
        vals = pd.to_numeric(df["calificacion_final"], errors="coerce").dropna()
        if not vals.empty:
            bins = max(6, min(20, int(np.sqrt(len(vals)))))
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(vals, bins=bins, edgecolor="black", alpha=0.9)
            ax.axvline(70, linestyle="--", color="red", linewidth=2, label="Límite 70")
            ax.set_xlabel("Calificación")
            ax.set_ylabel("Frecuencia")
            ax.set_title("Histograma de calificación final")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("No hay calificaciones finales numéricas para graficar.")

    # Gráfico de control por unidades
    st.markdown("**Gráfico de control por unidades**")
    cols_ctrl = [c for c in ["u1", "u2", "u3"] if c in df.columns]
    if cols_ctrl:
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        for c in cols_ctrl:
            y = pd.to_numeric(df[c], errors="coerce")
            ax2.plot(y.index, y, marker="o", label=c.upper())
        ax2.axhline(70, linestyle="--", color="gray", linewidth=1)
        ax2.set_ylabel("Calificación")
        ax2.set_title("Gráfico de control por unidades")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)

    # Tabla de alumnos y calificaciones
    st.markdown("### Alumnos y calificaciones del grupo")

    cols_est = ["id", "nombres", "apellido_paterno", "apellido_materno", "nombre"]
    dfe = analytics.df_estudiantes[cols_est].copy() if not analytics.df_estudiantes.empty else pd.DataFrame(columns=cols_est)

    tabla = df.merge(dfe, left_on="estudiante_id", right_on="id", how="left")

    def _nombre_alumno_para_tabla_row(r):
        piezas = []
        for k in ("nombres", "apellido_paterno", "apellido_materno"):
            if pd.notna(r.get(k)) and str(r.get(k)).strip():
                piezas.append(str(r.get(k)).strip())
        if piezas:
            return " ".join(piezas)
        if pd.notna(r.get("nombre")) and str(r.get("nombre")).strip():
            return str(r.get("nombre")).strip()
        return f"ID {int(r.get('estudiante_id', r.get('id', 0)))}"

    tabla["Alumno"] = tabla.apply(_nombre_alumno_para_tabla_row, axis=1)

    columnas_salida = ["estudiante_id", "Alumno"]
    for c in ["u1", "u2", "u3", "asistencia", "calificacion_final", "reprobado"]:
        if c in tabla.columns:
            columnas_salida.append(c)

    out = tabla[columnas_salida].rename(columns={
        "estudiante_id": "ID",
        "asistencia": "Asistencia",
        "calificacion_final": "Final",
        "reprobado": "Reprobado"
    })

    ordenar_por = "Final" if "Final" in out.columns else "ID"
    out = out.sort_values(by=ordenar_por, ascending=False, na_position="last")

    st.dataframe(out, use_container_width=True, hide_index=True)
