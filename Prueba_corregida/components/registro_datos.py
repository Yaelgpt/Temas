# components/registro_datos.py
import re
import unicodedata
import time
import streamlit as st
import pandas as pd
import numpy as np
from config.constants import CARRERAS, CATEGORIAS_FACTORES, SEMESTRES_INGRESO
from services.analytics import AnalyticsService

# RBAC
try:
    from services.rbac import es_docente, usuario_id
except Exception:
    def es_docente() -> bool:
        return False
    def usuario_id():
        return None

# ================= Helpers =================

def _norm_txt(x: str) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s).lower()
    return s

def _to_int(x, default=None):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default

def _to_float(x, default=None):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def _to_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("1", "si", "sí", "true", "t", "y", "yes")

def _student_key_row(row) -> tuple:
    return (
        _norm_txt(row.get("nombres", "")),
        _norm_txt(row.get("apellido_paterno", "")),
        _norm_txt(row.get("apellido_materno", "")),
        _to_int(row.get("carrera_id"), 0),
        str(row.get("ingreso_semestre", "")).strip(),
    )

def _ensure_columns(df: pd.DataFrame, required: list) -> list:
    if df is None or df.empty:
        return required[:]
    faltan = [c for c in required if c not in df.columns]
    return faltan

def _siguiente_matricula(analytics, prefijo="A"):
    """
    Sugerencia simple: A001, A002, ...
    Busca el numero mas alto al final de las matriculas existentes y suma 1.
    """
    try:
        df = analytics.df_estudiantes
        if df.empty or "matricula" not in df.columns:
            return f"{prefijo}001"

        nums = []
        for v in df["matricula"].dropna().astype(str).tolist():
            m = re.search(r"(\d+)$", v)
            if m:
                nums.append(int(m.group(1)))
        n = (max(nums) if nums else 0) + 1
        return f"{prefijo}{n:03d}"
    except Exception:
        return f"{prefijo}001"


def _siguiente_matricula_lote(usados: set[str], prefijo="A"):
    """
    Igual que la anterior, pero para usar durante importacion en lote,
    evitando chocar con valores ya usados en BD y en el propio lote.
    """
    maxn = 0
    for v in list(usados):
        m = re.search(r"(\d+)$", str(v))
        if m:
            maxn = max(maxn, int(m.group(1)))
    n = maxn + 1
    while True:
        cand = f"{prefijo}{n:03d}"
        if cand not in usados:
            usados.add(cand)
            return cand
        n += 1

# ================= Contenedor tabs =================

def mostrar_registro_datos(database_service):
    st.markdown('<div class="sub-header">📝 Registro de Datos</div>', unsafe_allow_html=True)
    analytics = AnalyticsService(database_service)

    base_tabs = [
        "📤 Importar desde Excel",
        "👥 Registrar Estudiante",
        "📚 Registrar Materias",
        "⚠️ Registrar Factores",
        "🧾 Inscribir Alumnos",
        "👩‍🏫 Asignar Docentes",
    ]
    if es_docente():
        base_tabs.insert(3, "📊 Registrar Calificaciones")

    tabs = st.tabs(base_tabs)
    idx = {t: i for i, t in enumerate(base_tabs)}

    with tabs[idx["📤 Importar desde Excel"]]:
        try:
            mostrar_importar_excel(analytics)
        except Exception as e:
            st.exception(e)

    with tabs[idx["👥 Registrar Estudiante"]]:
        try:
            mostrar_registro_estudiante(analytics)
        except Exception as e:
            st.exception(e)

    with tabs[idx["📚 Registrar Materias"]]:
        try:
            mostrar_registro_materias(analytics)
        except Exception as e:
            st.exception(e)

    if es_docente():
        with tabs[idx["📊 Registrar Calificaciones"]]:
            try:
                mostrar_registro_calificaciones(analytics)
            except Exception as e:
                st.exception(e)

    with tabs[idx["⚠️ Registrar Factores"]]:
        try:
            mostrar_registro_factores(analytics)
        except Exception as e:
            st.exception(e)

    with tabs[idx["🧾 Inscribir Alumnos"]]:
        try:
            mostrar_inscribir_alumnos(analytics)
        except Exception as e:
            st.exception(e)

    with tabs[idx["👩‍🏫 Asignar Docentes"]]:
        try:
            mostrar_asignar_docentes(analytics)
        except Exception as e:
            st.exception(e)

# ================= Importar Excel =================
def guardar_excel_validado(analytics, est_valid: pd.DataFrame, cal_valid: pd.DataFrame, fac_valid: pd.DataFrame):
    try:
        # ===== Estudiantes nuevos =====
        if not est_valid.empty:
            # Asegurar columna 'matricula'
            if "matricula" not in est_valid.columns:
                est_valid["matricula"] = ""

            # Conjunto de matriculas ya usadas en BD para no chocar
            try:
                usados = set(
                    analytics.df_estudiantes.get("matricula", pd.Series(dtype=str)).dropna().astype(str).tolist()
                )
            except Exception:
                usados = set()

            # Completar vacios con sugerencias
            for i, r in est_valid.iterrows():
                v = str(r.get("matricula", "")).strip()
                if not v:
                    est_valid.at[i, "matricula"] = _siguiente_matricula_lote(usados)

            # Insercion
            for _, r in est_valid.iterrows():
                analytics.db.insertar_estudiante({
                    "matricula": str(r["matricula"]).strip(),
                    "nombres": r["nombres"],
                    "apellido_paterno": r["apellido_paterno"],
                    "apellido_materno": r["apellido_materno"],
                    "carrera_id": int(r["carrera_id"]),
                    "ingreso_semestre": str(r["ingreso_semestre"]),
                    "horas_estudio": int(r["horas_estudio"]),
                    "desercion": bool(r["desercion"])
                })

        # ===== Calificaciones con upsert =====
        if not cal_valid.empty:
            df_rows = cal_valid.copy()

            # 1) Intento de resolver estudiante_id por matrícula si existe esa columna
            if "estudiante_id" in df_rows.columns and "matricula" in df_rows.columns:
                try:
                    df_e = analytics.df_estudiantes[["id", "matricula"]].copy()
                    df_e["matricula"] = df_e["matricula"].astype(str).str.strip().str.lower()
                    id_map = dict(zip(df_e["matricula"], df_e["id"]))

                    mask_noid = df_rows["estudiante_id"].isna()
                    df_rows.loc[mask_noid, "estudiante_id"] = (
                        df_rows.loc[mask_noid, "matricula"]
                        .astype(str).str.strip().str.lower()
                        .map(id_map)
                    )
                except Exception:
                    pass  # si falla, solo seguimos sin resolver

            guardadas = 0
            omitidas_sin_id = 0
            fallidas = []

            for _, r in df_rows.iterrows():
                sid = _to_int(r.get("estudiante_id"), None)
                mid = _to_int(r.get("materia_id"), None)
                per = str(r.get("periodo", "")).strip()
                gru = str(r.get("grupo", "")).strip()
                u1v = _to_float(r.get("u1"), 0.0)
                u2v = _to_float(r.get("u2"), 0.0)
                u3v = _to_float(r.get("u3"), 0.0)
                asis = _to_float(r.get("asistencia"), 0.0)
                fin  = _to_float(r.get("calificacion_final"), None)

                # Si no hay estudiante_id, NO intentamos guardar esa fila
                if sid is None:
                    omitidas_sin_id += 1
                    continue

                try:
                    analytics.db.insertar_calificacion({
                        "estudiante_id": int(sid),
                        "materia_id": int(mid) if mid is not None else None,
                        "periodo": per,
                        "grupo": gru,
                        "u1": float(u1v),
                        "u2": float(u2v),
                        "u3": float(u3v),
                        "asistencia": float(asis),
                        "calificacion_final": float(fin) if fin is not None else None
                    })
                    guardadas += 1
                except Exception as e:
                    fallidas.append({
                        "estudiante_id": sid,
                        "materia_id": mid,
                        "periodo": per,
                        "grupo": gru,
                        "error": str(e)
                    })

            # Mensajes de resultado
            if guardadas:
                st.success(f"Calificaciones guardadas: {guardadas}")
            if omitidas_sin_id:
                st.warning(f"Omitidas sin estudiante_id: {omitidas_sin_id}. "
                           f"Si agregas la columna 'matricula' en la hoja Calificaciones, "
                           f"podré resolver el ID automáticamente.")
            if fallidas:
                st.error("Algunas filas no pudieron guardarse")
                try:
                    st.dataframe(pd.DataFrame(fallidas), use_container_width=True, hide_index=True)
                except Exception:
                    st.write(fallidas)

        # ===== Factores =====
        if not fac_valid.empty:
            for _, r in fac_valid.iterrows():
                analytics.db.insertar_factor({
                    "categoria": r["categoria"],
                    "nombre": r["nombre"],
                    "inscripcion_id": _to_int(r["inscripcion_id"], None),
                    "gravedad": int(r["gravedad"])
                })

        analytics.actualizar_datos()
        st.success("Importación completada")
        st.rerun()

    except Exception as e:
        st.error(f"Error al guardar: {e}")

def mostrar_importar_excel(analytics):
    st.subheader("Importar Datos desde Excel")
    st.caption("El archivo puede incluir hojas Estudiantes, Calificaciones y Factores")

    archivo_excel = st.file_uploader(
        "Selecciona archivo Excel",
        type=["xlsx", "xls"],
        help="Se validan columnas, tipos y duplicados antes de guardar"
    )
    if not archivo_excel:
        return

    def _safe_read(sheet):
        try:
            return pd.read_excel(archivo_excel, sheet_name=sheet).rename(
                columns=lambda c: str(c).strip()
            )
        except Exception:
            return pd.DataFrame()

    df_est = _safe_read("Estudiantes")
    df_cal = _safe_read("Calificaciones")
    df_fac = _safe_read("Factores")

    COLS_EST = ["nombres", "apellido_paterno", "apellido_materno",
                "carrera_id", "ingreso_semestre", "horas_estudio", "desercion"]
    COLS_CAL = ["estudiante_id", "materia_id", "periodo", "grupo",
                "u1", "u2", "u3", "asistencia", "calificacion_final"]
    COLS_FAC = ["categoria", "nombre", "inscripcion_id", "gravedad"]

    # Estudiantes
    errores_est, dup_est_bd, dup_est_xlsx = [], 0, 0
    est_valid = pd.DataFrame()
    if not df_est.empty:
        faltan = _ensure_columns(df_est, COLS_EST)
        if faltan:
            st.error(f"Hoja Estudiantes: faltan columnas {faltan}")
        else:
            tmp = df_est.copy()
            tmp["nombres"] = tmp["nombres"].fillna("").astype(str).str.strip()
            tmp["apellido_paterno"] = tmp["apellido_paterno"].fillna("").astype(str).str.strip()
            tmp["apellido_materno"] = tmp["apellido_materno"].fillna("").astype(str).str.strip()
            tmp["carrera_id"] = tmp["carrera_id"].apply(lambda x: _to_int(x, None))
            tmp["ingreso_semestre"] = tmp["ingreso_semestre"].astype(str).str.strip()
            tmp["horas_estudio"] = tmp["horas_estudio"].apply(lambda x: _to_int(x, 0))
            tmp["desercion"] = tmp["desercion"].apply(_to_bool)

            mask_obli = np.logical_or(tmp["nombres"] == "", tmp["apellido_paterno"] == "")
            mask_carrera = tmp["carrera_id"].isna()
            mask_invalid = np.logical_or(mask_obli, mask_carrera)
            if mask_invalid.any():
                errores_est += tmp[mask_invalid].assign(_motivo="Campos obligatorios vacíos").to_dict("records")
                tmp = tmp[~mask_invalid].copy()

            tmp["__k__"] = tmp.apply(_student_key_row, axis=1)
            dup_est_xlsx = int(tmp.duplicated("__k__").sum())
            tmp = tmp.drop_duplicates("__k__", keep="first")

            exist = analytics.df_estudiantes.copy()
            if not exist.empty:
                exist["__k__"] = exist.apply(_student_key_row, axis=1)
                ya = set(exist["__k__"].tolist())
                tmp = tmp[~tmp["__k__"].isin(ya)].copy()
                dup_est_bd = int(len(df_est) - len(tmp) - dup_est_xlsx - len(errores_est))

            est_valid = tmp.drop(columns="__k__", errors="ignore")

            # Calificaciones
        errores_cal, dup_cal_xlsx = [], 0
        cal_valid = pd.DataFrame()
        if not df_cal.empty:
            # columnas mínimas siempre presentes
            COLS_CAL_MIN = ["materia_id", "periodo", "grupo", "u1", "u2", "u3", "asistencia", "calificacion_final"]
            faltan_min = _ensure_columns(df_cal, COLS_CAL_MIN)
            if faltan_min:
                st.error(f"Hoja Calificaciones: faltan columnas {faltan_min}")
            else:
                tmp = df_cal.copy()

                # normalización básica
                for col in ["materia_id"]:
                    tmp[col] = tmp[col].apply(lambda x: _to_int(x, None))
                tmp["periodo"] = tmp["periodo"].fillna("").astype(str).str.strip()
                tmp["grupo"]   = tmp["grupo"].fillna("").astype(str).str.strip().str.upper()
                for col in ["u1","u2","u3","asistencia","calificacion_final"]:
                    tmp[col] = tmp[col].apply(lambda x: _to_float(x, 0.0))

                # estudiante_id puede venir o no
                if "estudiante_id" not in tmp.columns:
                    tmp["estudiante_id"] = None
                else:
                    tmp["estudiante_id"] = tmp["estudiante_id"].apply(lambda x: _to_int(x, None))

                # si viene 'matricula', usarla para completar estudiante_id faltantes
                if "matricula" in tmp.columns:
                    cat = analytics.df_estudiantes[["id","matricula"]].copy()
                    cat["matricula"] = cat["matricula"].astype(str).str.strip().str.upper()
                    map_id = dict(zip(cat["matricula"], cat["id"]))

                    mask_na = tmp["estudiante_id"].isna() & tmp["matricula"].notna()
                    tmp.loc[mask_na, "estudiante_id"] = (
                        tmp.loc[mask_na, "matricula"]
                        .astype(str).str.strip().str.upper()
                        .map(map_id)
                    )

                # invalidación: falta de claves o id no resuelto
                m_id  = tmp["materia_id"].isna()
                m_per = tmp["periodo"] == ""
                m_grp = tmp["grupo"]   == ""
                m_std = tmp["estudiante_id"].isna()
                mask_invalid = np.logical_or.reduce([m_id, m_per, m_grp, m_std])
                if mask_invalid.any():
                    errores_cal += tmp[mask_invalid].assign(_motivo="Faltan claves o no se resolvió el estudiante").to_dict("records")
                    tmp = tmp[~mask_invalid].copy()

                # deduplicación por clave natural
                tmp["__k__"] = list(zip(tmp["estudiante_id"], tmp["materia_id"], tmp["periodo"], tmp["grupo"]))
                dup_cal_xlsx = int(tmp.duplicated("__k__").sum())
                tmp = tmp.drop_duplicates("__k__", keep="first")

                # preview amigable con nombre del alumno y nombre de materia
                try:
                    dm = analytics.df_materias[["id","nombre"]].rename(columns={"id":"materia_id","nombre":"Materia"})
                    de = analytics.df_estudiantes[["id","nombres","apellido_paterno","apellido_materno"]].rename(columns={"id":"estudiante_id"})
                    de["Alumno"] = (de["nombres"].fillna("") + " " + de["apellido_paterno"].fillna("") + " " + de["apellido_materno"].fillna("")).str.replace(r"\s+"," ", regex=True).str.strip()
                    prev = tmp.merge(de[["estudiante_id","Alumno"]], on="estudiante_id", how="left").merge(dm, on="materia_id", how="left")                                                             
                except Exception:
                    pass

                cal_valid = tmp.drop(columns="__k__", errors="ignore")


    # Factores
    errores_fac, dup_fac_xlsx = [], 0
    fac_valid = pd.DataFrame()
    if not df_fac.empty:
        faltan = _ensure_columns(df_fac, COLS_FAC)
        if faltan:
            st.error(f"Hoja Factores: faltan columnas {faltan}")
        else:
            tmp = df_fac.copy()
            tmp["categoria"] = tmp["categoria"].astype(str).str.strip()
            tmp["nombre"] = tmp["nombre"].astype(str).str.strip()
            tmp["inscripcion_id"] = tmp["inscripcion_id"].apply(lambda x: _to_int(x, None))
            tmp["gravedad"] = tmp["gravedad"].apply(lambda x: _to_int(x, None))

            m1 = tmp["categoria"] == ""
            m2 = ~tmp["categoria"].isin(CATEGORIAS_FACTORES)
            m3 = tmp["gravedad"].isna()
            m4 = tmp["gravedad"] < 1
            m5 = tmp["gravedad"] > 5
            mask_invalid = np.logical_or.reduce([m1, m2, m3, m4, m5])

            if mask_invalid.any():
                errores_fac += tmp[mask_invalid].assign(_motivo="Categoría inválida o gravedad fuera de 1..5").to_dict("records")
                tmp = tmp[~mask_invalid].copy()

            tmp["__k__"] = list(zip(tmp["categoria"], tmp["nombre"], tmp["inscripcion_id"]))
            dup_fac_xlsx = int(tmp.duplicated("__k__").sum())
            tmp = tmp.drop_duplicates("__k__", keep="first")
            fac_valid = tmp.drop(columns="__k__", errors="ignore")

    st.markdown("### Resumen de validación")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Estudiantes válidos", len(est_valid))
        st.caption(f"Duplicados en Excel: {dup_est_xlsx}. Duplicados vs BD: {dup_est_bd}. Inválidos: {len(errores_est)}")
    with cols[1]:
        st.metric("Calificaciones válidas", len(cal_valid))
        st.caption(f"Duplicados en Excel: {dup_cal_xlsx}. Inválidos: {len(errores_cal)}")
    with cols[2]:
        st.metric("Factores válidos", len(fac_valid))
        st.caption(f"Duplicados en Excel: {dup_fac_xlsx}. Inválidos: {len(errores_fac)}")

    with st.expander("Ver errores de Estudiantes", expanded=False):
        if errores_est:
            st.dataframe(pd.DataFrame(errores_est), use_container_width=True)
        else:
            st.write("Sin errores.")

    with st.expander("Ver errores de Calificaciones", expanded=False):
        if errores_cal:
            st.dataframe(pd.DataFrame(errores_cal), use_container_width=True)
        else:
            st.write("Sin errores.")

    with st.expander("Ver errores de Factores", expanded=False):
        if errores_fac:
            st.dataframe(pd.DataFrame(errores_fac), use_container_width=True)
        else:
            st.write("Sin errores.")

    # === Previsualización: ¿qué calificaciones quedaron válidas? ===
    if not cal_valid.empty:
            st.markdown("#### Calificaciones validadas (previa)")

            # Mapas de ids a nombres
            try:
                dm = analytics.df_materias[["id", "nombre"]].copy()
                dm["id"] = pd.to_numeric(dm["id"], errors="coerce").astype("Int64")
                map_mat = dict(zip(dm["id"].astype(int), dm["nombre"]))
            except Exception:
                map_mat = {}

            try:
                de = analytics.df_estudiantes[["id","nombres","apellido_paterno","apellido_materno","nombre"]].copy()
                de["id"] = pd.to_numeric(de["id"], errors="coerce").astype("Int64")
                def _full(r):
                    partes = [
                        str(r.get("nombres","")).strip(),
                        str(r.get("apellido_paterno","")).strip(),
                        str(r.get("apellido_materno","")).strip()
                    ]
                    nom = " ".join([p for p in partes if p])
                    return nom or str(r.get("nombre","")).strip()
                de["alumno"] = de.apply(_full, axis=1)
                map_est = dict(zip(de["id"].astype(int), de["alumno"]))
            except Exception:
                map_est = {}

            prev = cal_valid.copy()
            for c in ["estudiante_id","materia_id"]:
                if c in prev.columns:
                    prev[c] = pd.to_numeric(prev[c], errors="coerce").astype("Int64")

            prev["Alumno"]  = prev.get("estudiante_id").map(map_est).fillna(prev.get("estudiante_id").astype(str))
            prev["Materia"] = prev.get("materia_id").map(map_mat).fillna(prev.get("materia_id").astype(str))

            cols = [c for c in [
                "estudiante_id","Alumno","materia_id","Materia",
                "periodo","grupo","u1","u2","u3","asistencia","calificacion_final"
            ] if c in prev.columns]

            preview = prev[cols].sort_values(
                by=[c for c in ["materia_id","periodo","grupo","estudiante_id"] if c in prev.columns],
                na_position="last"
            )

            st.dataframe(preview, use_container_width=True, hide_index=True)

    st.markdown("---")
    if st.button("💾 Guardar válidos en la base de datos", type="primary"):
        guardar_excel_validado(analytics, est_valid, cal_valid, fac_valid)

# ================= Registrar Estudiante =================

def mostrar_registro_estudiante(analytics):
    st.subheader("Registrar Nuevo Estudiante")

    # Sugerimos una matricula por defecto
    matricula_sugerida = _siguiente_matricula(analytics)

    with st.form("form_estudiante"):
        col1, col2 = st.columns(2)
        with col1:
            # NUEVO: Matricula obligatoria
            matricula = st.text_input("Matrícula*", value=matricula_sugerida, placeholder="A001")
            nombres = st.text_input("Nombres*", placeholder="Juan")
            ap = st.text_input("Apellido paterno*", placeholder="García")
            am = st.text_input("Apellido materno", placeholder="López")
            ingreso_semestre = st.selectbox("Semestre de ingreso*", SEMESTRES_INGRESO, key="est_sem")
        with col2:
            carrera_id = st.selectbox(
                "Carrera*",
                options=list(CARRERAS.keys()),
                format_func=lambda x: f"{CARRERAS[x]}",
                key="est_carr"
            )
            horas_estudio = st.number_input("Horas de estudio semanales*", min_value=0, max_value=80, value=20)
            desercion = st.checkbox("Estudiante en riesgo de deserción")
        ok = st.form_submit_button("💾 Registrar Estudiante")

    if not ok:
        return

    # Validaciones basicas
    if not matricula or not nombres or not ap:
        st.error("Matrícula, Nombres y Apellido paterno son obligatorios")
        return

    # Unicidad de matricula
    try:
        df_e = analytics.df_estudiantes
        if not df_e.empty and "matricula" in df_e.columns:
            if df_e["matricula"].astype(str).str.lower().eq(matricula.strip().lower()).any():
                st.error("La matrícula ya existe, usa otra diferente")
                return
    except Exception:
        pass

    data = {
        "matricula": str(matricula).strip(),
        "nombres": nombres.strip(),
        "apellido_paterno": ap.strip(),
        "apellido_materno": am.strip(),
        "carrera_id": int(carrera_id),
        "ingreso_semestre": str(ingreso_semestre),
        "horas_estudio": int(horas_estudio),
        "desercion": bool(desercion)
    }
    res = analytics.db.insertar_estudiante(data)
    if res:
        analytics.actualizar_datos()
        st.success("Estudiante registrado")
        st.rerun()
    else:
        st.error("No se pudo registrar")

# ================= Registrar Materias =================

def mostrar_registro_materias(analytics):
    st.subheader("Registrar Nueva Materia")

    try:
        docentes = analytics.db.listar_docentes() or []
    except Exception:
        docentes = []

    opciones_doc = {"Sin asignar": None}
    for d in docentes:
        nombre_visible = ""
        if d.get("nombre") or d.get("apellidos"):
            nombre_visible = f"{str(d.get('nombre','')).strip()} {str(d.get('apellidos','')).strip()}".strip()
        if not nombre_visible:
            nombre_visible = str(d.get("usuario", f"doc_{d.get('id','')}")).strip()
        opciones_doc[f"{nombre_visible}  (ID {d.get('id')})"] = d.get("id")

    with st.form("form_materia"):
        col1, col2 = st.columns(2)
        with col1:
            nombre_materia = st.text_input("Nombre de la materia*", placeholder="Calidad en el Software")
            semestre = st.number_input("Semestre*", min_value=1, max_value=12, value=1, key="mat_sem")
        with col2:
            carrera_id = st.selectbox(
                "Carrera*",
                options=list(CARRERAS.keys()),
                format_func=lambda x: f"{CARRERAS[x]}",
                key="mat_carr"
            )
            docente_sel = st.selectbox(
                "Docente",
                options=list(opciones_doc.keys()),
                index=0,
                key="mat_doc_sel"
            )

        st.markdown("#### Grupo inicial")
        c1, c2 = st.columns(2)
        with c1:
            g_periodo = st.text_input("Periodo", placeholder="2025-1", help="Ejemplo: 2025-1")
        with c2:
            g_grupo = st.text_input("Grupo", placeholder="A", help="Ejemplo: A")

        ok = st.form_submit_button("📚 Registrar Materia")

    if not ok:
        return

    if not nombre_materia:
        st.error("El nombre de la materia es obligatorio")
        return

    docente_user_id = opciones_doc.get(docente_sel)
    docente_nombre_visible = ""
    if docente_user_id is not None:
        try:
            d = next((x for x in docentes if x.get("id") == docente_user_id), None)
        except Exception:
            d = None
        if d:
            if d.get("nombre") or d.get("apellidos"):
                docente_nombre_visible = f"{str(d.get('nombre','')).strip()} {str(d.get('apellidos','')).strip()}".strip()
            else:
                docente_nombre_visible = str(d.get("usuario","")).strip()

    data = {
        "nombre": nombre_materia.strip(),
        "semestre": int(semestre),
        "carrera_id": int(carrera_id),
        "docente": docente_nombre_visible
    }

    res = analytics.db.insertar_materia(data)
    if res:
        try:
            materia_id = int(res.get("id")) if isinstance(res, dict) and "id" in res else None
            if docente_user_id is not None and materia_id and hasattr(analytics.db, "asignar_docente_a_materia"):
                analytics.db.asignar_docente_a_materia(materia_id, int(docente_user_id))
            if materia_id and g_periodo.strip() and g_grupo.strip():
                analytics.db.crear_grupo(materia_id=materia_id, periodo=g_periodo.strip(), grupo=g_grupo.strip().upper())
        except Exception:
            pass

        analytics.actualizar_datos()
        st.session_state["LAST_DATA_UPDATE"] = time.time()
        st.success("Materia registrada")
        st.rerun()
    else:
        st.error("No se pudo registrar la materia")

# ================= Registrar Calificaciones DOCENTE =================

def mostrar_registro_calificaciones(analytics):
    st.subheader("Registrar Calificaciones por Materia (Docente)")

    if not es_docente():
        st.warning("Esta sección es exclusiva para docentes.")
        return

    # Catálogo base de materias del usuario
    mats = analytics.db.obtener_materias() or []
    if not mats:
        st.info("No tienes materias asignadas")
        return

    # Filtro por carrera
    carreras_ids = sorted({int(m.get("carrera_id")) for m in mats if m.get("carrera_id") is not None})
    etiquetas_carr = ["Todas"] + [f"ID {cid}: {CARRERAS.get(cid, f'Carrera {cid}')}" for cid in carreras_ids]
    carr_sel = st.selectbox("Carrera", etiquetas_carr, key="cal_carr_sel")
    carrera_id_filtro = None
    if carr_sel != "Todas":
        try:
            carrera_id_filtro = int(carr_sel.split(":")[0].split()[-1])
        except Exception:
            carrera_id_filtro = None

    if carrera_id_filtro is not None:
        mats = [m for m in mats if int(m.get("carrera_id", -1)) == carrera_id_filtro]

    if not mats:
        st.warning("No hay materias para la carrera seleccionada")
        return

    # Selección de materia
    opciones_mats = {f"ID {m['id']}: {m['nombre']}": m["id"] for m in mats}
    mat_label = st.selectbox("Materia*", list(opciones_mats.keys()), key="cal_mat_sel")
    materia_id = opciones_mats[mat_label]

    # Selección de grupo
    grupos = analytics.db.obtener_grupos(materia_id=materia_id) or []
    if not grupos:
        st.warning("La materia seleccionada no tiene grupos. Crea al menos uno.")
        return
    grp_labels = [f"{g['periodo']} - {g['grupo']}" for g in grupos]
    grp_sel = st.selectbox("Grupo*", grp_labels, key="cal_grp_sel")
    g = grupos[grp_labels.index(grp_sel)]
    periodo, grupo = g["periodo"], g["grupo"]

    # Alumnos inscritos en el grupo
    inscritos = analytics.db.alumnos_inscritos_para_calificacion(materia_id, periodo, grupo) or []
    if not inscritos:
        st.warning("No hay alumnos inscritos en este grupo. Ve a la pestaña Inscribir Alumnos.")
        return

    # Identificar quienes ya tienen calificación final para ocultarlos
    dfc = analytics.df_calificaciones
    ya_ids = set()
    if not dfc.empty and {'materia_id','periodo','grupo','estudiante_id','calificacion_final'}.issubset(dfc.columns):
        mask = (
            (dfc['materia_id'] == materia_id) &
            (dfc['periodo'] == periodo) &
            (dfc['grupo'] == grupo) &
            dfc['calificacion_final'].notna()
        )
        try:
            ya_ids = set(dfc.loc[mask, 'estudiante_id'].astype(int).tolist())
        except Exception:
            # fallback por seguridad
            ya_ids = set(dfc.loc[mask, 'estudiante_id'].tolist())

    # Mapa completo y candidatos sin calificar
    alumnos_map_full = {
        int(i["id"]): f"ID {int(i['id'])}: {i.get('nombre') or i.get('nombres','')}"
        for i in inscritos if i.get("id") is not None
    }
    candidatos_ids = [sid for sid in alumnos_map_full.keys() if sid not in ya_ids]

    st.caption(f"Pendientes por calificar: {len(candidatos_ids)}   Ya calificados: {len(ya_ids)}")

    if not candidatos_ids:
        st.success("Ya registraste todas las calificaciones de este grupo.")
        return

    # Lupa para buscar por nombre o ID
    q = st.text_input("Buscar alumno por nombre o ID", key="cal_search").strip()
    if q:
        nq = _norm_txt(q)
        def _match(aid: int) -> bool:
            label = alumnos_map_full[aid]
            return (q.isdigit() and str(aid).startswith(q)) or (nq in _norm_txt(label))
        candidatos_ids = [sid for sid in candidatos_ids if _match(sid)]

    if not candidatos_ids:
        st.info("No hay coincidencias con la búsqueda.")
        return

    with st.form("form_calif_por_materia"):
        col1, col2 = st.columns([2, 1])
        with col1:
            estudiante_id = st.selectbox(
                "Estudiante*",
                options=candidatos_ids,
                format_func=lambda x: alumnos_map_full[x],
                key="cal_est_sel_filtrado"
            )
        with col2:
            st.write("")

        st.subheader("Calificaciones por Unidad")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            u1 = st.number_input("Unidad 1", min_value=0.0, max_value=100.0, value=80.0, step=0.1, key="cal_u1")
        with c2:
            u2 = st.number_input("Unidad 2", min_value=0.0, max_value=100.0, value=80.0, step=0.1, key="cal_u2")
        with c3:
            u3 = st.number_input("Unidad 3", min_value=0.0, max_value=100.0, value=80.0, step=0.1, key="cal_u3")
        with c4:
            asistencia = st.number_input("Asistencia %", min_value=0.0, max_value=100.0, value=90.0, step=0.1, key="cal_asist")

        enviar = st.form_submit_button("📊 Registrar Calificaciones")

    if not enviar:
        return

    try:
        cal_final = round((u1 + u2 + u3) / 3.0, 2)
        payload = {
            "estudiante_id": int(estudiante_id),
            "materia_id": int(materia_id),
            "periodo": str(periodo),
            "grupo": str(grupo),
            "u1": float(u1),
            "u2": float(u2),
            "u3": float(u3),
            "asistencia": float(asistencia),
            "calificacion_final": float(cal_final)
        }
        res = analytics.db.insertar_calificacion(payload)
        if res:
            analytics.actualizar_datos()
            st.session_state["LAST_DATA_UPDATE"] = time.time()
            st.success(f"Calificaciones registradas. Final {cal_final}")
            st.rerun()
        else:
            st.error("No se pudo registrar la calificación")
    except Exception as e:
        st.error(f"Ocurrió un error al registrar: {e}")

# ================= Registrar Factores =================

def mostrar_registro_factores(analytics):
    st.subheader("Registrar Factores de Riesgo")

    df_est = analytics.df_estudiantes.copy()
    if df_est.empty:
        st.info("No hay estudiantes")
        return

    q = st.text_input("Buscar alumno por nombre, apellidos o ID", key="fac_q").strip()

    def _etiqueta(r):
        return f"ID {int(r['id'])}: {_nombre_estudiante_row(r)}"

    df_filtrado = df_est.copy()
    if q:
        qn = _norm_txt(q)
        if q.isdigit():
            df_filtrado = df_filtrado[df_filtrado["id"].astype(str).str.contains(q, na=False)]
        else:
            def _match(r):
                return qn in _norm_txt(_nombre_estudiante_row(r))
            df_filtrado = df_filtrado[df_filtrado.apply(_match, axis=1)]

    candidatos = df_filtrado if not df_filtrado.empty else df_est
    orden_cols = [c for c in ["nombres", "apellido_paterno", "apellido_materno", "id"] if c in candidatos.columns]
    if orden_cols:
        candidatos = candidatos.sort_values(orden_cols)

    opciones = {int(r["id"]): _etiqueta(r) for _, r in candidatos.iterrows()}
    alumno_id = st.selectbox(
        "Alumno",
        options=list(opciones.keys()),
        format_func=lambda x: opciones[x],
        key="fac_alumno_sel"
    )
    st.caption(f"Resultados: {len(opciones)}")

    with st.form("form_factor"):
        col1, col2 = st.columns(2)
        with col1:
            categoria = st.selectbox("Categoría*", CATEGORIAS_FACTORES, key="fac_cat_sel")
        with col2:
            gravedad = st.slider("Nivel de gravedad*", 1, 5, 3)

        descripcion = st.text_input("Descripción del factor*", placeholder="Ej. Falta de base en matemáticas")
        enviar = st.form_submit_button("⚠️ Registrar Factor")

    if not enviar:
        return

    if not descripcion.strip():
        st.error("La descripción es obligatoria")
        return

    try:
        payload = {
            "categoria": str(categoria),
            "nombre": descripcion.strip(),
            "inscripcion_id": None,
            "gravedad": int(gravedad),
        }
        res = analytics.db.insertar_factor(payload)
        if res:
            analytics.actualizar_datos()
            st.success(f"Factor registrado para {opciones[alumno_id]}")
            st.rerun()
        else:
            st.error("No se pudo registrar el factor")
    except Exception as e:
        st.error(f"No se pudo registrar: {e}")

# ================= Inscripciones =================

def _nombre_estudiante_row(row: dict) -> str:
    piezas = []
    for k in ("nombres", "apellido_paterno", "apellido_materno"):
        v = row.get(k, "")
        if pd.notna(v):
            s = str(v).strip()
            if s and s.lower() != "nan":
                piezas.append(s)
    if piezas:
        return " ".join(piezas)

    v = row.get("nombre")
    if pd.notna(v):
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return f"ID {row.get('id', '')}"

def mostrar_inscribir_alumnos(analytics):
    st.subheader("Inscribir alumnos a materias y grupos")

    # Traer datos frescos al entrar
    try:
        analytics.actualizar_datos()
    except Exception:
        pass

    # 1) Selección de carrera
    opciones_carr = {"Todas": None}
    for cid, nom in CARRERAS.items():
        opciones_carr[f"ID {cid}: {nom}"] = cid
    carr_label = st.selectbox("Carrera", list(opciones_carr.keys()), key="insc_carr_sel")
    carrera_id_sel = opciones_carr[carr_label]

    # 2) Catálogo de materias
    try:
        # si existe el método admin lo usamos, si no caemos en el normal
        mats = analytics.db.obtener_materias_admin() or []
        if not mats:
            mats = analytics.db.obtener_materias() or []
    except Exception:
        mats = analytics.db.obtener_materias() or []

    if not mats:
        st.info("No hay materias registradas")
        return

    # 3) Filtrar materias por carrera seleccionada
    if carrera_id_sel is not None:
        mats = [m for m in mats if str(m.get("carrera_id", "")) == str(carrera_id_sel)]

    if not mats:
        st.warning("No hay materias para la carrera seleccionada")
        return

    # 4) Selección de materia y grupo
    opciones_mats = {f"ID {m['id']}: {m['nombre']}": m["id"] for m in mats}
    mat_label = st.selectbox("Materia", list(opciones_mats.keys()), key="insc_mat_sel")
    materia_id = opciones_mats[mat_label]

    grupos = analytics.db.obtener_grupos(materia_id=materia_id) or []
    if not grupos:
        st.warning("La materia seleccionada no tiene grupos. Crea al menos uno en Materias.")
        return

    grp_labels = [f"{g['periodo']} - {g['grupo']}" for g in grupos]
    grp_sel = st.selectbox("Grupo", grp_labels, key="insc_grp_sel")
    g = grupos[grp_labels.index(grp_sel)]
    periodo, grupo = g["periodo"], g["grupo"]

    st.divider()
    st.markdown("### Agregar alumno")

    # Recargar lista de alumnos
    if st.button("🔄 Recargar lista de alumnos", key="insc_refresh"):
        try:
            analytics.actualizar_datos()
        except Exception:
            pass
        st.rerun()

    df_est = analytics.df_estudiantes.copy()

    # Filtrar alumnos por la misma carrera de la materia
    try:
        df_mat = analytics.df_materias
        fila = df_mat[df_mat["id"] == materia_id]
        if (not fila.empty) and ("carrera_id" in df_est.columns) and ("carrera_id" in df_mat.columns):
            carr_mat = int(fila.iloc[0]["carrera_id"])
            df_est["carrera_id"] = pd.to_numeric(df_est["carrera_id"], errors="coerce").astype("Int64")
            df_est = df_est[df_est["carrera_id"] == carr_mat]
    except Exception:
        pass  # si algo falla, mostramos todos

    # Excluir ya inscritos en ese grupo
    ya = analytics.db.listar_inscritos(materia_id, periodo, grupo) or []
    ids_ya = {int(r["estudiante_id"]) for r in ya if r.get("estudiante_id") is not None}

    if "id" in df_est.columns:
        df_est["id"] = pd.to_numeric(df_est["id"], errors="coerce").astype("Int64")
    df_candidatos = df_est[~df_est["id"].isin(ids_ya)].copy()

    if df_candidatos.empty:
        st.info("No hay alumnos elegibles para inscribir en este grupo")
    else:
        def _nombre_estudiante_row_local(row: dict) -> str:
            piezas = []
            for k in ("nombres", "apellido_paterno", "apellido_materno"):
                v = row.get(k, "")
                if pd.notna(v):
                    s = str(v).strip()
                    if s and s.lower() != "nan":
                        piezas.append(s)
            if piezas:
                return " ".join(piezas)

            v = row.get("nombre")
            if pd.notna(v):
                s = str(v).strip()
                if s and s.lower() != "nan":
                    return s
            return f"ID {row.get('id','')}"

        opciones_alumnos = {
            int(row["id"]): f"ID {int(row['id'])}: {_nombre_estudiante_row_local(row)}"
            for _, row in df_candidatos.iterrows()
            if pd.notna(row["id"])
        }

        alumno_id = st.selectbox(
            "Alumno a inscribir",
            options=list(opciones_alumnos.keys()),
            format_func=lambda x: opciones_alumnos[x],
            key="insc_alumno_sel"
        )

        if st.button("Inscribir", type="primary", key="insc_btn_add"):
            try:
                analytics.db.inscribir_estudiante(int(alumno_id), int(materia_id), str(periodo), str(grupo))
                analytics.actualizar_datos()
                st.success("Alumno inscrito")
                st.rerun()
            except Exception as e:
                st.error(f"No se pudo inscribir: {e}")

    st.divider()
    st.markdown("### Inscritos en el grupo")

    inscritos = analytics.db.listar_inscritos(materia_id, periodo, grupo) or []
    if not inscritos:
        st.info("Aún no hay alumnos inscritos")
        return

    df = pd.DataFrame(inscritos)
    def _nombre_row(r):
        piezas = []
        for k in ("nombres", "apellido_paterno", "apellido_materno"):
            v = r.get(k, "")
            if pd.notna(v):
                s = str(v).strip()
                if s and s.lower() != "nan":
                    piezas.append(s)
        return " ".join(piezas) if piezas else (r.get("nombre_simple") or f"ID {r.get('estudiante_id')}")


    df["Alumno"] = df.apply(_nombre_row, axis=1)
    df = df[["estudiante_id", "Alumno"]].rename(columns={"estudiante_id": "ID"})
    st.dataframe(df, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        des_id = st.selectbox("Seleccionar alumno para desinscribir", df["ID"].tolist(), key="insc_del_id")
    with col_b:
        if st.button("Quitar", key="insc_btn_del"):
            try:
                analytics.db.desinscribir_estudiante(int(des_id), int(materia_id), str(periodo), str(grupo))
                analytics.actualizar_datos()
                st.success("Alumno desinscrito")
                st.rerun()
            except Exception as e:
                st.error(f"No se pudo desinscribir: {e}")

# ================= Asignar Docentes =================

def mostrar_asignar_docentes(analytics):
    st.subheader("Asignar o quitar docentes de materias")

    try:
        docentes = analytics.db.listar_docentes() or []
        materias = analytics.db.obtener_materias_admin() or []
    except Exception as e:
        st.error(f"No se pudieron cargar catálogos: {e}")
        return

    if not materias:
        st.info("No hay materias registradas")
        return

    opciones_doc = {"Sin asignar": None}
    for d in docentes:
        etiqueta = str(d.get("usuario") or f"doc_{d.get('id')}")
        opciones_doc[f"{etiqueta}  (ID {d.get('id')})"] = d.get("id")

    opciones_mat = {f"ID {m['id']}: {m['nombre']}": m["id"] for m in materias}
    mat_label = st.selectbox("Materia", list(opciones_mat.keys()), key="asg_mat_sel")
    materia_id = opciones_mat[mat_label]

    actual = next((m for m in materias if m["id"] == materia_id), None)
    actual_id = actual.get("docente_user_id") if actual else None
    actual_txt = next((k for k, v in opciones_doc.items() if v == actual_id), "Sin asignar")
    st.caption(f"Docente actual: {actual_txt}")

    col1, col2 = st.columns([3, 1])
    with col1:
        doc_label = st.selectbox("Docente", list(opciones_doc.keys()), index=list(opciones_doc.keys()).index(actual_txt), key="asg_doc_sel")
        docente_id = opciones_doc[doc_label]
    with col2:
        st.write("")
        asignar = st.button("Guardar", key="asg_btn_save")

    quitar = st.button("Quitar docente de esta materia", key="asg_btn_quitar")

    if asignar:
        try:
            if docente_id is None:
                ok = analytics.db.quitar_docente_de_materia(materia_id)
            else:
                ok = analytics.db.set_docente_en_materia(materia_id, docente_id)
            if ok:
                st.success("Asignación actualizada")
                analytics.actualizar_datos()
                st.rerun()
            else:
                st.error("No se pudo actualizar la asignación")
        except Exception as e:
            st.error(f"Error al asignar: {e}")

    if quitar:
        try:
            ok = analytics.db.quitar_docente_de_materia(materia_id)
            if ok:
                st.success("Se quitó el docente de la materia")
                analytics.actualizar_datos()
                st.rerun()
            else:
                st.error("No se pudo quitar la asignación")
        except Exception as e:
            st.error(f"Error al quitar: {e}")

    st.divider()
    st.subheader("Listado de asignaciones")
    try:
        df_asg = analytics.db.listar_asignaciones()
        if df_asg is None or df_asg.empty:
            st.info("No hay asignaciones para mostrar")
        else:
            mostrar = df_asg.rename(columns={
                "materia_id": "ID Materia",
                "materia": "Materia",
                "docente_id": "ID Docente",
                "docente": "Docente"
            })
            st.dataframe(mostrar, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"No se pudo cargar el listado: {e}")