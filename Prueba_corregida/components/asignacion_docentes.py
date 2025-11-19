import streamlit as st
import pandas as pd
from services.rbac import es_docente

def mostrar_asignar_clases(database_service):
    if es_docente():
        st.info("Solo un administrador puede asignar clases")
        return

    st.subheader("Asignar clases a docentes")

    docentes = database_service.listar_docentes()
    materias = database_service.obtener_materias_admin()

    if not docentes:
        st.info("No hay docentes creados")
        return
    if not materias:
        st.info("No hay materias registradas")
        return

    colf1, colf2 = st.columns(2)
    with colf1:
        ver_solo_no_asignadas = st.checkbox("Ver solo materias sin docente", value=False)
    with colf2:
        filtro_texto = st.text_input("Filtrar materias por nombre")

    mats = materias
    if ver_solo_no_asignadas:
        mats = [m for m in mats if not m.get("docente_user_id")]
    if filtro_texto.strip():
        t = filtro_texto.lower()
        mats = [m for m in mats if t in str(m.get("nombre","")).lower()]

    col1, col2 = st.columns(2)
    with col1:
        docente_map = {f"{d['usuario']}  id {d['id']}": d["id"] for d in docentes}
        docente_label = st.selectbox("Docente", list(docente_map.keys()))
        docente_id = docente_map[docente_label]
    with col2:
        materia_map = {f"{m['nombre']}  id {m['id']}": m["id"] for m in mats}
        materia_label = st.selectbox("Materia", list(materia_map.keys()))
        materia_id = materia_map[materia_label]

    cbtn1, cbtn2 = st.columns(2)
    with cbtn1:
        if st.button("Asignar docente a materia", use_container_width=True):
            ok = database_service.set_docente_en_materia(materia_id, docente_id)
            if ok:
                st.success("Asignación guardada")
                st.rerun()
    with cbtn2:
        if st.button("Quitar asignación", use_container_width=True):
            ok = database_service.quitar_docente_de_materia(materia_id)
            if ok:
                st.success("Asignación eliminada")
                st.rerun()

    with st.expander("Crear grupo para la materia seleccionada"):
        p1, p2 = st.columns(2)
        with p1:
            periodo = st.text_input("Periodo", value="2025-1")
        with p2:
            grupo = st.text_input("Grupo", value="A", max_chars=5)
        if st.button("Crear grupo"):
            row = database_service.crear_grupo(materia_id, periodo, grupo)
            if row:
                st.success("Grupo creado")

    st.markdown("#### Asignaciones actuales")
    df = database_service.listar_asignaciones()
    if df.empty:
        st.info("Aún no hay asignaciones")
    else:
        df = df.sort_values(["docente", "materia"], na_position="last")
        st.dataframe(df, use_container_width=True, height=300)