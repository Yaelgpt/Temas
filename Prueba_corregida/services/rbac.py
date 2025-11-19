
import streamlit as st

def rol_usuario_actual() -> str:
    user = st.session_state.get("user") or {}
    return user.get("rol") or "docente"

def es_admin() -> bool:
    return rol_usuario_actual() == "admin"

def es_docente() -> bool:
    return rol_usuario_actual() == "docente"

def usuario_id():
    user = st.session_state.get("user") or {}
    return user.get("id")
