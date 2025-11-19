import os
import streamlit as st
from services.auth_service import AuthService

def mostrar_login():
    st.markdown("""
    <style>
      .login-card {max-width:420px;margin:6rem auto;padding:2rem;border:1px solid #ddd;border-radius:12px;background:#fff;box-shadow:0 8px 24px rgba(0,0,0,.06)}
    </style>
    """, unsafe_allow_html=True)

    if "user" not in st.session_state:
        st.session_state["user"] = None

    # Sesión activa
    if st.session_state["user"] is not None:
        u = st.session_state["user"]
        st.success(f"Sesión: {u['usuario']}  Rol: {u.get('rol','docente')}")
        if st.button("Cerrar sesión"):
            st.session_state["user"] = None
            st.rerun()
        return

    tab_login, tab_signup = st.tabs(["🔐 Iniciar sesión", "🆕 Crear usuario"])

    # Iniciar sesión
    with tab_login:
        with st.form("frm_login"):
            login_usuario = st.text_input("Usuario")
            login_password = st.text_input("Contraseña", type="password")
            login_ok = st.form_submit_button("Entrar")
        if login_ok:
            try:
                auth = AuthService()
                u = auth.verificar_login(login_usuario, login_password)
                if u:
                    st.session_state["user"] = u
                    st.success("Acceso concedido")
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos")
            except Exception as e:
                st.error(f"Error al iniciar sesión. Detalle: {e}")

    # Crear usuario con rol seleccionado
    with tab_signup:
        st.info("Crea un usuario en la tabla usuarios. Elige el rol al crearlo.")
        with st.form("frm_signup"):
            signup_usuario = st.text_input("Nuevo usuario")
            signup_pwd1 = st.text_input("Contraseña", type="password")
            signup_pwd2 = st.text_input("Repite la contraseña", type="password")
            signup_rol = st.radio("Rol", ["docente", "admin"], index=0, horizontal=True)
            signup_ok = st.form_submit_button("Crear")
        if signup_ok:
            if not signup_usuario or not signup_pwd1:
                st.error("Completa usuario y contraseña")
            elif signup_pwd1 != signup_pwd2:
                st.error("Las contraseñas no coinciden")
            else:
                try:
                    auth = AuthService()
                    creado = auth.crear_usuario(signup_usuario, signup_pwd1, rol=signup_rol)
                    if creado:
                        st.success(f"Usuario {signup_usuario} creado con rol {signup_rol}. Ahora inicia sesión.")
                    else:
                        st.warning("Ese usuario ya existe")
                except Exception as e:
                    st.error(f"Error al crear usuario. Detalle: {e}")