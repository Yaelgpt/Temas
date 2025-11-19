import streamlit as st
from supabase import create_client
import hashlib
import hmac
import os

def _get_supabase():
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Faltan credenciales de Supabase en secrets.toml")
    return create_client(url, key)

_SECRET = os.environ.get("APP_PWD_PEPPER", "cambio-esto-en-produc")

def _hash_password(password: str) -> str:
    return hmac.new(_SECRET.encode(), password.encode(), hashlib.sha256).hexdigest()

class AuthService:
    def __init__(self):
        self.client = _get_supabase()

    def crear_usuario(self, usuario: str, password: str, rol: str = "docente"):
        # normaliza rol para cumplir el CHECK de la tabla
        rol = "admin" if str(rol).lower() == "admin" else "docente"

        existe = self.client.table("usuarios").select("id").eq("usuario", usuario).execute()
        if existe.data:
            return None

        row = {
            "usuario": usuario,
            "password_hash": _hash_password(password),
            "activo": True,
            "rol": rol,
        }
        res = self.client.table("usuarios").insert(row).execute()
        return res.data[0] if res.data else None

    def verificar_login(self, usuario: str, password: str):
        hash_ = _hash_password(password)
        res = (self.client.table("usuarios")
               .select("id, usuario, activo, rol")
               .eq("usuario", usuario)
               .eq("password_hash", hash_)
               .eq("activo", True)
               .limit(1)
               .execute())
        return res.data[0] if res.data else None