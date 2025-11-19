
import re

REGEX_NOMBRE = re.compile(r"^[A-Za-zÁÉÍÓÚáéíóúÑñÜü\s]+$")

def validar_nombre(texto: str) -> bool:
    if not texto:
        return False
    return bool(REGEX_NOMBRE.fullmatch(texto.strip()))

def limpiar_texto(texto: str) -> str:
    return re.sub(r"\s+", " ", (texto or "")).strip()

def split_nombre_completo(nombre: str):
    nombre = limpiar_texto(nombre)
    if not nombre:
        return ("", "", "")
    partes = nombre.split(" ")
    if len(partes) == 1:
        return (partes[0], "", "")
    if len(partes) == 2:
        return (partes[0], partes[1], "")
    return (" ".join(partes[:-2]), partes[-2], partes[-1])
