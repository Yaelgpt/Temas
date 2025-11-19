import streamlit as st  
from services.database import DatabaseService
from services.analytics import AnalyticsService
from components.dashboard import mostrar_dashboard_principal
from components.registro_datos import (
    mostrar_registro_datos,
    mostrar_registro_calificaciones,  # << añadido para abrir solo calificaciones a docentes
)
from components.exportacion import mostrar_exportar_reportes
from components.login import mostrar_login
from services.rbac import es_docente, es_admin 
from components.analisis_calidad import (
    mostrar_analisis_calidad,
    analitica_histograma_y_control,
)

# intento de import compatible con versiones
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    try:
        from streamlit import st_autorefresh  # algunas builds lo exponen aquí
    except Exception:
        st_autorefresh = None  # si no existe, desactivamos la función

# ===== etiquetas de menú para evitar descalces por emojis =====
MENU_DASH = "🏠 Dashboard Principal"
MENU_QUAL = "📈 Análisis de Calidad"
MENU_REG  = "📝 Registro de Datos"
MENU_EXP  = "📦 Exportar Reportes"

# Estilos
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f3a60; text-align: center; margin-bottom: 2rem; font-weight: bold; }
    .sub-header { font-size: 1.8rem; color: #2c3e50; margin-bottom: 1rem; font-weight: bold; }
    .metric-card { background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #3498db; margin-bottom: 1rem; }
    .success-text { color: #27ae60; }
    .warning-text { color: #f39c12; }
    .danger-text { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def inicializar_servicios(_version: int = 2):
    db = DatabaseService()
    analytics = AnalyticsService(db)
    return db, analytics

def main():
    db, analytics = inicializar_servicios()

    st.markdown('<div class="main-header">🎓 SISTEMA DE ANÁLISIS EDUCATIVO - ITT</div>', unsafe_allow_html=True)

    # Sesión
    if "user" not in st.session_state:
        st.session_state["user"] = None

    if st.session_state["user"] is None:
        st.title("Acceso")
        mostrar_login()
        st.stop()

    # Sidebar
    with st.sidebar:
        st.caption(f"Usuario: **{st.session_state['user']['usuario']}**")
        if st.button("Cerrar sesión"):
            st.session_state["user"] = None
            st.rerun()

    with st.sidebar:
        st.image("https://www.tijuana.tecnm.mx/wp-content/themes/tecnm/images/logo_TECT.png", width=150)
        st.sidebar.markdown("### Navegación")

        menu_items = [MENU_DASH, MENU_QUAL]
        if es_docente() or es_admin():
            menu_items.append(MENU_REG)
        menu_items.append(MENU_EXP)

        opcion = st.sidebar.radio("Selecciona una opción:", menu_items, key="nav_main")

        st.divider()
        if st.button("🔄 Actualizar Datos", use_container_width=True):
            analytics.actualizar_datos()
            st.rerun()

        if st.button("Recargar servicios"):
            st.cache_resource.clear()
            st.rerun()

        st.divider()
        # controles de auto refresh
        auto_on = st.toggle("Auto actualizar", value=True, help="Refresca la vista de forma periódica")
        auto_secs = st.select_slider(
            "Intervalo",
            options=[10, 15, 30, 60, 120, 300],
            value=30,
            help="Frecuencia de actualización automática"
        )
        # texto informativo
        if auto_on:
            st.caption(f"Auto actualización cada {auto_secs} s")
        else:
            st.caption("Auto actualización desactivada")
        st.caption("Usa el botón 'Actualizar Datos' para forzar una actualización")

    # Auto refresh solo en vistas de lectura
    if auto_on and st_autorefresh is not None and opcion in (MENU_DASH, MENU_QUAL):
        # disparamos el refresh y actualizamos datos en cada ciclo
        st_autorefresh(interval=auto_secs * 1000, key="auto_refresh_main")
        try:
            analytics.actualizar_datos()
        except Exception:
            pass

    # Contenido
    try:
        if opcion == MENU_DASH:
            mostrar_dashboard_principal(analytics)

        elif opcion == MENU_QUAL:
            # Admin ve las herramientas completas
            if es_admin():
                mostrar_analisis_calidad(analytics)
            else:
                # Docente: oculta herramientas y muestra solo su análisis por materia y grupo
                st.markdown('<div class="sub-header">Análisis de Calidad</div>', unsafe_allow_html=True)
                analitica_histograma_y_control(analytics)

        elif opcion == MENU_REG:
            # Docente: solo la vista de registrar calificaciones
            if es_docente() and not es_admin():
                st.subheader("Registrar Calificaciones")
                mostrar_registro_calificaciones(analytics)
            else:
                # Admin: todo el módulo de registro
                mostrar_registro_datos(db)

        elif opcion == MENU_EXP:
            mostrar_exportar_reportes(db)

    except Exception as e:
        st.error(f"Error cargando la sección: {e}")

if __name__ == "__main__":
    main()