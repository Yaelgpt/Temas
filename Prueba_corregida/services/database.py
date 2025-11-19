import os
import streamlit as st
import pandas as pd

def _crear_supabase_client():
    url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("Faltan SUPABASE_URL y SUPABASE_KEY en .streamlit/secrets.toml o en variables de entorno")
    from supabase import create_client
    return create_client(url, key)

class DatabaseService:
    def __init__(self, supabase=None):
        self.supabase = supabase or _crear_supabase_client()

    # util
    def _to_df(self, data):
        try:
            return pd.DataFrame(data or [])
        except Exception:
            return pd.DataFrame()

    def limpiar_cache(self):
        # Si algún día cacheas lecturas aquí, límpialas también
        try:
            st.cache_data.clear()
        except Exception:
            pass
        return True

    # ===== CARGAS para AnalyticsService.cargar_datos
    def cargar_estudiantes(self) -> pd.DataFrame:
        try:
            # dentro de DatabaseService.cargar_estudiantes()
            res = self.supabase.table("estudiantes").select(
                "id, matricula, nombre, nombres, apellido_paterno, apellido_materno, carrera_id, "
                "ingreso_semestre, horas_estudio, desercion"
            ).order("id").execute()
            return self._to_df(res.data)
        except Exception as e:
            st.error(f"Error cargando estudiantes: {e}")
            return pd.DataFrame()

    def cargar_calificaciones(self) -> pd.DataFrame:
        try:
            res = self.supabase.table("registro_calificaciones").select(
                "id, estudiante_id, materia_id, periodo, grupo, "
                "calificacion_final, asistencia, u1, u2, u3, reprobado"
            ).order("id").execute()
            return self._to_df(res.data)
        except Exception as e:
            st.error(f"Error cargando calificaciones: {e}")
            return pd.DataFrame()

    def cargar_factores(self) -> pd.DataFrame:
        try:
            res = self.supabase.table("factores").select(
                "id, categoria, nombre, inscripcion_id, gravedad"
            ).order("id").execute()
            return self._to_df(res.data)
        except Exception as e:
            st.error(f"Error cargando factores: {e}")
            return pd.DataFrame()

    def cargar_materias(self) -> pd.DataFrame:
        try:
            q = self.supabase.table("materias").select(
                "id, nombre, semestre, carrera_id, docente, docente_user_id"
            )
            # RBAC docente
            try:
                from services.rbac import es_docente, usuario_id
                if es_docente() and usuario_id():
                    q = q.eq("docente_user_id", usuario_id())
            except Exception:
                pass
            res = q.order("nombre").execute()
            return self._to_df(res.data)
        except Exception as e:
            st.error(f"Error cargando materias: {e}")
            return pd.DataFrame()

    def cargar_grupos(self) -> pd.DataFrame:
        try:
            res = self.supabase.table("grupos").select(
                "id, materia_id, periodo, grupo"
            ).order("periodo", desc=True).execute()
            return self._to_df(res.data)
        except Exception as e:
            st.error(f"Error cargando grupos: {e}")
            return pd.DataFrame()

    # ===== OPERACIONES usadas en UI
    def insertar_estudiante(self, data: dict):
        try:
            if data.get("matricula"):
                res = self.supabase.table("estudiantes") \
                    .upsert(data, on_conflict="matricula", returning="representation") \
                    .execute()
            else:
                res = self.supabase.table("estudiantes").insert(data).execute()
            return (res.data or [None])[0]
        except Exception as e:
            st.error(f"Error insertando estudiante: {e}")
            return None
        
    def insertar_materia(self, data: dict):
        try:
            res = self.supabase.table("materias").insert(data).execute()
            return (res.data or [None])[0]
        except Exception as e:
            st.error(f"Error insertando materia: {e}")
            return None

    def actualizar_estudiante(self, estudiante_id: int, data: dict):
        try:
            self.supabase.table("estudiantes").update(data).eq("id", estudiante_id).execute()
        except Exception as e:
            st.error(f"Error actualizando estudiante: {e}")

    def actualizar_materia(self, materia_id: int, data: dict):
        try:
            self.supabase.table("materias").update(data).eq("id", materia_id).execute()
            return True
        except Exception as e:
            st.error(f"Error actualizando materia: {e}")
            return False

    def obtener_materias(self):
        try:
            df = self.cargar_materias()
            return df.to_dict("records")
        except Exception as e:
            st.error(f"Error obteniendo materias: {e}")
            return []

    def obtener_grupos(self, materia_id: int = None):
        try:
            q = self.supabase.table("grupos").select("id, materia_id, periodo, grupo")
            if materia_id:
                q = q.eq("materia_id", materia_id)
            res = q.order("periodo", desc=True).execute()
            return res.data or []
        except Exception as e:
            st.error(f"Error obteniendo grupos: {e}")
            return []

    def insertar_calificacion(self, payload: dict):
        """
        Upsert por clave compuesta.
        Se pide representación al servidor para obtener la fila inmediata.
        """
        try:
            return (
                self.supabase
                .table("registro_calificaciones")
                .upsert(
                    payload,
                    on_conflict="estudiante_id,materia_id,periodo,grupo",
                    returning="representation"
                )
                .execute()
            )
        except Exception as e:
            # Plan B upsert manual
            try:
                existe = (
                    self.supabase.table("registro_calificaciones")
                    .select("id")
                    .match({
                        "estudiante_id": payload["estudiante_id"],
                        "materia_id": payload["materia_id"],
                        "periodo": payload["periodo"],
                        "grupo": payload["grupo"],
                    })
                    .limit(1)
                    .execute()
                )
                if existe.data:
                    rid = existe.data[0]["id"]
                    self.supabase.table("registro_calificaciones").update(payload).eq("id", rid).execute()
                    return {"data": [{"id": rid}]}
                else:
                    res = self.supabase.table("registro_calificaciones").insert(payload).execute()
                    return res
            except Exception as e2:
                st.error(f"Error upsert calificación: {e2}")
                return None

    def upsert_calificacion(self, data: dict):
        try:
            res = (
                self.supabase.table("registro_calificaciones")
                .upsert(data, on_conflict="estudiante_id,materia_id,periodo,grupo", returning="representation")
                .execute()
            )
            return (res.data or [None])[0]
        except Exception as e:
            st.error(f"Error upsert calificación: {e}")
            return None

    def obtener_calificaciones_por(self, materia_id: int, periodo: str, grupo: str):
        try:
            res = (
                self.supabase.table("registro_calificaciones")
                .select("*")
                .eq("materia_id", materia_id)
                .eq("periodo", periodo)
                .eq("grupo", grupo)
                .execute()
            )
            return res.data or []
        except Exception as e:
            st.error(f"Error obteniendo calificaciones: {e}")
            return []

    def obtener_calificaciones_por_grupo(self, grupo_id: int):
        try:
            g = (
                self.supabase.table("grupos")
                .select("materia_id, periodo, grupo")
                .eq("id", grupo_id)
                .limit(1)
                .execute()
            )
            if not g.data:
                return []
            g0 = g.data[0]
            return self.obtener_calificaciones_por(g0["materia_id"], g0["periodo"], g0["grupo"])
        except Exception as e:
            st.error(f"Error obteniendo calificaciones por grupo: {e}")
            return []

    # ===== admin: asignar clases a docentes
    def listar_docentes(self):
        try:
            res = (self.supabase.table("usuarios")
                   .select("id, usuario, nombre, apellidos")
                   .eq("rol", "docente")
                   .eq("activo", True)
                   .order("usuario")
                   .execute())
            return res.data or []
        except Exception as e:
            st.error(f"Error listando docentes: {e}")
            return []

    def obtener_materias_admin(self):
        try:
            res = (self.supabase.table("materias")
                   .select("id, nombre, semestre, carrera_id, docente_user_id, docente")
                   .order("nombre")
                   .execute())
            return res.data or []
        except Exception as e:
            st.error(f"Error cargando materias: {e}")
            return []

    def set_docente_en_materia(self, materia_id: int, docente_id: int):
        try:
            self.supabase.table("materias").update(
                {"docente_user_id": int(docente_id)}
            ).eq("id", int(materia_id)).execute()
            return True
        except Exception as e:
            st.error(f"Error asignando docente: {e}")
            return False

    def quitar_docente_de_materia(self, materia_id: int):
        try:
            self.supabase.table("materias").update(
                {"docente_user_id": None}
            ).eq("id", int(materia_id)).execute()
            return True
        except Exception as e:
            st.error(f"Error quitando asignación: {e}")
            return False

    def listar_asignaciones(self) -> pd.DataFrame:
        try:
            mats = self.obtener_materias_admin()
            docs = {d["id"]: d.get("usuario") for d in (self.listar_docentes() or [])}
            rows = []
            for m in mats:
                did = m.get("docente_user_id")
                rows.append({
                    "materia_id": m["id"],
                    "materia": m["nombre"],
                    "docente_id": did,
                    "docente": docs.get(did)
                })
            return pd.DataFrame(rows)
        except Exception as e:
            st.error(f"Error listando asignaciones: {e}")
            return pd.DataFrame()

    def crear_grupo(self, materia_id: int, periodo: str, grupo: str):
        try:
            payload = {
                "materia_id": int(materia_id),
                "periodo": str(periodo).strip(),
                "grupo": str(grupo).strip().upper()
            }
            res = self.supabase.table("grupos").insert(payload).execute()
            return (res.data or [None])[0]
        except Exception as e:
            st.error(f"Error creando grupo: {e}")
            return None

    # =========================
    #   INSCRIPCIONES
    # =========================

    def listar_inscritos(self, materia_id: int, periodo: str, grupo: str):
        res = self.supabase.table("vw_inscripciones_detalle") \
            .select("*") \
            .eq("materia_id", int(materia_id)) \
            .eq("periodo", str(periodo)) \
            .eq("grupo", str(grupo)) \
            .order("estudiante_id") \
            .execute()
        return res.data or []

    def inscribir_estudiante(self, estudiante_id: int, materia_id: int, periodo: str, grupo: str):
        payload = {
            "estudiante_id": int(estudiante_id),
            "materia_id": int(materia_id),
            "periodo": str(periodo),
            "grupo": str(grupo)
        }
        return self.supabase.table("inscripciones").insert(payload).execute()

    def desinscribir_estudiante(self, estudiante_id: int, materia_id: int, periodo: str, grupo: str):
        return self.supabase.table("inscripciones") \
            .delete() \
            .eq("estudiante_id", int(estudiante_id)) \
            .eq("materia_id", int(materia_id)) \
            .eq("periodo", str(periodo)) \
            .eq("grupo", str(grupo)) \
            .execute()

    def alumnos_inscritos_para_calificacion(self, materia_id: int, periodo: str, grupo: str):
        filas = self.listar_inscritos(materia_id, periodo, grupo)
        salida = []
        for r in filas:
            piezas = []
            if r.get("nombres"): piezas.append(str(r["nombres"]).strip())
            if r.get("apellido_paterno"): piezas.append(str(r["apellido_paterno"]).strip())
            if r.get("apellido_materno"): piezas.append(str(r["apellido_materno"]).strip())
            nombre = " ".join(piezas) if piezas else (r.get("nombre_simple") or f"ID {r['estudiante_id']}")
            salida.append({"id": int(r["estudiante_id"]), "nombre": nombre})
        salida.sort(key=lambda x: x["nombre"])
        return salida

    # ===== Factores
    def insertar_factor(self, data: dict):
        try:
            res = self.supabase.table("factores").insert(data).execute()
            return (res.data or [None])[0]
        except Exception as e:
            st.error(f"Error insertando factor: {e}")
            return None