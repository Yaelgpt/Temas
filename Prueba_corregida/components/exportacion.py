import streamlit as st
import pandas as pd
from io import BytesIO
from services.analytics import AnalyticsService
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import tempfile
import os
import time
import errno

# ========= utilidades de reintento anti WinError 10035 =========
def _con_reintentos(fn, intentos=4, espera_inicial=0.6):
    """
    Ejecuta fn con reintentos si aparece un error no bloqueante tipo 10035.
    Incrementa la espera entre intentos.
    """
    espera = espera_inicial
    for i in range(intentos):
        try:
            return fn()
        except OSError as e:
            winerr = getattr(e, "winerror", None)
            ecode = getattr(e, "errno", None)
            if (winerr == 10035 or ecode in (errno.EWOULDBLOCK,)) and i < intentos - 1:
                time.sleep(espera)
                espera *= 1.7
                continue
            raise

def _df_est(analytics):
    return _con_reintentos(lambda: analytics.df_estudiantes.copy())

def _df_cal(analytics):
    return _con_reintentos(lambda: analytics.df_calificaciones.copy())

def _df_fac(analytics):
    return _con_reintentos(lambda: analytics.df_factores.copy())

def _df_mat(analytics):
    return _con_reintentos(lambda: analytics.df_materias.copy())


def mostrar_exportar_reportes(database_service):
    """Mostrar interfaz para exportar reportes"""
    st.markdown('<div class="sub-header">📤 Exportar Reportes</div>', unsafe_allow_html=True)

    analytics = AnalyticsService(database_service)
    # refresco con reintento
    try:
        _con_reintentos(lambda: analytics.actualizar_datos())
    except Exception as e:
        st.warning(f"No fue posible actualizar datos al abrir exportacion: {e}")

    st.subheader("Generar Reportes en Diferentes Formatos")

    tipo_reporte = st.selectbox(
        "Selecciona el tipo de reporte:",
        ["Reporte General", "Reporte de Estudiantes", "Reporte de Calificaciones", "Reporte de Factores de Riesgo", "Reporte Personalizado"]
    )

    col1, col2 = st.columns(2)

    with col1:
        if tipo_reporte in ["Reporte General", "Reporte de Calificaciones", "Reporte Personalizado"]:
            rango_calificaciones = st.slider(
                "Rango de Calificaciones:",
                min_value=0.0, max_value=100.0, value=(0.0, 100.0)
            )

        if tipo_reporte in ["Reporte General", "Reporte de Estudiantes", "Reporte Personalizado"]:
            filtro_desercion = st.selectbox(
                "Filtrar por estado de desercion:",
                ["Todos", "En riesgo", "No en riesgo"]
            )

    with col2:
        if tipo_reporte in ["Reporte General", "Reporte de Factores de Riesgo", "Reporte Personalizado"]:
            try:
                df_fac = _df_fac(analytics)
            except Exception:
                df_fac = pd.DataFrame()

            categorias_factores = df_fac['categoria'].unique().tolist() if not df_fac.empty else []
            if categorias_factores:
                filtro_categoria = st.multiselect(
                    "Filtrar por categoria de factores:",
                    options=categorias_factores,
                    default=categorias_factores
                )
            else:
                filtro_categoria = []
                st.info("No hay categorias de factores disponibles")

    formato = st.radio(
        "Formato de exportacion:",
        ["Excel (.xlsx)", "CSV (.csv)", "PDF (.pdf)"],
        horizontal=True
    )

    if formato == "PDF (.pdf)":
        col3, col4 = st.columns(2)
        with col3:
            incluir_graficas = st.checkbox("Incluir graficas", value=True)
        with col4:
            incluir_estadisticas = st.checkbox("Incluir estadisticas", value=True)

    if st.button("🔄 Generar Reporte", type="primary"):
        try:
            with st.spinner("Generando reporte..."):
                datos_filtrados = aplicar_filtros_y_generar_datos(analytics, tipo_reporte, {
                    'rango_calificaciones': rango_calificaciones if 'rango_calificaciones' in locals() else None,
                    'filtro_desercion': filtro_desercion if 'filtro_desercion' in locals() else None,
                    'filtro_categoria': filtro_categoria if 'filtro_categoria' in locals() else None
                })

                if datos_filtrados is None or (isinstance(datos_filtrados, dict) and all(v is None for v in datos_filtrados.values())):
                    st.warning("⚠️ No hay datos para generar el reporte con los filtros seleccionados.")
                    return

                if formato in ["Excel (.xlsx)", "CSV (.csv)"]:
                    archivo = generar_archivo_descarga(datos_filtrados, tipo_reporte, formato)

                    if archivo:
                        st.subheader("📊 Vista Previa del Reporte")
                        mostrar_vista_previa(datos_filtrados, tipo_reporte)

                        nombre_archivo = f"reporte_{tipo_reporte.lower().replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

                        if formato == "Excel (.xlsx)":
                            nombre_archivo += ".xlsx"
                            st.download_button(
                                label="📥 Descargar Reporte Excel",
                                data=archivo,
                                file_name=nombre_archivo,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            nombre_archivo += ".csv"
                            st.download_button(
                                label="📥 Descargar Reporte CSV",
                                data=archivo,
                                file_name=nombre_archivo,
                                mime="text/csv"
                            )

                        st.success("✅ Reporte generado")
                    else:
                        st.error("❌ Error al generar el archivo")

                elif formato == "PDF (.pdf)":
                    archivo_pdf = generar_pdf(analytics, datos_filtrados, tipo_reporte,
                                              incluir_graficas=incluir_graficas,
                                              incluir_estadisticas=incluir_estadisticas)

                    if archivo_pdf:
                        nombre_archivo = f"reporte_{tipo_reporte.lower().replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                        st.download_button(
                            label="📥 Descargar Reporte PDF",
                            data=archivo_pdf,
                            file_name=nombre_archivo,
                            mime="application/pdf"
                        )
                        st.success("✅ Reporte PDF generado")
                    else:
                        st.error("❌ Error al generar el reporte PDF")

        except Exception as e:
            st.error(f"❌ Error generando reporte: {e}")


def aplicar_filtros_y_generar_datos(analytics, tipo_reporte, filtros):
    """Aplicar filtros y generar datos para el reporte"""
    try:
        if tipo_reporte == "Reporte General":
            return generar_reporte_general(analytics, filtros)
        elif tipo_reporte == "Reporte de Estudiantes":
            return generar_reporte_estudiantes(analytics, filtros)
        elif tipo_reporte == "Reporte de Calificaciones":
            return generar_reporte_calificaciones(analytics, filtros)
        elif tipo_reporte == "Reporte de Factores de Riesgo":
            return generar_reporte_factores(analytics, filtros)
        elif tipo_reporte == "Reporte Personalizado":
            return generar_reporte_personalizado(analytics, filtros)
        else:
            return None
    except Exception as e:
        st.error(f"Error aplicando filtros: {e}")
        return None


def generar_reporte_general(analytics, filtros):
    """Generar reporte general consolidado"""
    try:
        df_est = _df_est(analytics)
        if filtros.get('filtro_desercion') == "En riesgo":
            df_est = df_est[df_est['desercion'] == True]
        elif filtros.get('filtro_desercion') == "No en riesgo":
            df_est = df_est[df_est['desercion'] == False]

        df_cal = _df_cal(analytics)
        if filtros.get('rango_calificaciones'):
            rango = filtros['rango_calificaciones']
            df_cal = df_cal[
                (df_cal['calificacion_final'] >= rango[0]) &
                (df_cal['calificacion_final'] <= rango[1])
            ]

        df_fac = _df_fac(analytics)
        if filtros.get('filtro_categoria'):
            df_fac = df_fac[df_fac['categoria'].isin(filtros['filtro_categoria'])]

        df_mat = _df_mat(analytics)

        return {
            'estudiantes': df_est,
            'calificaciones': df_cal,
            'factores': df_fac,
            'materias': df_mat
        }
    except Exception as e:
        st.error(f"Error generando reporte general: {e}")
        return None


def generar_reporte_estudiantes(analytics, filtros):
    """Generar reporte de estudiantes"""
    try:
        df_est = _df_est(analytics)
        if filtros.get('filtro_desercion') == "En riesgo":
            df_est = df_est[df_est['desercion'] == True]
        elif filtros.get('filtro_desercion') == "No en riesgo":
            df_est = df_est[df_est['desercion'] == False]
        return {'estudiantes': df_est}
    except Exception as e:
        st.error(f"Error generando reporte de estudiantes: {e}")
        return None


def generar_reporte_calificaciones(analytics, filtros):
    """Generar reporte de calificaciones"""
    try:
        df_cal = _df_cal(analytics)
        if filtros.get('rango_calificaciones'):
            rango = filtros['rango_calificaciones']
            df_cal = df_cal[
                (df_cal['calificacion_final'] >= rango[0]) &
                (df_cal['calificacion_final'] <= rango[1])
            ]
        return {'calificaciones': df_cal}
    except Exception as e:
        st.error(f"Error generando reporte de calificaciones: {e}")
        return None


def generar_reporte_factores(analytics, filtros):
    """Generar reporte de factores de riesgo"""
    try:
        df_fac = _df_fac(analytics)
        if filtros.get('filtro_categoria'):
            df_fac = df_fac[df_fac['categoria'].isin(filtros['filtro_categoria'])]
        return {'factores': df_fac}
    except Exception as e:
        st.error(f"Error generando reporte de factores: {e}")
        return None


def generar_reporte_personalizado(analytics, filtros):
    """Generar reporte personalizado"""
    try:
        df_cal = _df_cal(analytics)
        df_est = _df_est(analytics)

        datos_combinados = df_cal.merge(
            df_est,
            left_on='estudiante_id',
            right_on='id',
            suffixes=('_cal', '_est'),
            how='left'
        )

        if filtros.get('rango_calificaciones'):
            rango = filtros['rango_calificaciones']
            datos_combinados = datos_combinados[
                (datos_combinados['calificacion_final'] >= rango[0]) &
                (datos_combinados['calificacion_final'] <= rango[1])
            ]

        if filtros.get('filtro_desercion') == "En riesgo":
            datos_combinados = datos_combinados[datos_combinados['desercion'] == True]
        elif filtros.get('filtro_desercion') == "No en riesgo":
            datos_combinados = datos_combinados[datos_combinados['desercion'] == False]

        return {'personalizado': datos_combinados}
    except Exception as e:
        st.error(f"Error generando reporte personalizado: {e}")
        return None


def generar_archivo_descarga(datos, tipo_reporte, formato):
    """Generar archivo para descarga segun el formato"""
    try:
        if formato == "Excel (.xlsx)":
            return generar_excel(datos, tipo_reporte)
        elif formato == "CSV (.csv)":
            return generar_csv(datos, tipo_reporte)
        else:
            return None
    except Exception as e:
        st.error(f"Error generando archivo: {e}")
        return None


def generar_excel(datos, tipo_reporte):
    """Generar archivo Excel con multiples hojas"""
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            if tipo_reporte == "Reporte General":
                for nombre_hoja, df in datos.items():
                    if df is not None and not df.empty:
                        df.to_excel(writer, sheet_name=nombre_hoja.capitalize(), index=False)
            elif tipo_reporte == "Reporte de Estudiantes" and 'estudiantes' in datos:
                datos['estudiantes'].to_excel(writer, sheet_name='Estudiantes', index=False)
            elif tipo_reporte == "Reporte de Calificaciones" and 'calificaciones' in datos:
                datos['calificaciones'].to_excel(writer, sheet_name='Calificaciones', index=False)
            elif tipo_reporte == "Reporte de Factores de Riesgo" and 'factores' in datos:
                datos['factores'].to_excel(writer, sheet_name='Factores_Riesgo', index=False)
            elif tipo_reporte == "Reporte Personalizado" and 'personalizado' in datos:
                datos['personalizado'].to_excel(writer, sheet_name='Reporte_Personalizado', index=False)
        output.seek(0)
        return output.getvalue()
    except Exception as e:
        st.error(f"Error generando Excel: {e}")
        return None


def generar_csv(datos, tipo_reporte):
    """Generar archivo CSV"""
    try:
        if tipo_reporte == "Reporte General":
            if 'estudiantes' in datos and not datos['estudiantes'].empty:
                return datos['estudiantes'].to_csv(index=False).encode('utf-8')
        elif tipo_reporte == "Reporte de Estudiantes" and 'estudiantes' in datos:
            return datos['estudiantes'].to_csv(index=False).encode('utf-8')
        elif tipo_reporte == "Reporte de Calificaciones" and 'calificaciones' in datos:
            return datos['calificaciones'].to_csv(index=False).encode('utf-8')
        elif tipo_reporte == "Reporte de Factores de Riesgo" and 'factores' in datos:
            return datos['factores'].to_csv(index=False).encode('utf-8')
        elif tipo_reporte == "Reporte Personalizado" and 'personalizado' in datos:
            return datos['personalizado'].to_csv(index=False).encode('utf-8')
        return None
    except Exception as e:
        st.error(f"Error generando CSV: {e}")
        return None


def generar_pdf(analytics, datos, tipo_reporte, incluir_graficas=True, incluir_estadisticas=True):
    """Generar reporte PDF profesional"""
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch)
        elements = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            textColor=colors.HexColor('#2C3E50'),
            alignment=1
        )

        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=12,
            textColor=colors.HexColor('#34495E')
        )

        elements.append(Paragraph("SISTEMA DE ANALISIS ACADEMICO - ITT", title_style))
        elements.append(Paragraph(f"Reporte: {tipo_reporte}", subtitle_style))
        elements.append(Paragraph(f"Fecha: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
        elements.append(Spacer(1, 20))

        elements.append(Paragraph("RESUMEN EJECUTIVO", subtitle_style))

        if incluir_estadisticas:
            metricas = _con_reintentos(lambda: analytics.calcular_metricas_principales())
            stats_data = [
                ["Metrica", "Valor"],
                ["Total Estudiantes", str(metricas['total_estudiantes'])],
                ["Total Calificaciones", str(metricas['total_calificaciones'])],
                ["Tasa de Aprobacion", f"{metricas['tasa_aprobacion']}%"],
                ["Tasa de Reprobacion", f"{metricas['tasa_reprobacion']}%"],
                ["Tasa de Desercion", f"{metricas['tasa_desercion']}%"]
            ]
            stats_table = Table(stats_data, colWidths=[2.5*inch, 2*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(stats_table)
            elements.append(Spacer(1, 20))

        elements.append(Paragraph("DATOS DETALLADOS", subtitle_style))

        if tipo_reporte == "Reporte General":
            for seccion, df in datos.items():
                if df is not None and not df.empty:
                    elements.append(Paragraph(f"{seccion.upper()}", styles['Heading3']))
                    df_display = df.head(10)
                    table_data = [df_display.columns.tolist()] + df_display.values.tolist()
                    if table_data:
                        pdf_table = Table(table_data, repeatRows=1)
                        pdf_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, 0), 8),
                            ('FONTSIZE', (0, 1), (-1, -1), 6),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                        ]))
                        elements.append(pdf_table)
                        elements.append(Spacer(1, 10))

        elif tipo_reporte == "Reporte de Estudiantes" and 'estudiantes' in datos:
            df = datos['estudiantes']
            if not df.empty:
                df_display = df.head(15)
                table_data = [df.columns.tolist()] + df_display.values.tolist()
                pdf_table = crear_tabla_pdf(table_data)
                elements.append(pdf_table)

        elif tipo_reporte == "Reporte de Calificaciones" and 'calificaciones' in datos:
            df = datos['calificaciones']
            if not df.empty:
                df_display = df.head(15)
                table_data = [df.columns.tolist()] + df_display.values.tolist()
                pdf_table = crear_tabla_pdf(table_data)
                elements.append(pdf_table)

        elif tipo_reporte == "Reporte de Factores de Riesgo" and 'factores' in datos:
            df = datos['factores']
            if not df.empty:
                df_display = df.head(15)
                table_data = [df.columns.tolist()] + df_display.values.tolist()
                pdf_table = crear_tabla_pdf(table_data)
                elements.append(pdf_table)

        if incluir_graficas:
            try:
                df_cal = _df_cal(analytics)
                if not df_cal.empty:
                    elements.append(Spacer(1, 20))
                    elements.append(Paragraph("ANALISIS GRAFICO", subtitle_style))

                    fig1 = crear_grafica_distribucion(df_cal)
                    if fig1:
                        img_buffer1 = BytesIO()
                        fig1.savefig(img_buffer1, format='png', dpi=150, bbox_inches='tight')
                        img_buffer1.seek(0)
                        elements.append(Paragraph("Distribucion de Calificaciones", styles['Heading4']))
                        elements.append(Image(img_buffer1, width=6*inch, height=3*inch))
                        elements.append(Spacer(1, 10))

                fig2 = crear_grafica_tasas_carrera(analytics)
                if fig2:
                    img_buffer2 = BytesIO()
                    fig2.savefig(img_buffer2, format='png', dpi=150, bbox_inches='tight')
                    img_buffer2.seek(0)
                    elements.append(Paragraph("Tasas por Carrera", styles['Heading4']))
                    elements.append(Image(img_buffer2, width=6*inch, height=3*inch))

            except Exception as e:
                elements.append(Paragraph(f"Nota: No se pudieron incluir las graficas: {str(e)}", styles['Italic']))

        elements.append(Spacer(1, 20))
        elements.append(Paragraph("---", styles['Normal']))
        elements.append(Paragraph(
            "Reporte generado automaticamente por el Sistema de Analisis Academico",
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey)
        ))

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        st.error(f"Error generando PDF: {e}")
        return None


def crear_tabla_pdf(table_data):
    """Crear tabla estilizada para PDF"""
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
    ]))
    return table


def crear_grafica_distribucion(df_calificaciones):
    """Crear grafica de distribucion de calificaciones"""
    try:
        if df_calificaciones.empty or 'calificacion_final' not in df_calificaciones.columns:
            return None
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df_calificaciones['calificacion_final'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=70, color='red', linestyle='--', linewidth=2, label='Limite Aprobacion (70)')
        ax.set_xlabel('Calificacion Final')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribucion de Calificaciones')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    except Exception:
        return None


def crear_grafica_tasas_carrera(analytics):
    """Crear grafica de promedio general por carrera"""
    try:
        from config.constants import CARRERAS

        df_cal = _df_cal(analytics)
        df_est = _df_est(analytics)
        if df_cal.empty or df_est.empty:
            return None

        datos_combinados = df_cal.merge(
            df_est[['id', 'carrera_id']],
            left_on='estudiante_id',
            right_on='id',
            how='left'
        )

        promedios_por_carrera = []
        for carrera_id in datos_combinados['carrera_id'].dropna().unique():
            datos_carrera = datos_combinados[datos_combinados['carrera_id'] == carrera_id]
            if len(datos_carrera) > 0:
                promedio = datos_carrera['calificacion_final'].mean()
                nombre_carrera = CARRERAS.get(int(carrera_id), f'Carrera {carrera_id}')
                promedios_por_carrera.append((nombre_carrera, promedio))

        if not promedios_por_carrera:
            return None

        promedios_por_carrera.sort(key=lambda x: x[1], reverse=True)
        carreras, promedios = zip(*promedios_por_carrera)

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(carreras, promedios, color='mediumseagreen', edgecolor='black')

        for bar, valor in zip(bars, promedios):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{valor:.1f}", va='center', fontsize=8, color='black')

        ax.set_xlabel('Promedio General de Calificacion')
        ax.set_ylabel('Carreras')
        ax.set_title('Promedio General por Carrera')
        ax.set_xlim(0, 100)
        plt.tight_layout()
        return fig

    except Exception:
        return None


def mostrar_vista_previa(datos, tipo_reporte):
    """Mostrar vista previa del reporte"""
    try:
        if tipo_reporte == "Reporte General":
            col1, col2, col3 = st.columns(3)

            with col1:
                if 'estudiantes' in datos and not datos['estudiantes'].empty:
                    st.write("**Estudiantes:**", len(datos['estudiantes']))
                    st.dataframe(datos['estudiantes'].head(3))

            with col2:
                if 'calificaciones' in datos and not datos['calificaciones'].empty:
                    st.write("**Calificaciones:**", len(datos['calificaciones']))
                    st.dataframe(datos['calificaciones'].head(3))

            with col3:
                if 'factores' in datos and not datos['factores'].empty:
                    st.write("**Factores:**", len(datos['factores']))
                    st.dataframe(datos['factores'].head(3))

        elif tipo_reporte == "Reporte de Estudiantes" and 'estudiantes' in datos:
            st.dataframe(datos['estudiantes'].head(10))

        elif tipo_reporte == "Reporte de Calificaciones" and 'calificaciones' in datos:
            st.dataframe(datos['calificaciones'].head(10))

        elif tipo_reporte == "Reporte de Factores de Riesgo" and 'factores' in datos:
            st.dataframe(datos['factores'].head(10))

        elif tipo_reporte == "Reporte Personalizado" and 'personalizado' in datos:
            st.dataframe(datos['personalizado'].head(10))

    except Exception as e:
        st.error(f"Error mostrando vista previa: {e}")
