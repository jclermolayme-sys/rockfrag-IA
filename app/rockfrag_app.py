"""
RockFrag App — Interfaz web tipo SigmaFrag
Ejecutar con: streamlit run app/rockfrag_app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import json
import sys
from pathlib import Path

# Añadir el directorio raíz al path para importar core
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.segmentor import RockFragAnalyzer, RockFragVisualizer

# ─── Configuración de la página ──────────────────────────────────────────────

st.set_page_config(
    page_title="RockFrag AI — Análisis de Fragmentación",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado (estilo industrial/minero)
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stApp { background-color: #0f0f1a; }
    h1 { color: #00d4ff !important; font-family: 'Courier New', monospace; }
    h2, h3 { color: #e0e0e0 !important; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #00d4ff33;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin: 4px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #00d4ff; }
    .metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; }
    .p-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        color: #000;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        padding: 8px 24px;
    }
    .sidebar .stSlider { color: #00d4ff; }
</style>
""", unsafe_allow_html=True)


# ─── Cabecera ─────────────────────────────────────────────────────────────────

st.markdown("""
# ⛏️ RockFrag AI
### Análisis de Fragmentación de Roca por Visión Artificial
*Alternativa open source a SigmaFrag / WipFrag / Split Desktop*
""")

st.markdown("---")


# ─── Panel lateral (parámetros) ───────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Parámetros")
    
    st.subheader("📏 Escala de referencia")
    scale_mode = st.radio(
        "Modo de escala",
        ["Auto-detectar barra", "Ingresar manualmente"],
        help="Auto-detectar busca una barra negra rectangular en la imagen."
    )
    
    scale_ref_cm = st.number_input(
        "Longitud real de la referencia (cm)",
        min_value=1.0, max_value=500.0, value=30.0, step=1.0,
        help="Longitud real del objeto de escala: barra métrica, pelota de 25cm, etc."
    )
    
    manual_px_per_cm = None
    if scale_mode == "Ingresar manualmente":
        manual_px_per_cm = st.number_input(
            "Píxeles por cm (px/cm)",
            min_value=1.0, max_value=1000.0, value=20.0, step=0.5,
        )
    
    st.subheader("🔬 Segmentación")
    use_watershed = st.checkbox(
        "Usar Watershed (más preciso)",
        value=True,
        help="Watershed separa mejor fragmentos que se tocan. Más lento."
    )
    min_frag_px = st.slider(
        "Tamaño mínimo de fragmento (px²)",
        min_value=50, max_value=2000, value=300,
        help="Fragmentos más pequeños se ignoran (ruido)."
    )
    max_frag_ratio = st.slider(
        "Tamaño máximo (% de imagen)",
        min_value=10, max_value=80, value=60,
        help="Fragmentos más grandes que este % se ignoran (bordes/fondo)."
    )
    
    st.markdown("---")
    st.caption("RockFrag AI v1.0 | Open Source")
    st.caption("Equivalente a SigmaFrag/WipFrag")


# ─── Área principal ───────────────────────────────────────────────────────────

col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.subheader("📸 Imagen de entrada")
    
    uploaded = st.file_uploader(
        "Sube una foto de la pila de roca",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Foto post-voladura con objeto de escala visible (barra métrica, pelota, etc.)"
    )

# Imagen a analizar
img_to_analyze = None

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_to_analyze = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if img_to_analyze is not None:
    # Mostrar imagen original
    img_rgb = cv2.cvtColor(img_to_analyze, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Imagen cargada", use_column_width=True)
    
    st.markdown("---")
    
    # ── Botón de análisis ────────────────────────────────────────────────────
    if st.button("🔍 ANALIZAR FRAGMENTACIÓN", use_container_width=True):
        
        with st.spinner("Analizando fragmentos... ⏳"):
            
            try:
                # Guardar imagen temporal
                tmp_path = Path("/tmp/rockfrag_input.jpg")
                cv2.imwrite(str(tmp_path), img_to_analyze)
                
                # Configurar analizador
                analyzer = RockFragAnalyzer(
                    scale_reference_cm=scale_ref_cm,
                    min_fragment_area_px=min_frag_px,
                    max_fragment_ratio=max_frag_ratio / 100,
                )
                
                # Escala manual si aplica
                px_per_cm = manual_px_per_cm if scale_mode == "Ingresar manualmente" else None
                
                # Análisis
                result = analyzer.analyze(
                    str(tmp_path),
                    scale_px_per_cm=px_per_cm,
                    use_watershed=use_watershed,
                )
                
                # ── Métricas principales ────────────────────────────────────
                st.subheader("📊 Resultados")
                
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result.total_fragments}</div>
                        <div class="metric-label">Fragmentos detectados</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result.p50:.1f} cm</div>
                        <div class="metric-label">D50 (mediana)</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result.mean_diameter:.1f} cm</div>
                        <div class="metric-label">Diámetro promedio</div>
                    </div>""", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result.max_diameter:.1f} cm</div>
                        <div class="metric-label">Fragmento máximo</div>
                    </div>""", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # ── P20 / P50 / P80 ────────────────────────────────────────
                st.subheader("📐 Distribución granulométrica")
                
                pc1, pc2, pc3 = st.columns(3)
                with pc1:
                    st.markdown(f"""
                    <div style="background:#ff6b6b22;border:1px solid #ff6b6b;border-radius:8px;padding:16px;text-align:center">
                        <div style="font-size:1.8rem;font-weight:bold;color:#ff6b6b">
                            {result.p20:.1f} cm
                        </div>
                        <div style="color:#aaa;font-size:0.85rem">P20 — 20% del material pasa</div>
                    </div>""", unsafe_allow_html=True)
                with pc2:
                    st.markdown(f"""
                    <div style="background:#ffd93d22;border:1px solid #ffd93d;border-radius:8px;padding:16px;text-align:center">
                        <div style="font-size:1.8rem;font-weight:bold;color:#ffd93d">
                            {result.p50:.1f} cm
                        </div>
                        <div style="color:#aaa;font-size:0.85rem">P50 — Mediana granulométrica</div>
                    </div>""", unsafe_allow_html=True)
                with pc3:
                    st.markdown(f"""
                    <div style="background:#6bcb7722;border:1px solid #6bcb77;border-radius:8px;padding:16px;text-align:center">
                        <div style="font-size:1.8rem;font-weight:bold;color:#6bcb77">
                            {result.p80:.1f} cm
                        </div>
                        <div style="color:#aaa;font-size:0.85rem">P80 — 80% del material pasa</div>
                    </div>""", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # ── Imágenes de salida ──────────────────────────────────────
                col_seg, col_curve = st.columns(2)
                
                with col_seg:
                    st.subheader("🎨 Segmentación")
                    seg_img = RockFragVisualizer.draw_segmentation(img_to_analyze, result)
                    seg_rgb = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
                    st.image(seg_rgb, use_column_width=True, caption="Fragmentos detectados y coloreados por tamaño")
                
                with col_curve:
                    st.subheader("📈 Curva granulométrica")
                    curve_bytes = RockFragVisualizer.plot_grading_curve(result)
                    st.image(curve_bytes, use_column_width=True, caption="Curva acumulada — Distribución de tamaños")
                
                # ── Tabla de fragmentos ─────────────────────────────────────
                with st.expander("📋 Ver tabla completa de fragmentos"):
                    import pandas as pd
                    data = [
                        {
                            "ID": f.id,
                            "Diámetro (cm)": f.diameter_cm,
                            "Área (cm²)": f.area_cm2,
                            "Circularidad": f.circularity,
                        }
                        for f in result.fragments
                    ]
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                
                # ── Descargas ───────────────────────────────────────────────
                st.subheader("💾 Descargar resultados")
                
                dl1, dl2, dl3 = st.columns(3)
                
                with dl1:
                    result_json = json.dumps(
                        RockFragVisualizer.result_to_dict(result),
                        indent=2, ensure_ascii=False
                    )
                    st.download_button(
                        "⬇️ Datos JSON",
                        data=result_json,
                        file_name="rockfrag_resultado.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                
                with dl2:
                    _, seg_encoded = cv2.imencode('.png', seg_img)
                    st.download_button(
                        "⬇️ Imagen segmentada",
                        data=seg_encoded.tobytes(),
                        file_name="rockfrag_segmentacion.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                
                with dl3:
                    st.download_button(
                        "⬇️ Curva granulométrica",
                        data=curve_bytes,
                        file_name="rockfrag_curva.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                
                # CSV de fragmentos
                import pandas as pd
                df = pd.DataFrame([{
                    "id": f.id,
                    "diameter_cm": f.diameter_cm,
                    "area_cm2": f.area_cm2,
                    "circularity": f.circularity,
                } for f in result.fragments])
                
                st.download_button(
                    "⬇️ Tabla CSV de fragmentos",
                    data=df.to_csv(index=False),
                    file_name="rockfrag_fragmentos.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                
                st.success(f"✅ Análisis completado: {result.total_fragments} fragmentos procesados.")
            
            except Exception as e:
                st.error(f"❌ Error en el análisis: {str(e)}")
                st.info("💡 Intenta ajustar los parámetros de segmentación en el panel lateral.")
else:
    # Pantalla de bienvenida
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#555">
        <div style="font-size:4rem">⛏️</div>
        <h2 style="color:#444">Sube una foto para comenzar</h2>
        <br>
        <div style="display:flex;justify-content:center;gap:40px;margin-top:20px">
            <div style="text-align:center">
                <div style="font-size:2rem">📸</div>
                <div style="font-size:0.85rem;color:#666">Foto de pila de roca<br>con barra de escala</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:2rem">🤖</div>
                <div style="font-size:0.85rem;color:#666">IA segmenta<br>cada fragmento</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:2rem">📊</div>
                <div style="font-size:0.85rem;color:#666">Curva P20/P50/P80<br>automática</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
