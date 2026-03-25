"""
RockFrag Core - Motor de segmentación y medición de fragmentos de roca
Equivalente open source de SigmaFrag / WipFrag / Split Desktop
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
from dataclasses import dataclass, field
from typing import Optional
import io
import base64


# ─── Estructuras de datos ────────────────────────────────────────────────────

@dataclass
class Fragment:
    id: int
    area_px: float          # área en píxeles²
    area_cm2: float         # área real en cm²
    diameter_cm: float      # diámetro equivalente (esfera) en cm
    perimeter_px: float
    contour: np.ndarray
    bbox: tuple             # (x, y, w, h)
    circularity: float      # 0-1, qué tan redondo es

@dataclass
class AnalysisResult:
    image_path: str
    scale_px_per_cm: float
    fragments: list = field(default_factory=list)
    total_fragments: int = 0
    p20: float = 0.0        # 20% del material pasa por este tamaño
    p50: float = 0.0        # mediana
    p80: float = 0.0        # 80% pasa
    mean_diameter: float = 0.0
    max_diameter: float = 0.0
    min_diameter: float = 0.0


# ─── Motor principal ─────────────────────────────────────────────────────────

class RockFragAnalyzer:
    """
    Analiza fotos de pilas de roca y calcula distribución granulométrica.
    
    Uso:
        analyzer = RockFragAnalyzer(scale_reference_cm=30.0)
        result = analyzer.analyze("foto_voladura.jpg")
    """

    def __init__(
        self,
        scale_reference_cm: float = 30.0,  # tamaño del objeto de referencia en cm
        min_fragment_area_px: int = 200,   # filtrar ruido muy pequeño
        max_fragment_ratio: float = 0.8,   # ignorar fragmentos que sean >80% imagen
    ):
        self.scale_reference_cm = scale_reference_cm
        self.min_fragment_area_px = min_fragment_area_px
        self.max_fragment_ratio = max_fragment_ratio

    # ── Paso 1: Preprocesamiento ─────────────────────────────────────────────

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Convierte a gris, ecualiza y aplica desenfoque para reducir ruido."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Ecualización adaptativa (CLAHE) para mejorar contraste en fotos de campo
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Suavizado bilateral: preserva bordes mientras elimina ruido de textura
        blurred = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
        return blurred

    # ── Paso 2: Detección de bordes ──────────────────────────────────────────

    def detect_edges(self, preprocessed: np.ndarray) -> np.ndarray:
        """Detecta bordes entre fragmentos con Canny + morfología."""
        # Canny con umbrales auto-calculados (regla de Otsu)
        otsu_thresh, _ = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        edges = cv2.Canny(
            preprocessed,
            threshold1=otsu_thresh * 0.5,
            threshold2=otsu_thresh,
            apertureSize=3,
        )
        # Dilatar bordes para conectar líneas rotas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        return edges_dilated

    # ── Paso 3: Segmentación por watershed ──────────────────────────────────

    def segment_watershed(self, img: np.ndarray, preprocessed: np.ndarray) -> np.ndarray:
        """Usa watershed para separar fragmentos que se tocan."""
        _, binary = cv2.threshold(
            preprocessed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # Morfología para limpiar
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Fondo seguro
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Frente seguro (transformada de distancia)
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Región desconocida
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marcadores para watershed
        _, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(img, markers)
        return markers

    # ── Paso 4: Extracción de contornos y métricas ───────────────────────────

    def extract_fragments(
        self,
        img: np.ndarray,
        markers: np.ndarray,
        scale_px_per_cm: float,
    ) -> list:
        """Extrae cada fragmento y calcula sus métricas reales."""
        h, w = img.shape[:2]
        max_area = h * w * self.max_fragment_ratio
        fragments = []
        fid = 0

        unique_labels = np.unique(markers)
        for label in unique_labels:
            if label <= 1:  # 0=borde watershed, 1=fondo
                continue
            mask = np.uint8(markers == label) * 255
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            area_px = cv2.contourArea(cnt)

            if area_px < self.min_fragment_area_px or area_px > max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area_px / (perimeter ** 2)) if perimeter > 0 else 0
            
            # Convertir a unidades reales
            area_cm2 = area_px / (scale_px_per_cm ** 2)
            diameter_cm = 2 * np.sqrt(area_cm2 / np.pi)  # diámetro de círculo equivalente
            
            bbox = cv2.boundingRect(cnt)
            
            fragments.append(Fragment(
                id=fid,
                area_px=area_px,
                area_cm2=round(area_cm2, 2),
                diameter_cm=round(diameter_cm, 2),
                perimeter_px=round(perimeter, 1),
                contour=cnt,
                bbox=bbox,
                circularity=round(circularity, 3),
            ))
            fid += 1

        return sorted(fragments, key=lambda f: f.diameter_cm)

    # ── Paso 5: Detección de escala de referencia ────────────────────────────

    def detect_scale_bar(self, img: np.ndarray) -> Optional[float]:
        """
        Intenta detectar una barra de escala rectangular en la imagen.
        Busca el rectángulo más oscuro/claro bien definido.
        Retorna píxeles por cm si lo encuentra, None si no.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = img.shape[:2]
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500 or area > w * h * 0.05:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / ch if ch > 0 else 0
            # Buscamos rectángulos alargados horizontales (barras de escala)
            if 3 < aspect < 20:
                candidates.append((cw, cnt))
        
        if candidates:
            # Tomar el más grande
            bar_width_px = max(candidates, key=lambda c: c[0])[0]
            return bar_width_px / self.scale_reference_cm
        return None

    # ── Pipeline completo ────────────────────────────────────────────────────

    def analyze(
        self,
        image_path: str,
        scale_px_per_cm: Optional[float] = None,
        use_watershed: bool = True,
    ) -> AnalysisResult:
        """
        Pipeline completo de análisis.
        
        Args:
            image_path: ruta a la foto de la pila de roca
            scale_px_per_cm: píxeles por cm (si ya lo sabes). Si es None,
                             se intenta auto-detectar la barra de escala.
            use_watershed: usar watershed (mejor) o solo contornos (más rápido)
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar: {image_path}")
        
        h, w = img.shape[:2]
        
        # Determinar escala
        if scale_px_per_cm is None:
            scale_px_per_cm = self.detect_scale_bar(img)
            if scale_px_per_cm is None:
                # Fallback: asumir que la barra de referencia mide 10% del ancho
                scale_px_per_cm = (w * 0.10) / self.scale_reference_cm
        
        preprocessed = self.preprocess(img)
        
        if use_watershed:
            markers = self.segment_watershed(img, preprocessed)
            fragments = self.extract_fragments(img, markers, scale_px_per_cm)
        else:
            # Alternativa rápida: solo contornos con Canny
            edges = self.detect_edges(preprocessed)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            fragments = self._contours_to_fragments(contours, img, scale_px_per_cm)
        
        if not fragments:
            raise ValueError("No se detectaron fragmentos. Verifica la imagen o ajusta parámetros.")
        
        diameters = [f.diameter_cm for f in fragments]
        diameters_sorted = sorted(diameters)
        n = len(diameters_sorted)
        
        result = AnalysisResult(
            image_path=str(image_path),
            scale_px_per_cm=scale_px_per_cm,
            fragments=fragments,
            total_fragments=n,
            p20=float(np.percentile(diameters_sorted, 20)),
            p50=float(np.percentile(diameters_sorted, 50)),
            p80=float(np.percentile(diameters_sorted, 80)),
            mean_diameter=float(np.mean(diameters)),
            max_diameter=float(max(diameters)),
            min_diameter=float(min(diameters)),
        )
        return result

    def _contours_to_fragments(self, contours, img, scale_px_per_cm):
        h, w = img.shape[:2]
        max_area = h * w * self.max_fragment_ratio
        fragments = []
        for i, cnt in enumerate(contours):
            area_px = cv2.contourArea(cnt)
            if area_px < self.min_fragment_area_px or area_px > max_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area_px / (perimeter ** 2)) if perimeter > 0 else 0
            area_cm2 = area_px / (scale_px_per_cm ** 2)
            diameter_cm = 2 * np.sqrt(area_cm2 / np.pi)
            bbox = cv2.boundingRect(cnt)
            fragments.append(Fragment(
                id=i, area_px=area_px, area_cm2=round(area_cm2, 2),
                diameter_cm=round(diameter_cm, 2), perimeter_px=round(perimeter, 1),
                contour=cnt, bbox=bbox, circularity=round(circularity, 3),
            ))
        return sorted(fragments, key=lambda f: f.diameter_cm)


# ─── Visualización ───────────────────────────────────────────────────────────

class RockFragVisualizer:
    """Genera las imágenes de salida y la curva granulométrica."""

    @staticmethod
    def draw_segmentation(img: np.ndarray, result: AnalysisResult) -> np.ndarray:
        """Dibuja los contornos coloreados sobre la imagen original."""
        output = img.copy()
        n = len(result.fragments)
        
        for i, frag in enumerate(result.fragments):
            # Color en gradiente: azul (pequeño) → rojo (grande)
            ratio = i / max(n - 1, 1)
            b = int(255 * (1 - ratio))
            r = int(255 * ratio)
            g = int(255 * (1 - abs(2 * ratio - 1)))
            color = (b, g, r)
            
            cv2.drawContours(output, [frag.contour], -1, color, 2)
            
            # Etiqueta con diámetro en el centro del bounding box
            x, y, w, h = frag.bbox
            cx, cy = x + w // 2, y + h // 2
            label = f"{frag.diameter_cm:.1f}cm"
            cv2.putText(output, label, (cx - 20, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return output

    @staticmethod
    def plot_grading_curve(result: AnalysisResult) -> bytes:
        """Genera la curva granulométrica acumulada como imagen PNG."""
        diameters = sorted([f.diameter_cm for f in result.fragments])
        n = len(diameters)
        cumulative_pct = [(i + 1) / n * 100 for i in range(n)]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        
        ax.plot(diameters, cumulative_pct, color='#00d4ff', linewidth=2.5, label='Curva granulométrica')
        ax.fill_between(diameters, cumulative_pct, alpha=0.15, color='#00d4ff')
        
        # Líneas P20, P50, P80
        for pct, val, color in [
            (20, result.p20, '#ff6b6b'),
            (50, result.p50, '#ffd93d'),
            (80, result.p80, '#6bcb77'),
        ]:
            ax.axhline(pct, color=color, linestyle='--', alpha=0.7, linewidth=1.2)
            ax.axvline(val, color=color, linestyle='--', alpha=0.7, linewidth=1.2)
            ax.annotate(
                f'P{pct} = {val:.1f} cm',
                xy=(val, pct),
                xytext=(val + 0.5, pct + 3),
                color=color,
                fontsize=9,
                fontweight='bold',
            )
        
        ax.set_xlabel('Diámetro equivalente (cm)', color='white')
        ax.set_ylabel('Pasante acumulado (%)', color='white')
        ax.set_title('Curva Granulométrica — Análisis de Fragmentación', color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_color('#444')
        ax.spines['right'].set_color('#444')
        ax.set_ylim(0, 105)
        ax.set_xlim(0, max(diameters) * 1.05)
        ax.grid(True, alpha=0.2, color='#555')
        ax.legend(facecolor='#1a1a2e', labelcolor='white')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        buf.seek(0)
        return buf.read()

    @staticmethod
    def result_to_dict(result: AnalysisResult) -> dict:
        """Convierte el resultado a JSON serializable."""
        return {
            "image_path": result.image_path,
            "total_fragments": result.total_fragments,
            "scale_px_per_cm": round(result.scale_px_per_cm, 2),
            "granulometry": {
                "P20_cm": round(result.p20, 2),
                "P50_cm": round(result.p50, 2),
                "P80_cm": round(result.p80, 2),
                "mean_cm": round(result.mean_diameter, 2),
                "max_cm": round(result.max_diameter, 2),
                "min_cm": round(result.min_diameter, 2),
            },
            "fragments": [
                {
                    "id": f.id,
                    "diameter_cm": f.diameter_cm,
                    "area_cm2": f.area_cm2,
                    "circularity": f.circularity,
                }
                for f in result.fragments
            ],
        }
