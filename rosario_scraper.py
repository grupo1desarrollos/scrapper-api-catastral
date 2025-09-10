#!/usr/bin/env python3
"""
Web scraper para extraer enlaces de PDFs del sitio InfoMapa de Rosario
Busca una dirección y extrae el nombre del archivo PDF de la pestaña Cartobase
"""

import re
import json
import os
import logging
import requests
import io
import math
from io import BytesIO
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
import argparse
try:
    # Hacer pyproj opcional: en Windows/Python 3.13 puede no haber wheel precompilado
    from pyproj import Transformer as _PyprojTransformer
except Exception:
    _PyprojTransformer = None
try:
    from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None
    ImageEnhance = None
    ImageOps = None
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
try:
    from lxml import etree as _lxml_etree
    etree = _lxml_etree
except Exception:
    # Fallback a la biblioteca estándar (suficiente para parsear GML simple)
    import xml.etree.ElementTree as etree

# Las librerías de IA ya no son necesarias después de la limpieza
AI_AVAILABLE = False

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RosarioScraper:
    def __init__(self, driver_path, headless=True, default_crs: str = 'EPSG:22185'):
        if not driver_path:
            raise ValueError("The path to the ChromeDriver executable must be provided.")
        self.driver_path = driver_path
        self.headless = headless
        self.driver = None
        self.default_crs = default_crs
        # Guarda las últimas coordenadas X,Y (Easting, Northing) en SRS local (posiblemente EPSG:22185/5347)
        self.last_point = None  # tuple: (x, y)

    def _setup_driver(self):
        chrome_options = Options()
        if self.headless:
            # Modo headless moderno para Chrome 109+
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        # Apuntar al chrome.exe embebido junto al chromedriver
        try:
            chrome_bin = os.path.join(os.path.dirname(self.driver_path), "chrome.exe")
            if os.path.exists(chrome_bin):
                chrome_options.binary_location = chrome_bin
        except Exception:
            pass
        service = Service(executable_path=self.driver_path)
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

    def _close_driver(self):
        if self.driver:
            self.driver.quit()

    # --- Utilidades adicionales ---
    def _download_pdf(self, pdf_name: str, dest_dir: str = "output/pdfs") -> str:
        import os
        # Usar ruta absoluta basada en el directorio de trabajo
        if not os.path.isabs(dest_dir):
            dest_dir = os.path.join(os.getcwd(), dest_dir)
        
        # Crear directorio con permisos específicos
        try:
            os.makedirs(dest_dir, exist_ok=True, mode=0o775)
            try:
                os.chmod(dest_dir, 0o775)
                logger.info(f"Directorio {dest_dir} creado/actualizado con permisos 775")
            except (PermissionError, OSError):
                # Si no se pueden cambiar permisos, intentar con umask
                old_umask = os.umask(0o002)
                try:
                    os.makedirs(dest_dir, exist_ok=True)
                    logger.info(f"Directorio {dest_dir} creado con umask modificado")
                finally:
                    os.umask(old_umask)
        except Exception as e:
            logger.warning(f"Error creando directorio {dest_dir}: {e}")
            # Intentar crear sin permisos específicos como fallback
            os.makedirs(dest_dir, exist_ok=True)
        url = f"https://infomapa.rosario.gov.ar/emapa/servlets/verArchivo?path=manzanas/{pdf_name}"
        local_path = os.path.join(dest_dir, pdf_name)
        logger.info(f"Descargando PDF a: {local_path}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return local_path

    def _extract_first_polygon_from_gml(self, gml_text: str):
        try:
            root = etree.fromstring(gml_text.encode('latin-1', errors='ignore'))
        except Exception:
            root = etree.fromstring(gml_text.encode('utf-8', errors='ignore'))

        # Try GML3 posList first (namespace-agnostic)
        for node in root.iter():
            if isinstance(node.tag, str) and node.tag.endswith('posList') and node.text:
                try:
                    coords = list(map(float, node.text.strip().split()))
                    pts = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                    if pts:
                        return pts
                except Exception:
                    pass

        # Fallback to GML2 coordinates (x,y pairs separated by spaces), namespace-agnostic
        for node in root.iter():
            if isinstance(node.tag, str) and node.tag.endswith('coordinates') and node.text:
                txt = node.text.strip()
                # Handle both "x,y x,y ..." and "x,y,x,y,..."
                try:
                    if ' ' in txt:
                        pairs = txt.split()
                        pts = []
                        for p in pairs:
                            if ',' in p:
                                xy = p.split(',')
                                if len(xy) >= 2:
                                    pts.append((float(xy[0]), float(xy[1])))
                        if pts:
                            return pts
                    else:
                        nums = list(map(float, txt.split(',')))
                        pts = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
                        if pts:
                            return pts
                except Exception:
                    pass
        return None



    def _wms_getmap_image(self, bbox, srs: str, width=800):
        """Descarga imagen WMS para las capas base y parcelas en el bbox dado."""
        if Image is None:
            logger.warning("Pillow no está disponible; no se puede descargar ni abrir imagen WMS.")
            return None
        minx, miny, maxx, maxy = bbox
        # Mantener aspecto
        span_x = maxx - minx
        span_y = maxy - miny
        height = int(width * (span_y / span_x)) if span_x > 0 else width
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            # Fondo + trazado + etiquetas - Cambiamos el orden para asegurar que las parcelas sean visibles
            "LAYERS": "parcelas,segmentos_de_calle,nombres_de_calles,numeros_de_manzana,Fotos2013",
            "STYLES": "",
            "SRS": srs,
            "BBOX": f"{minx},{miny},{maxx},{maxy}",
            "WIDTH": width,
            "HEIGHT": height,
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE"
        }
        r = requests.get("https://www.rosario.gob.ar/wms/planobase", params=params, timeout=30)
        r.raise_for_status()
        # Guardar la respuesta para depuración
        os.makedirs("output/debug", exist_ok=True)
        with open("output/debug/wms_response.png", "wb") as f:
            f.write(r.content)
        logger.info(f"Imagen WMS descargada y guardada para depuración en output/debug/wms_response.png")
        return Image.open(BytesIO(r.content)).convert("RGBA")

    def _wms_getmap_image_bytes(self, bbox, srs: str, width=800):
        """Descarga imagen WMS (bytes PNG) sin requerir Pillow."""
        minx, miny, maxx, maxy = bbox
        span_x = maxx - minx
        span_y = maxy - miny
        height = int(width * (span_y / span_x)) if span_x > 0 else width
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "LAYERS": "parcelas,segmentos_de_calle,nombres_de_calles,numeros_de_manzana,Fotos2013",
            "STYLES": "",
            "SRS": srs,
            "BBOX": f"{minx},{miny},{maxx},{maxy}",
            "WIDTH": width,
            "HEIGHT": height,
            "FORMAT": "image/png",
            "TRANSPARENT": "FALSE"
        }
        r = requests.get("https://www.rosario.gob.ar/wms/planobase", params=params, timeout=30)
        r.raise_for_status()
        return r.content

    def _annotate_image_with_meta(self, image_path: str, address: str, info: dict):
        """Agrega datos del loteo sobre la imagen existente sin sobrescribirla (si Pillow está disponible)."""
        if Image is None:
            return
        try:
            # Abrir la imagen existente que ya contiene el PDF
            img = Image.open(image_path).convert("RGBA")
            width, height = img.size
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            # Preparar líneas: address + claves disponibles
            meta = info.get('cartobase_meta') or {}
            lines = []
            if address:
                lines.append(str(address))
            rg = meta.get('registro_grafico') or {}
            if rg.get('clave') or info.get('pdf_filename'):
                linea = f"Registro Gráfico: {rg.get('clave') or ''}"
                if info.get('pdf_filename'):
                    linea += f"  ({info['pdf_filename']})"
                lines.append(linea.strip())
            cat = meta.get('catastral') or {}
            if cat.get('clave'):
                lines.append(f"Catastral: {cat.get('clave')}")
            sec = meta.get('seccion') or {}
            if sec.get('clave'):
                lines.append(f"Sección: {sec.get('clave')}")
            # Agregar coordenadas Gauss-Krüger si están disponibles
            if 'gauss_kruger_coords' in info and info['gauss_kruger_coords']:
                lines.append(f"X: {info['gauss_kruger_coords']['x']:.2f}")
                lines.append(f"Y: {info['gauss_kruger_coords']['y']:.2f}")
            # Fuente y medidas
            # Intentar cargar una fuente más legible si está disponible
            font_path = None
            font_size = 20  # Tamaño mayor para mejor legibilidad
            try:
                font_locations = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    "C:\\Windows\\Fonts\\Arial Bold.ttf",
                    "C:\\Windows\\Fonts\\Arial.ttf",
                    "/System/Library/Fonts/Helvetica.ttc"
                ]
                for loc in font_locations:
                    if os.path.exists(loc):
                        font_path = loc
                        break
            except Exception:
                pass
            
            font = ImageFont.truetype(font_path, font_size) if font_path and ImageFont else ImageFont.load_default() if ImageFont else None
            padding = 10
            line_h = (font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) if font else 16
            box_h = padding * 2 + line_h * len(lines)
            box_w = int(width * 0.9)
            box_x0 = padding
            box_y0 = height - box_h - padding
            box_x1 = box_x0 + box_w
            box_y1 = height - padding
            # Caja semitransparente con mayor opacidad para mejor legibilidad
            draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill=(0, 0, 0, 180))
            # Escribir líneas
            y = box_y0 + padding
            for line in lines:
                draw.text((box_x0 + padding, y), line, font=font, fill=(255, 255, 255, 255))
                y += line_h
            # Agregar marca de agua en la esquina superior derecha
            watermark_text = "CATASTRO ROSARIO"
            watermark_font = ImageFont.truetype(font_path, 14) if font_path and ImageFont else font
            if watermark_font:
                # Calcular tamaño del texto para posicionamiento
                bbox = draw.textbbox((0, 0), watermark_text, font=watermark_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                # Posición en esquina superior derecha
                watermark_x = width - text_width - padding * 2
                watermark_y = padding
                # Dibujar texto de marca de agua
                draw.text((watermark_x, watermark_y), watermark_text, font=watermark_font, fill=(255, 255, 255, 200))
            out = Image.alpha_composite(img, overlay)
            out.save(image_path)
        except Exception as e:
            logger.warning(f"No se pudo anotar la imagen con metadatos: {e}")

    def _overlay_pdf_inset(self, image_path: str, pdf_path: str, inset_max_width_ratio: float = 0.45):
        """Inserta una miniatura de la primera página del PDF en la esquina inferior derecha del recorte.
        Requiere Pillow y PyMuPDF. Si alguna falta, se omite silenciosamente.
        """
        if Image is None or fitz is None:
            return
        try:
            base = Image.open(image_path).convert("RGBA")
            bw, bh = base.size
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                return
            page = doc[0]
            # Renderizar la página a pixmap con zoom moderado
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            mode = "RGBA" if pix.alpha else "RGB"
            pdf_img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            # Redimensionar manteniendo proporción
            max_w = int(bw * inset_max_width_ratio)
            scale = min(max_w / pdf_img.width, 1.0)
            new_w = int(pdf_img.width * scale)
            new_h = int(pdf_img.height * scale)
            pdf_img = pdf_img.resize((new_w, new_h), Image.LANCZOS).convert("RGBA")
            # Posicionar en esquina inferior derecha con margen y borde
            margin = 12
            x0 = bw - new_w - margin
            y0 = bh - new_h - margin
            # Borde blanco semitransparente
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            draw.rectangle([x0 - 4, y0 - 4, x0 + new_w + 4, y0 + new_h + 4], outline=(255, 255, 255, 220), width=3)
            base.alpha_composite(overlay)
            base.alpha_composite(pdf_img, dest=(x0, y0))
            base.save(image_path)
        except Exception as e:
            logger.warning(f"No se pudo insertar inset del PDF: {e}")

    def _extract_text_with_coordinates_from_pdf(self, pdf_path: str) -> list:
        """Extrae texto del PDF con coordenadas exactas para mantener posiciones originales."""
        if not pdf_path or fitz is None:
            return []
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                return []
            page = doc[0]
            
            # Obtener dimensiones de la página
            page_rect = page.rect
            
            # Extraer texto con coordenadas usando get_text("dict")
            text_dict = page.get_text("dict")
            text_items = []
            
            for block in text_dict["blocks"]:
                if "lines" in block:  # Es un bloque de texto
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text and (text.replace('.', '').replace(',', '').isdigit() or 
                                       any(c.isdigit() for c in text) or 
                                       text.startswith('PH') or 
                                       text.startswith('SP')):  # Solo números, PH, SP
                                bbox = span["bbox"]  # [x0, y0, x1, y1]
                                font_size = span["size"]
                                
                                # Calcular centro del texto
                                center_x = (bbox[0] + bbox[2]) / 2
                                center_y = (bbox[1] + bbox[3]) / 2
                                
                                text_items.append({
                                    "text": text,
                                    "x": center_x,
                                    "y": center_y,
                                    "bbox": bbox,
                                    "font_size": font_size,
                                    "page_width": page_rect.width,
                                    "page_height": page_rect.height
                                })
            
            return text_items
        except Exception as e:
            logger.info(f"No se pudieron extraer coordenadas del PDF: {e}")
            return []

    def _centroid_of_polygon(self, pts):
        """Centroid de un polígono no auto-intersecado. pts: lista [(x,y), ...]."""
        if not pts:
            return (0.0, 0.0)
        # Asegura cierre
        xys = list(pts)
        if xys[0] != xys[-1]:
            xys.append(xys[0])
        a = 0.0
        cx = 0.0
        cy = 0.0
        for i in range(len(xys) - 1):
            x0, y0 = xys[i]
            x1, y1 = xys[i + 1]
            cross = x0 * y1 - x1 * y0
            a += cross
            cx += (x0 + x1) * cross
            cy += (y0 + y1) * cross
        if abs(a) < 1e-9:
            # Fallback: promedio simple
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        a *= 0.5
        cx /= (6.0 * a)
        cy /= (6.0 * a)
        return (cx, cy)

    def _extract_pdf_block_contour(self, page, zoom_factor=1.0):
        """Extrae el contorno del bloque catastral filtrando rectángulos por proximidad al centro.
        Mejorado para detectar mejor el lote específico en lugar de toda la manzana."""
        paths = page.get_drawings()
        
        # Extraer todos los rectángulos del PDF
        rects = [fitz.Rect(p['rect']) for p in paths if p['type'] == 's']

        if len(rects) < 3: # Necesitamos al menos borde, pie de página y una parcela
            return None

        # Descartar el rectángulo más grande (borde de página) y el segundo más grande (posiblemente la manzana completa)
        rects.sort(key=lambda r: r.width * r.height, reverse=True)
        content_rects = rects[2:] if len(rects) > 2 else rects[1:]

        if not content_rects:
            return None

        # Extraer texto para buscar números de lote
        text_dict = page.get_text("dict")
        lote_numbers = []
        
        # Buscar números de lote en el texto
        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    # Buscar números que podrían ser lotes
                    if text.isdigit() or (len(text) <= 3 and any(c.isdigit() for c in text)):
                        lote_numbers.append({
                            'text': text,
                            'bbox': span.get("bbox"),
                            'center': ((span.get("bbox")[0] + span.get("bbox")[2])/2, 
                                      (span.get("bbox")[1] + span.get("bbox")[3])/2)
                        })
        
        # Si encontramos números de lote, usarlos para identificar el rectángulo del lote
        if lote_numbers:
            # Calcular el centro de la página
            page_width = page.rect.width
            page_height = page.rect.height
            page_center = (page_width/2, page_height/2)
            
            # Encontrar el número de lote más cercano al centro
            lote_numbers.sort(key=lambda x: ((x['center'][0] - page_center[0])**2 + 
                                           (x['center'][1] - page_center[1])**2))
            target_lote = lote_numbers[0]
            
            # Encontrar el rectángulo que contiene este número de lote
            target_rect = None
            min_distance = float('inf')
            
            for rect in content_rects:
                # Verificar si el número está dentro del rectángulo
                if (rect.x0 <= target_lote['center'][0] <= rect.x1 and 
                    rect.y0 <= target_lote['center'][1] <= rect.y1):
                    target_rect = rect
                    break
                else:
                    # Si no está dentro, calcular la distancia al centro del rectángulo
                    rect_center = ((rect.x0 + rect.x1)/2, (rect.y0 + rect.y1)/2)
                    dist = ((rect_center[0] - target_lote['center'][0])**2 + 
                           (rect_center[1] - target_lote['center'][1])**2)**0.5
                    if dist < min_distance:
                        min_distance = dist
                        target_rect = rect
            
            if target_rect:
                # Usar este rectángulo como contorno final
                final_contour = target_rect
            else:
                # Fallback al método original
                # Calcular el centroide de todos los rectángulos de contenido
                center_x = sum((r.x0 + r.x1) / 2 for r in content_rects) / len(content_rects)
                center_y = sum((r.y0 + r.y1) / 2 for r in content_rects) / len(content_rects)
                
                # Filtrar rectángulos cercanos al centro
                avg_width = sum(r.width for r in content_rects) / len(content_rects)
                avg_height = sum(r.height for r in content_rects) / len(content_rects)
                max_dist = 1.5 * (avg_width + avg_height) # Umbral heurístico
                
                central_rects = []
                for r in content_rects:
                    rect_center_x = (r.x0 + r.x1) / 2
                    rect_center_y = (r.y0 + r.y1) / 2
                    dist = ((rect_center_x - center_x)**2 + (rect_center_y - center_y)**2)**0.5
                    if dist < max_dist:
                        central_rects.append(r)
                
                if not central_rects:
                    return None
                
                # Calcular el bounding box que une solo los rectángulos centrales
                final_contour = fitz.Rect(central_rects[0])
                for rect in central_rects[1:]:
                    final_contour.include_rect(rect)
        else:
            # Método original si no se encuentran números de lote
            # Calcular el centroide de todos los rectángulos de contenido
            center_x = sum((r.x0 + r.x1) / 2 for r in content_rects) / len(content_rects)
            center_y = sum((r.y0 + r.y1) / 2 for r in content_rects) / len(content_rects)
            
            # Filtrar rectángulos cercanos al centro
            avg_width = sum(r.width for r in content_rects) / len(content_rects)
            avg_height = sum(r.height for r in content_rects) / len(content_rects)
            max_dist = 1.5 * (avg_width + avg_height) # Umbral heurístico
            
            central_rects = []
            for r in content_rects:
                rect_center_x = (r.x0 + r.x1) / 2
                rect_center_y = (r.y0 + r.y1) / 2
                dist = ((rect_center_x - center_x)**2 + (rect_center_y - center_y)**2)**0.5
                if dist < max_dist:
                    central_rects.append(r)
            
            if not central_rects:
                return None
            
            # Calcular el bounding box que une solo los rectángulos centrales
            final_contour = fitz.Rect(central_rects[0])
            for rect in central_rects[1:]:
                final_contour.include_rect(rect)
        
        # Devolver las 4 esquinas del contorno final
        return {
            'top_left': (final_contour.tl.x * zoom_factor, final_contour.tl.y * zoom_factor),
            'top_right': (final_contour.tr.x * zoom_factor, final_contour.tr.y * zoom_factor),
            'bottom_right': (final_contour.br.x * zoom_factor, final_contour.br.y * zoom_factor),
            'bottom_left': (final_contour.bl.x * zoom_factor, final_contour.bl.y * zoom_factor),
        }

    def _detect_map_lines_and_geometry(self, pdf_path, zoom_factor=8.0):
        """Detecta líneas y elementos geométricos del mapa en el PDF usando análisis de imagen.
        Retorna información sobre líneas detectadas, intersecciones y elementos geométricos."""
        if fitz is None or cv2 is None:
            logger.warning("PyMuPDF o OpenCV no disponibles para detección de líneas")
            return None
            
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                return None
                
            page = doc[0]
            mat = fitz.Matrix(zoom_factor, zoom_factor)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convertir a imagen OpenCV
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None
                
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Aplicar filtros para mejorar la detección de líneas
            # Reducir ruido
            denoised = cv2.medianBlur(gray, 3)
            
            # Detectar bordes usando Canny
            edges = cv2.Canny(denoised, 50, 150, apertureSize=3)
            
            # Detectar líneas usando transformada de Hough
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            detected_lines = []
            if lines is not None and len(lines) > 0:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Calcular longitud y ángulo de la línea
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    
                    detected_lines.append({
                        'start': (x1/zoom_factor, y1/zoom_factor),
                        'end': (x2/zoom_factor, y2/zoom_factor),
                        'length': length/zoom_factor,
                        'angle': angle,
                        'is_horizontal': abs(angle) < 15 or abs(angle) > 165,
                        'is_vertical': abs(abs(angle) - 90) < 15
                    })
            
            # Detectar contornos para elementos geométricos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            geometric_elements = []
            for contour in contours:
                # Filtrar contornos muy pequeños
                area = cv2.contourArea(contour)
                if area < 100:  # Área mínima ajustable
                    continue
                    
                # Aproximar el contorno a un polígono
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Calcular bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                geometric_elements.append({
                    'contour': [(pt[0][0]/zoom_factor, pt[0][1]/zoom_factor) for pt in approx],
                    'area': area/(zoom_factor**2),
                    'bbox': (x/zoom_factor, y/zoom_factor, w/zoom_factor, h/zoom_factor),
                    'vertices': len(approx),
                    'is_rectangle': len(approx) == 4,
                    'aspect_ratio': w/h if h > 0 else 0
                })
            
            # Detectar intersecciones de líneas principales
            intersections = self._find_line_intersections(detected_lines)
            
            # Guardar imagen de debug con líneas detectadas
            debug_img = img.copy()
            if lines is not None and len(lines) > 0:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Dibujar intersecciones
            for intersection in intersections:
                x, y = int(intersection['point'][0] * zoom_factor), int(intersection['point'][1] * zoom_factor)
                cv2.circle(debug_img, (x, y), 5, (255, 0, 0), -1)
            
            # Guardar imagen de debug
            os.makedirs("output/debug", exist_ok=True)
            cv2.imwrite("output/debug/detected_lines.png", debug_img)
            logger.info(f"Detectadas {len(detected_lines)} líneas y {len(geometric_elements)} elementos geométricos")
            
            return {
                'success': True,
                'lines': detected_lines,
                'geometric_elements': geometric_elements,
                'intersections': intersections,
                'image_size': (img.shape[1]/zoom_factor, img.shape[0]/zoom_factor),
                'image_center': [img.shape[1] // 2, img.shape[0] // 2],  # [x, y] centro de la imagen
                'original_image': img,  # Agregar la imagen original para el recorte
                'zoom_factor': zoom_factor
            }
            
        except Exception as e:
            logger.error(f"Error en detección de líneas del mapa: {e}")
            return {'success': False, 'error': str(e)}
    
    def _crop_image_based_on_detected_lines(self, map_analysis, address=None):
        """Recorta la imagen basándose en las líneas detectadas y elementos geométricos.
        Retorna la imagen recortada final."""
        if not map_analysis or 'original_image' not in map_analysis:
            logger.error("No hay datos de análisis del mapa o imagen original")
            return {'success': False, 'error': 'No hay datos de análisis del mapa o imagen original'}
            
        try:
            img = map_analysis['original_image']
            lines = map_analysis.get('lines', [])
            geometric_elements = map_analysis.get('geometric_elements', [])
            intersections = map_analysis.get('intersections', [])
            zoom_factor = map_analysis.get('zoom_factor', 8.0)
            
            logger.info(f"Procesando imagen de {img.shape[1]}x{img.shape[0]} píxeles")
            logger.info(f"Elementos disponibles: {len(lines)} líneas, {len(geometric_elements)} elementos geométricos, {len(intersections)} intersecciones")
            
            # Determinar área de interés basada en densidad de líneas y elementos
            crop_region = self._determine_crop_region_from_lines(img, lines, geometric_elements, intersections, zoom_factor)
            
            if not crop_region:
                logger.warning("No se pudo determinar región de recorte, usando centro de la imagen")
                # Fallback: recortar desde el centro
                h, w = img.shape[:2]
                crop_size = min(w, h) // 2
                x1 = max(0, w//2 - crop_size//2)
                y1 = max(0, h//2 - crop_size//2)
                x2 = min(w, x1 + crop_size)
                y2 = min(h, y1 + crop_size)
                crop_region = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            
            # Realizar el recorte
            x1, y1, x2, y2 = crop_region['x1'], crop_region['y1'], crop_region['x2'], crop_region['y2']
            cropped_img = img[y1:y2, x1:x2]
            
            logger.info(f"Imagen recortada: {cropped_img.shape[1]}x{cropped_img.shape[0]} píxeles")
            logger.info(f"Región de recorte: ({x1}, {y1}) a ({x2}, {y2})")
            
            # Guardar imagen recortada
            os.makedirs("output/crops", exist_ok=True)
            if address:
                safe_addr = address.replace(' ', '_').replace('/', '_')
                crop_filename = f"output/crops/crop_{safe_addr}.png"
            else:
                crop_filename = "output/crops/crop_final.png"
                
            cv2.imwrite(crop_filename, cropped_img)
            logger.info(f"Imagen recortada guardada en: {crop_filename}")
            
            # Crear imagen de visualización con la región de recorte marcada
            self._create_crop_visualization(img, crop_region, lines, geometric_elements, intersections, address)
            
            # Convertir a PIL Image para compatibilidad
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            
            return {
                'success': True,
                'image': cropped_pil,
                'crop_path': crop_filename,
                'crop_region': crop_region,
                'original_size': (img.shape[1], img.shape[0]),
                'cropped_size': (cropped_img.shape[1], cropped_img.shape[0])
            }
            
        except Exception as e:
            logger.error(f"Error al recortar imagen basada en líneas detectadas: {e}")
            return {'success': False, 'error': str(e)}
    
    def _determine_crop_region_from_lines(self, img, lines, geometric_elements, intersections, zoom_factor):
        """Determina la región de recorte basándose en la densidad de líneas y elementos geométricos."""
        try:
            h, w = img.shape[:2]
            
            # Si hay elementos geométricos (parcelas), usar el más central y grande
            if geometric_elements:
                # Filtrar elementos por tamaño mínimo
                valid_elements = [elem for elem in geometric_elements if elem['area'] > 50]
                
                if valid_elements:
                    # Encontrar el elemento más central
                    center_x, center_y = w // 2, h // 2
                    best_element = None
                    min_distance = float('inf')
                    
                    for elem in valid_elements:
                        bbox = elem['bbox']
                        elem_center_x = (bbox[0] + bbox[2]/2) * zoom_factor
                        elem_center_y = (bbox[1] + bbox[3]/2) * zoom_factor
                        
                        distance = math.sqrt((elem_center_x - center_x)**2 + (elem_center_y - center_y)**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_element = elem
                    
                    if best_element:
                        bbox = best_element['bbox']
                        # Expandir el bbox del elemento para incluir contexto
                        margin = 100  # píxeles de margen
                        x1 = max(0, int(bbox[0] * zoom_factor) - margin)
                        y1 = max(0, int(bbox[1] * zoom_factor) - margin)
                        x2 = min(w, int((bbox[0] + bbox[2]) * zoom_factor) + margin)
                        y2 = min(h, int((bbox[1] + bbox[3]) * zoom_factor) + margin)
                        
                        logger.info(f"Región de recorte basada en elemento geométrico central")
                        return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            
            # Si hay intersecciones de alta confianza, usar la más central
            if intersections:
                high_conf_intersections = [inter for inter in intersections if inter['confidence'] > 0.5]
                
                if high_conf_intersections:
                    center_x, center_y = w // (2 * zoom_factor), h // (2 * zoom_factor)
                    best_intersection = None
                    min_distance = float('inf')
                    
                    for inter in high_conf_intersections:
                        point = inter['point']
                        distance = math.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_intersection = inter
                    
                    if best_intersection:
                        point = best_intersection['point']
                        # Crear región de recorte centrada en la intersección
                        crop_size = 300  # píxeles
                        center_x = int(point[0] * zoom_factor)
                        center_y = int(point[1] * zoom_factor)
                        
                        x1 = max(0, center_x - crop_size // 2)
                        y1 = max(0, center_y - crop_size // 2)
                        x2 = min(w, x1 + crop_size)
                        y2 = min(h, y1 + crop_size)
                        
                        logger.info(f"Región de recorte basada en intersección de alta confianza")
                        return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            
            # Fallback: usar densidad de líneas
            if lines:
                # Crear mapa de densidad de líneas
                density_map = np.zeros((h // 10, w // 10), dtype=np.float32)
                
                for line in lines:
                    start = line['start']
                    end = line['end']
                    
                    # Convertir coordenadas a escala de densidad
                    x1 = int(start[0] * zoom_factor // 10)
                    y1 = int(start[1] * zoom_factor // 10)
                    x2 = int(end[0] * zoom_factor // 10)
                    y2 = int(end[1] * zoom_factor // 10)
                    
                    # Dibujar línea en mapa de densidad
                    if 0 <= x1 < density_map.shape[1] and 0 <= y1 < density_map.shape[0] and \
                       0 <= x2 < density_map.shape[1] and 0 <= y2 < density_map.shape[0]:
                        cv2.line(density_map, (x1, y1), (x2, y2), 1.0, 1)
                
                # Encontrar región de mayor densidad
                kernel = np.ones((5, 5), np.float32) / 25
                density_smooth = cv2.filter2D(density_map, -1, kernel)
                
                # Encontrar punto de máxima densidad
                max_loc = np.unravel_index(np.argmax(density_smooth), density_smooth.shape)
                max_y, max_x = max_loc
                
                # Convertir de vuelta a coordenadas de imagen
                center_x = max_x * 10
                center_y = max_y * 10
                
                # Crear región de recorte
                crop_size = 400  # píxeles
                x1 = max(0, center_x - crop_size // 2)
                y1 = max(0, center_y - crop_size // 2)
                x2 = min(w, x1 + crop_size)
                y2 = min(h, y1 + crop_size)
                
                logger.info(f"Región de recorte basada en densidad de líneas")
                return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            
            return None
            
        except Exception as e:
            logger.error(f"Error al determinar región de recorte: {e}")
            return None
    

    
    def _find_line_intersections(self, lines):
        """Encuentra intersecciones entre líneas detectadas."""
        intersections = []
        
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines[i+1:], i+1):
                intersection = self._line_intersection(line1, line2)
                if intersection:
                    intersections.append({
                        'point': intersection,
                        'line1_idx': i,
                        'line2_idx': j,
                        'confidence': self._calculate_intersection_confidence(line1, line2, intersection)
                    })
        
        return intersections
    
    def _line_intersection(self, line1, line2):
        """Calcula la intersección entre dos líneas."""
        x1, y1 = line1['start']
        x2, y2 = line1['end']
        x3, y3 = line2['start']
        x4, y4 = line2['end']
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None  # Líneas paralelas
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
        
        # Verificar si la intersección está dentro de ambos segmentos
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t*(x2-x1)
            y = y1 + t*(y2-y1)
            return (x, y)
        
        return None
    
    def _calculate_intersection_confidence(self, line1, line2, intersection):
        """Calcula la confianza de una intersección basada en las características de las líneas."""
        # Ángulo entre líneas (mejor si son perpendiculares)
        angle_diff = abs(line1['angle'] - line2['angle'])
        angle_diff = min(angle_diff, 180 - angle_diff)  # Normalizar a 0-90
        angle_score = abs(angle_diff - 90) / 90  # Mejor si es cercano a 90 grados
        
        # Longitud de las líneas (líneas más largas son más confiables)
        length_score = min(line1['length'], line2['length']) / max(line1['length'], line2['length'])
        
        return (1 - angle_score) * length_score
    
    def _correlate_pdf_coordinates_with_lines(self, pdf_path, map_analysis, web_coords):
        """Correlaciona las coordenadas geográficas del sitio web con las líneas detectadas para mejorar la geolocalización."""
        if not map_analysis or not web_coords or not self.last_point:
            return None
            
        try:
            # Obtener coordenadas del sitio web
            if not web_coords or 'x' not in web_coords or 'y' not in web_coords:
                logger.warning("No se encontraron coordenadas válidas del sitio web")
                return None
                
            # Coordenadas del punto de interés
            target_x, target_y = web_coords['x'], web_coords['y']
            
            # Extraer líneas del mapa web para comparación
            web_map_analysis = self._analyze_web_map_lines()
            
            # Buscar correspondencias entre líneas del PDF y del mapa web
            line_correspondences = self._find_line_correspondences(map_analysis, web_map_analysis)
            
            if line_correspondences and len(line_correspondences) >= 2:
                # Calcular transformación basada en correspondencias de líneas
                transformation = self._calculate_line_based_transformation(line_correspondences)
                
                # Calcular el factor de escala mejorado
                scale_factor = self._calculate_enhanced_scale_factor(line_correspondences, web_coords)
                
                logger.info(f"Correlación mejorada encontrada con {len(line_correspondences)} correspondencias")
                
                return {
                    'line_correspondences': line_correspondences,
                    'transformation': transformation,
                    'scale_factor': scale_factor,
                    'web_coords': web_coords,
                    'target_point': (target_x, target_y),
                    'confidence': min(1.0, len(line_correspondences) * 0.3)
                }
            else:
                # Fallback al método original
                closest_intersection = None
                min_distance = float('inf')
                
                for intersection in map_analysis.get('intersections', []):
                    dist = math.sqrt(
                        (intersection['point'][0] - map_analysis.get('image_center', [0, 0])[0])**2 +
                        (intersection['point'][1] - map_analysis.get('image_center', [0, 0])[1])**2
                    )
                    
                    if dist < min_distance and intersection['confidence'] > 0.7:
                        min_distance = dist
                        closest_intersection = intersection
                
                if closest_intersection:
                    scale_factor = self._calculate_geographic_scale_factor(web_coords, map_analysis)
                    
                    return {
                        'target_intersection': closest_intersection,
                        'scale_factor': scale_factor,
                        'web_coords': web_coords,
                        'target_point': (target_x, target_y),
                        'confidence': closest_intersection['confidence']
                    }
                else:
                    logger.warning("No se encontraron intersecciones de alta confianza")
                    return None
                
        except Exception as e:
            logger.error(f"Error correlacionando coordenadas del PDF con líneas: {e}")
            return None
    
    def _analyze_web_map_lines(self):
        """Analiza las líneas del mapa web actual para encontrar correspondencias."""
        try:
            # Obtener la imagen WMS actual para análisis
            if hasattr(self, 'last_wms_image_path') and os.path.exists(self.last_wms_image_path):
                return self._detect_map_lines_and_geometry(self.last_wms_image_path, zoom_factor=4.0)
            else:
                logger.warning("No se encontró imagen WMS para análisis de líneas")
                return None
        except Exception as e:
            logger.error(f"Error analizando líneas del mapa web: {e}")
            return None
    
    def _find_line_correspondences(self, pdf_analysis, web_analysis):
        """Encuentra correspondencias entre líneas del PDF y del mapa web."""
        if not pdf_analysis or not web_analysis:
            return []
            
        try:
            pdf_lines = pdf_analysis.get('lines', [])
            web_lines = web_analysis.get('lines', [])
            correspondences = []
            
            # Buscar líneas similares por ángulo y longitud relativa
            for pdf_line in pdf_lines:
                pdf_angle = pdf_line.get('angle', 0)
                pdf_length = pdf_line.get('length', 0)
                
                best_match = None
                best_score = 0
                
                for web_line in web_lines:
                    web_angle = web_line.get('angle', 0)
                    web_length = web_line.get('length', 0)
                    
                    # Calcular similitud de ángulo (normalizado)
                    angle_diff = abs(pdf_angle - web_angle)
                    angle_diff = min(angle_diff, 180 - angle_diff)  # Considerar líneas opuestas
                    angle_similarity = 1.0 - (angle_diff / 90.0)
                    
                    # Calcular similitud de longitud relativa
                    if pdf_length > 0 and web_length > 0:
                        length_ratio = min(pdf_length, web_length) / max(pdf_length, web_length)
                    else:
                        length_ratio = 0
                    
                    # Puntuación combinada
                    score = (angle_similarity * 0.7 + length_ratio * 0.3)
                    
                    if score > best_score and score > 0.6:  # Umbral de similitud
                        best_score = score
                        best_match = web_line
                
                if best_match:
                    correspondences.append({
                        'pdf_line': pdf_line,
                        'web_line': best_match,
                        'confidence': best_score
                    })
            
            # Ordenar por confianza y tomar las mejores
            correspondences.sort(key=lambda x: x['confidence'], reverse=True)
            return correspondences[:6]  # Máximo 6 correspondencias
            
        except Exception as e:
            logger.error(f"Error encontrando correspondencias de líneas: {e}")
            return []
    
    def _calculate_line_based_transformation(self, correspondences):
        """Calcula una transformación basada en correspondencias de líneas."""
        try:
            if len(correspondences) < 2:
                return None
                
            # Extraer puntos de correspondencia
            pdf_points = []
            web_points = []
            
            for corr in correspondences[:4]:  # Usar máximo 4 correspondencias
                pdf_line = corr['pdf_line']
                web_line = corr['web_line']
                
                # Usar puntos medios de las líneas
                pdf_mid = (
                    (pdf_line['start'][0] + pdf_line['end'][0]) / 2,
                    (pdf_line['start'][1] + pdf_line['end'][1]) / 2
                )
                web_mid = (
                    (web_line['start'][0] + web_line['end'][0]) / 2,
                    (web_line['start'][1] + web_line['end'][1]) / 2
                )
                
                pdf_points.append(pdf_mid)
                web_points.append(web_mid)
            
            if len(pdf_points) >= 2:
                # Calcular transformación afín simple
                pdf_array = np.array(pdf_points, dtype=np.float32)
                web_array = np.array(web_points, dtype=np.float32)
                
                # Usar los primeros dos puntos para calcular escala y rotación
                if len(pdf_points) >= 2:
                    return cv2.getAffineTransform(pdf_array[:3], web_array[:3]) if len(pdf_points) >= 3 else None
                    
            return None
            
        except Exception as e:
            logger.error(f"Error calculando transformación basada en líneas: {e}")
            return None
    
    def _calculate_enhanced_scale_factor(self, correspondences, web_coords):
        """Calcula un factor de escala mejorado basado en correspondencias de líneas."""
        try:
            if not correspondences:
                return 1.0
                
            scale_factors = []
            
            for corr in correspondences:
                pdf_line = corr['pdf_line']
                web_line = corr['web_line']
                
                pdf_length = pdf_line.get('length', 1)
                web_length = web_line.get('length', 1)
                
                if pdf_length > 0 and web_length > 0:
                    scale_factor = web_length / pdf_length
                    scale_factors.append(scale_factor)
            
            if scale_factors:
                # Usar la mediana para robustez
                scale_factors.sort()
                median_scale = scale_factors[len(scale_factors) // 2]
                logger.info(f"Factor de escala calculado: {median_scale:.4f} (basado en {len(scale_factors)} líneas)")
                return median_scale
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculando factor de escala mejorado: {e}")
            return 1.0
    
    def _ai_feature_alignment(self, pdf_img, web_img):
        """Utiliza IA para detectar características y alinear imágenes automáticamente."""
        try:
            logger.warning("Librerías de IA no disponibles, usando método tradicional")
            return self._traditional_alignment(pdf_img, web_img)
            
            # Convertir imágenes a formato OpenCV si es necesario
            if isinstance(pdf_img, Image.Image):
                pdf_cv = cv2.cvtColor(np.array(pdf_img), cv2.COLOR_RGB2BGR)
            else:
                pdf_cv = pdf_img
                
            if isinstance(web_img, Image.Image):
                web_cv = cv2.cvtColor(np.array(web_img), cv2.COLOR_RGB2BGR)
            else:
                web_cv = web_img
            
            # Convertir a escala de grises
            pdf_gray = cv2.cvtColor(pdf_cv, cv2.COLOR_BGR2GRAY)
            web_gray = cv2.cvtColor(web_cv, cv2.COLOR_BGR2GRAY)
            
            # Detectar características usando ORB (más robusto que SIFT)
            orb = cv2.ORB_create(nfeatures=5000)
            
            # Encontrar keypoints y descriptores
            kp1, des1 = orb.detectAndCompute(pdf_gray, None)
            kp2, des2 = orb.detectAndCompute(web_gray, None)
            
            if des1 is None or des2 is None:
                logger.warning("No se pudieron detectar características suficientes")
                return None
            
            # Matcher FLANN para mejor rendimiento
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                              table_number=6,
                              key_size=12,
                              multi_probe_level=1)
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Filtrar buenos matches usando ratio test de Lowe
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 10:
                logger.warning(f"Pocas correspondencias encontradas: {len(good_matches)}")
                return None
            
            # Extraer puntos correspondientes
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Calcular homografía usando RANSAC
            homography, mask = cv2.findHomography(src_pts, dst_pts, 
                                                cv2.RANSAC, 5.0)
            
            if homography is None:
                logger.warning("No se pudo calcular la homografía")
                return None
            
            # Calcular métricas de calidad
            inliers = np.sum(mask)
            inlier_ratio = inliers / len(good_matches)
            
            logger.info(f"Alineación IA: {inliers}/{len(good_matches)} inliers ({inlier_ratio:.2%})")
            
            return {
                'homography': homography,
                'inliers': int(inliers),
                'total_matches': len(good_matches),
                'inlier_ratio': inlier_ratio,
                'quality_score': inlier_ratio * min(1.0, inliers / 50)
            }
            
        except Exception as e:
            logger.error(f"Error en alineación con IA: {e}")
            return None
    
    def _apply_ai_transformation(self, pdf_img, transformation_data):
        """Aplica la transformación calculada por IA a la imagen PDF."""
        try:
            if transformation_data is None:
                return pdf_img
            
            homography = transformation_data['homography']
            
            # Convertir PIL a OpenCV si es necesario
            if isinstance(pdf_img, Image.Image):
                pdf_cv = cv2.cvtColor(np.array(pdf_img), cv2.COLOR_RGB2BGR)
            else:
                pdf_cv = pdf_img
            
            # Aplicar transformación
            h, w = pdf_cv.shape[:2]
            transformed = cv2.warpPerspective(pdf_cv, homography, (w, h))
            
            # Convertir de vuelta a PIL
            transformed_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
            return Image.fromarray(transformed_rgb)
            
        except Exception as e:
            logger.error(f"Error aplicando transformación IA: {e}")
            return pdf_img
    
    def _traditional_alignment(self, pdf_img, web_img):
        """Método de alineación tradicional como respaldo."""
        try:
            # Implementación básica usando correlación de plantillas
            if isinstance(pdf_img, Image.Image):
                pdf_cv = cv2.cvtColor(np.array(pdf_img), cv2.COLOR_RGB2GRAY)
            else:
                pdf_cv = cv2.cvtColor(pdf_img, cv2.COLOR_BGR2GRAY)
                
            if isinstance(web_img, Image.Image):
                web_cv = cv2.cvtColor(np.array(web_img), cv2.COLOR_RGB2GRAY)
            else:
                web_cv = cv2.cvtColor(web_img, cv2.COLOR_BGR2GRAY)
            
            # Redimensionar para correlación más rápida
            h, w = pdf_cv.shape
            pdf_small = cv2.resize(pdf_cv, (w//4, h//4))
            web_small = cv2.resize(web_cv, (web_cv.shape[1]//4, web_cv.shape[0]//4))
            
            # Correlación de plantillas
            result = cv2.matchTemplate(web_small, pdf_small, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.3:  # Umbral de confianza
                # Calcular transformación simple (traslación y escala)
                scale_x = web_cv.shape[1] / pdf_cv.shape[1]
                scale_y = web_cv.shape[0] / pdf_cv.shape[0]
                tx, ty = max_loc[0] * 4, max_loc[1] * 4
                
                # Crear matriz de transformación afín
                transform_matrix = np.float32([[scale_x, 0, tx], [0, scale_y, ty]])
                
                return {
                    'transform_matrix': transform_matrix,
                    'confidence': max_val,
                    'method': 'traditional'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error en alineación tradicional: {e}")
            return None
    
    def _optimize_image_contrast(self, img, method='adaptive'):
        """Optimiza el contraste de la imagen usando técnicas avanzadas."""
        try:
            if isinstance(img, Image.Image):
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            else:
                img_cv = img
            
            # Convertir a escala de grises para análisis
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            if method == 'adaptive':
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
            elif method == 'histogram':
                # Ecualización de histograma estándar
                enhanced = cv2.equalizeHist(gray)
            else:
                # Normalización simple
                enhanced = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            
            # Convertir de vuelta a color manteniendo la información de contraste
            enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            # Mezclar con imagen original para mantener información de color
            alpha = 0.7  # Peso de la imagen mejorada
            result = cv2.addWeighted(enhanced_color, alpha, img_cv, 1-alpha, 0)
            
            # Convertir de vuelta a PIL
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            return Image.fromarray(result_rgb)
            
        except Exception as e:
            logger.error(f"Error optimizando contraste: {e}")
            return img
     
    def _calculate_geographic_scale_factor(self, web_coords, map_analysis):
        """Calcula el factor de escala geográfico basado en las coordenadas del sitio web."""
        try:
            # Obtener las coordenadas del sitio web
            if 'x' not in web_coords or 'y' not in web_coords:
                return 1.0
                
            # Para calcular escala, usamos las coordenadas del sitio web y las dimensiones de la imagen
            web_x = web_coords['x']
            web_y = web_coords['y']
            
            # Obtener las dimensiones de la imagen del PDF
            image_width = map_analysis.get('image_size', [800, 600])[0]
            image_height = map_analysis.get('image_size', [800, 600])[1]
            
            # Estimar la escala basada en dimensiones típicas de un lote catastral
            # Un lote típico en Rosario puede tener entre 8-20 metros de frente
            # Asumimos que la imagen del PDF cubre aproximadamente 100-200 metros
            estimated_coverage_meters = 150.0  # metros que cubre la imagen completa
            
            # Calcular factor de escala (metros por pixel)
            scale_factor = estimated_coverage_meters / max(image_width, image_height)
            
            logger.info(f"Factor de escala calculado: {scale_factor:.4f} metros/pixel")
            
            return scale_factor
            
        except Exception as e:
            logger.error(f"Error calculando factor de escala geográfico: {e}")
            return 1.0
    
    def _calculate_intelligent_bbox(self, correlation_data, base_bbox, zoom_factor=1.5):
        """Calcula un bbox inteligente basado en las líneas detectadas y coordenadas geográficas."""
        if not correlation_data:
            return base_bbox
            
        try:
            target_x, target_y = correlation_data['target_point']
            scale_factor = correlation_data['scale_factor']
            intersection = correlation_data['target_intersection']
            
            # Calcular el área de interés basada en la intersección más cercana
            # Usar el factor de escala para convertir píxeles a metros
            pixel_radius = 100  # Radio en píxeles alrededor de la intersección
            meter_radius = pixel_radius * scale_factor * zoom_factor
            
            # Crear bbox centrado en el punto de interés con el radio calculado
            intelligent_bbox = (
                target_x - meter_radius,
                target_y - meter_radius,
                target_x + meter_radius,
                target_y + meter_radius
            )
            
            logger.info(f"Bbox inteligente calculado: radio {meter_radius:.1f}m alrededor del punto ({target_x:.1f}, {target_y:.1f})")
            return intelligent_bbox
            
        except Exception as e:
            logger.error(f"Error calculando bbox inteligente: {e}")
            return base_bbox
    
    def _create_enhanced_transformation(self, pdf_contour, map_analysis, base_img_size):
        """Crea una transformación mejorada usando líneas detectadas y elementos geométricos."""
        if not map_analysis or not pdf_contour:
            return None
            
        W, H = base_img_size
        
        # Obtener puntos del contorno del PDF
        src_pts = [
            pdf_contour['top_left'], pdf_contour['top_right'],
            pdf_contour['bottom_right'], pdf_contour['bottom_left']
        ]
        
        # Analizar las líneas para mejorar la alineación
        horizontal_lines = [line for line in map_analysis['lines'] if line['is_horizontal']]
        vertical_lines = [line for line in map_analysis['lines'] if line['is_vertical']]
        
        # Usar intersecciones de alta confianza como puntos de referencia
        high_confidence_intersections = [
            intersection for intersection in map_analysis['intersections'] 
            if intersection['confidence'] > 0.5
        ]
        
        logger.info(f"Líneas horizontales: {len(horizontal_lines)}, verticales: {len(vertical_lines)}")
        logger.info(f"Intersecciones de alta confianza: {len(high_confidence_intersections)}")
        
        # Si tenemos suficientes líneas y intersecciones, ajustar la transformación
        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2 and len(high_confidence_intersections) >= 2:
            # Calcular el centro de masa de las intersecciones principales
            center_x = sum(inter['point'][0] for inter in high_confidence_intersections) / len(high_confidence_intersections)
            center_y = sum(inter['point'][1] for inter in high_confidence_intersections) / len(high_confidence_intersections)
            
            # Calcular la orientación promedio de las líneas principales
            avg_horizontal_angle = sum(line['angle'] for line in horizontal_lines) / len(horizontal_lines)
            avg_vertical_angle = sum(line['angle'] for line in vertical_lines) / len(vertical_lines)
            
            # Ajustar los puntos de destino basándose en la orientación detectada
            rotation_correction = avg_horizontal_angle if abs(avg_horizontal_angle) < 45 else 0
            
            # Crear puntos de destino ajustados
            margin = 0.1
            cos_r = np.cos(np.radians(rotation_correction))
            sin_r = np.sin(np.radians(rotation_correction))
            
            # Puntos base sin rotación
            base_pts = [
                (W * margin, H * margin), (W * (1 - margin), H * margin),
                (W * (1 - margin), H * (1 - margin)), (W * margin, H * (1 - margin))
            ]
            
            # Aplicar corrección de rotación
            center_dst_x, center_dst_y = W/2, H/2
            dst_pts = []
            for x, y in base_pts:
                # Trasladar al origen
                x_rel, y_rel = x - center_dst_x, y - center_dst_y
                # Rotar
                x_rot = x_rel * cos_r - y_rel * sin_r
                y_rot = x_rel * sin_r + y_rel * cos_r
                # Trasladar de vuelta
                dst_pts.append((x_rot + center_dst_x, y_rot + center_dst_y))
            
            logger.info(f"Transformación mejorada con corrección de rotación: {rotation_correction:.2f}°")
            return self._calculate_perspective_transform(src_pts, dst_pts)
        
        # Fallback al método original si no hay suficiente información
        margin = 0.1
        dst_pts = [
            (W * margin, H * margin), (W * (1 - margin), H * margin),
            (W * (1 - margin), H * (1 - margin)), (W * margin, H * (1 - margin))
        ]
        
        return self._calculate_perspective_transform(src_pts, dst_pts)

    def _extract_pdf_street_info(self, pdf_path):
        """Extrae información de calles del PDF para orientación espacial."""
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                return None
                
            page = doc[0]
            text_dict = page.get_text("dict")
            
            # Buscar nombres de calles y sus posiciones
            street_info = {}
            street_names = ['SALTA', 'CATAMARCA', 'ENTRE RIOS', 'CORRIENTES']
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip().upper()
                            for street in street_names:
                                if street in text:
                                    bbox = span["bbox"]
                                    street_info[street] = {
                                        'bbox': bbox,
                                        'x': (bbox[0] + bbox[2]) / 2,
                                        'y': (bbox[1] + bbox[3]) / 2
                                    }
            
            return street_info if street_info else None
            
        except Exception as e:
            logger.info(f"No se pudo extraer información de calles del PDF: {e}")
            return None

    def _calculate_perspective_transform(self, src_pts, dst_pts):
        """Calcula la matriz de transformación de perspectiva a partir de 4 pares de puntos."""
        if np is None:
            raise ImportError("Numpy no está instalado, no se puede calcular la transformación.")

        # Para la transformación de perspectiva, el sistema es Ax=b donde x son los 8 coeficientes.
        # x' = (a*x + b*y + c) / (g*x + h*y + 1)
        # y' = (d*x + e*y + f) / (g*x + h*y + 1)
        # Reordenando para resolver con linalg.solve:
        # a*x + b*y + c - g*x*x' - h*y*x' = x'
        # d*x + e*y + f - g*x*y' - h*y*y' = y'
        A = []
        b_vec = []
        for (x, y), (xp, yp) in zip(src_pts, dst_pts):
            A.append([x, y, 1, 0, 0, 0, -x*xp, -y*xp])
            A.append([0, 0, 0, x, y, 1, -x*yp, -y*yp])
            b_vec.append(xp)
            b_vec.append(yp)
        
        A = np.array(A)
        b_vec = np.array(b_vec)
        
        try:
            # Resolvemos para los 8 coeficientes de la transformación
            transform_coeffs = np.linalg.solve(A, b_vec)
            # Pillow espera los 8 coeficientes directamente
            return transform_coeffs
        except np.linalg.LinAlgError:
            logger.error("No se pudo resolver el sistema de ecuaciones para la transformación de perspectiva.")
            return None

    def _extract_map_from_pdf(self, pdf_path, zoom_factor=8.0):
        """Extrae y procesa el mapa del PDF usando OpenCV para detectar elementos cartográficos."""
        if fitz is None or cv2 is None:
            return None
            
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                return None
            page = doc[0]
            
            # Renderizar PDF a imagen con alta resolución
            mat = fitz.Matrix(zoom_factor, zoom_factor)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_data = pix.tobytes("png")
            
            # Convertir a formato OpenCV
            nparr = np.frombuffer(img_data, np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if cv_img is None:
                return None
                
            # Detectar elementos del mapa usando OpenCV
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Detectar líneas usando HoughLinesP para elementos cartográficos
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            
            # Preprocesamiento mejorado para detectar parcelas catastrales
            # Aplicar filtro bilateral para reducir ruido manteniendo bordes
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Umbralización adaptativa para mejor detección de contornos
            thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # Operaciones morfológicas para cerrar líneas discontinuas
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Detectar contornos para identificar parcelas
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos por área para encontrar parcelas (parámetros más permisivos)
            parcels = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 100000:  # Rango más amplio para parcelas catastrales
                    # Aproximar contorno a polígono
                    epsilon = 0.03 * cv2.arcLength(contour, True)  # Más tolerante
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) >= 3:  # Más permisivo con vértices
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        # Filtrar por relación de aspecto razonable
                        if 0.1 < aspect_ratio < 10:
                            parcels.append({
                                'contour': contour,
                                'area': area,
                                'vertices': approx,
                                'bbox': (x, y, w, h)
                            })
            
            # Detectar texto usando PyMuPDF para extraer texto real
            text_regions = []
            try:
                # Extraer texto con posiciones del PDF
                text_dict = page.get_text("dict")
                for block in text_dict.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                text_content = span.get("text", "").strip()
                                if text_content and len(text_content) > 0:
                                    # Convertir coordenadas del PDF a coordenadas de imagen
                                    bbox = span.get("bbox", [])
                                    if len(bbox) == 4:
                                        x0, y0, x1, y1 = bbox
                                        # Escalar coordenadas según el zoom_factor
                                        x = int(x0 * zoom_factor)
                                        y = int(y0 * zoom_factor)
                                        w = int((x1 - x0) * zoom_factor)
                                        h = int((y1 - y0) * zoom_factor)
                                        
                                        text_regions.append({
                                            'bbox': (x, y, w, h),
                                            'text': text_content,
                                            'area': w * h
                                        })
                                        logger.debug(f"Texto extraído: '{text_content}' en posición ({x}, {y})")
            except Exception as e:
                logger.warning(f"Error extrayendo texto del PDF: {e}")
                # Fallback: usar morfología para detectar regiones de texto
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                text_contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in text_contours:
                    area = cv2.contourArea(contour)
                    if 50 < area < 2000:  # Área típica de texto
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        if 0.2 < aspect_ratio < 5:  # Relación de aspecto típica de texto
                            text_regions.append({'bbox': (x, y, w, h), 'text': '', 'area': area})
            
            return {
                'image': cv_img,
                'gray': gray,
                'edges': edges,
                'lines': lines if lines is not None and len(lines) > 0 else [],
                'parcels': parcels,
                'text_regions': text_regions,
                'dimensions': (cv_img.shape[1], cv_img.shape[0])  # width, height
            }
            
        except Exception as e:
            logger.error(f"Error al extraer mapa del PDF: {e}")
            return None
    
    def _correlate_pdf_map_with_web_coords(self, map_data, web_coords):
        """Correlaciona elementos del mapa del PDF con coordenadas web usando análisis geométrico inteligente."""
        try:
            if not map_data['parcels'] or not web_coords:
                return None
                
            width, height = map_data['dimensions']
            
            # Análisis multi-criterio para encontrar la parcela objetivo
            best_parcel = None
            best_score = 0
            
            # Extraer coordenadas web
            target_web_x = web_coords.get('x', 0)
            target_web_y = web_coords.get('y', 0)
            
            logger.info(f"Correlacionando con coordenadas web: ({target_web_x}, {target_web_y})")
            
            for parcel in map_data['parcels']:
                score = 0
                x, y, w, h = parcel['bbox']
                parcel_center = (x + w/2, y + h/2)
                
                # Criterio 1: Posición relativa en el PDF (30% del peso)
                # Convertir coordenadas web a posición relativa esperada en PDF
                if hasattr(self, 'last_point') and self.last_point:
                    # Usar transformación aproximada basada en coordenadas conocidas
                    expected_pdf_x = width * 0.5  # Centro como fallback
                    expected_pdf_y = height * 0.5
                    
                    # Calcular distancia normalizada
                    distance = ((parcel_center[0] - expected_pdf_x)**2 + (parcel_center[1] - expected_pdf_y)**2)**0.5
                    max_distance = (width**2 + height**2)**0.5
                    position_score = max(0, 1 - (distance / max_distance)) * 0.3
                    score += position_score
                
                # Criterio 2: Tamaño de parcela (25% del peso)
                # Parcelas de tamaño medio tienen mayor probabilidad
                area = parcel['area']
                avg_area = sum(p['area'] for p in map_data['parcels']) / len(map_data['parcels'])
                size_score = max(0, 1 - abs(area - avg_area) / avg_area) * 0.25
                score += size_score
                
                # Criterio 3: Forma de parcela (20% del peso)
                # Parcelas rectangulares son más comunes en zonas urbanas
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
                ideal_aspect = 1.5  # Relación ideal
                shape_score = max(0, 1 - abs(aspect_ratio - ideal_aspect) / ideal_aspect) * 0.2
                score += shape_score
                
                # Criterio 4: Densidad de elementos cercanos (25% del peso)
                # Parcelas con más elementos de texto/líneas cercanas son más relevantes
                nearby_elements = 0
                search_radius = max(w, h) * 1.5
                
                for text_region in map_data['text_regions']:
                    tx, ty, tw, th = text_region['bbox']
                    text_center = (tx + tw/2, ty + th/2)
                    text_distance = ((parcel_center[0] - text_center[0])**2 + (parcel_center[1] - text_center[1])**2)**0.5
                    if text_distance <= search_radius:
                        nearby_elements += 1
                
                density_score = min(1.0, nearby_elements / 3.0) * 0.25
                score += density_score
                
                logger.debug(f"Parcela en ({x},{y}) - Score: {score:.3f} (pos:{position_score:.3f}, size:{size_score:.3f}, shape:{shape_score:.3f}, density:{density_score:.3f})")
                
                if score > best_score:
                    best_score = score
                    best_parcel = parcel
            
            if not best_parcel:
                logger.warning("No se encontró parcela objetivo, usando parcela central")
                # Fallback: usar parcela más central
                pdf_center = (width / 2, height / 2)
                min_distance = float('inf')
                for parcel in map_data['parcels']:
                    x, y, w, h = parcel['bbox']
                    parcel_center = (x + w/2, y + h/2)
                    distance = ((parcel_center[0] - pdf_center[0])**2 + (parcel_center[1] - pdf_center[1])**2)**0.5
                    if distance < min_distance:
                        min_distance = distance
                        best_parcel = parcel
                        best_score = 0.5  # Score moderado para fallback
            
            if not best_parcel:
                return None
                
            # Calcular factor de escala inteligente
            parcel_area_pixels = best_parcel['area']
            
            # Estimar escala basada en el contexto urbano de Rosario
            # Parcelas típicas: 200-800 m², promedio ~400 m²
            estimated_real_area = 400
            
            # Ajustar estimación basada en el tamaño relativo de la parcela
            if parcel_area_pixels > avg_area * 1.5:
                estimated_real_area = 600  # Parcela grande
            elif parcel_area_pixels < avg_area * 0.7:
                estimated_real_area = 250  # Parcela pequeña
            
            scale_factor = (estimated_real_area / parcel_area_pixels) ** 0.5
            
            # Calcular confianza basada en múltiples factores
            confidence = best_score  # Score base
            
            # Bonus por calidad de detección
            detection_quality = min(1.0, len(map_data['lines']) / 15.0) * 0.3
            detection_quality += min(1.0, len(map_data['parcels']) / 8.0) * 0.2
            detection_quality += min(1.0, len(map_data['text_regions']) / 10.0) * 0.2
            
            confidence = min(1.0, confidence + detection_quality)
            
            logger.info(f"Parcela objetivo seleccionada con score {best_score:.3f} y confianza {confidence:.3f}")
            
            return {
                'target_parcel': best_parcel,
                'scale_factor': scale_factor,
                'confidence': confidence,
                'selection_score': best_score,
                'web_coords': web_coords,
                'estimated_area': estimated_real_area
            }
            
        except Exception as e:
            logger.error(f"Error en correlación PDF-web: {e}")
            return None
    





    



    
    def _convert_to_high_contrast_bw(self, img):
        """Convierte la imagen a blanco y negro con alto contraste para mejor visualización."""
        try:
            # Convertir a escala de grises
            gray_img = img.convert('L')
            
            # Aumentar el contraste
            enhancer = ImageEnhance.Contrast(gray_img)
            contrast_img = enhancer.enhance(1.5)  # Aumentar contraste 50%
            
            # Ajustar brillo para mejor definición
            brightness_enhancer = ImageEnhance.Brightness(contrast_img)
            bright_img = brightness_enhancer.enhance(1.1)
            
            # Aplicar umbralización adaptativa para líneas más definidas
            img_array = np.array(bright_img)
            
            # Usar umbralización de Otsu para separación automática
            _, threshold_img = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convertir de vuelta a PIL
            result_img = Image.fromarray(threshold_img, mode='L')
            
            # Convertir a RGB para mantener compatibilidad
            result_img = result_img.convert('RGB')
            
            return result_img
            
        except Exception as e:
            logger.error(f"Error convirtiendo a blanco y negro: {e}")
            return img  # Retornar imagen original si falla
    
    def _enhance_pdf_visibility(self, overlay_img, pdf_img):
        """Mejora la visibilidad del PDF en el overlay manteniendo colores."""
        try:
            # Convertir a arrays para procesamiento
            overlay_array = np.array(overlay_img.convert('RGB'))
            
            # Mejorar contraste selectivamente
            # Convertir a espacio de color LAB para mejor control
            lab_img = cv2.cvtColor(overlay_array, cv2.COLOR_RGB2LAB)
            
            # Separar canales
            l_channel, a_channel, b_channel = cv2.split(lab_img)
            
            # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization) solo al canal L
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l_channel = clahe.apply(l_channel)
            
            # Recombinar canales
            enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
            
            # Convertir de vuelta a RGB
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Aplicar un ligero aumento de saturación para mejor visibilidad
            hsv_img = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV)
            hsv_img[:,:,1] = cv2.multiply(hsv_img[:,:,1], 1.2)  # Aumentar saturación 20%
            enhanced_rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
            
            # Aplicar filtro de nitidez sutil
            kernel = np.array([[-0.1, -0.1, -0.1],
                              [-0.1,  1.8, -0.1],
                              [-0.1, -0.1, -0.1]])
            sharpened = cv2.filter2D(enhanced_rgb, -1, kernel)
            
            # Mezclar imagen original con la mejorada (70% mejorada, 30% original)
            result = cv2.addWeighted(enhanced_rgb, 0.7, overlay_array, 0.3, 0)
            
            return Image.fromarray(result)
            
        except Exception as e:
            logger.error(f"Error mejorando visibilidad del PDF: {e}")
            return overlay_img  # Retornar imagen original si falla
    
    def _create_simple_overlay(self, base_img, pdf_img):
        """
        Crea una superposición inteligente del PDF sobre la imagen base
        usando IA para alineación automática y optimización de contraste.
        """
        try:
            logger.info("Iniciando superposición con IA y optimización de contraste")
            
            # Optimizar contraste de ambas imágenes primero
            base_optimized = self._optimize_image_contrast(base_img, method='adaptive')
            pdf_optimized = self._optimize_image_contrast(pdf_img, method='adaptive')
            
            # Usar método tradicional directamente
            ai_alignment = None
            logger.info("Usando alineación tradicional...")
            traditional_alignment = self._traditional_alignment(pdf_optimized, base_optimized)
            
            if traditional_alignment and traditional_alignment.get('confidence', 0) > 0.3:
                # Aplicar transformación tradicional
                pdf_cv = cv2.cvtColor(np.array(pdf_optimized), cv2.COLOR_RGB2BGR)
                transform_matrix = traditional_alignment['transform_matrix']
                
                h, w = base_optimized.size[1], base_optimized.size[0]
                pdf_transformed = cv2.warpAffine(pdf_cv, transform_matrix, (w, h))
                pdf_aligned = Image.fromarray(cv2.cvtColor(pdf_transformed, cv2.COLOR_BGR2RGB))
                logger.info(f"Alineación tradicional aplicada (confianza: {traditional_alignment['confidence']:.3f})")
            else:
                # Fallback a método simple mejorado
                logger.info("Usando método de superposición simple mejorado")
                pdf_aligned = self._simple_overlay_fallback(pdf_optimized, base_optimized)
            
            # Crear superposición final con mejor blending
            result_img = self._create_advanced_overlay(base_optimized, pdf_aligned)
            
            # Aplicar optimización final de contraste manteniendo colores
            result_img = self._optimize_image_contrast(result_img, method='histogram')
            
            # Mejorar la visibilidad del PDF sin convertir a B&N
            result_img = self._enhance_pdf_visibility(result_img, pdf_aligned)
            
            # Guardar imagen de depuración
            os.makedirs("output/debug", exist_ok=True)
            result_img.save("output/debug/ai_enhanced_overlay.png", format="PNG")
            logger.info("Superposición con IA y optimización completada")
            
            return result_img
            
        except Exception as e:
            logger.error(f"Error en superposición con IA: {e}")
            # Fallback al método original
            return self._simple_overlay_fallback(pdf_img, base_img)
    
    def _simple_overlay_fallback(self, pdf_img, base_img):
        """Método de respaldo para superposición simple cuando IA falla."""
        try:
            # Calcular mejor escala basada en el contenido
            base_w, base_h = base_img.size
            pdf_w, pdf_h = pdf_img.size
            
            # Usar una escala ligeramente menor para mejor ajuste
            scale_factor = min(base_w / pdf_w, base_h / pdf_h) * 0.85
            new_w = int(pdf_w * scale_factor)
            new_h = int(pdf_h * scale_factor)
            
            # Redimensionar PDF con la nueva escala
            pdf_resized = pdf_img.resize((new_w, new_h), Image.LANCZOS)
            
            # Crear imagen base para la superposición
            result_img = base_img.copy()
            
            # Calcular posición centrada
            x_offset = (base_w - new_w) // 2
            y_offset = (base_h - new_h) // 2
            
            # Convertir a arrays para procesamiento
            base_array = np.array(result_img.convert('RGB'))
            pdf_array = np.array(pdf_resized.convert('RGB'))
            
            # Crear máscara para áreas no blancas del PDF
            pdf_gray = np.mean(pdf_array, axis=2)
            mask = pdf_gray < 240  # Detectar líneas y contenido no blanco
            
            # Aplicar superposición solo en áreas con contenido
            for y in range(new_h):
                for x in range(new_w):
                    base_y = y + y_offset
                    base_x = x + x_offset
                    
                    if (0 <= base_y < base_h and 0 <= base_x < base_w and mask[y, x]):
                        # Mezclar colores con mayor peso al PDF para líneas más visibles
                        alpha = 0.7
                        base_array[base_y, base_x] = (
                            (1 - alpha) * base_array[base_y, base_x] + 
                            alpha * pdf_array[y, x]
                        ).astype(np.uint8)
            
            # Convertir de vuelta a imagen PIL
            return Image.fromarray(base_array)
            
        except Exception as e:
            logger.error(f"Error en superposición de respaldo: {e}")
            return base_img
    
    def _create_advanced_overlay(self, base_img, pdf_img):
        """Crea una superposición avanzada con mejor blending y alineación."""
        try:
            # Calcular mejor escala para el PDF manteniendo proporciones
            base_w, base_h = base_img.size
            pdf_w, pdf_h = pdf_img.size
            
            # Usar una escala más conservadora para mejor alineación
            scale_factor = min(base_w / pdf_w, base_h / pdf_h) * 0.75
            new_w = int(pdf_w * scale_factor)
            new_h = int(pdf_h * scale_factor)
            
            # Redimensionar PDF con mejor calidad
            pdf_resized = pdf_img.resize((new_w, new_h), Image.LANCZOS)
            
            # Crear imagen resultado del tamaño de la base
            result_img = base_img.copy().convert('RGBA')
            
            # Calcular posición centrada con ligero ajuste hacia arriba
            x_offset = (base_w - new_w) // 2
            y_offset = max(0, (base_h - new_h) // 2 - 20)  # Ajuste hacia arriba
            
            # Convertir a arrays
            base_array = np.array(result_img)
            pdf_array = np.array(pdf_resized.convert('RGBA'))
            
            # Crear máscara inteligente para contenido del PDF
            pdf_gray = cv2.cvtColor(pdf_array[:,:,:3], cv2.COLOR_RGB2GRAY)
            
            # Detectar líneas y contenido importante
            edges = cv2.Canny(pdf_gray, 30, 100)
            
            # Crear máscara para áreas no blancas (contenido)
            content_mask = (pdf_gray < 250).astype(np.float32)
            
            # Combinar con detección de bordes
            edge_mask = (edges > 0).astype(np.float32)
            combined_mask = np.maximum(content_mask, edge_mask)
            
            # Suavizar máscara para transiciones más naturales
            combined_mask = cv2.GaussianBlur(combined_mask, (3, 3), 0.5)
            
            # Aplicar superposición solo en el área del PDF redimensionado
            for y in range(new_h):
                for x in range(new_w):
                    base_y = y + y_offset
                    base_x = x + x_offset
                    
                    if (0 <= base_y < base_h and 0 <= base_x < base_w):
                        mask_value = combined_mask[y, x]
                        if mask_value > 0.1:  # Solo aplicar donde hay contenido
                            # Blending con mayor peso al PDF para líneas más visibles
                            alpha = min(0.8, mask_value * 0.9)
                            
                            for c in range(3):  # RGB channels
                                base_array[base_y, base_x, c] = (
                                    (1 - alpha) * base_array[base_y, base_x, c] + 
                                    alpha * pdf_array[y, x, c]
                                ).astype(np.uint8)
            
            return Image.fromarray(base_array[:,:,:3])  # Remover canal alpha
            
        except Exception as e:
            logger.error(f"Error en superposición avanzada: {e}")
            return base_img
     
    def _extract_pdf_as_image(self, pdf_path, zoom_factor=4.0, address=None):
        """Extrae el PDF completo como imagen sin recortes."""
        if fitz is None:
            logger.warning("PyMuPDF no está disponible; no se puede extraer imagen del PDF.")
            return None
            
        try:
            logger.info("Extrayendo PDF completo como imagen")
            return self._extract_full_pdf_as_fallback(pdf_path, zoom_factor)
        except Exception as e:
            logger.error(f"Error al extraer PDF como imagen: {e}")
            return None

    def _extract_and_crop_pdf_by_lines(self, pdf_path, zoom_factor=8.0, address=None):
        """Extrae el PDF como imagen y lo recorta basándose en líneas detectadas."""
        if fitz is None or cv2 is None:
            logger.warning("PyMuPDF o OpenCV no están disponibles; usando fallback.")
            return self._extract_full_pdf_as_fallback(pdf_path, zoom_factor=4.0)
            
        try:
            logger.info("Iniciando extracción y recorte del PDF basado en líneas detectadas")
            
            # Paso 1: Detectar líneas y geometría en el PDF
            map_analysis = self._detect_map_lines_and_geometry(pdf_path, zoom_factor)
            
            if not map_analysis or not map_analysis.get('success'):
                logger.warning("No se pudieron detectar líneas en el PDF, usando fallback")
                return self._extract_full_pdf_as_fallback(pdf_path, zoom_factor=4.0)
            
            # Paso 2: Determinar región de recorte basada en líneas
            crop_result = self._crop_image_based_on_detected_lines(map_analysis, address)
            
            if not crop_result or not crop_result.get('success'):
                logger.warning("No se pudo determinar región de recorte, usando fallback")
                return self._extract_full_pdf_as_fallback(pdf_path, zoom_factor=4.0)
            
            # Paso 3: Extraer la imagen recortada
            cropped_img = crop_result.get('image')
            
            if cropped_img is not None:
                logger.info(f"PDF recortado exitosamente: {cropped_img.size[0]}x{cropped_img.size[1]} píxeles")
                
                # Mejorar la imagen recortada
                if ImageEnhance is not None:
                    try:
                        # Mejorar contraste
                        enhancer = ImageEnhance.Contrast(cropped_img)
                        cropped_img = enhancer.enhance(1.2)
                        
                        # Mejorar nitidez
                        enhancer = ImageEnhance.Sharpness(cropped_img)
                        cropped_img = enhancer.enhance(1.1)
                        
                        logger.info("Imagen recortada mejorada")
                    except Exception as e:
                        logger.warning(f"No se pudo mejorar la imagen recortada: {e}")
                
                return cropped_img
            else:
                logger.warning("La imagen recortada es None, usando fallback")
                return self._extract_full_pdf_as_fallback(pdf_path, zoom_factor=4.0)
                
        except Exception as e:
            logger.error(f"Error en extracción y recorte basado en líneas: {e}")
            return self._extract_full_pdf_as_fallback(pdf_path, zoom_factor=4.0)
    
    def _extract_full_pdf_as_fallback(self, pdf_path, zoom_factor=4.0):
        """Función de fallback para extraer el PDF completo cuando no se puede hacer detección automática."""
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                logger.warning("El PDF no tiene páginas")
                return None
                
            page = doc[0]  # Primera página
            
            # Renderizar con zoom alto para mejor calidad
            mat = fitz.Matrix(zoom_factor, zoom_factor)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convertir a imagen PIL
            mode = "RGBA" if pix.alpha else "RGB"
            pdf_img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            
            # Mejorar contraste y nitidez para mejor visualización
            if ImageEnhance is not None:
                try:
                    # Mejorar contraste
                    enhancer = ImageEnhance.Contrast(pdf_img)
                    pdf_img = enhancer.enhance(1.3)
                    
                    # Mejorar nitidez
                    enhancer = ImageEnhance.Sharpness(pdf_img)
                    pdf_img = enhancer.enhance(1.2)
                    
                    logger.info(f"PDF extraído como imagen (fallback): {pdf_img.size[0]}x{pdf_img.size[1]} píxeles")
                except Exception as e:
                    logger.warning(f"No se pudo mejorar la imagen del PDF: {e}")
            
            doc.close()
            return pdf_img
            
        except Exception as e:
            logger.error(f"Error al extraer imagen del PDF (fallback): {e}")
            return None

    def _overlay_pdf_as_georeferenced_image(self, image_path, pdf_path, bbox_world, transparency=0.6, address=None):
        """Superpone el PDF como imagen georreferenciada usando detección avanzada de elementos del mapa.
        Utiliza OpenCV para detectar y alinear elementos cartográficos."""
        if Image is None or not image_path or not os.path.exists(image_path) or fitz is None:
            return
            
        try:
            # Extraer mapa del PDF usando OpenCV
            map_data = self._extract_map_from_pdf(pdf_path, zoom_factor=8.0)
            if not map_data:
                logger.warning("No se pudo extraer el mapa del PDF")
                return
                
            # Convertir imagen OpenCV a PIL
            cv_img = map_data['image']
            pdf_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)).convert("RGBA")
            
            logger.info(f"Mapa extraído del PDF: {len(map_data['lines'])} líneas, {len(map_data['parcels'])} parcelas, {len(map_data['text_regions'])} regiones de texto")
            
            # Guardar imagen de depuración con elementos detectados
            os.makedirs("output/debug", exist_ok=True)
            debug_img = cv_img.copy()
            
            # Dibujar líneas detectadas
            for line in map_data['lines']:
                x1, y1, x2, y2 = line[0]
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Dibujar parcelas detectadas
            for parcel in map_data['parcels']:
                cv2.drawContours(debug_img, [parcel['contour']], -1, (255, 0, 0), 2)
                # Marcar centro de la parcela
                x, y, w, h = parcel['bbox']
                center = (x + w//2, y + h//2)
                cv2.circle(debug_img, center, 5, (0, 0, 255), -1)
            
            # Dibujar regiones de texto
            for text_region in map_data['text_regions']:
                x, y, w, h = text_region['bbox']
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 255, 0), 1)
            
            # Guardar imagen de depuración
            cv2.imwrite("output/debug/pdf_map_analysis.png", debug_img)
            logger.info("Análisis del mapa guardado en output/debug/pdf_map_analysis.png")
            
            # Correlacionar con coordenadas web si están disponibles
            correlation_data = None
            if hasattr(self, 'last_point') and self.last_point:
                web_coords = {'x': self.last_point[0], 'y': self.last_point[1]}
                correlation_data = self._correlate_pdf_map_with_web_coords(map_data, web_coords)
                if correlation_data:
                    logger.info(f"Correlación exitosa: confianza {correlation_data['confidence']:.3f}, escala {correlation_data['scale_factor']:.4f} m/px")
            
            # Encontrar la parcela más probable basada en la correlación
            target_parcel = self._find_target_parcel(map_data, correlation_data, address=address)
            
            if target_parcel:
                # Usar la parcela detectada para crear el recorte enfocado
                x, y, w, h = target_parcel['bbox']
                
                # Calcular el centro de la parcela
                center_x = x + w / 2
                center_y = y + h / 2
                
                # Calcular el área de recorte con margen alrededor de la parcela
                zoom_margin = 0.5  # Margen para mostrar contexto alrededor de la parcela
                crop_width = w * (1 + zoom_margin)
                crop_height = h * (1 + zoom_margin)
                
                # Asegurar que el recorte no exceda los límites de la imagen
                crop_x0 = max(0, center_x - crop_width / 2)
                crop_y0 = max(0, center_y - crop_height / 2)
                crop_x1 = min(pdf_img.width, center_x + crop_width / 2)
                crop_y1 = min(pdf_img.height, center_y + crop_height / 2)
                
                # Recortar la imagen del PDF enfocándose en la parcela objetivo
                pdf_cropped = pdf_img.crop((int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)))
                
                # Crear imagen de depuración mostrando la parcela seleccionada
                debug_crop = cv_img[int(crop_y0):int(crop_y1), int(crop_x0):int(crop_x1)].copy()
                # Dibujar contorno de la parcela en la imagen recortada
                parcel_contour_adjusted = target_parcel['contour'] - np.array([int(crop_x0), int(crop_y0)])
                cv2.drawContours(debug_crop, [parcel_contour_adjusted], -1, (0, 255, 0), 3)
                cv2.imwrite("output/debug/pdf_target_parcel.png", debug_crop)
                logger.info(f"Parcela objetivo detectada: área {target_parcel['area']} px², centro ({center_x:.1f}, {center_y:.1f})")
                
                # Guardar la imagen recortada para depuración
                pdf_cropped.save("output/debug/pdf_cropped.png", format="PNG")
                logger.info("Imagen recortada del PDF guardada para depuración en output/debug/pdf_cropped.png")
                
                # Redimensionar al tamaño de la imagen base para mantener proporciones
                base_img = Image.open(image_path).convert("RGBA")
                W, H = base_img.size
                pdf_resized = pdf_cropped.resize((W, H), Image.Resampling.LANCZOS)
                
                # Guardar la imagen base para depuración
                base_img.save("output/debug/base_img.png", format="PNG")
                logger.info("Imagen base guardada para depuración en output/debug/base_img.png")
                
                # Combinar la imagen base con el PDF recortado en lugar de reemplazarla
                # Esto asegura que se vea tanto el mapa base como el PDF
                try:
                    # Crear una nueva imagen con el mismo tamaño que la base
                    # Usamos fondo negro para mejor contraste con los detalles del PDF
                    result_img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
                    
                    # En lugar de usar la imagen base, usar directamente el PDF para máxima visibilidad
                    # No pegar la imagen base para evitar que interfiera con los datos del PDF
                    
                    # Asegurar que el PDF tenga opacidad completa
                    if len(pdf_resized.split()) == 4:  # Verificar que tenga canal alfa
                        # Opacidad completa para máxima visibilidad de los datos del PDF
                        pdf_alpha = pdf_resized.split()[3].point(lambda p: 255 if p > 0 else 0)  # Opacidad completa donde hay contenido
                        pdf_resized.putalpha(pdf_alpha)
                    
                    # Mejorar el contraste y nitidez de manera segura
                    try:
                        # Mejorar el contraste de la imagen para que los detalles sean más visibles
                        enhancer = ImageEnhance.Contrast(pdf_resized)
                        pdf_resized = enhancer.enhance(1.8)  # Aumentar contraste moderadamente
                        
                        # Mejorar la nitidez para que los textos sean más legibles
                        enhancer = ImageEnhance.Sharpness(pdf_resized)
                        pdf_resized = enhancer.enhance(1.5)  # Aumentar nitidez moderadamente
                    except Exception as e:
                        logger.warning(f"No se pudo mejorar la imagen: {e}")
                    
                    # Usar directamente el PDF recortado como imagen principal
                    # Crear una imagen con fondo blanco del tamaño correcto
                    final_img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
                    final_img.paste(pdf_resized, (0, 0), pdf_resized)
                    
                    # Implementar superposición georreferenciada usando la correlación
                    result = self._create_georeferenced_overlay(final_img, pdf_resized, correlation_data, target_parcel)
                    
                    if result:
                        result.save(image_path, format="PNG")
                        logger.info("PDF superpuesto exitosamente sobre la imagen usando georreferenciación")
                    else:
                        # Fallback: usar la imagen combinada simple
                        final_img.save(image_path, format="PNG")
                        logger.info("Usando superposición simple como respaldo")
                        
                except Exception as e:
                    logger.error(f"Error al combinar imágenes: {e}")
                    # Fallback: guardar solo el PDF recortado
                    pdf_resized.save(image_path, format="PNG")
                logger.info("Imagen combinada del PDF con zoom en la zona geolocalizada aplicada correctamente.")
                return
            
            # Si no se pudo extraer la parcela objetivo, usar método de respaldo simple
            logger.warning("No se pudo detectar parcela objetivo, usando superposición simple")
            
            # Cargar imagen base para el respaldo
            base_img = Image.open(image_path).convert("RGBA")
            W, H = base_img.size
            
            # Guardar la imagen base para depuración
            base_img.save("output/debug/base_img_fallback.png", format="PNG")
            logger.info("Imagen base guardada para depuración en output/debug/base_img_fallback.png")
            
            # Usar superposición simple como respaldo
            try:
                result = self._create_simple_overlay(base_img, pdf_img)
                if result:
                    result.save(image_path, format="PNG")
                    logger.info("Superposición simple aplicada como respaldo")
                else:
                    # Último respaldo: redimensionar y guardar solo el PDF
                    scale = min(W / pdf_img.width, H / pdf_img.height) * 0.8
                    new_w, new_h = int(pdf_img.width * scale), int(pdf_img.height * scale)
                    pdf_resized = pdf_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    x_offset = (W - new_w) // 2
                    y_offset = (H - new_h) // 2
                    
                    # Crear imagen final con el PDF centrado
                    result_img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
                    result_img.paste(base_img, (0, 0))
                    result_img.paste(pdf_resized, (x_offset, y_offset), pdf_resized)
                    result_img.save(image_path, format="PNG")
                    logger.info("Respaldo final: PDF centrado aplicado")
                    
            except Exception as e:
                logger.error(f"Error en superposición de respaldo: {e}")
                # Guardar solo la imagen base si todo falla
                base_img.save(image_path, format="PNG")
                logger.info("Se guardó solo la imagen base como fallback final.")
            
        except Exception as e:
            logger.error(f"No se pudo superponer PDF como imagen georreferenciada: {e}", exc_info=True)

    def _crop_image_by_polygon(self, img, bbox, polygon_xy):
        """Recorta la imagen según un polígono, con mejoras para asegurar que se muestre correctamente la zona del PDF."""
        if Image is None:
            logger.warning("Pillow no está disponible; se omite recorte del polígono.")
            return None
        
        minx, miny, maxx, maxy = bbox
        width, height = img.size
        
        def world_to_px(wx, wy):
            px = (wx - minx) * (width - 1) / (maxx - minx)
            py = (maxy - wy) * (height - 1) / (maxy - miny)
            return (px, py)

        # Convertir coordenadas del polígono a píxeles
        poly_px = [world_to_px(x, y) for (x, y) in polygon_xy]

        # Calcular el centro del polígono
        center_x = sum(p[0] for p in poly_px) / len(poly_px)
        center_y = sum(p[1] for p in poly_px) / len(poly_px)
        
        # Calcular el ancho y alto del polígono
        xs = [p[0] for p in poly_px]
        ys = [p[1] for p in poly_px]
        
        # Aumentar el margen para mostrar más contexto alrededor del lote
        margin = 50  # Margen en píxeles aumentado para mostrar más contexto
        
        # Calcular el bounding box del polígono con margen
        min_px = max(int(min(xs)) - margin, 0)
        max_px = min(int(max(xs)) + margin, width)
        min_py = max(int(min(ys)) - margin, 0)
        max_py = min(int(max(ys)) + margin, height)
        
        # Si el área es muy pequeña, ampliar usando el centro como referencia
        min_area = 300 * 300  # Área mínima en píxeles cuadrados (aumentada)
        area = (max_px - min_px) * (max_py - min_py)
        
        if area < min_area:
            # Calcular un nuevo tamaño basado en el centro
            target_size = int(math.sqrt(min_area))
            half_size = target_size // 2
            
            # Recalcular los límites basados en el centro
            min_px = max(int(center_x - half_size), 0)
            max_px = min(int(center_x + half_size), width)
            min_py = max(int(center_y - half_size), 0)
            max_py = min(int(center_y + half_size), height)
        
        # Recortar la imagen según el bounding box calculado
        sub = img.crop((min_px, min_py, max_px, max_py))
        
        # Reproyectar polígono relativo a la subimagen
        poly_sub = [(x - min_px, y - min_py) for (x, y) in poly_px]

        # Crear una imagen con fondo transparente para preservar detalles del PDF
        out = Image.new("RGBA", sub.size, (255, 255, 255, 255))  # Fondo blanco para mejor contraste
        out.paste(sub, (0, 0))
        
        # Dibujar un borde más grueso y visible alrededor del polígono
        overlay = Image.new("RGBA", sub.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Dibujar un borde exterior más grueso para mayor visibilidad
        draw.polygon(poly_sub, outline=(255, 0, 0, 255), width=5)  # Borde rojo grueso
        
        # Dibujar un área semitransparente para resaltar el polígono sin ocultar detalles
        draw.polygon(poly_sub, fill=(255, 255, 0, 60))  # Relleno amarillo más transparente
        
        # Combinar las imágenes
        out = Image.alpha_composite(out, overlay)
        
        # Añadir una etiqueta para identificar claramente que es el lote
        draw = ImageDraw.Draw(out)
        draw.text((10, 10), "LOTE SELECCIONADO", fill=(0, 0, 0, 255), stroke_width=2, stroke_fill=(255, 255, 255, 255))
        
        return out

    def _transform_to_wgs84(self, x, y):
        """Transforma (E,N) desde POSGAR Faja 5 a WGS84 probando EPSG 5347 y 22185 y valida por bbox de Rosario."""
        if _PyprojTransformer is None:
            logger.warning("pyproj no está disponible; se omite transformación a WGS84.")
            return None, None, None
        candidates = ["EPSG:5347", "EPSG:22185"]
        for epsg in candidates:
            try:
                transformer = _PyprojTransformer.from_crs(epsg, "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(x, y)
                # Rosario approx bbox
                if -61.0 < lon < -60.4 and -33.3 < lat < -32.7:
                    return lat, lon, epsg
            except Exception:
                continue
        # Default best-effort
        transformer = _PyprojTransformer.from_crs("EPSG:22185", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return lat, lon, "EPSG:22185"

    def _get_pdf_filename(self, address):
        wait = WebDriverWait(self.driver, 45) # Increased wait time
        logger.info(f"Iniciando búsqueda para: {address}")
        self.driver.get("https://infomapa.rosario.gov.ar/emapa/")

        logger.info("Ingresando dirección...")
        search_input = wait.until(EC.visibility_of_element_located((By.ID, "txtDireccionesLugares")))
        search_input.clear()
        search_input.send_keys(address)

        logger.info("Haciendo clic en la primera sugerencia y capturando coordenadas...")
        first_suggestion = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "li.ubicaciones-li:first-child")))
        # Capturar el hidden-value que contiene un JSON con geometry
        hidden_value = first_suggestion.get_attribute("hidden-value")
        logger.info(f"hidden-value capturado: {'sí' if hidden_value else 'no'}")
        self.driver.execute_script("arguments[0].click();", first_suggestion)

        # Intentar extraer coordenadas del hidden-value
        punto_x = None
        punto_y = None
        if hidden_value:
            try:
                # El sitio guarda JSON con comillas simples, las normalizamos
                normalized = hidden_value.replace("'", '"')
                data = json.loads(normalized)
                # Puede venir como Feature completo o solo geometry
                if isinstance(data, dict):
                    if data.get('type') == 'Feature' and 'geometry' in data:
                        coords = data['geometry'].get('coordinates')
                    elif data.get('type') == 'Point' and 'coordinates' in data:
                        coords = data.get('coordinates')
                    else:
                        coords = None
                    if coords and len(coords) >= 2:
                        punto_x, punto_y = coords[0], coords[1]
                        logger.info(f"Coordenadas extraídas del hidden-value: x={punto_x}, y={punto_y}")
            except Exception as e:
                logger.warning(f"No se pudo parsear hidden-value: {e}")

        # Si tenemos coordenadas, consultar el endpoint AJAX directamente (más robusto)
        if punto_x is not None and punto_y is not None:
            logger.info("Consultando endpoint AJAX 'direccion/cartobase.htm' con las coordenadas extraídas...")
            ajax_url = "https://infomapa.rosario.gov.ar/emapa/direccion/cartobase.htm"
            try:
                r = requests.post(ajax_url, data={
                    'punto_x': punto_x,
                    'punto_y': punto_y
                }, timeout=25)
                r.raise_for_status()
                data = r.json()
                # Buscar el primer registro_grafico disponible
                href = None
                cartobase_meta = {}
                if isinstance(data, list):
                    for item in data:
                        try:
                            # Guardar claves/links de interés
                            def first_entry(lst):
                                return lst[0] if isinstance(lst, list) and lst else None
                            rg = first_entry(item.get('registro_grafico') or [])
                            cat = first_entry(item.get('catastral') or [])
                            sec = first_entry(item.get('seccion') or [])
                            lin = first_entry(item.get('lineas') or [])
                            may = first_entry(item.get('mayor_area') or [])
                            cartobase_meta = {
                                'registro_grafico': rg,
                                'catastral': cat,
                                'seccion': sec,
                                'lineas': lin,
                                'mayor_area': may,
                            }
                            if rg and rg.get('link'):
                                href = rg['link']
                                break
                        except Exception:
                            continue
                if not href:
                    raise Exception("No se encontró enlace 'Registro Gráfico' en la respuesta AJAX.")
                # Guardar punto para usos posteriores
                self.last_point = (punto_x, punto_y)
            except Exception as e:
                logger.error(f"Fallo al consultar 'cartobase.htm': {e}")
                raise
        else:
            # Fallback: intentar mediante el DOM del popup si no se pudieron leer coordenadas
            logger.info("No se pudieron extraer coordenadas. Volviendo a método DOM como fallback.")
            # Step 1: Wait for the popup container and jQuery tabs to exist
            logger.info("Paso F1: Esperando el contenedor '#tabspopup'...")
            wait.until(EC.presence_of_element_located((By.ID, "tabspopup")))
            # Step F2: Buscar el enlace 'Registro Gráfico' directamente
            logger.info("Paso F2: Buscando enlace 'Registro Gráfico' en el popup...")
            pdf_link = wait.until(EC.visibility_of_element_located((By.XPATH, "//div[@id='tabspopup']//a[starts-with(normalize-space(text()), 'Registro Gráfico')]")))  
            href = pdf_link.get_attribute('href')

        if not href:
            raise Exception("El enlace 'Registro Gráfico' no tiene atributo href.")

        match = re.search(r'([^/]+\.pdf)$', href)
        if not match:
            raise Exception(f"No se pudo extraer el nombre del PDF desde el href: {href}")

        pdf_filename = match.group(1)
        pdf_url = f"https://infomapa.rosario.gov.ar/emapa/servlets/verArchivo?path=manzanas/{pdf_filename}"

        # Preparar retorno con coords en formato estándar (Easting/Northing) EPSG:22185/5347
        # En Rosario, típicamente: Easting ~ 5.42e6 - 5.46e6; Northing ~ 6.34e6 - 6.37e6
        easting = None
        northing = None
        if 'punto_x' in locals() and 'punto_y' in locals():
            x_val, y_val = punto_x, punto_y
            # Heurística por rangos
            if 5.3e6 <= x_val <= 5.7e6 and 6.2e6 <= y_val <= 6.6e6:
                # X parece Easting, Y parece Northing
                easting, northing = x_val, y_val
            elif 5.3e6 <= y_val <= 5.7e6 and 6.2e6 <= x_val <= 6.6e6:
                # Y parece Easting, X parece Northing
                easting, northing = y_val, x_val
            else:
                # Fallback conservador: usar (x,y) como (E,N)
                easting, northing = x_val, y_val

        return {
            'pdf_filename': pdf_filename,
            'pdf_url': pdf_url,
            'easting': easting,
            'northing': northing,
            'crs': 'EPSG:22185',
            'cartobase_meta': cartobase_meta if 'cartobase_meta' in locals() else None,
        }

    def _correlate_pdf_with_reference_map(self, pdf_path: str, target_x: float, target_y: float) -> dict:
        """Correlaciona el PDF con un mapa de referencia para encontrar la posición exacta del punto objetivo."""
        try:
            # Analizar elementos del PDF
            map_data = self._extract_map_from_pdf(pdf_path, zoom_factor=8.0)
            if not map_data:
                return None
                
            # Obtener dimensiones del PDF
            doc = fitz.open(pdf_path)
            page = doc[0]
            page_rect = page.rect
            doc.close()
            
            # Buscar patrones geométricos que puedan correlacionarse con mapas de referencia
            lines = map_data.get('lines', [])
            parcels = map_data.get('parcels', [])
            
            if len(lines) < 10:  # Muy pocas líneas para correlación confiable
                return None
                
            # Calcular densidad de líneas para estimar escala
            total_line_length = 0
            for line in lines:
                try:
                    if hasattr(line, '__len__') and len(line) > 0 and hasattr(line[0], '__len__') and len(line[0]) >= 4:
                        x1, y1, x2, y2 = line[0][:4]
                        total_line_length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                except (IndexError, TypeError):
                    continue
            avg_line_length = total_line_length / len(lines) if lines else 0
            
            # Estimar factor de escala basado en densidad de elementos
            # Más líneas y elementos = mayor detalle = menor escala (más metros por pixel)
            if len(lines) > 2000:  # Muy detallado
                estimated_scale = 0.1  # 0.1 metros/pixel
            elif len(lines) > 1000:  # Detallado
                estimated_scale = 0.15
            elif len(lines) > 500:  # Moderado
                estimated_scale = 0.2
            else:  # Poco detallado
                estimated_scale = 0.3
                
            logger.info(f"Análisis del PDF: {len(lines)} líneas, escala estimada: {estimated_scale} m/px")
            
            # Buscar la zona más densa de elementos (probablemente la zona urbana objetivo)
            target_position = self._find_densest_area(map_data, page_rect.width, page_rect.height)
            
            if target_position:
                return {
                    'success': True,
                    'target_pixel_x': target_position['x'],
                    'target_pixel_y': target_position['y'],
                    'scale_factor': estimated_scale,
                    'confidence': target_position['confidence'],
                    'method': 'density_analysis'
                }
            else:
                # Fallback: usar centro del PDF
                return {
                    'success': True,
                    'target_pixel_x': page_rect.width / 2,
                    'target_pixel_y': page_rect.height / 2,
                    'scale_factor': estimated_scale,
                    'confidence': 0.3,
                    'method': 'center_fallback'
                }
                
        except Exception as e:
            logger.error(f"Error en correlación con mapa de referencia: {e}")
            return None
            
    def _find_densest_area(self, map_data: dict, width: int, height: int) -> dict:
        """Encuentra el área con mayor densidad de elementos geométricos."""
        try:
            lines = map_data.get('lines', [])
            parcels = map_data.get('parcels', [])
            
            if lines is None or len(lines) == 0:
                return None
                
            # Dividir la imagen en una grilla y calcular densidad en cada celda
            grid_size = 20  # 20x20 grid
            cell_width = width / grid_size
            cell_height = height / grid_size
            
            density_grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
            
            # Contar líneas en cada celda
            for line in lines:
                try:
                    if hasattr(line, '__len__') and len(line) > 0 and hasattr(line[0], '__len__') and len(line[0]) >= 4:
                        x1, y1, x2, y2 = line[0][:4]
                        # Punto medio de la línea
                        mid_x = (x1 + x2) / 2
                        mid_y = (y1 + y2) / 2
                        
                        # Determinar celda
                        grid_x = min(int(mid_x / cell_width), grid_size - 1)
                        grid_y = min(int(mid_y / cell_height), grid_size - 1)
                        
                        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                            density_grid[grid_y][grid_x] += 1
                except (IndexError, TypeError):
                    continue
                
            # Encontrar la celda con mayor densidad
            max_density = 0
            best_x, best_y = width // 2, height // 2
            
            for y in range(grid_size):
                for x in range(grid_size):
                    if density_grid[y][x] > max_density:
                        max_density = density_grid[y][x]
                        best_x = (x + 0.5) * cell_width
                        best_y = (y + 0.5) * cell_height
                        
            # Calcular confianza basada en la densidad relativa
            total_lines = len(lines)
            confidence = min(1.0, max_density / (total_lines / (grid_size * grid_size)) if total_lines > 0 else 0)
            
            logger.info(f"Área más densa encontrada en ({best_x:.1f}, {best_y:.1f}) con densidad {max_density} (confianza: {confidence:.3f})")
            
            return {
                'x': best_x,
                'y': best_y,
                'confidence': confidence,
                'density': max_density
            }
            
        except Exception as e:
            logger.error(f"Error encontrando área más densa: {e}")
            return None
            
    def _number_to_words(self, number_str: str) -> str:
        """Convierte un número a su representación en palabras en español."""
        number_words = {
            '1': 'uno', '2': 'dos', '3': 'tres', '4': 'cuatro', '5': 'cinco',
            '6': 'seis', '7': 'siete', '8': 'ocho', '9': 'nueve', '10': 'diez',
            '11': 'once', '12': 'doce', '13': 'trece', '14': 'catorce', '15': 'quince',
            '16': 'dieciseis', '17': 'diecisiete', '18': 'dieciocho', '19': 'diecinueve',
            '20': 'veinte', '21': 'veintiuno', '22': 'veintidos', '23': 'veintitres',
            '24': 'veinticuatro', '25': 'veinticinco', '26': 'veintiseis', '27': 'veintisiete',
            '28': 'veintiocho', '29': 'veintinueve', '30': 'treinta', '31': 'treinta y uno',
            '100': 'cien', '200': 'doscientos', '300': 'trescientos', '400': 'cuatrocientos',
            '500': 'quinientos', '600': 'seiscientos', '700': 'setecientos', '800': 'ochocientos',
            '900': 'novecientos', '1000': 'mil'
        }
        
        # Casos especiales para números comunes en direcciones
        special_cases = {
            '1': ['uno', 'primero', 'primera'],
            '2': ['dos', 'segundo', 'segunda'],
            '3': ['tres', 'tercero', 'tercera'],
            '4': ['cuatro', 'cuarto', 'cuarta'],
            '5': ['cinco', 'quinto', 'quinta'],
            '9': ['nueve', 'noveno', 'novena']
        }
        
        if number_str in special_cases:
            return special_cases[number_str]
        elif number_str in number_words:
            return [number_words[number_str]]
        else:
            return [number_str]  # Fallback al número original
    
    def _find_target_position_in_pdf(self, map_data: dict, target_x: float, target_y: float, address: str = None) -> dict:
        """Encuentra la posición del punto objetivo en el PDF basándose en análisis de elementos mejorado."""
        try:
            width = map_data.get('dimensions', [800, 600])[0]
            height = map_data.get('dimensions', [800, 600])[1]
            
            # Método 1: Usar coordenadas geográficas si están disponibles
            if hasattr(self, 'last_point') and self.last_point:
                geo_x, geo_y = self.last_point
                logger.info(f"Usando coordenadas geográficas: ({geo_x}, {geo_y})")
                
                # Convertir coordenadas geográficas a posición en el PDF
                pdf_position = self._convert_geo_coords_to_pdf_position(geo_x, geo_y, map_data, width, height)
                if pdf_position:
                    logger.info(f"Posición convertida en PDF: ({pdf_position['x']:.1f}, {pdf_position['y']:.1f})")
                    return {
                        'x': pdf_position['x'],
                        'y': pdf_position['y'],
                        'scale_factor': 0.08,  # Zoom más preciso para mejor centrado
                        'confidence': 0.95,
                        'method': 'geographic_coordinates'
                    }
            
            # Método 2: Búsqueda mejorada por dirección con análisis de proximidad
            if address:
                import re
                # Extraer nombre de la calle y número
                address_parts = address.lower().strip().split()
                
                # Buscar números en la dirección
                numbers = re.findall(r'\d+', address.lower())
                target_number = numbers[0] if numbers else ""
                
                # Extraer el nombre de la calle (todo excepto el número)
                street_parts = []
                for part in address_parts:
                    if not re.search(r'\d', part):  # Si no contiene números
                        street_parts.append(part)
                
                # Convertir números en el nombre de la calle a palabras
                converted_street_parts = []
                for part in address_parts:
                    if re.match(r'^\d+$', part):  # Si es solo un número
                        word_variants = self._number_to_words(part)
                        converted_street_parts.extend(word_variants)
                    elif not re.search(r'\d', part):  # Si no contiene números
                        converted_street_parts.append(part)
                
                street_name_numeric = ' '.join(street_parts) if street_parts else ""
                street_name_words = ' '.join(converted_street_parts) if converted_street_parts else ""
                
                logger.info(f"Buscando calle numérica: '{street_name_numeric}', calle en palabras: '{street_name_words}', número: '{target_number}' en el PDF")
                
                # Buscar coincidencias de texto con análisis de proximidad
                text_matches = []
                for text_region in map_data.get('text_regions', []):
                    text_content = text_region.get('text', '').strip().lower()
                    score = 0
                    
                    # Puntuación por coincidencia de nombre de calle en palabras (prioridad alta)
                    if street_name_words:
                        for word in street_name_words.split():
                            if word and len(word) >= 3 and word in text_content:
                                score += 4.0
                                logger.info(f"Palabra de calle '{word}' encontrada en texto '{text_content}'")
                    
                    # Puntuación por coincidencia de nombre de calle numérico (prioridad media)
                    if street_name_numeric:
                        for word in street_name_numeric.split():
                            if word and len(word) >= 3 and word in text_content:
                                score += 2.0
                                logger.info(f"Calle numérica '{word}' encontrada en texto '{text_content}'")
                    
                    # Puntuación por coincidencia de número de altura
                    if target_number and target_number in text_content:
                        score += 1.5
                        logger.info(f"Número '{target_number}' encontrado en texto '{text_content}'")
                    
                    if score > 0:
                        tx, ty, tw, th = text_region['bbox']
                        text_matches.append({
                            'region': text_region,
                            'score': score,
                            'center_x': tx + tw / 2,
                            'center_y': ty + th / 2
                        })
                
                # Si encontramos coincidencias, usar análisis de proximidad
                if text_matches:
                    # Ordenar por puntuación
                    text_matches.sort(key=lambda x: x['score'], reverse=True)
                    best_match = text_matches[0]
                    
                    # Si hay múltiples coincidencias, calcular centroide ponderado
                    if len(text_matches) > 1 and best_match['score'] >= 2.0:
                        weighted_x = sum(m['center_x'] * m['score'] for m in text_matches[:3])
                        weighted_y = sum(m['center_y'] * m['score'] for m in text_matches[:3])
                        total_weight = sum(m['score'] for m in text_matches[:3])
                        
                        final_x = weighted_x / total_weight
                        final_y = weighted_y / total_weight
                        
                        logger.info(f"Centroide ponderado calculado: ({final_x:.1f}, {final_y:.1f}) con {len(text_matches)} coincidencias")
                        return {
                            'x': final_x,
                            'y': final_y,
                            'scale_factor': 0.15,
                            'confidence': min(0.95, best_match['score'] / 5.0),
                            'method': 'weighted_text_analysis'
                        }
                    elif best_match['score'] >= 2.0:
                        logger.info(f"Mejor coincidencia encontrada con puntuación {best_match['score']} en posición ({best_match['center_x']:.1f}, {best_match['center_y']:.1f})")
                        return {
                            'x': best_match['center_x'],
                            'y': best_match['center_y'],
                            'scale_factor': 0.15,
                            'confidence': min(0.95, best_match['score'] / 5.0),
                            'method': 'street_and_number_match'
                        }
            
            # Usar análisis de densidad como método secundario
            densest_area = self._find_densest_area(map_data, width, height)
            
            if densest_area and densest_area['confidence'] > 0.4:
                return {
                    'x': densest_area['x'],
                    'y': densest_area['y'],
                    'scale_factor': 0.18,  # Escala estimada
                    'confidence': densest_area['confidence'],
                    'method': 'density_analysis'
                }
            else:
                # Fallback: buscar parcelas cerca del centro
                parcels = map_data.get('parcels', [])
                if parcels:
                    # Encontrar la parcela más central
                    center_x, center_y = width / 2, height / 2
                    best_parcel = None
                    min_distance = float('inf')
                    
                    for parcel in parcels:
                        x, y, w, h = parcel['bbox']
                        parcel_center_x = x + w / 2
                        parcel_center_y = y + h / 2
                        
                        distance = math.sqrt((parcel_center_x - center_x)**2 + (parcel_center_y - center_y)**2)
                        if distance < min_distance:
                            min_distance = distance
                            best_parcel = parcel
                            
                    if best_parcel:
                        x, y, w, h = best_parcel['bbox']
                        return {
                            'x': x + w / 2,
                            'y': y + h / 2,
                            'scale_factor': 0.18,
                            'confidence': 0.6,
                            'method': 'central_parcel'
                        }
                        
                # Último fallback: centro del PDF
                return {
                    'x': width / 2,
                    'y': height / 2,
                    'scale_factor': 0.2,
                    'confidence': 0.3,
                    'method': 'center_fallback'
                }
                
        except Exception as e:
            logger.error(f"Error encontrando posición objetivo en PDF: {e}")
            return None
    
    def _convert_geo_coords_to_pdf_position(self, geo_x: float, geo_y: float, map_data: dict, pdf_width: int, pdf_height: int) -> dict:
        """Convierte coordenadas geográficas a posición en el PDF usando análisis de elementos."""
        try:
            # Buscar elementos de referencia geográfica en el PDF
            text_regions = map_data.get('text_regions', [])
            parcels = map_data.get('parcels', [])
            
            # Buscar coordenadas de referencia en el texto del PDF
            reference_coords = []
            import re
            
            for text_region in text_regions:
                text_content = text_region.get('text', '')
                
                # Buscar patrones de coordenadas (formato típico: 5xxxxxx.xx, 6xxxxxx.xx)
                coord_patterns = re.findall(r'([56]\d{6}\.?\d*)', text_content)
                if len(coord_patterns) >= 2:
                    try:
                        coords = [float(c.replace('.', '')) if '.' not in c else float(c) for c in coord_patterns[:2]]
                        # Normalizar coordenadas si están sin punto decimal
                        normalized_coords = []
                        for coord in coords:
                            if coord > 1000000:  # Coordenada sin punto decimal
                                normalized_coords.append(coord / 100)
                            else:
                                normalized_coords.append(coord)
                        
                        tx, ty, tw, th = text_region['bbox']
                        reference_coords.append({
                            'geo_coords': normalized_coords,
                            'pdf_pos': (tx + tw/2, ty + th/2)
                        })
                        logger.info(f"Coordenadas de referencia encontradas: {normalized_coords} en posición PDF ({tx + tw/2:.1f}, {ty + th/2:.1f})")
                    except ValueError:
                        continue
            
            if len(reference_coords) >= 2:
                # Usar interpolación bilineal para estimar la posición
                return self._interpolate_position_from_references(geo_x, geo_y, reference_coords, pdf_width, pdf_height)
            elif len(reference_coords) == 1:
                # Usar una sola referencia con estimación de escala
                ref = reference_coords[0]
                ref_geo_x, ref_geo_y = ref['geo_coords']
                ref_pdf_x, ref_pdf_y = ref['pdf_pos']
                
                # Estimar escala basada en el tamaño típico del PDF
                estimated_scale = 0.15  # metros por pixel (estimación)
                
                # Calcular desplazamiento
                delta_x_meters = geo_x - ref_geo_x
                delta_y_meters = geo_y - ref_geo_y
                
                delta_x_pixels = delta_x_meters / estimated_scale
                delta_y_pixels = -delta_y_meters / estimated_scale  # Y invertida en PDF
                
                target_x = ref_pdf_x + delta_x_pixels
                target_y = ref_pdf_y + delta_y_pixels
                
                # Validar que esté dentro de los límites
                if 0 <= target_x <= pdf_width and 0 <= target_y <= pdf_height:
                    logger.info(f"Posición estimada con una referencia: ({target_x:.1f}, {target_y:.1f})")
                    return {'x': target_x, 'y': target_y}
            
            # Fallback: usar análisis de densidad mejorado con coordenadas geográficas
            return self._estimate_position_by_geographic_context(geo_x, geo_y, map_data, pdf_width, pdf_height)
            
        except Exception as e:
            logger.error(f"Error convirtiendo coordenadas geográficas a posición PDF: {e}")
            return None
    
    def _interpolate_position_from_references(self, target_x: float, target_y: float, references: list, pdf_width: int, pdf_height: int) -> dict:
        """Interpola la posición usando múltiples puntos de referencia."""
        try:
            # Encontrar los dos puntos de referencia más cercanos
            distances = []
            for ref in references:
                ref_x, ref_y = ref['geo_coords']
                dist = math.sqrt((target_x - ref_x)**2 + (target_y - ref_y)**2)
                distances.append((dist, ref))
            
            distances.sort(key=lambda x: x[0])
            closest_refs = distances[:2]
            
            if len(closest_refs) >= 2:
                ref1 = closest_refs[0][1]
                ref2 = closest_refs[1][1]
                
                # Calcular escala y orientación entre las referencias
                geo1_x, geo1_y = ref1['geo_coords']
                geo2_x, geo2_y = ref2['geo_coords']
                pdf1_x, pdf1_y = ref1['pdf_pos']
                pdf2_x, pdf2_y = ref2['pdf_pos']
                
                # Calcular vectores
                geo_vector = (geo2_x - geo1_x, geo2_y - geo1_y)
                pdf_vector = (pdf2_x - pdf1_x, pdf2_y - pdf1_y)
                
                # Calcular escala
                geo_distance = math.sqrt(geo_vector[0]**2 + geo_vector[1]**2)
                pdf_distance = math.sqrt(pdf_vector[0]**2 + pdf_vector[1]**2)
                
                if geo_distance > 0 and pdf_distance > 0:
                    scale = pdf_distance / geo_distance
                    
                    # Interpolar posición
                    target_vector = (target_x - geo1_x, target_y - geo1_y)
                    
                    # Aplicar escala y rotación si es necesaria
                    target_pdf_x = pdf1_x + target_vector[0] * scale
                    target_pdf_y = pdf1_y - target_vector[1] * scale  # Y invertida
                    
                    # Validar límites
                    if 0 <= target_pdf_x <= pdf_width and 0 <= target_pdf_y <= pdf_height:
                        logger.info(f"Posición interpolada: ({target_pdf_x:.1f}, {target_pdf_y:.1f}) con escala {scale:.6f}")
                        return {'x': target_pdf_x, 'y': target_pdf_y}
            
            return None
            
        except Exception as e:
            logger.error(f"Error en interpolación: {e}")
            return None
    
    def _estimate_position_by_geographic_context(self, geo_x: float, geo_y: float, map_data: dict, pdf_width: int, pdf_height: int) -> dict:
        """Estima la posición basándose en el contexto geográfico general."""
        try:
            # Usar análisis de densidad mejorado
            densest_area = self._find_densest_area(map_data, pdf_width, pdf_height)
            
            if densest_area and densest_area['confidence'] > 0.3:
                # Ajustar la posición basándose en las coordenadas geográficas
                # Si las coordenadas están en el rango típico de Rosario
                if 5.42e6 <= geo_x <= 5.46e6 and 6.34e6 <= geo_y <= 6.37e6:
                    # Aplicar un pequeño offset basado en la posición relativa en Rosario
                    x_offset_ratio = (geo_x - 5.42e6) / (5.46e6 - 5.42e6)  # 0-1
                    y_offset_ratio = (geo_y - 6.34e6) / (6.37e6 - 6.34e6)  # 0-1
                    
                    # Ajustar la posición del área más densa
                    adjusted_x = densest_area['x'] + (x_offset_ratio - 0.5) * pdf_width * 0.1
                    adjusted_y = densest_area['y'] + (0.5 - y_offset_ratio) * pdf_height * 0.1  # Y invertida
                    
                    # Mantener dentro de límites
                    adjusted_x = max(0, min(pdf_width, adjusted_x))
                    adjusted_y = max(0, min(pdf_height, adjusted_y))
                    
                    logger.info(f"Posición ajustada por contexto geográfico: ({adjusted_x:.1f}, {adjusted_y:.1f})")
                    return {'x': adjusted_x, 'y': adjusted_y}
                
                return {'x': densest_area['x'], 'y': densest_area['y']}
            
            return None
            
        except Exception as e:
            logger.error(f"Error estimando posición por contexto geográfico: {e}")
            return None
    

            
    def _extract_gauss_kruger_coordinates_from_pdf(self, pdf_path: str) -> dict:
        """Extrae las coordenadas Gauss-Krüger de la pestaña 'Datos Útiles' del PDF."""
        if not pdf_path or fitz is None:
            return None
        try:
            doc = fitz.open(pdf_path)
            if len(doc) < 2:  # Asumimos que 'Datos Útiles' está en la segunda página
                return None
            page = doc[1]  # Segunda página
            text = page.get_text()
            
            # Expresiones regulares para encontrar las coordenadas X e Y
            x_match = re.search(r"X\s*=\s*([\d\.,]+)", text)
            y_match = re.search(r"Y\s*=\s*([\d\.,]+)", text)
            
            if x_match and y_match:
                x_coord = float(x_match.group(1).replace('.', '').replace(',', '.'))
                y_coord = float(y_match.group(1).replace('.', '').replace(',', '.'))
                return {"x": x_coord, "y": y_coord}
            
            return None
        except Exception as e:
            logger.info(f"No se pudieron extraer las coordenadas Gauss-Krüger del PDF: {e}")
            return None

    def scrape_address(self, address):
        try:
            logger.info(f"--- Iniciando scraping para: {address} ---")
            self._setup_driver()
            info = self._get_pdf_filename(address)
        except Exception as e:
            logger.error(f"El scraping falló: {e}")
            if self.driver:
                try:
                    # Usar ruta absoluta para el screenshot
                    screenshot_path = os.path.join(os.getcwd(), "debug_screenshot.png")
                    self.driver.save_screenshot(screenshot_path)
                except Exception as e_ss:
                    logger.error(f"No se pudo guardar el screenshot: {e_ss}")
            return {
                'address': address,
                'success': False,
                'pdf_filename': None,
                'full_pdf_url': None,
                'pdf_path': None,
                'crop_image_path': None,
                'latlon': None,
                'error': str(e)
            }
        finally:
            self._close_driver()
        
        # Descargar PDF
        pdf_path = self._download_pdf(info['pdf_filename'])

        # Extraer coordenadas Gauss-Krüger del PDF
        gauss_kruger_coords = self._extract_gauss_kruger_coordinates_from_pdf(pdf_path)

        # Extraer PDF como imagen completa
        crop_path = None
        lat = lon = None
        used_srs = None
        try:
            if self.last_point:
                x, y = self.last_point
                # Transformación a WGS84 (para referencia)
                lat, lon, epsg_used = self._transform_to_wgs84(x, y)
                used_srs = epsg_used
                
            # Crear directorio de salida
            crops_dir = "output/crops"
            # Convertir a ruta absoluta si es relativa
            if not os.path.isabs(crops_dir):
                crops_dir = os.path.join(os.getcwd(), crops_dir)
            
            # Crear directorio con permisos específicos
            try:
                os.makedirs(crops_dir, exist_ok=True, mode=0o775)
                try:
                    os.chmod(crops_dir, 0o775)
                    logger.info(f"Directorio {crops_dir} creado/actualizado con permisos 775")
                except (PermissionError, OSError):
                    # Si no se pueden cambiar permisos, intentar con umask
                    old_umask = os.umask(0o002)
                    try:
                        os.makedirs(crops_dir, exist_ok=True)
                        logger.info(f"Directorio {crops_dir} creado con umask modificado")
                    finally:
                        os.umask(old_umask)
            except Exception as e:
                logger.warning(f"Error creando directorio {crops_dir}: {e}")
                # Intentar crear sin permisos específicos como fallback
                os.makedirs(crops_dir, exist_ok=True)
            safe_addr = re.sub(r"[^a-zA-Z0-9_-]+", "_", address)
            crop_path = os.path.join(crops_dir, f"{safe_addr}_{info['pdf_filename'].replace('.pdf','')}_full.png")
            
            # Extraer PDF como imagen completa
            try:
                pdf_img = self._extract_pdf_as_image(pdf_path, zoom_factor=4.0, address=address)
                if pdf_img is not None:
                    pdf_img.save(crop_path)
                    logger.info(f"PDF extraído como imagen completa y guardado en: {crop_path}")
                else:
                    logger.warning("No se pudo extraer el PDF como imagen")
            except Exception as e_pdf:
                logger.error(f"Error al extraer PDF como imagen: {e_pdf}")
                # Fallback: crear imagen vacía
                if Image is not None:
                    fallback_img = Image.new('RGB', (800, 600), 'white')
                    fallback_img.save(crop_path)
                    logger.info("Imagen fallback creada.")
        except Exception as ee:
            logger.warning(f"No se pudo extraer el PDF como imagen: {ee}")

        logger.info(f"ÉXITO! PDF encontrado: {info['pdf_filename']}")
        return {
            'address': address,
            'success': True,
            'pdf_filename': info['pdf_filename'],
            'full_pdf_url': info['pdf_url'],
            'pdf_path': str(pdf_path) if pdf_path else None,
            'crop_image_path': crop_path,
            'lat': lat,
            'lon': lon,
            'source_srs': used_srs,
            'gauss_kruger_coords': gauss_kruger_coords,
            'error': None
        }

def main():
    # Detectar automáticamente el path del ChromeDriver según el entorno
    if os.path.exists("/usr/local/bin/chromedriver"):
        # Entorno de contenedor/Ubuntu
        CHROME_DRIVER_PATH = "/usr/local/bin/chromedriver"
    elif os.path.exists("C:/Users/Usuario/Documents/Proyectos/Scrapper/Final/CATASTRAL/chrome-win64/chromedriver.exe"):
        # Entorno local Windows
        CHROME_DRIVER_PATH = "C:/Users/Usuario/Documents/Proyectos/Scrapper/Final/CATASTRAL/chrome-win64/chromedriver.exe"
    else:
        # Buscar en el directorio actual
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_chromedriver = os.path.join(current_dir, "chrome-win64", "chromedriver.exe")
        if os.path.exists(local_chromedriver):
            CHROME_DRIVER_PATH = local_chromedriver
        else:
            # Fallback: usar variable de entorno o path por defecto
            CHROME_DRIVER_PATH = os.environ.get('CHROMEDRIVER_PATH', '/usr/local/bin/chromedriver')

    parser = argparse.ArgumentParser(description="Scraper de InfoMapa Rosario para obtener PDF de Cartobase")
    parser.add_argument("-a", "--address", help="Dirección a buscar, por ejemplo: 'corrientes 241'", default=None)
    parser.add_argument("--headless", action="store_true", help="Ejecutar el navegador en modo headless")
    args = parser.parse_args()

    # Si no se pasa por CLI, consultamos por input interactivo
    if not args.address:
        while True:
            try:
                entered = input("Ingrese la dirección a buscar (ej: corrientes 241): ").strip()
            except EOFError:
                entered = ""
            if entered:
                address = entered
                break
            print("La dirección no puede estar vacía. Intente nuevamente.")
    else:
        address = args.address

    scraper = RosarioScraper(driver_path=CHROME_DRIVER_PATH, headless=args.headless is True)
    result = scraper.scrape_address(address)
    
    print("\n" + "="*50)
    print("RESULTADO FINAL DEL SCRAPING")
    print("="*50)
    print(f"Dirección: {result['address']}")
    print(f"Éxito: {result['success']}")
    
    if result['success']:
        print(f"Archivo PDF: {result['pdf_filename']}")
        print(f"URL completa: {result['full_pdf_url']}")
        if result.get('pdf_path'):
            print(f"PDF descargado en: {result['pdf_path']}")
        if result.get('crop_image_path'):
            print(f"Recorte del lote guardado en: {result['crop_image_path']}")
        if result.get('lat') is not None and result.get('lon') is not None:
            print(f"Centro aproximado (WGS84): lat={result['lat']:.6f}, lon={result['lon']:.6f} (SRS origen: {result.get('source_srs')})")
    else:
        print(f"Error: {result['error']}")
    
    print("="*50)

if __name__ == "__main__":
    main()
