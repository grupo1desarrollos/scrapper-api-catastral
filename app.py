from flask import Flask, request, jsonify, send_file
import os
import subprocess
import tempfile
import logging
from datetime import datetime
import json
from pathlib import Path

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definir directorios - usar /tmp para evitar problemas de permisos
BASE_DIR = Path(__file__).parent
if os.path.exists('/tmp'):  # Sistema Linux/Unix
    OUTPUT_DIR = Path('/tmp/scraper_output')
else:  # Sistema Windows (desarrollo local)
    OUTPUT_DIR = BASE_DIR / "output"
PDFS_DIR = OUTPUT_DIR / "pdfs"
CROPS_DIR = OUTPUT_DIR / "crops"

# Crear directorios necesarios con permisos específicos
for directory in [OUTPUT_DIR, PDFS_DIR, CROPS_DIR]:
    try:
        # Crear directorio con permisos 775 directamente
        directory.mkdir(parents=True, exist_ok=True, mode=0o775)
        # Intentar cambiar permisos si ya existe
        try:
            os.chmod(directory, 0o775)
            logger.info(f"Directorio {directory} creado/actualizado con permisos 775")
        except (PermissionError, OSError):
            # Si no se pueden cambiar permisos, intentar con umask
            old_umask = os.umask(0o002)  # Permite escritura para grupo
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directorio {directory} creado con umask modificado")
            finally:
                os.umask(old_umask)  # Restaurar umask original
    except PermissionError:
        logger.warning(f"No se pudieron crear permisos para {directory}, usando directorio existente")
    except Exception as e:
        logger.error(f"Error creando directorio {directory}: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de salud para verificar que el servicio está funcionando"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'rosario-scraper-api'
    })

@app.route('/scrape', methods=['POST'])
def scrape_address():
    """Endpoint principal para procesar direcciones via webhook"""
    try:
        # Obtener datos del request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se recibieron datos JSON'}), 400
        
        # Extraer dirección del payload
        address = data.get('address') or data.get('direccion')
        
        if not address:
            return jsonify({'error': 'Campo "address" o "direccion" requerido'}), 400
        
        logger.info(f"Procesando dirección: {address}")
        
        # Ejecutar el scraper
        cmd = [
            'python3', 
            str(BASE_DIR / 'rosario_scraper.py'),
            '-a', address,
            '--headless'
        ]
        
        logger.info(f"Ejecutando comando: {' '.join(cmd)}")
        
        # Ejecutar el comando
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=BASE_DIR,
            timeout=300  # 5 minutos timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Error en scraper: {result.stderr}")
            return jsonify({
                'error': 'Error ejecutando scraper',
                'details': result.stderr
            }), 500
        
        # Buscar la imagen más reciente en crops
        crop_files = list(CROPS_DIR.glob('*.png'))
        if not crop_files:
            return jsonify({'error': 'No se generó imagen de resultado'}), 500
        
        # Obtener el archivo más reciente
        latest_crop = max(crop_files, key=os.path.getctime)
        
        logger.info(f"Imagen generada: {latest_crop}")
        
        # Retornar la imagen
        return send_file(
            latest_crop,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'catastral_{address.replace(" ", "_")}.png'
        )
        
    except subprocess.TimeoutExpired:
        logger.error("Timeout ejecutando scraper")
        return jsonify({'error': 'Timeout procesando dirección'}), 504
    
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        return jsonify({
            'error': 'Error interno del servidor',
            'details': str(e)
        }), 500

@app.route('/scrape/json', methods=['POST'])
def scrape_address_json():
    """Endpoint alternativo que retorna información en JSON"""
    try:
        # Obtener datos del request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se recibieron datos JSON'}), 400
        
        # Extraer dirección del payload
        address = data.get('address') or data.get('direccion')
        
        if not address:
            return jsonify({'error': 'Campo "address" o "direccion" requerido'}), 400
        
        logger.info(f"Procesando dirección (JSON): {address}")
        
        # Ejecutar el scraper
        cmd = [
            'python3', 
            str(BASE_DIR / 'rosario_scraper.py'),
            '-a', address,
            '--headless'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=BASE_DIR,
            timeout=300
        )
        
        if result.returncode != 0:
            logger.error(f"Error en scraper: {result.stderr}")
            return jsonify({
                'success': False,
                'error': 'Error ejecutando scraper',
                'details': result.stderr
            }), 500
        
        # Buscar archivos generados
        crop_files = list(CROPS_DIR.glob('*.png'))
        pdf_files = list(PDFS_DIR.glob('*.pdf'))
        
        if not crop_files:
            return jsonify({
                'success': False,
                'error': 'No se generó imagen de resultado'
            }), 500
        
        # Obtener archivos más recientes
        latest_crop = max(crop_files, key=os.path.getctime)
        latest_pdf = max(pdf_files, key=os.path.getctime) if pdf_files else None
        
        response_data = {
            'success': True,
            'address': address,
            'timestamp': datetime.now().isoformat(),
            'files': {
                'image': str(latest_crop.relative_to(BASE_DIR)),
                'pdf': str(latest_pdf.relative_to(BASE_DIR)) if latest_pdf else None
            },
            'download_urls': {
                'image': f'/download/image/{latest_crop.name}',
                'pdf': f'/download/pdf/{latest_pdf.name}' if latest_pdf else None
            }
        }
        
        return jsonify(response_data)
        
    except subprocess.TimeoutExpired:
        logger.error("Timeout ejecutando scraper")
        return jsonify({
            'success': False,
            'error': 'Timeout procesando dirección'
        }), 504
    
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Error interno del servidor',
            'details': str(e)
        }), 500

@app.route('/download/image/<filename>', methods=['GET'])
def download_image(filename):
    """Endpoint para descargar imágenes generadas"""
    try:
        file_path = CROPS_DIR / filename
        if not file_path.exists():
            return jsonify({'error': 'Archivo no encontrado'}), 404
        
        return send_file(
            file_path,
            mimetype='image/png',
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/pdf/<filename>', methods=['GET'])
def download_pdf(filename):
    """Endpoint para descargar PDFs generados"""
    try:
        file_path = PDFS_DIR / filename
        if not file_path.exists():
            return jsonify({'error': 'Archivo no encontrado'}), 404
        
        return send_file(
            file_path,
            mimetype='application/pdf',
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Configuración para desarrollo
    app.run(host='0.0.0.0', port=5000, debug=False)