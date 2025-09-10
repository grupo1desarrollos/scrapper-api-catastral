# Rosario Scraper API - Deployment en Coolify

## Descripción

API REST para extraer información catastral de InfoMapa Rosario mediante webhooks. La aplicación recibe una dirección, ejecuta el scraper y retorna la imagen del lote catastral.

## Arquitectura

- **Backend**: Flask API con Gunicorn
- **Scraper**: Selenium + Chrome headless
- **Containerización**: Docker + Docker Compose
- **Deployment**: Coolify en Ubuntu

## Estructura del Proyecto

```
CATASTRAL/
├── app.py                 # API Flask principal
├── rosario_scraper.py     # Scraper de InfoMapa
├── requirements.txt       # Dependencias Python
├── Dockerfile            # Configuración del contenedor
├── docker-compose.yml    # Orquestación de servicios
├── start.sh             # Script de inicio
├── output/              # Archivos generados
│   ├── crops/          # Imágenes recortadas
│   ├── pdfs/           # PDFs descargados
│   └── debug/          # Archivos de debug
└── README.md           # Documentación original
```

## Deployment en Coolify

### 1. Preparación del Servidor

Asegúrate de que tu servidor Ubuntu tenga:
- Docker instalado
- Coolify configurado
- Puertos disponibles (5000 por defecto)

### 2. Configuración en Coolify

1. **Crear nuevo proyecto en Coolify**:
   - Tipo: Docker Compose
   - Repositorio: Tu repositorio Git con este código

2. **Variables de entorno** (opcional):
   ```bash
   FLASK_ENV=production
   CHROMEDRIVER_PATH=/usr/local/bin/chromedriver
   ```

3. **Configuración de red**:
   - Puerto interno: 5000
   - Puerto externo: Asignado por Coolify
   - Protocolo: HTTP

### 3. Deploy Automático

Coolify detectará automáticamente el `docker-compose.yml` y desplegará la aplicación.

## Endpoints de la API

### 1. Health Check
```http
GET /health
```

**Respuesta**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "service": "rosario-scraper-api"
}
```

### 2. Scraping con Imagen (Principal)
```http
POST /scrape
Content-Type: application/json

{
  "address": "alem 1538"
}
```

**Respuesta**: Imagen PNG directa (binary)
- Content-Type: `image/png`
- Filename: `catastral_alem_1538.png`

### 3. Scraping con JSON
```http
POST /scrape/json
Content-Type: application/json

{
  "address": "alem 1538"
}
```

**Respuesta**:
```json
{
  "success": true,
  "address": "alem 1538",
  "timestamp": "2024-01-15T10:30:00",
  "files": {
    "image": "output/crops/alem_1538_crop.png",
    "pdf": "output/pdfs/alem_1538.pdf"
  },
  "download_urls": {
    "image": "/download/image/alem_1538_crop.png",
    "pdf": "/download/pdf/alem_1538.pdf"
  }
}
```

### 4. Descarga de Archivos
```http
GET /download/image/{filename}
GET /download/pdf/{filename}
```

## Uso con Webhooks

### Ejemplo con curl
```bash
# Obtener imagen directamente
curl -X POST https://tu-dominio.coolify.app/scrape \
  -H "Content-Type: application/json" \
  -d '{"address": "alem 1538"}' \
  --output catastral.png

# Obtener información en JSON
curl -X POST https://tu-dominio.coolify.app/scrape/json \
  -H "Content-Type: application/json" \
  -d '{"address": "alem 1538"}'
```

### Ejemplo con n8n
```json
{
  "method": "POST",
  "url": "https://tu-dominio.coolify.app/scrape",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": {
    "address": "{{ $json.direccion }}"
  }
}
```

### Ejemplo con Python
```python
import requests

url = "https://tu-dominio.coolify.app/scrape"
data = {"address": "alem 1538"}

response = requests.post(url, json=data)

if response.status_code == 200:
    with open("catastral.png", "wb") as f:
        f.write(response.content)
    print("Imagen guardada como catastral.png")
else:
    print(f"Error: {response.status_code}")
```

## Monitoreo y Logs

### Ver logs en Coolify
1. Accede al dashboard de Coolify
2. Selecciona tu aplicación
3. Ve a la pestaña "Logs"

### Logs de la aplicación
```bash
# Logs del contenedor
docker logs rosario-scraper-api

# Logs en tiempo real
docker logs -f rosario-scraper-api
```

### Métricas importantes
- Tiempo de respuesta (típicamente 30-60 segundos)
- Uso de memoria (pico ~1.5GB durante scraping)
- Uso de CPU (pico durante procesamiento de imágenes)

## Troubleshooting

### Problemas Comunes

1. **Error de ChromeDriver**:
   ```bash
   # Verificar que Chrome esté instalado
   docker exec rosario-scraper-api google-chrome --version
   
   # Verificar ChromeDriver
   docker exec rosario-scraper-api /usr/local/bin/chromedriver --version
   ```

2. **Timeout en requests**:
   - Aumentar timeout en el cliente
   - Verificar conectividad a InfoMapa Rosario

3. **Memoria insuficiente**:
   - Aumentar límites en docker-compose.yml
   - Monitorear uso con `docker stats`

4. **Permisos de archivos**:
   ```bash
   # Verificar permisos del directorio output
   docker exec rosario-scraper-api ls -la /app/output
   ```

### Comandos Útiles

```bash
# Reiniciar servicio
docker-compose restart

# Reconstruir imagen
docker-compose build --no-cache

# Acceder al contenedor
docker exec -it rosario-scraper-api bash

# Ver estado de servicios
docker-compose ps

# Limpiar archivos generados
docker exec rosario-scraper-api rm -rf /app/output/crops/*
docker exec rosario-scraper-api rm -rf /app/output/pdfs/*
```

## Escalabilidad

### Configuración de Producción

1. **Múltiples workers**:
   ```bash
   # En start.sh, aumentar workers
   gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
   ```

2. **Load balancer**:
   - Configurar múltiples instancias en Coolify
   - Usar nginx como proxy reverso

3. **Persistencia de datos**:
   - Configurar volúmenes persistentes
   - Backup automático de archivos generados

### Optimizaciones

1. **Cache de resultados**:
   - Implementar Redis para cachear resultados
   - Evitar re-scraping de direcciones recientes

2. **Queue system**:
   - Usar Celery para procesamiento asíncrono
   - Manejar múltiples requests concurrentes

## Seguridad

### Recomendaciones

1. **Rate limiting**:
   ```python
   from flask_limiter import Limiter
   limiter = Limiter(app, key_func=get_remote_address)
   
   @app.route('/scrape', methods=['POST'])
   @limiter.limit("10 per minute")
   def scrape_address():
       # ...
   ```

2. **Autenticación**:
   - Implementar API keys
   - Usar JWT tokens

3. **HTTPS**:
   - Configurar SSL en Coolify
   - Forzar conexiones seguras

## Contacto y Soporte

Para problemas o mejoras:
1. Revisar logs de la aplicación
2. Verificar conectividad a InfoMapa Rosario
3. Consultar documentación de Coolify

---

**Nota**: Esta API está optimizada para el sitio InfoMapa de Rosario. Cambios en el sitio web pueden requerir actualizaciones del scraper.