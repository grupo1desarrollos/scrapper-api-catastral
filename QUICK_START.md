# 🚀 Quick Start - Rosario Scraper API

## Deployment en Coolify (Ubuntu)

### 1. Preparación
```bash
# Clonar el repositorio en tu servidor
git clone <tu-repositorio>
cd CATASTRAL
```

### 2. Deploy en Coolify
1. **Crear nuevo proyecto** en Coolify Dashboard
2. **Tipo**: Docker Compose
3. **Repositorio**: Tu repositorio Git
4. **Branch**: main/master
5. **Deploy** automático

### 3. Verificar Deployment
```bash
# Health check
curl https://tu-dominio.coolify.app/health
```

## 🔥 Uso Rápido

### Comando equivalente a CLI
```bash
# Antes (CLI local):
python rosario_scraper.py -a "alem 1538" --headless

# Ahora (API):
curl -X POST https://tu-dominio.coolify.app/scrape \
  -H "Content-Type: application/json" \
  -d '{"address": "alem 1538"}' \
  --output catastral.png
```

### Webhook para n8n/Zapier
```json
{
  "method": "POST",
  "url": "https://tu-dominio.coolify.app/scrape",
  "headers": {"Content-Type": "application/json"},
  "body": {"address": "{{ direccion }}"}
}
```

## 📁 Archivos Creados

- `app.py` - API Flask principal
- `Dockerfile` - Configuración del contenedor
- `docker-compose.yml` - Orquestación
- `start.sh` - Script de inicio
- `requirements.txt` - Dependencias actualizadas
- `DEPLOYMENT.md` - Documentación completa
- `.dockerignore` - Optimización del build

## ⚡ Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/scrape` | POST | Retorna imagen PNG |
| `/scrape/json` | POST | Retorna JSON con URLs |
| `/download/image/{file}` | GET | Descarga imagen |
| `/download/pdf/{file}` | GET | Descarga PDF |

## 🔧 Troubleshooting

```bash
# Ver logs
docker logs rosario-scraper-api

# Reiniciar
docker-compose restart

# Reconstruir
docker-compose build --no-cache
```

## 📊 Performance

- **Tiempo de respuesta**: 30-60 segundos
- **Memoria**: ~1.5GB durante scraping
- **Concurrencia**: 2 workers por defecto
- **Timeout**: 5 minutos

---

✅ **¡Listo para producción!** Tu scraper ahora es un servicio API escalable.