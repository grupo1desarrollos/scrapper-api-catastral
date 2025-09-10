#!/bin/bash

# Script de inicio para la aplicación Rosario Scraper API

echo "Iniciando Rosario Scraper API..."

# Iniciar Xvfb para el display virtual (necesario para Chrome headless)
echo "Iniciando Xvfb..."
Xvfb :99 -screen 0 1920x1080x24 &
XVFB_PID=$!

# Esperar un momento para que Xvfb se inicie
sleep 2

# Verificar que ChromeDriver está disponible
echo "Verificando ChromeDriver..."
if [ -f "/usr/local/bin/chromedriver" ]; then
    echo "ChromeDriver encontrado en /usr/local/bin/chromedriver"
    /usr/local/bin/chromedriver --version
else
    echo "ERROR: ChromeDriver no encontrado"
    exit 1
fi

# Verificar que Chrome está disponible
echo "Verificando Google Chrome..."
if command -v google-chrome &> /dev/null; then
    echo "Google Chrome encontrado"
    google-chrome --version
else
    echo "ERROR: Google Chrome no encontrado"
    exit 1
fi

# Función para limpiar procesos al salir
cleanup() {
    echo "Cerrando aplicación..."
    if [ ! -z "$XVFB_PID" ]; then
        kill $XVFB_PID 2>/dev/null
    fi
    exit 0
}

# Configurar trap para limpiar al salir
trap cleanup SIGTERM SIGINT

# Actualizar el path del ChromeDriver en rosario_scraper.py si es necesario
echo "Configurando ChromeDriver path..."
sed -i 's|chromedriver_path = .*|chromedriver_path = "/usr/local/bin/chromedriver"|g' /app/rosario_scraper.py

# Iniciar la aplicación Flask con Gunicorn
echo "Iniciando aplicación Flask con Gunicorn..."
exec gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 600 --keep-alive 2 --max-requests 100 --max-requests-jitter 10 app:app