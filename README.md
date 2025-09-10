# Scraper InfoMapa Rosario

Este scraper automatiza la extracción de nombres de archivos PDF del sitio web InfoMapa de Rosario (https://infomapa.rosario.gov.ar/emapa/mapa.htm).

## Funcionalidad

El scraper:
1. Busca una dirección específica en el sitio web
2. Navega a la pestaña "Cartobase" 
3. Extrae el nombre del archivo PDF del enlace "Registro Gráfico"
4. Devuelve el nombre del archivo (ej: "01038.pdf") y la URL completa para descarga

## Instalación

### 1. Descargar Google Chrome

Asegúrate de tener instalado el navegador [Google Chrome](https://www.google.com/chrome/).

### 2. Descargar ChromeDriver

El scraper necesita un archivo llamado `chromedriver.exe` para poder controlar Google Chrome.

1.  **Verifica tu versión de Chrome**:
    *   Abre Chrome, ve a `chrome://settings/help` y anota el número de versión (ej: `128.0.6613.84`).

2.  **Descarga el ChromeDriver correspondiente**:
    *   Visita el [Dashboard de ChromeDriver](https://googlechromelabs.github.io/chrome-for-testing/).
    *   Busca la sección que coincida con tu versión de Chrome.
    *   En la columna `chromedriver`, haz clic en la URL de `win64` para descargar el archivo ZIP.

3.  **Extrae el archivo**:
    *   Descomprime el archivo ZIP que descargaste.
    *   Dentro de la carpeta, encontrarás `chromedriver.exe`.
    *   Copia este archivo a una ubicación segura en tu sistema (ej: `C:\Users\TuUsuario\Documents\drivers\`).

### 3. Instalar dependencias de Python

```bash
pip install -r requirements.txt
```

## Configuración y Uso

### 1. Configurar la ruta del driver

Antes de ejecutar el scraper, debes indicarle dónde se encuentra el archivo `chromedriver.exe` que descargaste.

Abre el archivo `example_usage.py` (o `rosario_scraper.py` si lo usas directamente) y modifica la siguiente línea:

```python
# Reemplaza esta ruta con la ubicación real de tu chromedriver.exe
CHROME_DRIVER_PATH = "C:/path/to/your/chromedriver.exe"
```

Por ejemplo:

```python
CHROME_DRIVER_PATH = "C:/Users/Usuario/Documents/drivers/chromedriver.exe"
```

### 2. Ejecutar el scraper

Una vez configurada la ruta, puedes ejecutar el script de ejemplo:

```bash
python example_usage.py
```

### 3. Integración con n8n

Para usarlo en un nodo de n8n, puedes adaptar la función `extract_pdf_for_n8n` del archivo `example_usage.py`. Asegúrate de pasar la ruta al `chromedriver.exe` como un argumento o variable de entorno.

```python
from rosario_scraper import RosarioScraper

# La ruta al driver puede ser un valor fijo o venir de una variable de n8n
chrome_driver_path = "/path/on/n8n/server/chromedriver.exe"
address_to_search = "corrientes 241"

def extract_pdf_for_n8n(address, driver_path):
    scraper = RosarioScraper(driver_path=driver_path, headless=True)
    return scraper.scrape_address(address)

# Llamar a la función
result = extract_pdf_for_n8n(address_to_search, chrome_driver_path)

# El resultado estará disponible en la variable 'result'
```

## Estructura del resultado

```python
{
    'address': 'corrientes 241',
    'success': True/False,
    'pdf_filename': '01038.pdf',  # Nombre del archivo
    'full_pdf_url': 'https://infomapa.rosario.gov.ar/emapa/servlets/verArchivo?path=manzanas/01038.pdf',
    'error': None  # Mensaje de error si success=False
}
```

## Ejemplos de direcciones

- "corrientes 241"
- "san martin 1234" 
- "mitre 500"
- "francia 50 bis"
- "corrientes 4550 A"

## Notas importantes

- **Ruta del Driver**: El error más común será una ruta incorrecta al `chromedriver.exe`. Asegúrate de que la ruta en los scripts es la correcta.
- **Versión de Chrome**: Si actualizas Google Chrome, es posible que necesites descargar una nueva versión de ChromeDriver que sea compatible.
- **Modo Headless**: Para automatización (como en n8n), se recomienda usar `headless=True` para que no se abra una ventana del navegador.

## Troubleshooting

Si encuentras errores:
1. Verifica que Chrome esté instalado
2. Revisa los logs para detalles del error
3. Prueba con modo headless=False para debug visual
4. Asegúrate de que la dirección existe en Rosario
