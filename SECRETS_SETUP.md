# Configuración del Asistente IA

Para habilitar el Asistente IA en Streamlit Cloud:

1. Ve a tu app en Streamlit Cloud
2. Click en **Settings** (⚙️)
3. En la sección **Secrets**, agrega:

```toml
GROQ_API_KEY = "tu_api_key_de_groq_aqui"
```

4. Guarda y la app se reiniciará automáticamente con el Asistente IA habilitado.

## Obtener API Key
Visita [Groq Console](https://console.groq.com) para obtener tu API key gratuita.

## Nota de seguridad
El archivo `.streamlit/secrets.toml` está en `.gitignore` para no subir la API key a GitHub.
