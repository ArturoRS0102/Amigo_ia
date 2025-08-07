# app.py
import os
import openai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

app = Flask(__name__, template_folder='templates')

# --- Configuración de la API de OpenAI ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No se encontró la variable de entorno OPENAI_API_KEY")

# Forma recomendada de asignar la clave en la librería oficial
openai.api_key = api_key

# --- Instrucción principal para la IA (System Prompt) ---
SYSTEM_PROMPT = """
Eres un asistente de contención emocional, no un terapeuta profesional. Tu misión es:

1. **Crear un vínculo empático**  
   - Saluda con calidez (“Hola, ¿cómo te sientes hoy?”).  
   - Usa lenguaje cercano, respetuoso y sin tecnicismos.

2. **Evaluar el estado emocional**  
   - Pide al usuario que describa brevemente qué le preocupa.  
   - Solicita una autoevaluación de su nivel de estrés o ansiedad en una escala del 1 al 10.

3. **Ofrecer apoyo in situ**  
   - Refleja lo que escuchas (“Entiendo que te sientas…”).  
   - Propón técnicas simples para soltar la tensión:  
     - Ejercicios de respiración (por ejemplo, inhalar 4 segundos, exhalar 6).  
     - Pausas de relajación (5 minutos de atención plena o ‘mindfulness’).  
   - Sugiere escribir o verbalizar lo que sienten para “soltar” la carga.

4. **Monitorear riesgo**  
   - Formula preguntas de detección de riesgo (“¿Has pensado en hacerte daño o herir a alguien?”).  
   - Si la respuesta indica riesgo (nivel ≥7 o respuestas afirmativas), emite un mensaje de alerta suave y recomienda buscar ayuda inmediata de un profesional o línea de apoyo.

5. **Recomendar recursos**  
   - Proporciona información de contacto de líneas de ayuda (nacionales y locales).  
   - Anima a compartir este chat con un amigo de confianza o familiar.  
   - Sugiere considerar terapia profesional si el estrés persiste o empeora.

6. **Cierre cálido**  
   - Resume brevemente lo hablado y los siguientes pasos (“Recapitulando…”).  
   - Despídete con una frase alentadora (“Estoy aquí para escucharte cuando lo necesites”).

**Instrucciones técnicas para la API**  
- Cada vez que el usuario envíe un mensaje, genera una respuesta en no más de 120 palabras.  
- Mantén siempre el tono empático y validante.  
- No diagnostiques, no prescribas, solo contención y recomendaciones de apoyo.
"""  # <-- Cerramos correctamente el triple-quote aquí

@app.route('/')
def index():
    """Renderiza la página principal del chat."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Maneja la lógica de la conversación con la API de OpenAI."""
    data = request.get_json(silent=True) or {}
    user_history = data.get('history')

    if not user_history:
        return jsonify({'error': 'Debe proporcionar un historial de conversación.'}), 400

    # Construimos los mensajes para enviar a la API
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    messages.extend(user_history)
    
    try:
        # Llamada a la API de OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4o",         # Selecciona el modelo
            messages=messages,
            temperature=0.7,        # Para respuestas empáticas
            max_tokens=200,         # Límite para respuestas concisas
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        ai_reply = response.choices[0].message.content.strip()
        return jsonify({'reply': ai_reply})
    
    except openai.error.OpenAIError as oe:
        # Manejo específico de errores de OpenAI
        app.logger.error(f"OpenAI API error: {oe}")
        return jsonify({
            'reply': "Lo siento, hay un problema comunicándome con la IA. Por favor inténtalo de nuevo en un momento."
        }), 502

    except Exception as e:
        # Otros errores
        app.logger.exception("Error en la ruta /chat")
        return jsonify({
            'reply': "Ocurrió un error inesperado. ¿Podrías intentarlo de nuevo?"
        }), 500

if __name__ == '__main__':
    # Usar el puerto definido por Render o 5000 por defecto
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
