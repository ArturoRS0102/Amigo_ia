# app.py
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__, template_folder='templates')

# --- Configuración de la API de OpenAI ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No se encontró la variable de entorno OPENAI_API_KEY")

# Instanciar el cliente con la nueva librería openai>=1.0.0
client = OpenAI(api_key=api_key)

# --- System Prompt ---
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
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    history = data.get('history')
    if not history:
        return jsonify({'error': 'Debe proporcionar un historial de conversación.'}), 400

    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}] + history

    try:
        # Usamos la nueva interfaz client.chat.completions.create
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        reply = response.choices[0].message.content.strip()
        return jsonify({'reply': reply})

    except OpenAIError as oe:
        app.logger.error(f"OpenAI API error: {oe}")
        return jsonify({
            'reply': "Lo siento, hay un problema comunicándome con la IA. Por favor inténtalo de nuevo más tarde."
        }), 502

    except Exception as e:
        app.logger.exception("Error en la ruta /chat")
        return jsonify({
            'reply': "Ocurrió un error inesperado. ¿Podrías intentarlo de nuevo?"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
