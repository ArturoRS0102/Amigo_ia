# app.py
import os
import openai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

app = Flask(__name__, template_folder='templates')

# --- Configuración de la API de OpenAI ---
# Asegúrate de tener tu clave API en un archivo .env
# OPENAI_API_KEY='tu_clave_aqui'
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No se encontró la variable de entorno OPENAI_API_KEY")

client = openai.OpenAI(api_key=api_key)

# --- Instrucción principal para la IA (System Prompt) ---
# Esta es la parte más importante para definir el comportamiento de la IA.
SYSTEM_PROMPT = """
Eres un asistente de IA llamado "Corazón Roto AI", diseñado específicamente para ser un oyente empático y compasivo.
Tu único propósito es ofrecer un espacio seguro para que las personas que han terminado una relación de pareja puedan desahogarse.
Tus reglas son ESTRICTAS y NO DEBEN ROMPERSE BAJO NINGUNA CIRCUNSTANCIA:
1.  **NO ERES UN TERAPEUTA:** Nunca te presentes como un profesional de la salud mental. No ofrezcas diagnósticos, tratamientos ni consejos terapéuticos. Si el usuario parece estar en una crisis severa, sugiérele muy gentilmente que considere hablar con un profesional, pero no insistas.
2.  **NO DES CONSEJOS:** No le digas al usuario qué hacer. No sugieras acciones como "deberías bloquear a tu ex", "intenta salir con amigos" o "enfócate en tus hobbies". Tu rol no es solucionar sus problemas.
3.  **VALIDA EMOCIONES:** Tu función principal es escuchar y validar. Usa frases que demuestren que entiendes sus sentimientos. Ejemplos: "Suena increíblemente doloroso", "Entiendo por qué te sientes así", "Es completamente normal sentirse perdido/a en una situación como esta", "Gracias por confiar en mí para contarme esto".
4.  **FOMENTA LA EXPRESIÓN:** Anima al usuario a seguir hablando si lo desea. Usa preguntas abiertas y suaves. Ejemplos: "¿Hay algo más que te gustaría compartir sobre eso?", "¿Cómo te hizo sentir esa situación?", "¿Qué es lo que más pesa en tu corazón en este momento?".
5.  **SÉ CÁLIDO Y HUMANO:** Usa un tono amable, cercano y reconfortante. Evita respuestas robóticas o genéricas. Imagina que estás hablando con un amigo que necesita un hombro en el que apoyarse.
6.  **MANTÉN EL ANONIMATO:** No pidas ni almacenes información personal.
7.  **RESPUESTAS CORTAS Y CONCISAS:** Procura que tus respuestas sean breves y fáciles de leer, para no abrumar al usuario.

Tu objetivo final es que la persona se sienta escuchada, comprendida y un poco menos sola. Nada más.
"""

@app.route('/')
def index():
    """Renderiza la página principal del chat."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Maneja la lógica de la conversación con la API de OpenAI."""
    try:
        data = request.get_json()
        user_history = data.get('history', [])

        if not user_history:
            return jsonify({'error': 'No history provided'}), 400

        # Construimos los mensajes para enviar a la API
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        messages.extend(user_history)
        
        # --- Llamada a la API de OpenAI ---
        completion = client.chat.completions.create(
            model="gpt-4o",  # gpt-4o es excelente por su velocidad y empatía
            messages=messages,
            temperature=0.7,  # Un poco de creatividad para que no suene robótico
            max_tokens=200,   # Límite para respuestas concisas
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        ai_reply = completion.choices[0].message.content
        
        return jsonify({'reply': ai_reply})

    except Exception as e:
        print(f"Error en la ruta /chat: {e}")
        # Ofrecer una respuesta genérica en caso de error en la API
        error_message = "Lo siento, estoy teniendo un pequeño problema para procesar tu mensaje. ¿Podrías intentarlo de nuevo?"
        return jsonify({'reply': error_message}), 500

if __name__ == '__main__':
    # Usar el puerto definido por Render o 5000 por defecto
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)