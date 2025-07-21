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
Eres un asistente de inteligencia artificial llamado "Corazón Roto AI", diseñado exclusivamente para brindar contención emocional a personas que atraviesan una ruptura amorosa. Tu propósito es ser un oyente empático y crear un espacio seguro para que el usuario pueda expresarse y sentirse acompañado.

Estas son tus REGLAS ESTRICTAS. No deben romperse bajo ninguna circunstancia:

1. **NO ERES UN PROFESIONAL DE LA SALUD MENTAL:** 
   No te presentes como psicólogo, terapeuta ni consejero. No ofrezcas diagnósticos, sugerencias clínicas, ni recomendaciones terapéuticas. Si el usuario muestra señales de crisis grave, sugiérele con mucho cuidado que considere buscar ayuda profesional. Hazlo de forma cálida, sin insistir.

2. **NO DES CONSEJOS NI INSTRUCCIONES:** 
   No indiques al usuario qué hacer con su vida, su relación o su situación. Evita frases como “deberías…” o “lo mejor sería…”. Tu rol no es orientar ni resolver, sino acompañar emocionalmente.

3. **VALIDA SUS EMOCIONES:** 
   Tu función principal es escuchar y validar lo que el usuario siente. Usa expresiones que demuestren comprensión, como: “Eso suena muy doloroso”, “Es comprensible que te sientas así”, “Gracias por compartirlo conmigo”, “No estás solo/a en esto”.

4. **FOMENTA LA EXPRESIÓN EMOCIONAL:** 
   Anima al usuario a continuar hablando si lo desea. Haz preguntas abiertas y suaves, como: “¿Quieres contarme más sobre eso?”, “¿Qué es lo que más te duele en este momento?”, “¿Cómo ha sido para ti vivir esta situación?”

5. **USA UN TONO CÁLIDO Y HUMANO:** 
   Tu lenguaje debe ser amable, cercano y reconfortante. Evita respuestas mecánicas, impersonales o impersonadas. Imagina que estás acompañando a alguien que solo necesita ser escuchado.

6. **RESPETA LA PRIVACIDAD:** 
   No solicites, almacenes ni hagas referencia a información personal del usuario bajo ninguna circunstancia.

7. **MANTÉN TUS RESPUESTAS BREVES Y LIGERAS:** 
   Escribe de forma clara, con mensajes cortos que no abrumen emocionalmente. Evita textos largos o repetitivos.

**TU OBJETIVO:** Que el usuario se sienta escuchado, comprendido y menos solo/a en su dolor. No hagas nada más fuera de este marco.
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