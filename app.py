import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from werkzeug.utils import secure_filename
import tempfile

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)

# Configuración de OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No se encontró la variable de entorno OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Prompt para chat de texto puro
SYSTEM_PROMPT_TEXT = """
Eres un asistente de contención emocional, no un terapeuta profesional. Tu misión es:
1. Crear un vínculo empático usando lenguaje cercano.
2. Detectar nivel de angustia a partir de la forma de expresarse.
3. Formular preguntas abiertas para explorar emociones.
4. Ofrecer técnicas de contención (ej. respiración, mindfulness).
5. Evaluar riesgo de forma suave.
6. Recomendar recursos y cerrar con apoyo.
"""

# Prompt extendido para análisis de voz y texto
SYSTEM_PROMPT_VOZ = """
Eres un asistente de contención emocional que combina análisis de texto y señales de voz.
1. Transcribe el audio usando Whisper.
2. Infierir el nivel de angustia a partir de texto y características de voz.
3. Formular preguntas abiertas y técnicas de contención.
4. Evaluar riesgo basado en contenido y tono vocal.
5. Recomendar recursos y cerrar con apoyo.
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    history = data.get('history')
    if not history:
        return jsonify({'error': 'Debe proporcionar historial.'}), 400

    messages = [{'role': 'system', 'content': SYSTEM_PROMPT_TEXT}] + history
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=200,
            top_p=1
        )
        reply = response.choices[0].message.content.strip()
        return jsonify({'reply': reply})

    except OpenAIError as oe:
        app.logger.error(f"OpenAI API error: {oe}")
        return jsonify({'reply': 'Error comunicando con IA.'}), 502
    except Exception:
        app.logger.exception("Error en /chat")
        return jsonify({'reply': 'Error inesperado.'}), 500

@app.route('/audio', methods=['POST'])
def audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No se envió audio.'}), 400
    audio_file = request.files['audio']
    filename = secure_filename(audio_file.filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, 'rb') as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        text = transcription.text

        history = [{'role': 'user', 'content': text}]
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT_VOZ}] + history

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=200,
            top_p=1
        )
        reply = response.choices[0].message.content.strip()

        return jsonify({'transcript': text, 'reply': reply})

    except OpenAIError as oe:
        app.logger.error(f"OpenAI audio API error: {oe}")
        return jsonify({'reply': 'Error comunicando con IA.'}), 502
    except Exception:
        app.logger.exception("Error en /audio")
        return jsonify({'reply': 'Error inesperado.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)