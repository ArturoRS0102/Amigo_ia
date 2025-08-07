import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from werkzeug.utils import secure_filename
import tempfile

# Cargar variables de entorno\load_dotenv()
app = Flask(__name__)

# Configuración de OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No se encontró la variable de entorno OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# System Prompt extendido para texto y voz
SYSTEM_PROMPT_VOZ = """
Eres un asistente de contención emocional. Analiza tanto el contenido del usuario como las señales emocionales de su voz.
1. Transcribe el audio usando Whisper.
2. Infier e indirectamente su nivel de angustia basándote en texto y características de voz.
3. Formula preguntas abiertas, ofrece técnicas de contención y evalúa riesgo según el tono y contenido.
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

    messages = [{'role': 'system', 'content': SYSTEM_PROMPT_VOZ}] + history
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
    except Exception as e:
        app.logger.exception("Error en /chat")
        return jsonify({'reply': 'Error inesperado.'}), 500

@app.route('/audio', methods=['POST'])
def audio():
    # Recibe archivo de audio
    if 'audio' not in request.files:
        return jsonify({'error': 'No se envió audio.'}), 400
    audio_file = request.files['audio']
    filename = secure_filename(audio_file.filename)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        audio_file.save(tmp.name)
        # Transcripción con Whisper
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=tmp.name
        )
    text = transcription.text

    # Preparar historia inicial con transcripción
    history = [{'role': 'user', 'content': text}]
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT_VOZ}] + history
    try:
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
    except Exception as e:
        app.logger.exception("Error en /audio")
        return jsonify({'reply': 'Error inesperado.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
