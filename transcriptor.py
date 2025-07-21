import argparse
import os
import whisper
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import time

# --- Configuración de modelos Whisper ---
# Puedes encontrar más información sobre los modelos en la documentación de Whisper
# 'tiny', 'base', 'small', 'medium', 'large'
# También versiones en inglés: 'tiny.en', 'base.en', 'small.en', 'medium.en'
DEFAULT_WHISPER_MODEL = "small" # Buen equilibrio entre velocidad y precisión para empezar

# --- Funciones principales ---

def convert_m4a_to_wav(m4a_path):
    """
    Convierte un archivo M4A a formato WAV temporal.
    Retorna la ruta al archivo WAV temporal o None si falla.
    """
    try:
        audio = AudioSegment.from_file(m4a_path, format="m4a")
        # Crea un nombre de archivo temporal único para evitar conflictos
        temp_wav_path = os.path.join(os.path.dirname(m4a_path), "temp_" + os.path.basename(m4a_path).replace(".m4a", ".wav"))
        audio.export(temp_wav_path, format="wav")
        return temp_wav_path
    except FileNotFoundError:
        print(f"Error: FFmpeg no encontrado o no está en el PATH. Asegúrate de instalarlo.")
        return None
    except CouldntDecodeError:
        print(f"Error: No se pudo decodificar el archivo M4A: '{m4a_path}'. ¿Está corrupto o es un formato no soportado por FFmpeg?")
        return None
    except Exception as e:
        print(f"Error al convertir '{m4a_path}' a WAV: {e}")
        return None

def transcribe_audio_file(audio_path, whisper_model_name):
    """
    Transcribe un archivo de audio usando el modelo Whisper especificado.
    Retorna el texto transcribido o None si hay un error.
    """
    # Carga el modelo de Whisper solo una vez por proceso de transcripción para eficiencia
    # Esto se hará en la función principal para evitar recargar el modelo en cada archivo.
    # Esta función asume que ya recibe un modelo cargado o lo carga una vez por llamada.
    # Para nuestra implementación, el modelo se cargará una vez en main y se pasará.

    try:
        # La librería Whisper puede manejar varios formatos, pero la conversión a WAV
        # asegura la compatibilidad y a veces mejora el rendimiento con pydub primero.
        # Sin embargo, Whisper es bastante robusto y puede intentar leer M4A directamente
        # si FFmpeg está bien configurado en el sistema.
        # Para ser explícitos y manejar posibles problemas de M4A, lo convertiremos a WAV.

        print(f"  > Convirtiendo '{os.path.basename(audio_path)}' a WAV...")
        temp_wav_path = convert_m4a_to_wav(audio_path)
        if not temp_wav_path:
            return None

        # Cargar el modelo de Whisper si no se pasó (esto no debería pasar en nuestra implementación final)
        # o usar el modelo ya cargado
        model = whisper.load_model(whisper_model_name)

        print(f"  > Transcribiendo '{os.path.basename(audio_path)}' con modelo '{whisper_model_name}'...")
        start_time = time.time()
        
        # Realiza la transcripción. Whisper detecta el idioma automáticamente.
        result = model.transcribe(temp_wav_path)
        
        end_time = time.time()
        print(f"  > Transcripción completada en {end_time - start_time:.2f} segundos.")
        
        # Elimina el archivo WAV temporal
        os.remove(temp_wav_path)

        return result["text"]

    except Exception as e:
        print(f"Error al transcribir '{audio_path}': {e}")
        # Asegurarse de limpiar el archivo temporal incluso si hay un error
        if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe archivos de audio M4A a texto usando OpenAI Whisper.",
        formatter_class=argparse.RawTextHelpFormatter # Para que el texto de ayuda mantenga formato
    )
    parser.add_argument(
        "directory_input",
        type=str,
        help="Ruta al directorio que contiene los archivos M4A a transcribir."
    )
    parser.add_argument(
        "--directory_output",
        type=str,
        default="transcripciones_salida", # Por defecto, crea esta carpeta en el mismo lugar que el script
        help="Ruta al directorio donde se guardarán los archivos de texto.\n"
             "Por defecto: 'transcripciones_salida' (creado en la ubicación del script)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large",
                 "tiny.en", "base.en", "small.en", "medium.en"],
        help=f"Modelo de Whisper a usar para la transcripción.\n"
             f"Opciones: tiny, base, small, medium, large\n"
             f"Para inglés solamente: tiny.en, base.en, small.en, medium.en\n"
             f"Por defecto: '{DEFAULT_WHISPER_MODEL}'."
    )

    args = parser.parse_args()

    input_dir = args.directory_input
    output_dir = args.directory_output
    selected_model = args.model

    # Verificar si el directorio de entrada existe
    if not os.path.isdir(input_dir):
        print(f"Error: El directorio de entrada '{input_dir}' no existe.")
        return

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directorio de salida para transcripciones: {os.path.abspath(output_dir)}")
    print(f"Modelo de Whisper seleccionado: {selected_model}")
    print("-" * 50)

    # --- Cargar el modelo de Whisper una sola vez para todos los archivos ---
    print(f"Cargando el modelo Whisper '{selected_model}' (esto puede tardar la primera vez)...")
    try:
        # Se asegura de que el modelo se cargue con soporte GPU si está disponible.
        # Whisper lo detecta automáticamente si CUDA está bien configurado.
        whisper_model = whisper.load_model(selected_model)
        print("Modelo Whisper cargado exitosamente.")
    except Exception as e:
        print(f"Error al cargar el modelo Whisper '{selected_model}': {e}")
        print("Asegúrate de que el nombre del modelo es correcto y tienes conexión a internet si es la primera descarga.")
        print("Si estás en WSL, verifica que PyTorch identifique tu GPU.")
        return

    # --- Procesar archivos M4A ---
    processed_count = 0
    error_count = 0

    # Iterar sobre los archivos en el directorio de entrada
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".m4a"):
            m4a_full_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0] # Nombre del archivo sin extensión
            output_txt_path = os.path.join(output_dir, f"{base_name}.txt")

            print(f"\nProcesando: '{filename}'")

            # Verificar si el archivo de salida ya existe para evitar re-transcribir innecesariamente
            if os.path.exists(output_txt_path):
                print(f"  > El archivo de salida '{os.path.basename(output_txt_path)}' ya existe. Saltando.")
                processed_count += 1
                continue

            transcribed_text = transcribe_audio_file(m4a_full_path, selected_model) # Pasamos el nombre del modelo
            
            if transcribed_text:
                try:
                    with open(output_txt_path, "w", encoding="utf-8") as f:
                        f.write(transcribed_text)
                    print(f"  > Transcripción guardada en: '{os.path.basename(output_txt_path)}'")
                    processed_count += 1
                except Exception as e:
                    print(f"  > Error al guardar la transcripción para '{filename}': {e}")
                    error_count += 1
            else:
                print(f"  > No se pudo transcribir '{filename}'.")
                error_count += 1
        else:
            print(f"  > Saltando '{filename}': No es un archivo M4A.")

    print("\n" + "=" * 50)
    print("PROCESO COMPLETADO")
    print(f"Archivos M4A procesados: {processed_count}")
    print(f"Archivos con errores o no transcribidos: {error_count}")
    print("=" * 50)

if __name__ == "__main__":
    main()