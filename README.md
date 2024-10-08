# Aplicación de Transcripción de Batería

Esta aplicación permite a los usuarios subir un archivo de audio, aislar la pista de batería y transcribirla en partitura y formato MIDI. La aplicación está construida utilizando Python y Streamlit, ofreciendo una interfaz simple para generar transcripciones de batería de calidad profesional.

## Características

- Carga de Audio: Los usuarios pueden subir un archivo de audio en formato .wav o .mp3.
- Separación de Pista de Batería: La aplicación utiliza Demucs para aislar la pista de batería del audio.
- Detección de Onsets: Detecta los golpes de batería y los alinea con los ritmos.
- Generación de Partituras: Genera automáticamente partituras de batería utilizando LilyPond y Abjad.
- Generación de Archivos MIDI: Convierte la transcripción a un archivo MIDI.
- Salidas Descargables: Los usuarios pueden descargar la partitura en formatos .pdf, así como el archivo MIDI y el archivo de Lilypond.

## Instalación

1. Clona el repositorio:
```
git clone https://github.com/tuusuario/drum-transcription-app.git
cd drum-transcription-app
```

2. Instala las dependencias requeridas:
```
pip install -r requirements.txt
```

3. Instala LilyPond: Para generar partituras, debes tener LilyPond instalado. Puedes descargarlo desde el [sitio oficial de LilyPond](https://lilypond.org/).

5. Ejecuta la applicación:
```
streamlit run app.py
```

## Uso

1. Abre la aplicación en tu navegador navegando a http://localhost:8501.
2. Sube tu archivo de audio (preferiblemente en formato .wav para mejor precisión).
3. Espera a que la pista de batería sea separada y procesada.
4. Visualiza y descarga la partitura generada y el archivo MIDI.

## Agradecimientos

- [Demucs](https://github.com/facebookresearch/demucs) por la separación de pistas de batería.
- [Abjad](https://abjad.github.io/) y [LilyPond](https://lilypond.org/) por la generación de partituras.
