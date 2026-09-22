import os
import shutil
import subprocess


class TTSNotifier:
    """
    Reemplaza a AudioNotifier (audio_player.py): en vez de buscar un .wav
    pregrabado en audio_mapper.json, arma la frase en el momento y la hace
    hablar con eSpeak NG.

    IMPORTANTE: eSpeak NG NO es una librería de Python, es un programa aparte
    (el .msi que instalaste). Por eso esta clase lo llama por afuera con
    subprocess, como si lo tipearas en la consola. Requisitos:

      1. Tener eSpeak NG instalado (https://github.com/espeak-ng/espeak-ng/releases).
      2. Que 'espeak-ng.exe' esté en el PATH de Windows, o en una de las
         RUTAS_CANDIDATAS de más abajo (las rutas default del instalador).

    Para confirmar que quedó bien instalado, abrí una consola (cmd/PowerShell)
    y corré:   espeak-ng --version
    Si eso no funciona, reinstalá tildando "Add espeak-ng to PATH" si el
    instalador te lo ofrece, o agregá manualmente la carpeta de instalación
    a la variable de entorno PATH.
    """

    RUTAS_CANDIDATAS = [
        r"C:\Program Files\eSpeak NG\espeak-ng.exe",
        r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
    ]

    # Nuestro idioma interno -> código de voz de eSpeak NG.
    VOZ_ESPEAK = {"es": "es", "en": "en", "gn": "gn"}

    VELOCIDAD_PALABRAS_MIN = 300   # -s de espeak-ng: palabras por minuto. Rango típico útil: 80-450.
                                    # Más alto = habla más rápido (y más "atropellado").
    VOLUMEN = 100                  # -a de espeak-ng: volumen, 0-200 (100 = normal, 200 = el doble).
    TONO = 50                      # -p de espeak-ng: tono/pitch, 0-99 (50 = normal). Más alto = voz
                                    # más aguda, más bajo = más grave.
    PAUSA_ENTRE_PALABRAS = 0       # -g de espeak-ng: pausa extra entre palabras, en unidades de 10ms.
                                    # Subilo (ej. 5-10) si querés que hable más "pausado y claro",
                                    # útil para asistencia visual donde la claridad importa más que
                                    # la velocidad natural.

    # Variante de voz: se pega al código de idioma con un "+", ej. "es+f3" = español, voz femenina 3.
    # Dejalo vacío ("") para la voz default de cada idioma. Variantes comunes de eSpeak NG:
    #   +m1 a +m7  -> variantes de voz masculina (m3 suena más grave, m1 más neutra)
    #   +f1 a +f4  -> variantes de voz femenina
    #   +croak, +whisper, +klatt  -> efectos especiales (no recomendados para uso real, son curiosidades)
    # Podés probar variantes directo en consola ANTES de tocar el código, por ejemplo:
    #   espeak-ng -v es+f3 -p 60 -s 150 -a 100 "hola, silla al frente, muy cerca"
    # y una vez que te guste alguna combinación, la copiás acá:
    VARIANTE = "+f3"

    # ------------------------------------------------------------------
    # FRASES: acá se arma lo que se dice, en vez de buscar un archivo.
    # Aviso honesto: no soy hablante nativo de guaraní. "ava" (persona),
    # "apyka" (silla) y "okẽ" (puerta) los mantuve porque ya los habías
    # usado vos en los nombres de archivo originales, pero las palabras de
    # POSICIONES y DISTANCIAS en guaraní son mi mejor estimación y están
    # marcadas VERIFICAR: si tenés (o conseguís) un hablante de guaraní
    # paraguayo que las revise, mejor. Mientras tanto el programa funciona
    # igual, solo puede sonar raro en esas palabras puntuales.
    # ------------------------------------------------------------------
    OBJETOS = {
        "es": {"person": "persona", "chair": "silla", "bench": "banco",
               "table": "mesa", "door": "puerta", "stairs": "escalera",
               "obstacle": "obstáculo", "couch": "sillón", "bed": "cama",
               "tv": "televisor", "laptop": "notebook", "cell_phone": "celular",
               "bottle": "botella", "cup": "taza", "backpack": "mochila",
               "handbag": "cartera", "suitcase": "valija",
               "refrigerator": "heladera", "oven": "horno",
               "microwave": "microondas", "sink": "pileta",
               "toilet": "inodoro", "book": "libro", "clock": "reloj",
               "vase": "florero", "potted_plant": "planta"},
        "en": {"person": "person", "chair": "chair", "bench": "bench",
               "table": "table", "door": "door", "stairs": "stairs",
               "obstacle": "obstacle", "couch": "couch", "bed": "bed",
               "tv": "television", "laptop": "laptop", "cell_phone": "cell phone",
               "bottle": "bottle", "cup": "cup", "backpack": "backpack",
               "handbag": "handbag", "suitcase": "suitcase",
               "refrigerator": "refrigerator", "oven": "oven",
               "microwave": "microwave", "sink": "sink",
               "toilet": "toilet", "book": "book", "clock": "clock",
               "vase": "vase", "potted_plant": "plant"},
        "gn": {"person": "ava", "chair": "apyka", "bench": "banco",
               "table": "mesa", "door": "okẽ",
               "stairs": "jupiha",     # VERIFICAR
               "obstacle": "mba'e",    # VERIFICAR (significa "cosa" genérica)
               "couch": "sillón",      # VERIFICAR (probable préstamo del español)
               "bed": "tupa",          # VERIFICAR
               "tv": "televisor",      # VERIFICAR (probable préstamo)
               "laptop": "notebook",   # VERIFICAR (probable préstamo)
               "cell_phone": "celular",  # VERIFICAR (probable préstamo)
               "bottle": "ita'y",      # VERIFICAR
               "cup": "kagua",         # VERIFICAR
               "backpack": "mochila",  # VERIFICAR (probable préstamo)
               "handbag": "cartera",   # VERIFICAR (probable préstamo)
               "suitcase": "valija",   # VERIFICAR (probable préstamo)
               "refrigerator": "heladera",  # VERIFICAR (probable préstamo)
               "oven": "horno",        # VERIFICAR (probable préstamo)
               "microwave": "microondas",  # VERIFICAR (probable préstamo)
               "sink": "pileta",       # VERIFICAR (probable préstamo)
               "toilet": "inodoro",    # VERIFICAR (probable préstamo)
               "book": "kuatiañe'ẽ",   # VERIFICAR
               "clock": "aravo",       # VERIFICAR
               "vase": "florero",      # VERIFICAR (probable préstamo)
               "potted_plant": "yvoty"},  # VERIFICAR
    }
    POSICIONES = {
        "es": {"left": "a la izquierda", "front": "al frente", "right": "a la derecha"},
        "en": {"left": "on your left", "front": "ahead", "right": "on your right"},
        "gn": {"left": "asuguio",        # VERIFICAR
               "front": "tenondeguio",   # VERIFICAR
               "right": "akatúaguio"},   # VERIFICAR
    }
    DISTANCIAS = {
        "es": {"close": "muy cerca", "media": "a media distancia", "far": "lejos"},
        "en": {"close": "very close", "media": "at medium distance", "far": "far away"},
        "gn": {"close": "aguĩ eterei",  # VERIFICAR
               "media": "mbyte",         # VERIFICAR
               "far": "mombyry"},        # VERIFICAR
    }

    def __init__(self):
        self.espeak_path = self._encontrar_espeak()
        print(f"[TTSNotifier] Usando eSpeak NG en: {self.espeak_path}")
        self.proceso_actual = None
        self.ultimo_texto = None
        # Copia por instancia (no compartida entre instancias) de los diccionarios de
        # idioma, para poder sincronizar el español con NOMBRE_VISUAL sin tocar la
        # constante de clase.
        self.OBJETOS = {lang: dict(mapa) for lang, mapa in TTSNotifier.OBJETOS.items()}

    def sincronizar_nombres_es(self, nombre_visual: dict):
        """Hace que la voz en español diga EXACTAMENTE las mismas palabras que
        aparecen dibujadas en pantalla (el diccionario NOMBRE_VISUAL del script
        principal), en vez de mantener una traducción aparte que se puede
        desincronizar. Se llama una vez, después de crear el notifier."""
        for clave, palabra in nombre_visual.items():
            self.OBJETOS["es"][clave] = palabra.lower()
        print("[TTSNotifier] Nombres en español sincronizados con NOMBRE_VISUAL.")

    def _encontrar_espeak(self):
        en_path = shutil.which("espeak-ng")
        if en_path:
            return en_path
        for ruta in self.RUTAS_CANDIDATAS:
            if os.path.exists(ruta):
                return ruta
        raise FileNotFoundError(
            "No encontré 'espeak-ng.exe'. Instalá el .msi y fijate que quede en el PATH, "
            "o agregá manualmente la carpeta 'C:\\Program Files\\eSpeak NG' al PATH de Windows. "
            "Podés probar si ya está en el PATH corriendo 'espeak-ng --version' en una consola."
        )

    def ocupado(self):
        """Equivalente a pygame.mixer.music.get_busy() de la versión anterior:
        True mientras eSpeak NG todavía está diciendo la frase anterior."""
        return self.proceso_actual is not None and self.proceso_actual.poll() is None

    def play_warning(self, obj_class: str, position: str, distance: str, lang: str = "es"):
        """Misma firma que AudioNotifier.play_warning, para que el script
        principal no tenga que cambiar cómo la llama."""
        if self.ocupado():
            return  # ya está hablando algo, no lo interrumpimos ni lo pisamos

        voz = self.VOZ_ESPEAK.get(lang, "es")
        objeto = self.OBJETOS.get(lang, self.OBJETOS["es"]).get(obj_class, obj_class)
        pos_txt = self.POSICIONES.get(lang, self.POSICIONES["es"]).get(position, position)
        dist_txt = self.DISTANCIAS.get(lang, self.DISTANCIAS["es"]).get(distance, distance)

        texto = f"{objeto}, {pos_txt}, {dist_txt}"
        voz_final = f"{voz}{self.VARIANTE}" if self.VARIANTE else voz

        try:
            self.proceso_actual = subprocess.Popen(
                [self.espeak_path,
                 "-v", voz_final,
                 "-s", str(self.VELOCIDAD_PALABRAS_MIN),
                 "-a", str(self.VOLUMEN),
                 "-p", str(self.TONO),
                 "-g", str(self.PAUSA_ENTRE_PALABRAS),
                 texto],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.ultimo_texto = texto
            print(f"Hablando ({lang}, voz={voz_final}, vel={self.VELOCIDAD_PALABRAS_MIN}, "
                  f"tono={self.TONO}): {texto}")
        except Exception as e:
            print(f"[TTSNotifier] Error al hablar: {e}")

    # ------------------------------------------------------------------
    # Ajustes en caliente, para poder tunear la voz probando mientras el
    # programa corre, sin reiniciar. Llamalos desde el script principal
    # asociados a alguna tecla (ver ejemplo en asistente_visual_con_audios.py).
    # ------------------------------------------------------------------
    def ajustar_velocidad(self, delta):
        self.VELOCIDAD_PALABRAS_MIN = max(80, min(450, self.VELOCIDAD_PALABRAS_MIN + delta))
        print(f"[TTSNotifier] Velocidad ahora: {self.VELOCIDAD_PALABRAS_MIN} palabras/min")

    def ajustar_tono(self, delta):
        self.TONO = max(0, min(99, self.TONO + delta))
        print(f"[TTSNotifier] Tono ahora: {self.TONO}")

    def ajustar_volumen(self, delta):
        self.VOLUMEN = max(0, min(200, self.VOLUMEN + delta))
        print(f"[TTSNotifier] Volumen ahora: {self.VOLUMEN}")


if __name__ == "__main__":
    import time
    notifier = TTSNotifier()
    print("Probando TTSNotifier...")
    notifier.play_warning(obj_class="chair", position="front", distance="close", lang="es")
    while notifier.ocupado():
        time.sleep(0.1)
    notifier.play_warning(obj_class="person", position="left", distance="media", lang="en")
    while notifier.ocupado():
        time.sleep(0.1)
    notifier.play_warning(obj_class="door", position="right", distance="far", lang="gn")
    while notifier.ocupado():
        time.sleep(0.1)