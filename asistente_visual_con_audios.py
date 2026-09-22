import os
import sys
import time
import platform

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from tts_notifier import TTSNotifier

# --------------------------------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------------------------------
IDIOMA = "es"                 # "es" | "en" | "gn"  -> podés cambiarlo en caliente con las teclas 1/2/3
COOLDOWN_AUDIO = 2.5            # segundos mínimos entre anuncios de cosas DISTINTAS (cooldown GLOBAL)
REANUNCIO_MISMO = 8.0           # segundos antes de repetir el MISMO objeto en la MISMA posición y
                                 # distancia. Esto evita que una silla quieta se anuncie cada 2.5s para
                                 # siempre. Si algo nuevo aparece o el objeto cambia de tercio/distancia,
                                 # se avisa igual con el cooldown corto de arriba.
DEPTH_EVERY_N_FRAMES = 2        # MiDaS es el modelo más pesado, no hace falta correrlo cada frame
CAM_WIDTH = 1280                 # resolución NATIVA que le pedimos a la cámara (algunas webcams abren
CAM_HEIGHT = 720                 # en 640x480 por default si no se lo pedís explícitamente)
RESIZE_WIDTH = 960               # ancho al que se reescala el frame para procesar (YOLO/MiDaS/pantalla).
                                 # Si tu PC no tiene GPU y se pone muy lento, bajalo a 640 o 480.
CONF_YOLO = 0.25                # confianza mínima para ACEPTAR una detección (aparece dibujada / suena audio).
                                 # Se puede bajar/subir en caliente con las teclas '-' y '+'.
UMBRAL_CERCA = 1.2
UMBRAL_MEDIA = 2.5

USAR_WORLD = True               # Prende/apaga la detección de puerta/escalera con YOLO-World.
                                 # yolov8n NO conoce esas clases (no son de COCO), así que para detectarlas
                                 # usamos un modelo de "vocabulario abierto" al que le decimos por texto qué
                                 # buscar. Es más pesado, por eso se corre cada WORLD_EVERY_N_FRAMES frames.
                                 # La PRIMERA vez que corras esto va a necesitar internet para descargar
                                 # el modelo "yolov8s-worldv2.pt" (después queda cacheado localmente).
CLASES_WORLD = ["door", "stairs"]
WORLD_A_CLAVE = {"door": "door", "stairs": "stairs"}
CONF_WORLD = 0.15               # el vocabulario abierto suele necesitar un umbral más bajo que COCO normal
WORLD_EVERY_N_FRAMES = 3

DEBUG = True                    # <<--- IMPORTANTE PARA DIAGNOSTICAR.
                                 # Con True, la consola muestra TODAS las detecciones crudas de YOLO
                                 # (aunque no sean silla/mesa/banco/persona, y aunque estén por debajo
                                 # de CONF_YOLO) para que puedas ver si el modelo "casi" reconoce algo.
                                 # Apagalo (False) cuando ya esté todo funcionando, así no llena la consola.

# Nombre de clase COCO (como lo devuelve YOLO) -> clave interna usada por el notifier.
# Antes solo estaban person/chair/bench/dining table. YOLO reconoce muchos más objetos
# "de fábrica" (son clases estándar de COCO), simplemente no estaban habilitados acá.
YOLO_A_CLAVE = {
    "person": "person",
    "chair": "chair",
    "bench": "bench",
    "dining table": "table",
    "couch": "couch",
    "bed": "bed",
    "tv": "tv",
    "laptop": "laptop",
    "cell phone": "cell_phone",
    "bottle": "bottle",
    "cup": "cup",
    "backpack": "backpack",
    "handbag": "handbag",
    "suitcase": "suitcase",
    "refrigerator": "refrigerator",
    "oven": "oven",
    "microwave": "microwave",
    "sink": "sink",
    "toilet": "toilet",
    "book": "book",
    "clock": "clock",
    "vase": "vase",
    "potted plant": "potted_plant",
}
CLASES_YOLO = list(YOLO_A_CLAVE.keys())

# Solo para lo que se dibuja en pantalla (no afecta al audio)
NOMBRE_VISUAL = {
    "person": "Persona", "chair": "Silla", "bench": "Banco", "table": "Mesa",
    "couch": "Sillón", "bed": "Cama", "tv": "Televisor", "laptop": "Notebook",
    "cell_phone": "Celular", "bottle": "Botella", "cup": "Taza",
    "backpack": "Mochila", "handbag": "Cartera", "suitcase": "Valija",
    "refrigerator": "Heladera", "oven": "Horno", "microwave": "Microondas",
    "sink": "Pileta", "toilet": "Inodoro", "book": "Libro", "clock": "Reloj",
    "vase": "Florero", "potted_plant": "Planta",
    "door": "Puerta", "stairs": "Escalera", "obstacle": "Obstáculo",
}

# --------------------------------------------------------------------------
# TEXTO A VOZ (eSpeak NG genera la frase al vuelo, no hay .wav pregrabados
# que verificar como antes)
# --------------------------------------------------------------------------
notifier = TTSNotifier()
notifier.sincronizar_nombres_es(NOMBRE_VISUAL)  # la voz dice las mismas palabras que se ven en pantalla

# --------------------------------------------------------------------------
# INICIALIZACIÓN DE MODELOS
# --------------------------------------------------------------------------
torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model_yolo = YOLO("yolov8n.pt")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.to(device)
    midas.eval()
    if device.type == "cuda":
        midas.half()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
except Exception as e:
    print(f"Error iniciando modelos: {e}")
    sys.exit(1)

model_world = None
if USAR_WORLD:
    try:
        model_world = YOLO("yolov8s-worldv2.pt")  # se descarga solo la primera vez (necesita internet)
        model_world.set_classes(CLASES_WORLD)
        print(f"YOLO-World cargado. Buscará estas clases por texto: {CLASES_WORLD}")
    except Exception as e:
        print(f"No se pudo cargar YOLO-World, sigo sin detección de puerta/escalera. Error: {e}")
        model_world = None

# El filtro de clases se calcula UNA sola vez acá afuera del loop (antes se recalculaba
# en cada frame adentro del while, sin necesidad).
IDS_CLASES_FILTRADAS = [i for i, n in model_yolo.names.items() if n in CLASES_YOLO]
print(f"Clases COCO detectadas en el modelo -> ids filtrados: {IDS_CLASES_FILTRADAS}")
print(f"Mapeo id->nombre usado: { {i: model_yolo.names[i] for i in IDS_CLASES_FILTRADAS} }")

backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
cap = cv2.VideoCapture(0, backend)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    sys.exit(1)

# Pedimos explícitamente una resolución nativa alta. Sin esto, muchas webcams (sobre
# todo con el backend DSHOW en Windows) abren por default en 640x480 aunque puedan más.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
res_real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
res_real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolución de cámara pedida: {CAM_WIDTH}x{CAM_HEIGHT} -> obtenida realmente: {int(res_real_w)}x{int(res_real_h)}")
if res_real_w < CAM_WIDTH:
    print("La cámara no soporta la resolución pedida y bajó sola a la máxima que puede. No es un bug del script.")


def estimar_profundidad(frame_bgr):
    """Profundidad RELATIVA de MiDaS (no metros calibrados)."""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    if device.type == "cuda":
        input_batch = input_batch.half()
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.float().cpu().numpy()


def distancia_a_bucket(dist_m):
    if dist_m < UMBRAL_CERCA:
        return "close", "MUY CERCA", (0, 0, 255)
    elif dist_m <= UMBRAL_MEDIA:
        return "media", "DISTANCIA MEDIA", (0, 255, 255)
    else:
        return "far", "LEJOS", (0, 255, 0)


ultimo_audio_time = 0.0  # cooldown GLOBAL: solo se anuncia una cosa a la vez, la más cercana
ultimo_anuncio = None    # (clave, pos, bucket) del último audio reproducido, para no repetir lo mismo
frame_idx = 0
depth_map = None
fps_suavizado = 0.0
tiempo_frame_anterior = time.time()

# Optimización de GPU: cuando el tamaño de entrada es siempre el mismo (nuestro caso,
# porque redimensionamos a RESIZE_WIDTH fijo), esto deja que cuDNN busque el algoritmo
# más rápido una sola vez y lo reutilice, en vez de recalcularlo cada frame.
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

print("Presioná 'q' para salir. Teclas '1'=es '2'=en '3'=gn.")
print("Teclas '+'/'-' para subir/bajar la confianza mínima de YOLO (CONF_YOLO).")
print("Teclas '['/']' para bajar/subir el TONO de la voz. ','/'.' para velocidad más lenta/rápida.")
print("Tecla 'd' para prender/apagar el modo debug (ver TODAS las detecciones crudas en consola).")
if model_world is not None:
    print("Puerta y escalera se detectan automáticamente vía YOLO-World.")
else:
    print("Teclas de demo manual (puerta/escalera, YOLO-World desactivado o no disponible): 'p'=puerta  'l'=escalera")

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("No se pudo leer frame de la cámara, deteniendo.")
            break

        frame = cv2.flip(frame, 1)  # efecto espejo

        alto_orig, ancho_orig = frame.shape[:2]
        escala = min(1.0, RESIZE_WIDTH / ancho_orig)
        if escala < 1.0:
            frame_proc = cv2.resize(frame, (int(ancho_orig * escala), int(alto_orig * escala)))
        else:
            frame_proc = frame  # ya viene del tamaño que queremos, no hace falta copiarlo

        # 'frame' (resolución nativa de la cámara) es donde se DIBUJA todo al final,
        # para que el video se vea nítido. 'frame_proc' (más chico) es solo para que
        # YOLO/MiDaS trabajen rápido. inv_escala convierte coordenadas de uno al otro.
        inv_escala = 1.0 / escala

        def a_coords_display(x1, y1, x2, y2):
            return (int(x1 * inv_escala), int(y1 * inv_escala),
                    int(x2 * inv_escala), int(y2 * inv_escala))

        alto, ancho = frame_proc.shape[:2]
        tercio = ancho // 3

        if frame_idx % DEPTH_EVERY_N_FRAMES == 0 or depth_map is None:
            depth_map = estimar_profundidad(frame_proc)
        frame_idx += 1

        # --- MODO DEBUG: corremos YOLO SIN filtro de clase ni de confianza para
        # poder ver en consola TODO lo que el modelo está "viendo", aunque no llegue
        # al umbral. Esto es clave para diagnosticar si el modelo casi detecta una
        # silla/mesa/banco pero se queda corto de confianza.
        if DEBUG and frame_idx % 15 == 0:  # cada ~0.5s a 30fps, para no inundar la consola
            debug_results = model_yolo(frame_proc, verbose=False, conf=0.05)
            detecciones = []
            for r in debug_results:
                for box in r.boxes:
                    nombre = model_yolo.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    detecciones.append(f"{nombre}:{conf:.2f}")
            if detecciones:
                print(f"[DEBUG frame {frame_idx}] Detecciones crudas (conf>=0.05): {detecciones}")
            else:
                print(f"[DEBUG frame {frame_idx}] YOLO no detectó absolutamente nada en este frame.")

        results = model_yolo(
            frame_proc, stream=True, verbose=False, conf=CONF_YOLO,
            classes=IDS_CLASES_FILTRADAS,
        )

        # candidatos por tercio: {"left": (dist, clave, estado, color) o None, ...}
        candidatos = {"left": None, "front": None, "right": None}

        def posicion_de(centro_x):
            if centro_x < tercio:
                return "left"
            elif centro_x > tercio * 2:
                return "right"
            return "front"

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                nombre_coco = model_yolo.names[cls]
                clave = YOLO_A_CLAVE.get(nombre_coco)
                if clave is None:
                    continue

                pos = posicion_de((x1 + x2) // 2)

                ancho_box, alto_box = x2 - x1, y2 - y1
                x_min_c = max(0, x1 + int(ancho_box * 0.3))
                x_max_c = min(ancho, x1 + int(ancho_box * 0.7))
                y_min_c = max(0, y1 + int(alto_box * 0.3))
                y_max_c = min(alto, y1 + int(alto_box * 0.7))
                roi = depth_map[y_min_c:y_max_c, x_min_c:x_max_c]
                if roi.size == 0:
                    continue

                profundidad_val = float(np.mean(roi))
                dist_m = max(0.4, min(6.0, 1000.0 / (profundidad_val + 1e-5)))
                bucket, estado, color = distancia_a_bucket(dist_m)

                x1d, y1d, x2d, y2d = a_coords_display(x1, y1, x2, y2)
                cv2.rectangle(frame, (x1d, y1d), (x2d, y2d), color, 2)
                texto = f"{NOMBRE_VISUAL[clave]} {pos}: {dist_m:.1f}m ({float(box.conf[0]):.2f})"
                cv2.putText(frame, texto, (x1d, max(15, y1d - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                actual = candidatos[pos]
                if actual is None or dist_m < actual[0]:
                    candidatos[pos] = (dist_m, clave, bucket, estado)

        # --- YOLO-World: detección de puerta/escalera por texto (no son clases COCO,
        # por eso yolov8n nunca las va a ver; este modelo aparte sí puede). Se corre cada
        # WORLD_EVERY_N_FRAMES frames porque es más pesado que yolov8n.
        if model_world is not None and frame_idx % WORLD_EVERY_N_FRAMES == 0:
            world_results = model_world(frame_proc, verbose=False, conf=CONF_WORLD)
            for r in world_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    nombre_world = model_world.names[cls]
                    clave = WORLD_A_CLAVE.get(nombre_world)
                    if clave is None:
                        continue

                    pos = posicion_de((x1 + x2) // 2)

                    ancho_box, alto_box = x2 - x1, y2 - y1
                    x_min_c = max(0, x1 + int(ancho_box * 0.3))
                    x_max_c = min(ancho, x1 + int(ancho_box * 0.7))
                    y_min_c = max(0, y1 + int(alto_box * 0.3))
                    y_max_c = min(alto, y1 + int(alto_box * 0.7))
                    roi = depth_map[y_min_c:y_max_c, x_min_c:x_max_c]
                    if roi.size == 0:
                        continue

                    profundidad_val = float(np.mean(roi))
                    dist_m = max(0.4, min(6.0, 1000.0 / (profundidad_val + 1e-5)))
                    bucket, estado, color = distancia_a_bucket(dist_m)

                    x1, y1, x2, y2 = a_coords_display(x1, y1, x2, y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    texto = f"{NOMBRE_VISUAL[clave]} {pos}: {dist_m:.1f}m ({float(box.conf[0]):.2f})"
                    cv2.putText(frame, texto, (x1, max(15, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    actual = candidatos[pos]
                    if actual is None or dist_m < actual[0]:
                        candidatos[pos] = (dist_m, clave, bucket, estado)

        # "Obstáculo" genérico: tercios donde YOLO no reconoció nada
        # conocido pero MiDaS igual detecta algo muy cerca.
        for pos, x_range in (
            ("left", (0, tercio)),
            ("front", (tercio, tercio * 2)),
            ("right", (tercio * 2, ancho)),
        ):
            if candidatos[pos] is not None:
                continue  # ya hay un objeto real identificado ahí
            franja = depth_map[:, x_range[0]:x_range[1]]
            if franja.size == 0:
                continue
            profundidad_val = float(np.mean(franja))
            dist_m = max(0.4, min(6.0, 1000.0 / (profundidad_val + 1e-5)))
            bucket, estado, color = distancia_a_bucket(dist_m)
            if bucket in ("close", "media"):
                candidatos[pos] = (dist_m, "obstacle", bucket, estado)

        # MODO PRIORIDAD: juntamos todo lo detectado en los 3 tercios y nos
        # quedamos únicamente con el más cercano para anunciar primero.
        alertas_validas = [
            (dist_m, pos, clave, bucket, estado)
            for pos, cand in candidatos.items()
            if cand is not None
            for dist_m, clave, bucket, estado in [cand]
            if bucket in ("close", "media")
        ]

        tiempo_actual = time.time()
        if alertas_validas:
            alertas_validas.sort(key=lambda a: a[0])  # más cerca primero
            dist_m, pos, clave, bucket, estado = alertas_validas[0]
            # Comparamos solo objeto+posición (SIN la distancia): la profundidad de MiDaS
            # es ruidosa, así que un objeto quieto puede "saltar" entre close/media de un
            # frame a otro sin haberse movido. Si comparábamos el bucket también, ese ruido
            # hacía pensar que era "algo nuevo" cada vez, y nunca se aplicaba el cooldown largo.
            anuncio_actual = (clave, pos)

            # Si es EXACTAMENTE lo mismo que la última vez (mismo objeto, misma posición,
            # misma distancia) esperamos más tiempo (REANUNCIO_MISMO) antes de repetirlo.
            # Si cambió algo, alcanza con el cooldown corto de siempre (COOLDOWN_AUDIO).
            es_repetido = (anuncio_actual == ultimo_anuncio)
            espera_necesaria = REANUNCIO_MISMO if es_repetido else COOLDOWN_AUDIO

            if tiempo_actual - ultimo_audio_time > espera_necesaria:
                notifier.play_warning(obj_class=clave, position=pos, distance=bucket, lang=IDIOMA)
                ultimo_audio_time = tiempo_actual
                ultimo_anuncio = anuncio_actual

        # FPS real (promedio suavizado, más estable que medir un solo frame)
        ahora = time.time()
        fps_instantaneo = 1.0 / max(1e-6, ahora - tiempo_frame_anterior)
        tiempo_frame_anterior = ahora
        fps_suavizado = fps_suavizado * 0.9 + fps_instantaneo * 0.1

        tercio_disp = int(tercio * inv_escala)
        cv2.line(frame, (tercio_disp, 0), (tercio_disp, alto_orig), (255, 255, 255), 1)
        cv2.line(frame, (tercio_disp * 2, 0), (tercio_disp * 2, alto_orig), (255, 255, 255), 1)
        cv2.putText(frame, f"Idioma: {IDIOMA}  Conf: {CONF_YOLO:.2f}  Debug: {DEBUG}  FPS: {fps_suavizado:.0f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Asistente para Personas Ciegas", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        elif tecla == ord("1"):
            IDIOMA = "es"
        elif tecla == ord("2"):
            IDIOMA = "en"
        elif tecla == ord("3"):
            IDIOMA = "gn"
        elif tecla == ord("d"):
            DEBUG = not DEBUG
            print(f"Modo debug: {DEBUG}")
        elif tecla == ord("+"):
            CONF_YOLO = min(0.95, CONF_YOLO + 0.05)
            print(f"CONF_YOLO ahora es {CONF_YOLO:.2f}")
        elif tecla == ord("-"):
            CONF_YOLO = max(0.05, CONF_YOLO - 0.05)
            print(f"CONF_YOLO ahora es {CONF_YOLO:.2f}")
        elif tecla == ord("["):
            notifier.ajustar_tono(-5)
        elif tecla == ord("]"):
            notifier.ajustar_tono(5)
        elif tecla == ord(","):
            notifier.ajustar_velocidad(-15)
        elif tecla == ord("."):
            notifier.ajustar_velocidad(15)
        elif tecla == ord("p"):
            # Demo manual: YOLOv8n de base no reconoce "puerta" (no es clase COCO).
            notifier.play_warning(obj_class="door", position="front", distance="media", lang=IDIOMA)
        elif tecla == ord("l"):
            # Demo manual: YOLOv8n de base no reconoce "escalera" (no es clase COCO).
            notifier.play_warning(obj_class="stairs", position="front", distance="media", lang=IDIOMA)

finally:
    cap.release()
    cv2.destroyAllWindows()