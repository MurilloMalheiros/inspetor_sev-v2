"""Monitor de sonolencia em tempo real com YOLO e MediaPipe."""

from __future__ import annotations

import argparse
import io
import os
import queue
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame
import torch
from ultralytics import YOLO

from audio_tones import siren_wav, tone_wav
from fatigue_monitor import DriverState, FatigueMonitor, FrameObservation


EXPECTED_CLASSES = {"bocaAberta", "olhoDireito", "olhoEsquerdo", "rosto"}
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


@dataclass(frozen=True)
class FaceMetrics:
    face: bool = False
    left_eye_open: bool = False
    right_eye_open: bool = False
    mouth_open: bool = False
    # Coordenadas normalizadas (x1, y1, x2, y2). Elas continuam validas
    # mesmo quando a camera entrega uma resolucao diferente da solicitada.
    box: tuple[float, float, float, float] | None = None
    updated_at: float = 0.0


@dataclass(frozen=True)
class YoloDetection:
    mouth_open: bool = False
    boxes: tuple = ()
    updated_at: float = 0.0


class EyeStateFilter:
    """Converte a leitura continua em ABERTO/FECHADO sem oscilacoes."""

    def __init__(self, closed_threshold: float) -> None:
        self.closed_threshold = closed_threshold
        self.open_threshold = max(0.10, closed_threshold - 0.20)
        self.is_open = True
        self._smoothed = 0.0
        self._candidate_since: float | None = None

    def reset(self) -> None:
        self.is_open = True
        self._smoothed = 0.0
        self._candidate_since = None

    def update(self, blink_score: float, now: float) -> bool:
        # A media exponencial absorve falhas isoladas, comuns sobretudo quando
        # um dos olhos esta parcialmente de lado para a camera.
        self._smoothed = 0.75 * self._smoothed + 0.25 * blink_score
        threshold = self.open_threshold if not self.is_open else self.closed_threshold
        wants_change = self._smoothed <= threshold if not self.is_open else self._smoothed >= threshold
        if not wants_change:
            self._candidate_since = None
            return self.is_open

        if self._candidate_since is None:
            self._candidate_since = now
        confirmation_time = 0.08 if not self.is_open else 0.15
        if now - self._candidate_since >= confirmation_time:
            self.is_open = not self.is_open
            self._candidate_since = None
        return self.is_open


class FaceBoxFilter:
    """Suaviza a caixa facial sem permitir saltos ou deformacoes bruscas."""

    def __init__(self, smoothing: float = 0.22) -> None:
        self.smoothing = smoothing
        self._box: tuple[float, float, float, float] | None = None

    def reset(self) -> None:
        self._box = None

    def update(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if self._box is None:
            self._box = box
            return box

        # Filtrar centro e dimensoes evita que os quatro cantos variem de
        # maneira independente, que era o efeito visual de caixa "esticando".
        old_x1, old_y1, old_x2, old_y2 = self._box
        x1, y1, x2, y2 = box
        old_values = (
            (old_x1 + old_x2) / 2,
            (old_y1 + old_y2) / 2,
            old_x2 - old_x1,
            old_y2 - old_y1,
        )
        new_values = (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            x2 - x1,
            y2 - y1,
        )
        alpha = self.smoothing
        cx, cy, width, height = (
            old + alpha * (new - old)
            for old, new in zip(old_values, new_values)
        )
        self._box = (
            max(0.0, cx - width / 2),
            max(0.0, cy - height / 2),
            min(1.0, cx + width / 2),
            min(1.0, cy + height / 2),
        )
        return self._box


def combined_blink_score(left: float, right: float) -> float:
    """Combina os olhos sem deixar uma leitura fraca bloquear a deteccao.

    O sinal mais forte recebe peso maior. Assim, se um olho estiver parcialmente
    oculto ou falhar, o outro ainda consegue indicar fechamento; uma oscilacao
    curta continua sendo removida pelo EyeStateFilter e pelos 3 s do monitor.
    """

    strongest = max(left, right)
    weakest = min(left, right)
    return 0.75 * strongest + 0.25 * weakest


class SoundPlayer:
    """Mixer persistente em processo; a thread de audio nao bloqueia o video."""

    PATTERNS = {
        "startup": [(880, 120), (1320, 180)],
        "sleep": [(1500, 250), (2100, 250), (1500, 350)],
        "yawn": [(1050, 140), (1350, 180)],
        "fatigue1": [(900, 220), (1100, 220)],
        "fatigue2": [(1000, 250), (1400, 250), (1700, 300)],
        "fatigue3": [(1300, 300), (1700, 300), (2100, 450)],
    }

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = False
        self._continuous = False
        self._stop = threading.Event()
        self._items: queue.Queue[str | None] = queue.Queue(maxsize=4)
        self._channel = None
        self._siren_channel = None
        self._sounds: dict[tuple[int, int], pygame.mixer.Sound] = {}
        self._siren = None
        self._thread: threading.Thread | None = None
        if not enabled:
            return

        try:
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(2)
            self._channel = pygame.mixer.Channel(0)
            self._siren_channel = pygame.mixer.Channel(1)
            for pattern in self.PATTERNS.values():
                for frequency, duration in pattern:
                    self._sound(frequency, duration)
            self._siren = pygame.mixer.Sound(file=io.BytesIO(siren_wav()))
            self.enabled = True
            self._thread = threading.Thread(target=self._run, name="audio-alerts", daemon=True)
            self._thread.start()
            print("[audio] mixer Pygame inicializado", flush=True)
        except pygame.error as error:
            pygame.mixer.quit()
            print(f"[audio] nao foi possivel iniciar o mixer Pygame: {error}", flush=True)

    def _sound(self, frequency: int, duration: int) -> pygame.mixer.Sound:
        key = (frequency, duration)
        sound = self._sounds.get(key)
        if sound is None:
            sound = pygame.mixer.Sound(file=io.BytesIO(tone_wav(frequency, duration)))
            self._sounds[key] = sound
        return sound

    @staticmethod
    def _tone_wav(frequency: int, duration: int) -> bytes:
        return tone_wav(frequency, duration)

    def play(self, kind: str) -> None:
        if not self.enabled or self._continuous:
            return
        try:
            self._items.put_nowait(kind)
        except queue.Full:
            try:
                self._items.get_nowait()
                self._items.put_nowait(kind)
            except queue.Empty:
                pass

    def set_continuous(self, active: bool) -> None:
        active = self.enabled and active
        if active == self._continuous:
            return
        self._continuous = active
        if active:
            while True:
                try:
                    self._items.get_nowait()
                except queue.Empty:
                    break
            self._channel.stop()
            self._siren_channel.play(self._siren, loops=-1)
        else:
            self._siren_channel.stop()

    def close(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        self._continuous = False
        self._channel.stop()
        self._siren_channel.stop()
        try:
            self._items.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        pygame.mixer.quit()
        self.enabled = False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                kind = self._items.get(timeout=0.1)
            except queue.Empty:
                continue
            if kind is None:
                return
            for frequency, duration in self.PATTERNS.get(kind, self.PATTERNS["fatigue1"]):
                if self._stop.is_set() or self._continuous:
                    break
                sound = self._sound(frequency, duration)
                self._channel.play(sound)
                while self._channel.get_busy() and not self._stop.is_set() and not self._continuous:
                    time.sleep(0.005)


def alarm_decision(update) -> tuple[bool, str | None]:
    """Retorna (sirene_continua, aviso) para uma atualizacao do monitor."""

    continuous = update.critical_alarm or update.state is DriverState.ASLEEP
    if continuous:
        return True, None
    if update.fatigue_level_changed:
        level = FatigueMonitor.fatigue_level(update.fatigue)
        return False, f"fatigue{level}"
    if update.new_yawn:
        return False, "yawn"
    if update.sleep_alarm:
        return False, "sleep"
    return False, None


class FaceAnalyzer:
    """MediaPipe assincrono: mede o fechamento real das palpebras."""

    def __init__(self, model_path: Path, closed_threshold: float, mouth_threshold: float) -> None:
        self._lock = threading.Lock()
        self._latest = FaceMetrics()
        self._eyes = EyeStateFilter(closed_threshold)
        self._face_box = FaceBoxFilter()
        self._mouth_threshold = mouth_threshold
        self._mouth_open = False
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.45,
            output_face_blendshapes=True,
            result_callback=self._on_result,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def _on_result(self, result, _image, _timestamp_ms: int) -> None:
        if not result.face_landmarks or not result.face_blendshapes:
            self._eyes.reset()
            self._face_box.reset()
            self._mouth_open = False
            metrics = FaceMetrics(updated_at=time.monotonic())
        else:
            values = {item.category_name: float(item.score) for item in result.face_blendshapes[0]}
            left = values.get("eyeBlinkLeft", 0.0)
            right = values.get("eyeBlinkRight", 0.0)
            now = time.monotonic()
            eyes_open = self._eyes.update(combined_blink_score(left, right), now)
            jaw_open = values.get("jawOpen", 0.0)
            # Histerese: depois de aberta, a boca so fecha com uma leitura
            # claramente menor. Isso evita cortes durante um bocejo.
            close_threshold = max(0.05, self._mouth_threshold - 0.14)
            self._mouth_open = (
                jaw_open >= close_threshold
                if self._mouth_open
                else jaw_open >= self._mouth_threshold
            )

            landmarks = result.face_landmarks[0]
            min_x = min(point.x for point in landmarks)
            max_x = max(point.x for point in landmarks)
            min_y = min(point.y for point in landmarks)
            max_y = max(point.y for point in landmarks)
            width = max_x - min_x
            height = max_y - min_y
            raw_box = (
                max(0.0, min_x - width * 0.10),
                max(0.0, min_y - height * 0.18),
                min(1.0, max_x + width * 0.10),
                min(1.0, max_y + height * 0.10),
            )
            metrics = FaceMetrics(
                face=True,
                left_eye_open=eyes_open,
                right_eye_open=eyes_open,
                mouth_open=self._mouth_open,
                box=self._face_box.update(raw_box),
                updated_at=now,
            )
        with self._lock:
            self._latest = metrics

    def submit(self, frame, timestamp_ms: int) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._landmarker.detect_async(image, timestamp_ms)

    def latest(self) -> FaceMetrics:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._landmarker.close()


class YoloWorker:
    """Executa o modelo pesado em paralelo e sempre usa o quadro mais recente."""

    def __init__(self, model_path: Path, confidence: float, device: str, imgsz: int) -> None:
        if device == "cpu":
            torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        self.model = YOLO(str(model_path))
        missing = EXPECTED_CLASSES.difference(self.model.names.values())
        if missing:
            raise RuntimeError(f"O modelo nao possui as classes esperadas: {sorted(missing)}")
        self.confidence = confidence
        self.device = device
        self.imgsz = imgsz
        self._frames: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest = YoloDetection()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame) -> None:
        if self._frames.full():
            return
        try:
            self._frames.put_nowait(frame.copy())
        except queue.Full:
            pass

    def latest(self) -> YoloDetection:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.1)
            except queue.Empty:
                continue
            result = self.model.predict(frame, conf=self.confidence, imgsz=self.imgsz, device=self.device, verbose=False)[0]
            boxes = []
            mouth_open = False
            if result.boxes is not None:
                for box in result.boxes:
                    score = float(box.conf.item())
                    class_id = int(box.cls.item())
                    label = result.names[class_id]
                    coords = tuple(int(value) for value in box.xyxy[0].tolist())
                    boxes.append((coords, label, score))
                    mouth_open = mouth_open or label == "bocaAberta"
            detection = YoloDetection(mouth_open=mouth_open, boxes=tuple(boxes), updated_at=time.monotonic())
            with self._lock:
                self._latest = detection


def ensure_face_model(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path
    print("Baixando o modelo leve de palpebras (somente na primeira execucao)...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(".download")
    try:
        urllib.request.urlretrieve(FACE_MODEL_URL, temporary)
        temporary.replace(model_path)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Nao foi possivel baixar o modelo facial: {error}") from error
    return model_path


def parse_source(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def draw_box(frame, coords, label: str) -> None:
    x1, y1, x2, y2 = coords
    color = (40, 200, 40) if label != "bocaAberta" else (0, 180, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def draw_face_box(frame, box: tuple[float, float, float, float] | None) -> None:
    if box is None:
        return
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    draw_box(
        frame,
        (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)),
        "rosto",
    )


UI_WIDTH = 1600
UI_HEIGHT = 900
UI_BG = (23, 16, 9)
UI_CARD = (39, 26, 16)
UI_BORDER = (62, 43, 28)
UI_MUTED = (178, 169, 157)
UI_WHITE = (242, 242, 244)
UI_RED = (69, 69, 248)
UI_GREEN = (154, 207, 30)
UI_BLUE = (255, 181, 55)
UI_ORANGE = (0, 167, 255)


def draw_rounded_rect(image, x: int, y: int, width: int, height: int, color, radius: int = 14, thickness: int = -1) -> None:
    """Desenha um painel arredondado usando apenas primitivas do OpenCV."""

    if thickness != -1:
        cv2.line(image, (x + radius, y), (x + width - radius, y), color, thickness)
        cv2.line(image, (x + radius, y + height), (x + width - radius, y + height), color, thickness)
        cv2.line(image, (x, y + radius), (x, y + height - radius), color, thickness)
        cv2.line(image, (x + width, y + radius), (x + width, y + height - radius), color, thickness)
        cv2.ellipse(image, (x + radius, y + radius), (radius, radius), 0, 180, 270, color, thickness)
        cv2.ellipse(image, (x + width - radius, y + radius), (radius, radius), 0, 270, 360, color, thickness)
        cv2.ellipse(image, (x + radius, y + height - radius), (radius, radius), 0, 90, 180, color, thickness)
        cv2.ellipse(image, (x + width - radius, y + height - radius), (radius, radius), 0, 0, 90, color, thickness)
        return
    cv2.rectangle(image, (x + radius, y), (x + width - radius, y + height), color, -1)
    cv2.rectangle(image, (x, y + radius), (x + width, y + height - radius), color, -1)
    cv2.circle(image, (x + radius, y + radius), radius, color, -1)
    cv2.circle(image, (x + width - radius, y + radius), radius, color, -1)
    cv2.circle(image, (x + radius, y + height - radius), radius, color, -1)
    cv2.circle(image, (x + width - radius, y + height - radius), radius, color, -1)


def draw_card(image, x: int, y: int, width: int, height: int, title: str) -> None:
    draw_rounded_rect(image, x, y, width, height, UI_CARD)
    draw_rounded_rect(image, x, y, width, height, UI_BORDER, thickness=1)
    cv2.putText(image, title, (x + 24, y + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.48, UI_MUTED, 1, cv2.LINE_AA)


def draw_badge(image, x: int, y: int, width: int, text: str, color) -> None:
    draw_rounded_rect(image, x, y, width, 30, color, radius=8)
    cv2.putText(image, text, (x + 14, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, UI_WHITE, 1, cv2.LINE_AA)


def resize_cover(image, width: int, height: int):
    """Preenche a area da camera preservando a proporcao, com corte central."""

    source_height, source_width = image.shape[:2]
    scale = max(width / source_width, height / source_height)
    resized = cv2.resize(
        image,
        (max(width, int(source_width * scale)), max(height, int(source_height * scale))),
        interpolation=cv2.INTER_LINEAR,
    )
    start_x = (resized.shape[1] - width) // 2
    start_y = (resized.shape[0] - height) // 2
    return resized[start_y:start_y + height, start_x:start_x + width]


def render_interface(frame, update, fps: float, face: FaceMetrics, sleep_seconds: float, source_name: str):
    """Monta o painel visual sem modificar a logica do monitoramento."""

    canvas = np.full((UI_HEIGHT, UI_WIDTH, 3), UI_BG, dtype=np.uint8)

    # Cabecalho
    draw_rounded_rect(canvas, 25, 25, 1550, 59, UI_CARD)
    draw_rounded_rect(canvas, 25, 25, 1550, 59, UI_BORDER, thickness=1)
    cv2.putText(canvas, "INSPETOR DE TRANSITO", (50, 60), cv2.FONT_HERSHEY_DUPLEX, 0.76, UI_WHITE, 1, cv2.LINE_AA)
    cv2.line(canvas, (350, 43), (350, 65), UI_MUTED, 1)
    cv2.putText(canvas, "SISTEMA DE MONITORAMENTO DE FADIGA DO MOTORISTA", (364, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, UI_MUTED, 1, cv2.LINE_AA)

    alert_text = "MONITORANDO"
    alert_color = (69, 117, 36)
    if update.state is DriverState.ASLEEP or update.critical_alarm:
        alert_text, alert_color = "ALERTA", (36, 32, 177)
    elif update.state is DriverState.DROWSY or update.fatigue >= 65:
        alert_text, alert_color = "ATENCAO", (25, 83, 192)
    elif update.state is DriverState.DISTRACTED:
        alert_text, alert_color = "SEM ROSTO", (25, 83, 192)
    draw_rounded_rect(canvas, 1400, 40, 150, 30, alert_color, radius=8)
    cv2.circle(canvas, (1419, 55), 6, UI_RED if alert_text != "MONITORANDO" else UI_GREEN, -1)
    cv2.putText(canvas, alert_text, (1434, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, UI_WHITE, 1, cv2.LINE_AA)

    # Area principal da camera
    draw_rounded_rect(canvas, 25, 100, 984, 768, UI_CARD)
    draw_rounded_rect(canvas, 25, 100, 984, 768, UI_BORDER, thickness=1)
    camera_x, camera_y, camera_width, camera_height = 42, 149, 950, 691
    camera_view = resize_cover(frame, camera_width, camera_height)
    canvas[camera_y:camera_y + camera_height, camera_x:camera_x + camera_width] = camera_view
    cv2.rectangle(canvas, (camera_x, camera_y), (camera_x + camera_width, camera_y + camera_height), UI_BORDER, 1)

    cv2.putText(canvas, "CAM 01 - CABINE DO MOTORISTA", (67, 183), cv2.FONT_HERSHEY_SIMPLEX, 0.48, UI_BLUE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"REC  LIVE | {frame.shape[1]}x{frame.shape[0]} | {fps:.1f} FPS", (67, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.40, UI_RED, 1, cv2.LINE_AA)

    warning = None
    if update.state is DriverState.ASLEEP:
        warning = "PERIGO: MOTORISTA DORMINDO - PARE EM LOCAL SEGURO!"
    elif update.critical_alarm or update.fatigue >= 85:
        warning = "PERIGO: CANSACO CRITICO - PARE EM LOCAL SEGURO!"
    elif update.state is DriverState.DROWSY:
        warning = "SONOLENCIA DETECTADA - MANTENHA-SE ALERTA!"
    if warning:
        draw_rounded_rect(canvas, 142, 758, 750, 60, (35, 31, 169), radius=12)
        draw_rounded_rect(canvas, 142, 758, 750, 60, UI_RED, radius=12, thickness=1)
        text_size = cv2.getTextSize(warning, cv2.FONT_HERSHEY_DUPLEX, 0.56, 1)[0]
        cv2.putText(canvas, warning, (517 - text_size[0] // 2, 794), cv2.FONT_HERSHEY_DUPLEX, 0.56, UI_WHITE, 1, cv2.LINE_AA)

    # Cartao: nivel de cansaco
    right_x, right_width = 1025, 551
    draw_card(canvas, right_x, 100, right_width, 160, "NIVEL DE CANSACO")
    fatigue = max(0.0, min(100.0, update.fatigue))
    if fatigue >= 85:
        fatigue_label, fatigue_color = "CRITICO", UI_RED
    elif fatigue >= 65:
        fatigue_label, fatigue_color = "ALTO", UI_ORANGE
    elif fatigue >= 40:
        fatigue_label, fatigue_color = "ATENCAO", (0, 205, 255)
    else:
        fatigue_label, fatigue_color = "NORMAL", UI_GREEN
    cv2.putText(canvas, f"{fatigue:.0f}%", (1050, 177), cv2.FONT_HERSHEY_DUPLEX, 1.20, fatigue_color, 2, cv2.LINE_AA)
    draw_badge(canvas, 1175, 162, 105, fatigue_label, (35, 31, 169) if fatigue >= 85 else (54, 70, 38))
    cv2.rectangle(canvas, (1050, 216), (1552, 234), (48, 62, 79), -1)
    fill_width = int(502 * fatigue / 100.0)
    if fill_width:
        cv2.rectangle(canvas, (1050, 216), (1050 + fill_width, 234), fatigue_color, -1)

    # Cartao: olhos
    draw_card(canvas, right_x, 276, right_width, 159, "ESTADO DOS OLHOS")
    if not face.face:
        eye_label, eye_color = "SEM ROSTO", UI_MUTED
    elif face.left_eye_open and face.right_eye_open:
        eye_label, eye_color = "ABERTOS", UI_GREEN
    else:
        eye_label, eye_color = "FECHADOS", UI_RED
    cv2.putText(canvas, "Status:", (1050, 342), cv2.FONT_HERSHEY_SIMPLEX, 0.47, UI_WHITE, 1, cv2.LINE_AA)
    draw_badge(canvas, 1117, 323, 104, eye_label, (35, 31, 169) if eye_label == "FECHADOS" else (54, 70, 38))
    cv2.putText(canvas, "Tempo Fechado:", (1050, 383), cv2.FONT_HERSHEY_SIMPLEX, 0.47, UI_WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{update.eyes_closed_seconds:.1f}s / {sleep_seconds:.1f}s", (1175, 383), cv2.FONT_HERSHEY_SIMPLEX, 0.57, eye_color, 2, cv2.LINE_AA)

    # Cartao: eventos
    draw_card(canvas, right_x, 451, right_width, 150, "EVENTOS & FADIGA")
    cv2.putText(canvas, "Contagem de Bocejos:", (1050, 517), cv2.FONT_HERSHEY_SIMPLEX, 0.47, UI_WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{update.yawn_count} bocejo{'s' if update.yawn_count != 1 else ''}", (1250, 517), cv2.FONT_HERSHEY_SIMPLEX, 0.55, UI_ORANGE, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Estado do Motorista:", (1050, 558), cv2.FONT_HERSHEY_SIMPLEX, 0.47, UI_WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, update.state.value, (1250, 558), cv2.FONT_HERSHEY_SIMPLEX, 0.43, UI_BLUE, 1, cv2.LINE_AA)

    # Cartao: telemetria
    draw_card(canvas, right_x, 617, right_width, 251, "TELEMETRIA DE IA & HARDWARE")
    cv2.putText(canvas, "Taxa de Quadros (FPS):", (1050, 689), cv2.FONT_HERSHEY_SIMPLEX, 0.47, UI_WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{fps:.1f} FPS", (1260, 689), cv2.FONT_HERSHEY_SIMPLEX, 0.55, UI_GREEN, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Resolucao da Camera:", (1050, 731), cv2.FONT_HERSHEY_SIMPLEX, 0.47, UI_WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{frame.shape[1]} x {frame.shape[0]}", (1260, 731), cv2.FONT_HERSHEY_SIMPLEX, 0.49, UI_BLUE, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Dispositivo de Video:", (1050, 773), cv2.FONT_HERSHEY_SIMPLEX, 0.47, UI_WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, source_name, (1260, 773), cv2.FONT_HERSHEY_SIMPLEX, 0.47, UI_WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Q / ESC: sair     R: zerar fadiga", (1050, 829), cv2.FONT_HERSHEY_SIMPLEX, 0.43, UI_MUTED, 1, cv2.LINE_AA)
    return canvas


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor de sonolencia do motorista")
    parser.add_argument("--model", default="best.pt", help="caminho do modelo YOLO")
    parser.add_argument("--face-model", default="face_landmarker.task", help="modelo facial MediaPipe")
    parser.add_argument("--source", default="0", help="indice da camera ou caminho de video")
    parser.add_argument("--confidence", type=float, default=0.45, help="confianca minima do YOLO")
    parser.add_argument("--eye-closed-threshold", type=float, default=0.55, help="limite de palpebra fechada")
    parser.add_argument("--mouth-open-threshold", type=float, default=0.38, help="limite de abertura da boca no MediaPipe")
    parser.add_argument("--sleep-seconds", type=float, default=3.0, help="tempo de olhos fechados para alarme")
    parser.add_argument("--yawn-seconds", type=float, default=0.9, help="duracao minima de um bocejo")
    parser.add_argument("--fatigue-per-yawn", type=float, default=20.0, help="percentual somado por bocejo")
    parser.add_argument("--yolo-imgsz", type=int, default=416, help="resolucao da inferencia YOLO")
    parser.add_argument("--width", type=int, default=960, help="largura solicitada da camera")
    parser.add_argument("--height", type=int, default=540, help="altura solicitada da camera")
    parser.add_argument("--no-sound", action="store_true", help="desativa os sons")
    parser.add_argument("--test-sound", action="store_true", help="testa os alertas de audio e encerra")
    parser.add_argument("--device", default="cpu", help="cpu ou indice da GPU, como 0")
    return parser


def run(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")
    if not 0 < args.confidence <= 1 or not 0 < args.eye_closed_threshold < 1 or not 0 < args.mouth_open_threshold < 1:
        raise ValueError("Os limites de confianca devem estar entre 0 e 1")

    face_model = ensure_face_model(Path(args.face_model))
    yolo = YoloWorker(model_path, args.confidence, args.device, args.yolo_imgsz)
    face_analyzer = FaceAnalyzer(face_model, args.eye_closed_threshold, args.mouth_open_threshold)
    source = parse_source(args.source)
    capture = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
    if not capture.isOpened():
        yolo.close()
        face_analyzer.close()
        raise RuntimeError(f"Nao foi possivel abrir a fonte de video: {args.source}")
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, 30)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    monitor = FatigueMonitor(
        sleep_seconds=args.sleep_seconds,
        drowsy_seconds=min(0.7, args.sleep_seconds / 2),
        yawn_seconds=args.yawn_seconds,
        fatigue_per_yawn=args.fatigue_per_yawn,
    )
    sound = SoundPlayer(enabled=not args.no_sound)
    if sound.enabled:
        # Confirma ja na inicializacao que a aplicacao alcancou a saida de
        # audio; os alertas seguintes usam exatamente o mesmo caminho.
        sound.play("startup")
    previous_time = time.monotonic()
    start_time = previous_time
    smoothed_fps = 0.0
    source_name = f"Camera USB {source}" if isinstance(source, int) else Path(source).name
    window_name = "Inspetor de Transito - Monitor do Motorista"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, UI_WIDTH, UI_HEIGHT)

    print("Monitor iniciado. Q/Esc sai; R zera a fadiga apos o descanso.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if isinstance(source, str):
                    break
                return 2

            now = time.monotonic()
            timestamp_ms = int((now - start_time) * 1000)
            face_analyzer.submit(frame, timestamp_ms)
            yolo.submit(frame)
            face = face_analyzer.latest()
            detection = yolo.latest()
            face_current = face.face and now - face.updated_at < 1.0
            yolo_mouth_open = detection.mouth_open and now - detection.updated_at < 1.0
            mouth_open = face_current and (face.mouth_open or yolo_mouth_open)
            observation = FrameObservation(
                face=face_current,
                left_eye=face_current and face.left_eye_open,
                right_eye=face_current and face.right_eye_open,
                mouth_open=mouth_open,
            )
            update = monitor.update(observation, now)

            elapsed = max(now - previous_time, 1e-6)
            instant_fps = 1.0 / elapsed
            smoothed_fps = instant_fps if not smoothed_fps else 0.9 * smoothed_fps + 0.1 * instant_fps
            previous_time = now

            continuous_alarm, sound_alert = alarm_decision(update)
            sound.set_continuous(continuous_alarm)
            if sound_alert is not None:
                sound.play(sound_alert)

            for coords, label, _score in detection.boxes:
                # As caixas de olhos do YOLO apenas localizam a regiao e podem
                # piscar na tela. O estado aberto/fechado vem do MediaPipe.
                if label not in {"olhoDireito", "olhoEsquerdo", "rosto"}:
                    draw_box(frame, coords, label)
            if face_current:
                draw_face_box(frame, face.box)
            display_face = face if face_current else FaceMetrics()
            interface = render_interface(
                frame,
                update,
                smoothed_fps,
                display_face,
                args.sleep_seconds,
                source_name,
            )
            cv2.imshow(window_name, interface)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                monitor.reset_fatigue()
                sound.set_continuous(update.state is DriverState.ASLEEP)
    finally:
        capture.release()
        cv2.destroyAllWindows()
        sound.close()
        yolo.close()
        face_analyzer.close()
    return 0


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    if args.test_sound:
        sound = SoundPlayer(enabled=True)
        print("Teste de audio: dois tons curtos e uma sirene alternada.")
        sound.play("startup")
        time.sleep(0.7)
        sound.set_continuous(True)
        time.sleep(1.2)
        sound.set_continuous(False)
        sound.close()
        return 0
    try:
        return run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
