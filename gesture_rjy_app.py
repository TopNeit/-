#!/usr/bin/env python3
"""
Распознавание динамических жестов РЖЯ в реальном времени (камера -> текст -> голос).

Что внутри:
- OpenCV: поток с веб-камеры.
- MediaPipe Hands: трекинг до двух рук.
- Динамическое распознавание:
    1) ML-режим (LSTM/GRU), если есть model.h5 + labels.txt;
    2) Расширенный fallback-режим (движение + форма кисти) с увеличенным словарём.
- Tkinter GUI: видео, распознанный текст, история, кнопки управления.
- pyttsx3: озвучка текста.
- history.txt: сохранение истории распознаваний.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk


if importlib.util.find_spec("tensorflow") is not None:
    from tensorflow.keras.models import load_model
else:
    load_model = None


MODEL_PATH = "model.h5"
LABELS_PATH = "labels.txt"
HISTORY_PATH = "history.txt"
LEXICON_DB_PATH = "lexicon.db"
USER_GESTURES_PATH = "user_gestures.json"

SEQUENCE_LEN = 30
PREDICTION_THRESHOLD = 0.90
PREDICTION_COOLDOWN_SEC = 0.8

WINDOW_TITLE = "Распознавание динамических жестов РЖЯ"
CAMERA_ID = 0
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540

# Увеличенный fallback-словарь (используется без ML модели).
FALLBACK_VOCAB = [
    "ПРИВЕТ", "ПОКА", "ДА", "НЕТ", "СПАСИБО", "ПОЖАЛУЙСТА", "Я", "ТЫ", "МЫ",
    "СТОП", "ПОМОГИ", "ОТЛИЧНО", "ВПРАВО", "ВЛЕВО", "ВВЕРХ", "ВНИЗ",
]


def number_to_russian(n: int) -> str:
    """Преобразует число 0..9999 в русскую текстовую форму."""
    if n == 0:
        return "ноль"

    ones_m = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
    ones_f = ["", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
    teens = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
    tens = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
    hundreds = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]

    parts: List[str] = []

    if n >= 1000:
        t = n // 1000
        if t == 1:
            parts.append("тысяча")
        elif t == 2:
            parts.append("две тысячи")
        elif t in (3, 4):
            parts.append(f"{ones_f[t]} тысячи")
        else:
            parts.append(f"{ones_m[t]} тысяч")
        n = n % 1000

    h = n // 100
    if h:
        parts.append(hundreds[h])
    n = n % 100

    if 10 <= n <= 19:
        parts.append(teens[n - 10])
        return " ".join([p for p in parts if p]).strip()

    t = n // 10
    if t:
        parts.append(tens[t])
    o = n % 10
    if o:
        parts.append(ones_m[o])

    return " ".join([p for p in parts if p]).strip()


class LexiconDB:
    """SQLite-база словаря (1000+ слов/фраз) для пост-обработки меток."""

    def __init__(self, db_path: str = LEXICON_DB_PATH) -> None:
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lexicon (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL
                )
                """
            )
            conn.commit()

        if self.count_tokens() < 1000:
            self._bootstrap_tokens()

    def _bootstrap_tokens(self) -> None:
        common_words = [
            "привет", "пока", "да", "нет", "спасибо", "пожалуйста", "извини", "помоги", "стоп", "быстро",
            "медленно", "вверх", "вниз", "влево", "вправо", "я", "ты", "мы", "они", "он", "она",
            "работа", "дом", "школа", "университет", "больница", "аптека", "магазин", "метро", "автобус",
            "поезд", "машина", "улица", "город", "деревня", "друг", "семья", "мама", "папа", "ребенок",
            "врач", "учитель", "студент", "еда", "вода", "чай", "кофе", "завтрак", "обед", "ужин",
            "сегодня", "завтра", "вчера", "утро", "день", "вечер", "ночь", "хорошо", "плохо", "отлично",
            "опасно", "безопасно", "тепло", "холодно", "рад", "грустно", "злой", "тихо", "громко", "повтори",
        ]

        # 1201 числовая фраза: от 0 до 1200.
        number_tokens = [number_to_russian(i) for i in range(0, 1201)]

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO lexicon(token, category) VALUES(?, ?)",
                [(w, "common") for w in common_words],
            )
            conn.executemany(
                "INSERT OR IGNORE INTO lexicon(token, category) VALUES(?, ?)",
                [(w, "number") for w in number_tokens],
            )
            conn.commit()

    def count_tokens(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM lexicon")
            return int(cur.fetchone()[0])

    def token_by_id(self, idx: int) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT token FROM lexicon WHERE id = ?", (idx,))
            row = cur.fetchone()
            return row[0] if row else None

    def resolve(self, raw_label: str) -> str:
        """Пытается преобразовать метку в слово из базы:
        - если метка числовая (например, '42'), возвращает token по id;
        - иначе возвращает как есть.
        """
        txt = raw_label.strip()
        if txt.isdigit():
            token = self.token_by_id(int(txt))
            if token:
                return token
        return raw_label


class UserGestureConfig:
    """JSON-конфиг для пользовательских жестов и переименования меток.

    Файл user_gestures.json можно редактировать вручную:
    - label_map: переименование ML-меток (или fallback-меток) в удобный текст.
    - fallback_rules: простые правила для эвристического режима.
    """

    DEFAULT_DATA = {
        "label_map": {
            "ПРИВЕТ": "Привет!",
            "ПОКА": "Пока!"
        },
        "fallback_rules": [
            {
                "name": "custom_help",
                "motion": "still",
                "fingers": [1, 0, 0, 0, 1],
                "output": "ПОМОГИТЕ МНЕ"
            }
        ]
    }

    def __init__(self, path: str = USER_GESTURES_PATH) -> None:
        self.path = path
        self.data = self._load_or_create()

    def _load_or_create(self) -> Dict:
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.DEFAULT_DATA, f, ensure_ascii=False, indent=2)
            return dict(self.DEFAULT_DATA)

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                return dict(self.DEFAULT_DATA)
            loaded.setdefault("label_map", {})
            loaded.setdefault("fallback_rules", [])
            return loaded
        except Exception:
            return dict(self.DEFAULT_DATA)

    def reload(self) -> None:
        self.data = self._load_or_create()

    def map_label(self, label: str) -> str:
        return self.data.get("label_map", {}).get(label, label)

    def match_fallback_rule(
        self,
        motion_tag: str,
        finger_pattern: Optional[Tuple[int, int, int, int, int]],
    ) -> Optional[str]:
        rules = self.data.get("fallback_rules", [])
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            rule_motion = rule.get("motion")
            rule_fingers = rule.get("fingers")
            rule_output = rule.get("output")
            if not rule_output:
                continue
            if rule_motion and rule_motion != motion_tag:
                continue
            if rule_fingers is not None:
                if finger_pattern is None:
                    continue
                if list(finger_pattern) != list(rule_fingers):
                    continue
            return str(rule_output)
        return None


@dataclass
class RecognitionEvent:
    text: str
    timestamp: datetime

    def to_history_line(self) -> str:
        return f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {self.text}"


class DynamicGestureRecognizer:
    """Распознавание динамических жестов по последовательности landmarks."""

    def __init__(self, lexicon_db: Optional[LexiconDB] = None, user_cfg: Optional[UserGestureConfig] = None) -> None:
        self.lexicon_db = lexicon_db
        self.user_cfg = user_cfg
        self.model = None
        self.labels: List[str] = []
        self.use_ml_model = False

        self.sequence: Deque[np.ndarray] = deque(maxlen=SEQUENCE_LEN)
        self.prediction_buffer: Deque[str] = deque(maxlen=8)
        self.last_emit_time = 0.0

        # Буферы для fallback-режима.
        self.wrist_path_left: Deque[Tuple[float, float]] = deque(maxlen=SEQUENCE_LEN)
        self.wrist_path_right: Deque[Tuple[float, float]] = deque(maxlen=SEQUENCE_LEN)
        self.fingers_left: Deque[Tuple[int, int, int, int, int]] = deque(maxlen=SEQUENCE_LEN)
        self.fingers_right: Deque[Tuple[int, int, int, int, int]] = deque(maxlen=SEQUENCE_LEN)

        self._load_model_if_exists()

    def _load_model_if_exists(self) -> None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
            return

        if load_model is None:
            print("[WARN] TensorFlow не найден. Используется fallback-словарь.")
            return

        self.model = load_model(MODEL_PATH)
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            self.labels = [line.strip() for line in f if line.strip()]

        if not self.labels:
            self.model = None
            print("[WARN] labels.txt пуст. Используется fallback-словарь.")
            return

        self.use_ml_model = True
        print(f"[INFO] Загружена ML-модель. Классов: {len(self.labels)}")

    def reset(self) -> None:
        self.sequence.clear()
        self.prediction_buffer.clear()
        self.wrist_path_left.clear()
        self.wrist_path_right.clear()
        self.fingers_left.clear()
        self.fingers_right.clear()
        self.last_emit_time = 0.0

    @staticmethod
    def _extract_two_hand_features(multi_hand_landmarks, multi_handedness) -> np.ndarray:
        left = np.zeros(63, dtype=np.float32)
        right = np.zeros(63, dtype=np.float32)

        if not multi_hand_landmarks or not multi_handedness:
            return np.concatenate([left, right])

        for hand_lms, handedness in zip(multi_hand_landmarks, multi_handedness):
            coords = []
            for lm in hand_lms.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            coords_arr = np.array(coords, dtype=np.float32)
            side = handedness.classification[0].label.lower()
            if side == "left":
                left = coords_arr
            else:
                right = coords_arr

        return np.concatenate([left, right])

    @staticmethod
    def _finger_state(hand_landmarks, hand_label: str) -> Tuple[int, int, int, int, int]:
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [3, 6, 10, 14, 18]

        states: List[int] = []

        thumb_tip = hand_landmarks.landmark[tip_ids[0]]
        thumb_pip = hand_landmarks.landmark[pip_ids[0]]
        if hand_label.lower() == "right":
            states.append(1 if thumb_tip.x < thumb_pip.x else 0)
        else:
            states.append(1 if thumb_tip.x > thumb_pip.x else 0)

        for tip_id, pip_id in zip(tip_ids[1:], pip_ids[1:]):
            tip = hand_landmarks.landmark[tip_id]
            pip = hand_landmarks.landmark[pip_id]
            states.append(1 if tip.y < pip.y else 0)

        return tuple(states)

    def _update_fallback_buffers(self, multi_hand_landmarks, multi_handedness) -> None:
        if not multi_hand_landmarks or not multi_handedness:
            return

        for hand_lms, handedness in zip(multi_hand_landmarks, multi_handedness):
            side = handedness.classification[0].label.lower()
            wrist = hand_lms.landmark[0]
            fingers = self._finger_state(hand_lms, side)

            if side == "left":
                self.wrist_path_left.append((wrist.x, wrist.y))
                self.fingers_left.append(fingers)
            else:
                self.wrist_path_right.append((wrist.x, wrist.y))
                self.fingers_right.append(fingers)

    @staticmethod
    def _majority_finger_pattern(buffer: Deque[Tuple[int, int, int, int, int]]) -> Optional[Tuple[int, int, int, int, int]]:
        if len(buffer) < 10:
            return None
        return Counter(buffer).most_common(1)[0][0]

    @staticmethod
    def _motion_metrics(path: Deque[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
        if len(path) < SEQUENCE_LEN:
            return None

        xs = np.array([p[0] for p in path], dtype=np.float32)
        ys = np.array([p[1] for p in path], dtype=np.float32)
        dx = float(xs[-1] - xs[0])
        dy = float(ys[-1] - ys[0])
        amp_x = float(xs.max() - xs.min())
        amp_y = float(ys.max() - ys.min())
        return dx, dy, amp_x, amp_y

    def _decode_static_phrase(self) -> Optional[str]:
        left = self._majority_finger_pattern(self.fingers_left)
        right = self._majority_finger_pattern(self.fingers_right)

        # Две руки: расширенные шаблоны
        if left == (1, 1, 1, 1, 1) and right == (1, 1, 1, 1, 1):
            return "МЫ"
        if left == (0, 0, 0, 0, 0) and right == (0, 0, 0, 0, 0):
            return "СТОП"

        dominant = right or left
        if dominant is None:
            return None

        static_map = {
            (1, 1, 1, 1, 1): "ПРИВЕТ",
            (0, 0, 0, 0, 0): "НЕТ",
            (0, 1, 0, 0, 0): "ТЫ",
            (1, 0, 0, 0, 0): "Я",
            (0, 1, 1, 0, 0): "ДА",
            (1, 1, 0, 0, 1): "ОТЛИЧНО",
            (1, 0, 0, 0, 1): "ПОМОГИ",
        }
        return static_map.get(dominant)

    def _current_dominant_fingers(self) -> Optional[Tuple[int, int, int, int, int]]:
        right = self._majority_finger_pattern(self.fingers_right)
        left = self._majority_finger_pattern(self.fingers_left)
        return right or left

    def _motion_tag(self) -> str:
        path = self.wrist_path_right if len(self.wrist_path_right) >= len(self.wrist_path_left) else self.wrist_path_left
        metrics = self._motion_metrics(path)
        if metrics is None:
            return "unknown"
        dx, dy, amp_x, amp_y = metrics
        if amp_x > 0.20 and amp_y < 0.20:
            return "horizontal"
        if amp_y > 0.22 and amp_x < 0.20:
            return "vertical"
        if dx > 0.20:
            return "right"
        if dx < -0.20:
            return "left"
        if dy < -0.18:
            return "up"
        if dy > 0.18:
            return "down"
        if abs(dx) < 0.06 and abs(dy) < 0.06 and (amp_x + amp_y) < 0.12:
            return "still"
        return "unknown"

    def _decode_motion_phrase(self) -> Optional[str]:
        # Выбираем более длинную траекторию как доминирующую руку.
        path = self.wrist_path_right if len(self.wrist_path_right) >= len(self.wrist_path_left) else self.wrist_path_left
        metrics = self._motion_metrics(path)
        if metrics is None:
            return None

        dx, dy, amp_x, amp_y = metrics

        # Расширенный словарь динамики.
        if amp_x > 0.20 and amp_y < 0.20:
            return "ПОКА"
        if amp_y > 0.22 and amp_x < 0.20:
            return "СПАСИБО"
        if dx > 0.20:
            return "ВПРАВО"
        if dx < -0.20:
            return "ВЛЕВО"
        if dy < -0.18:
            return "ВВЕРХ"
        if dy > 0.18:
            return "ВНИЗ"
        if abs(dx) < 0.06 and abs(dy) < 0.06 and (amp_x + amp_y) < 0.12:
            return "ПОЖАЛУЙСТА"
        return None

    def _fallback_prediction(self) -> Optional[str]:
        motion_tag = self._motion_tag()
        dominant_fingers = self._current_dominant_fingers()

        if self.user_cfg is not None:
            custom = self.user_cfg.match_fallback_rule(motion_tag, dominant_fingers)
            if custom:
                return custom

        motion = self._decode_motion_phrase()
        static = self._decode_static_phrase()

        # Приоритет динамике, затем статике.
        return motion or static

    def _emit_with_voting(self, label: str) -> str:
        self.prediction_buffer.append(label)
        return Counter(self.prediction_buffer).most_common(1)[0][0]

    def predict(self, multi_hand_landmarks, multi_handedness) -> Optional[str]:
        features = self._extract_two_hand_features(multi_hand_landmarks, multi_handedness)
        self.sequence.append(features)
        self._update_fallback_buffers(multi_hand_landmarks, multi_handedness)

        if len(self.sequence) < SEQUENCE_LEN:
            return None

        now = time.time()
        if now - self.last_emit_time < PREDICTION_COOLDOWN_SEC:
            return None

        if self.use_ml_model and self.model is not None:
            input_seq = np.expand_dims(np.array(self.sequence, dtype=np.float32), axis=0)
            probs = self.model.predict(input_seq, verbose=0)[0]
            best_idx = int(np.argmax(probs))
            best_prob = float(probs[best_idx])
            if best_prob >= PREDICTION_THRESHOLD and best_idx < len(self.labels):
                resolved = self.lexicon_db.resolve(self.labels[best_idx]) if self.lexicon_db else self.labels[best_idx]
                resolved = self.user_cfg.map_label(resolved) if self.user_cfg else resolved
                voted = self._emit_with_voting(resolved)
                self.last_emit_time = now
                return voted
            return None

        fallback_label = self._fallback_prediction()
        if fallback_label:
            fallback_label = self.user_cfg.map_label(fallback_label) if self.user_cfg else fallback_label
            voted = self._emit_with_voting(fallback_label)
            self.last_emit_time = now
            return voted
        return None


class SignLanguageApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1200x760")

        self.lexicon_db = LexiconDB()
        self.user_cfg = UserGestureConfig()
        self.recognizer = DynamicGestureRecognizer(self.lexicon_db, self.user_cfg)

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1,
        )

        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError("Не удалось открыть веб-камеру.")

        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 170)

        self.current_text_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value=self._status_text())

        self.history_events: List[RecognitionEvent] = []

        self._build_ui()
        self._load_history()

        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_frame()

    def _status_text(self) -> str:
        if self.recognizer.use_ml_model:
            return "Режим: ML-модель (рекомендуется для точности >=90%)"
        return f"Режим: fallback-словарь ({len(FALLBACK_VOCAB)} фраз) + БД {self.lexicon_db.count_tokens()} слов + user_gestures.json"

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(main, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        self.video_label = ttk.Label(left)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        text_frame = ttk.LabelFrame(left, text="Распознанный текст", padding=10)
        text_frame.pack(fill=tk.X, pady=8)

        ttk.Label(
            text_frame,
            textvariable=self.current_text_var,
            font=("Arial", 16, "bold"),
            foreground="#0a4b78",
            anchor="w",
        ).pack(fill=tk.X)

        ttk.Label(left, textvariable=self.status_var, foreground="#666").pack(fill=tk.X, pady=(2, 8))

        control_frame = ttk.Frame(left)
        control_frame.pack(fill=tk.X)

        ttk.Button(control_frame, text="🔊 Озвучить текст", command=self.speak_current_text).pack(side=tk.LEFT, padx=4)
        ttk.Button(control_frame, text="🧹 Очистить историю", command=self.clear_history).pack(side=tk.LEFT, padx=4)
        ttk.Button(control_frame, text="🔄 Перезапуск распознавания", command=self.restart_recognition).pack(side=tk.LEFT, padx=4)
        ttk.Button(control_frame, text="📥 Перезагрузить жесты", command=self.reload_user_gestures).pack(side=tk.LEFT, padx=4)

        history_frame = ttk.LabelFrame(right, text="История сообщений", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True)

        self.history_list = tk.Listbox(history_frame, font=("Consolas", 10), height=34)
        self.history_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_list.config(yscrollcommand=scrollbar.set)

    def _load_history(self) -> None:
        if not os.path.exists(HISTORY_PATH):
            return
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            for line in [line.strip() for line in f if line.strip()][-100:]:
                self.history_list.insert(tk.END, line)

    def _append_history(self, text: str) -> None:
        event = RecognitionEvent(text=text, timestamp=datetime.now())
        line = event.to_history_line()
        self.history_events.append(event)
        self.history_list.insert(tk.END, line)
        self.history_list.yview_moveto(1.0)
        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    def _fingers_state_text(hand_landmarks, hand_label: str) -> str:
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [3, 6, 10, 14, 18]

        vals: List[int] = []
        thumb_tip = hand_landmarks.landmark[tip_ids[0]]
        thumb_pip = hand_landmarks.landmark[pip_ids[0]]
        if hand_label.lower() == "right":
            vals.append(1 if thumb_tip.x < thumb_pip.x else 0)
        else:
            vals.append(1 if thumb_tip.x > thumb_pip.x else 0)

        for tip, pip in zip(tip_ids[1:], pip_ids[1:]):
            vals.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y else 0)

        names = ["T", "I", "M", "R", "P"]
        return " ".join(f"{n}:{v}" for n, v in zip(names, vals))

    def _draw_hand_hints(self, frame_bgr: np.ndarray, results) -> None:
        if not results.multi_hand_landmarks or not results.multi_handedness:
            return

        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            self.mp_draw.draw_landmarks(
                frame_bgr,
                hand_lms,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )
            label = handedness.classification[0].label
            score = handedness.classification[0].score
            wrist = hand_lms.landmark[0]
            h, w, _ = frame_bgr.shape
            x_px, y_px = int(wrist.x * w), int(wrist.y * h)
            finger_info = self._fingers_state_text(hand_lms, label)
            cv2.putText(
                frame_bgr,
                f"{label} ({score:.2f}) {finger_info}",
                (x_px - 90, max(20, y_px - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (60, 220, 60),
                2,
                cv2.LINE_AA,
            )

    def speak_current_text(self) -> None:
        text = self.current_text_var.get().strip()
        if not text:
            messagebox.showinfo("Озвучка", "Нет текста для озвучки.")
            return

        def _speak_worker() -> None:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

        threading.Thread(target=_speak_worker, daemon=True).start()

    def clear_history(self) -> None:
        self.history_events.clear()
        self.history_list.delete(0, tk.END)
        open(HISTORY_PATH, "w", encoding="utf-8").close()

    def restart_recognition(self) -> None:
        self.recognizer.reset()
        self.current_text_var.set("")
        self.status_var.set(self._status_text() + " | Буферы очищены")

    def reload_user_gestures(self) -> None:
        self.user_cfg.reload()
        self.status_var.set(self._status_text() + " | user_gestures.json перезагружен")

    def _process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        self._draw_hand_hints(frame_bgr, results)

        recognized = self.recognizer.predict(results.multi_hand_landmarks, results.multi_handedness)
        if recognized:
            self.current_text_var.set(recognized)
            self._append_history(recognized)

        return frame_bgr

    def update_frame(self) -> None:
        if not self.running:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.status_var.set("Ошибка чтения кадра с камеры")
            self.root.after(50, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        frame = self._process_frame(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tk_image = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        self.video_label.imgtk = tk_image
        self.video_label.configure(image=tk_image)

        self.root.after(10, self.update_frame)

    def on_close(self) -> None:
        self.running = False
        self.cap.release()
        self.hands.close()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        SignLanguageApp(root)
    except Exception as exc:
        messagebox.showerror("Ошибка запуска", str(exc))
        return
    root.mainloop()


if __name__ == "__main__":
    main()
