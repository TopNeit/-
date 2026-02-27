#!/usr/bin/env python3
"""Интерактивное добавление нового жеста в user_gestures.json.

Сценарий:
1) Запускает камеру и MediaPipe Hands.
2) Пользователь показывает жест.
3) Нажимает SPACE для захвата.
4) Скрипт просит ввести название жеста.
5) Добавляет правило в user_gestures.json (fallback_rules + label_map).
"""

from __future__ import annotations

import json
import os
from collections import Counter, deque
from typing import Deque, List, Optional, Tuple

import cv2
import mediapipe as mp

USER_GESTURES_PATH = "user_gestures.json"
SEQUENCE_FRAMES = 20


def finger_state(hand_landmarks, hand_label: str) -> Tuple[int, int, int, int, int]:
    """Возвращает состояние 5 пальцев: 1 поднят / 0 опущен."""
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


def load_or_create_config(path: str) -> dict:
    default_data = {"label_map": {}, "fallback_rules": []}
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_data, f, ensure_ascii=False, indent=2)
        return default_data

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return default_data
    data.setdefault("label_map", {})
    data.setdefault("fallback_rules", [])
    return data


def save_config(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    pattern_buffer: Deque[Tuple[int, int, int, int, int]] = deque(maxlen=SEQUENCE_FRAMES)

    print("Покажите жест в камеру. Нажмите SPACE для сохранения, Q для выхода.")

    captured_pattern: Optional[Tuple[int, int, int, int, int]] = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_lms = results.multi_hand_landmarks[0]
            hand_label = results.multi_handedness[0].classification[0].label
            state = finger_state(hand_lms, hand_label)
            pattern_buffer.append(state)

            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            cv2.putText(
                frame,
                f"Fingers: {state}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.putText(frame, "SPACE: capture gesture | Q: quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("add_new_gesture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32:
            if len(pattern_buffer) < 5:
                print("Недостаточно стабильных кадров. Подержите жест 1-2 секунды и нажмите SPACE снова.")
                continue
            captured_pattern = Counter(pattern_buffer).most_common(1)[0][0]
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    if captured_pattern is None:
        print("Жест не сохранён.")
        return

    print(f"Распознана форма пальцев: {captured_pattern}")
    gesture_name = input("Введите название жеста: ").strip()
    if not gesture_name:
        print("Пустое название. Отмена.")
        return

    config = load_or_create_config(USER_GESTURES_PATH)
    rule_name = f"custom_{gesture_name.lower().replace(' ', '_')}"

    new_rule = {
        "name": rule_name,
        "motion": "still",
        "fingers": list(captured_pattern),
        "output": gesture_name,
    }

    # Удаляем старые правила с тем же именем, чтобы не дублировать.
    config["fallback_rules"] = [r for r in config["fallback_rules"] if not (isinstance(r, dict) and r.get("name") == rule_name)]
    config["fallback_rules"].append(new_rule)

    # И сразу добавляем map, чтобы текст был единообразный.
    config["label_map"][gesture_name] = gesture_name

    save_config(USER_GESTURES_PATH, config)

    print("Готово! Жест добавлен в user_gestures.json")
    print("В основном приложении нажмите кнопку '📥 Перезагрузить жесты'.")


if __name__ == "__main__":
    main()
