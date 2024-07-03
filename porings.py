import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import time
import os
import json
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "CATCHING_RADIUS": 60,
    "INITIAL_LIVES": 5,
    "INITIAL_PORING_COUNT": 5,
    "INITIAL_PORING_LIFETIME": 5,
    "DIFFICULTY_INCREASE_RATE": 0.98,
    "LEVEL_UP_SCORE": 10,
    "SCREEN_WIDTH": 1280,
    "SCREEN_HEIGHT": 720,
    "FPS": 60,
    "ASSET_PATH": "assets",
    "SOUND_PATH": "sounds",
    "SPAWN_MARGIN": 100,
    "MAX_PORING_SPEED": 5,
    "SPEED_INCREASE_RATE": 1.05
}

# Initialize Pygame for audio
pygame.init()
pygame.mixer.init()

# Load sound effects
try:
    catch_sound = pygame.mixer.Sound(
        os.path.join(CONFIG["SOUND_PATH"], "poi3.mp3"))
    level_up_sound = pygame.mixer.Sound(
        os.path.join(CONFIG["SOUND_PATH"], "levelup.wav"))
    life_lost_sound = pygame.mixer.Sound(
        os.path.join(CONFIG["SOUND_PATH"], "Acolyte_Heal.wav"))
    background_music = pygame.mixer.Sound(
        os.path.join(CONFIG["SOUND_PATH"], "prontera.mp3"))

    background_music.set_volume(0.5)  # Set volume to 50%
except pygame.error as e:
    logger.error(f"Error loading sound files: {e}")
    exit(1)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Load Poring images


def load_poring_images():
    images = []
    for filename in os.listdir(CONFIG["ASSET_PATH"]):
        if filename.startswith("RO_Poring(SD)") and filename.endswith(".png"):
            try:
                image_path = os.path.join(CONFIG["ASSET_PATH"], filename)
                image = Image.open(image_path).convert("RGBA")
                image = image.resize((120, 120))
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
                images.append(image)
            except IOError as e:
                logger.error(f"Error loading image {filename}: {e}")
    return images


poring_images = load_poring_images()
if not poring_images:
    logger.error("Failed to load any Poring images. Exiting.")
    exit(1)

# Poring class


class Poring:
    def __init__(self, level):
        margin = CONFIG["SPAWN_MARGIN"]
        self.position = np.array([
            random.randint(margin, CONFIG["SCREEN_WIDTH"] - margin),
            random.randint(margin, CONFIG["SCREEN_HEIGHT"] - margin)
        ], dtype=np.float32)
        speed = min(1 + (level - 1) * 0.2, CONFIG["MAX_PORING_SPEED"])
        self.velocity = np.random.uniform(-speed, speed, 2)
        self.start_time = time.time()
        self.image = random.choice(poring_images)
        self.value = random.randint(1, 5)
        self.angle = random.uniform(0, 2 * np.pi)
        self.angular_speed = random.uniform(0.5, 1.5)

    def move(self):
        t = time.time() - self.start_time
        self.angle += self.angular_speed * 0.1

        # Use sine wave for smoother movement
        offset = np.array([
            np.sin(self.angle) * 5,
            np.cos(self.angle) * 5
        ])

        new_position = self.position + self.velocity + offset

        # Bounce off the edges
        if new_position[0] < 0 or new_position[0] > CONFIG["SCREEN_WIDTH"]:
            self.velocity[0] *= -1
        if new_position[1] < 0 or new_position[1] > CONFIG["SCREEN_HEIGHT"]:
            self.velocity[1] *= -1

        self.position = np.clip(new_position, [0, 0], [
                                CONFIG["SCREEN_WIDTH"], CONFIG["SCREEN_HEIGHT"]])

    def draw(self, frame):
        x, y = self.position.astype(int)
        y_offset = int(20 * np.sin(time.time() * 5))

        y_start = max(0, y - 60 + y_offset)
        y_end = min(frame.shape[0], y + 60 + y_offset)
        x_start = max(0, x - 60)
        x_end = min(frame.shape[1], x + 60)

        poring_resized = cv2.resize(
            self.image, (x_end - x_start, y_end - y_start))
        alpha_s = poring_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[y_start:y_end, x_start:x_end, c] = (alpha_s * poring_resized[:, :, c] +
                                                      alpha_l * frame[y_start:y_end, x_start:x_end, c])

# Game state


class GameState:
    def __init__(self):
        self.score = 0
        self.lives = CONFIG["INITIAL_LIVES"]
        self.level = 1
        self.game_over = False
        self.paused = False
        self.porings = [Poring(self.level)
                        for _ in range(CONFIG["INITIAL_PORING_COUNT"])]
        self.poring_lifetime = CONFIG["INITIAL_PORING_LIFETIME"]
        self.combo = 0
        self.last_catch_time = 0
        self.spawn_timer = 0
        self.spawn_interval = 1.0  # Decreased initial spawn interval
        self.min_porings = CONFIG["INITIAL_PORING_COUNT"]
        self.max_porings = CONFIG["INITIAL_PORING_COUNT"] * 2

    def reset(self):
        self.__init__()

# High score management


def load_high_scores():
    try:
        with open("high_scores.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_high_score(score):
    high_scores = load_high_scores()
    high_scores.append(
        {"score": score, "date": time.strftime("%Y-%m-%d %H:%M:%S")})
    high_scores.sort(key=lambda x: x["score"], reverse=True)
    high_scores = high_scores[:10]  # Keep only top 10
    with open("high_scores.json", "w") as f:
        json.dump(high_scores, f)

# Game logic functions


def handle_poring_catch(game_state, poring):
    game_state.score += poring.value
    catch_sound.play()
    current_time = time.time()
    if current_time - game_state.last_catch_time < 1.0:  # Combo system
        game_state.combo += 1
        game_state.score += game_state.combo
    else:
        game_state.combo = 0
    game_state.last_catch_time = current_time
    game_state.porings.remove(poring)
    if game_state.score % CONFIG["LEVEL_UP_SCORE"] == 0:
        game_state.level += 1
        level_up_sound.play()
        game_state.poring_lifetime *= CONFIG["DIFFICULTY_INCREASE_RATE"]
        game_state.spawn_interval *= 0.95  # Decrease spawn interval
        game_state.min_porings = min(game_state.min_porings + 1, 15)
        game_state.max_porings = min(game_state.max_porings + 2, 25)


def handle_poring_escape(game_state, poring):
    game_state.lives -= 1
    life_lost_sound.play()
    game_state.porings.remove(poring)


def center_text(frame, text, font, scale, thickness, y_offset=0):
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2 + y_offset
    cv2.putText(frame, text, (text_x, text_y), font,
                scale, (255, 255, 255), thickness)


def draw_game(frame, game_state):
    for poring in game_state.porings:
        poring.draw(frame)
        elapsed_time = time.time() - poring.start_time
        if elapsed_time > game_state.poring_lifetime - 2:
            if int(elapsed_time * 5) % 2 == 0:
                cv2.circle(frame, tuple(poring.position.astype(int)),
                           60, (0, 0, 255), 3)

    font_scale = CONFIG["SCREEN_HEIGHT"] / \
        480  # Scale font based on screen height
    cv2.putText(frame, f"Score: {game_state.score}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    cv2.putText(frame, f"Lives: {game_state.lives}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    cv2.putText(frame, f"Level: {game_state.level}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    cv2.putText(frame, f"Combo: {game_state.combo}", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)


def show_start_screen(frame):
    font_scale = CONFIG["SCREEN_HEIGHT"] / 480
    center_text(frame, "Poring Catching Game",
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 2, 3, -100)
    center_text(frame, "Catch Porings with both hands!",
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2, -20)
    center_text(frame, "Press SPACE to start",
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2, 40)
    center_text(frame, "Press ESC to quit",
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2, 80)


def show_game_over_screen(frame, game_state):
    font_scale = CONFIG["SCREEN_HEIGHT"] / 480
    center_text(frame, "GAME OVER", cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 2.5, 3, -100)
    center_text(frame, f"Final Score: {
                game_state.score}", cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.5, 2, -20)
    center_text(frame, "Press SPACE to restart",
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2, 40)
    center_text(frame, "Press ESC to quit",
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2, 80)


def show_pause_screen(frame):
    font_scale = CONFIG["SCREEN_HEIGHT"] / 480
    center_text(frame, "PAUSED", cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 2.5, 3, -100)
    center_text(frame, "Press P to resume",
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2, -20)
    center_text(frame, "Press R to restart",
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2, 20)
    center_text(frame, "Press ESC to quit",
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2, 60)


def get_hand_area(landmarks):
    wrist = np.array([landmarks[mp_hands.HandLandmark.WRIST].x,
                     landmarks[mp_hands.HandLandmark.WRIST].y])
    thumb_tip = np.array([landmarks[mp_hands.HandLandmark.THUMB_TIP].x,
                         landmarks[mp_hands.HandLandmark.THUMB_TIP].y])
    pinky_tip = np.array([landmarks[mp_hands.HandLandmark.PINKY_TIP].x,
                         landmarks[mp_hands.HandLandmark.PINKY_TIP].y])
    return np.array([wrist, thumb_tip, pinky_tip])


def point_in_triangle(point, triangle):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(point, triangle[0], triangle[1])
    d2 = sign(point, triangle[1], triangle[2])
    d3 = sign(point, triangle[2], triangle[0])

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

# Main game loop


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["SCREEN_WIDTH"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["SCREEN_HEIGHT"])

    background_music.play(-1)  # Play background music on loop

    game_state = GameState()
    start_screen = True
    clock = pygame.time.Clock()

    cv2.namedWindow('Poring Catching Game', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Poring Catching Game',
                     CONFIG["SCREEN_WIDTH"], CONFIG["SCREEN_HEIGHT"])

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(
            frame, (CONFIG["SCREEN_WIDTH"], CONFIG["SCREEN_HEIGHT"]))

        if start_screen:
            show_start_screen(frame)

        elif game_state.game_over:
            show_game_over_screen(frame, game_state)
        elif game_state.paused:
            show_pause_screen(frame)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_area = get_hand_area(hand_landmarks.landmark)
                    hand_area[:, 0] *= CONFIG["SCREEN_WIDTH"]
                    hand_area[:, 1] *= CONFIG["SCREEN_HEIGHT"]
                    hand_area = hand_area.astype(int)

                    cv2.polylines(frame, [hand_area], True, (0, 255, 0), 3)

                    for poring in game_state.porings.copy():
                        if point_in_triangle(poring.position, hand_area):
                            handle_poring_catch(game_state, poring)

            for poring in game_state.porings.copy():
                poring.move()
                elapsed_time = time.time() - poring.start_time
                if elapsed_time > game_state.poring_lifetime:
                    handle_poring_escape(game_state, poring)

            # Spawn new Porings based on spawn interval and current Poring count
            game_state.spawn_timer += 1 / CONFIG["FPS"]
            if game_state.spawn_timer >= game_state.spawn_interval or len(game_state.porings) < game_state.min_porings:
                if len(game_state.porings) < game_state.max_porings:
                    game_state.porings.append(Poring(game_state.level))
                    game_state.spawn_timer = 0

            if game_state.lives <= 0:
                game_state.game_over = True
                save_high_score(game_state.score)

            draw_game(frame, game_state)

        cv2.imshow('Poring Catching Game', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):
            if start_screen:
                start_screen = False
            elif game_state.game_over:
                game_state.reset()
        elif key == ord('p'):
            game_state.paused = not game_state.paused
        elif key == ord('r'):
            game_state.reset()

        # FPS control
        clock.tick(CONFIG["FPS"])

        # Performance logging
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        logger.debug(f"FPS: {fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    background_music.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
