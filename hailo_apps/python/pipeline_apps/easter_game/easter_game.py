"""Easter Eggs & Afikoman Game — catch items with your hands for points."""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import cv2
import hailo
import math
import multiprocessing
import numpy as np
import random
import time

from hailo_apps.python.core.common.buffer_utils import get_caps_from_pad
from hailo_apps.python.core.common.core import get_resource_path
from hailo_apps.python.core.common.defines import RESOURCES_PHOTOS_DIR_NAME
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.common.parser import get_pipeline_parser
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.python.pipeline_apps.pose_estimation.pose_estimation_pipeline import (
    GStreamerPoseEstimationApp,
)

logger = get_logger(__name__)

# --- Constants ---
GAME_DURATION = 90          # seconds
ITEM_TIMEOUT = 3.0          # seconds before item despawns
CATCH_RADIUS = 55           # pixels — how close a wrist must be to catch
EGG_PROBABILITY = 0.70      # 70 % eggs, 30 % afikoman
EGG_POINTS = 20
AFIKOMAN_POINTS = 10
POPUP_DURATION = 0.8        # seconds the "+N" text floats
RESTART_DELAY = 5           # seconds to show final scores before restart

MAX_FRAME_BYTES = 1920 * 1080 * 3

# Only drop complete frames outside the cropper/aggregator branches.
LEAKY_QUEUES = (
    "source_scale_q",
    "source_convert_q",
    "inference_wrapper_output_q",
    "hailo_tracker_q",
    "identity_callback_q",
)

# Wrist keypoint indices (COCO 17-keypoint model)
LEFT_WRIST = 9
RIGHT_WRIST = 10

# Player name pool
PLAYER_NAMES = [
    "Red Fox", "Blue Jay", "Gold Cat", "Green Owl", "Pink Bear",
    "Teal Bat", "Plum Elk", "Jade Ram", "Ruby Ant", "Mint Bee",
]

# --- Egg colour palette (RGB) ---
EGG_COLORS = [
    (255, 80, 80),     # red
    (80, 255, 80),     # green
    (80, 160, 255),    # blue
    (255, 200, 60),    # yellow
    (200, 100, 255),   # purple
    (255, 140, 200),   # pink
    (60, 220, 220),    # cyan
    (255, 160, 60),    # orange
]

# Stripe accent colours per egg (lighter tint)
EGG_STRIPES = [
    (255, 180, 180),
    (180, 255, 180),
    (180, 210, 255),
    (255, 235, 150),
    (230, 180, 255),
    (255, 200, 230),
    (170, 240, 240),
    (255, 210, 150),
]


# ─── Game item ──────────────────────────────────────────────────────────────
class GameItem:
    """One collectible item on screen."""

    def __init__(self, kind, x, y, color_idx=0):
        self.kind = kind          # "egg" or "afikoman"
        self.x = x
        self.y = y
        self.spawn_time = time.time()
        self.color_idx = color_idx  # for eggs

    @property
    def points(self):
        return EGG_POINTS if self.kind == "egg" else AFIKOMAN_POINTS

    def expired(self):
        return (time.time() - self.spawn_time) >= ITEM_TIMEOUT


# ─── Popup text ─────────────────────────────────────────────────────────────
class Popup:
    def __init__(self, text, x, y, colour):
        self.text = text
        self.x = x
        self.y = y
        self.colour = colour
        self.spawn = time.time()

    def alive(self):
        return (time.time() - self.spawn) < POPUP_DURATION

    def alpha(self):
        age = time.time() - self.spawn
        return max(0.0, 1.0 - age / POPUP_DURATION)

    def current_y(self):
        """Float upward over time."""
        age = time.time() - self.spawn
        return int(self.y - age * 60)


# ─── Drawing helpers ────────────────────────────────────────────────────────
def draw_easter_egg(img, cx, cy, color_idx):
    """Draw a colourful decorated Easter egg (oval)."""
    color = EGG_COLORS[color_idx % len(EGG_COLORS)]
    stripe = EGG_STRIPES[color_idx % len(EGG_STRIPES)]
    a, b = 30, 40  # semi-axes (width, height)

    # Main egg body
    cv2.ellipse(img, (cx, cy), (a, b), 0, 0, 360, color, -1, cv2.LINE_AA)
    # Dark outline
    cv2.ellipse(img, (cx, cy), (a, b), 0, 0, 360, (40, 40, 40), 2, cv2.LINE_AA)
    # Decorative stripes
    for dy in (-14, 0, 14):
        cv2.ellipse(img, (cx, cy + dy), (a - 4, 5), 0, 0, 360, stripe, -1, cv2.LINE_AA)
    # Shine highlight
    cv2.ellipse(img, (cx - 10, cy - 16), (6, 8), -30, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)


def draw_afikoman(img, cx, cy):
    """Draw a golden matzah rectangle with texture lines."""
    w, h = 50, 30
    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = cx + w // 2, cy + h // 2

    # Golden fill
    cv2.rectangle(img, (x1, y1), (x2, y2), (210, 180, 100), -1, cv2.LINE_AA)
    # Darker border
    cv2.rectangle(img, (x1, y1), (x2, y2), (140, 110, 50), 2, cv2.LINE_AA)
    # Texture dots (matzah perforations)
    for row in range(3):
        for col in range(5):
            px = x1 + 8 + col * 9
            py = y1 + 7 + row * 9
            if x1 < px < x2 and y1 < py < y2:
                cv2.circle(img, (px, py), 2, (160, 130, 60), -1)
    # Light sheen
    cv2.line(img, (x1 + 4, y1 + 4), (x2 - 4, y1 + 4), (240, 220, 160), 1, cv2.LINE_AA)


# ─── Callback class ─────────────────────────────────────────────────────────
class EasterGameCallback(app_callback_class):
    """Holds all game state across frames."""

    def __init__(self, background_path):
        super().__init__()
        self.use_frame = True
        self.frame_shm = multiprocessing.RawArray("B", MAX_FRAME_BYTES)
        self.frame_dims = multiprocessing.RawArray("i", 2)
        self.frame_lock = multiprocessing.Lock()
        self.frame_ready = multiprocessing.Event()
        self.display_stop = multiprocessing.Event()

        # Background
        raw = cv2.imread(background_path)
        if raw is None:
            logger.warning("Could not load background %s — using dark fallback", background_path)
            self.background_orig = None
        else:
            # Store as RGB to match GStreamer frame colour space
            self.background_orig = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        self.background = None  # resized per-frame if needed

        # Player state  {track_id: {"name": str, "score": int}}
        self.players = {}
        self._name_idx = 0

        # Current item
        self.current_item = None

        # Popups
        self.popups = []

        # Timer
        self.game_start = None
        self.game_over = False
        self.game_over_time = None

        # Frame dimensions cache
        self.fw = 0
        self.fh = 0

    # --- helpers ---
    def set_frame(self, frame):
        """Publish the latest BGR frame without serializing it."""
        h, w = frame.shape[:2]
        if frame.nbytes > MAX_FRAME_BYTES:
            logger.warning("Frame %dx%d exceeds shared buffer; dropping", w, h)
            return
        with self.frame_lock:
            shared = np.frombuffer(self.frame_shm, dtype=np.uint8)
            shared[:frame.size] = frame.reshape(-1)
            self.frame_dims[:] = h, w
            self.frame_ready.set()

    def _get_bg(self, w, h):
        """Return resized background (RGB) or black frame."""
        if self.fw != w or self.fh != h or self.background is None:
            self.fw, self.fh = w, h
            if self.background_orig is not None:
                self.background = cv2.resize(self.background_orig, (w, h))
            else:
                self.background = np.zeros((h, w, 3), dtype=np.uint8)
        return self.background.copy()

    def _get_or_create_player(self, track_id):
        if track_id not in self.players:
            name = PLAYER_NAMES[self._name_idx % len(PLAYER_NAMES)]
            self._name_idx += 1
            self.players[track_id] = {"name": name, "score": 0}
        return self.players[track_id]

    def spawn_item(self):
        """Spawn a new random item in the playable area."""
        margin = 80
        lb_width = 220  # keep items out of leaderboard area on right
        x = random.randint(margin, max(margin + 1, self.fw - lb_width - margin))
        y = random.randint(margin + 60, max(margin + 61, self.fh - margin))
        if random.random() < EGG_PROBABILITY:
            kind = "egg"
        else:
            kind = "afikoman"
        color_idx = random.randint(0, len(EGG_COLORS) - 1)
        self.current_item = GameItem(kind, x, y, color_idx)

    def restart(self):
        """Reset game state for a new round."""
        self.players = {}
        self._name_idx = 0
        self.current_item = None
        self.popups = []
        self.game_start = None
        self.game_over = False
        self.game_over_time = None


def display_game_frame(user_data):
    """Display consistent snapshots of the latest shared frame."""
    shared = np.frombuffer(user_data.frame_shm, dtype=np.uint8)
    try:
        while not user_data.display_stop.is_set():
            if user_data.frame_ready.wait(timeout=0.02):
                with user_data.frame_lock:
                    h, w = user_data.frame_dims
                    frame = shared[:h * w * 3].reshape(h, w, 3).copy()
                    user_data.frame_ready.clear()
                cv2.imshow("Easter Hunt", frame)
            cv2.waitKey(1)
    finally:
        cv2.destroyAllWindows()


# ─── Callback function ──────────────────────────────────────────────────────
def app_callback(element, buffer, user_data):
    pad = element.get_static_pad("src")
    _, width, height = get_caps_from_pad(pad)
    if not (user_data.use_frame and width and height):
        return Gst.FlowReturn.OK

    now = time.time()

    # Lazy init game clock
    if user_data.game_start is None:
        user_data.game_start = now
        user_data.fw = width
        user_data.fh = height

    elapsed = now - user_data.game_start
    remaining = max(0.0, GAME_DURATION - elapsed)

    # --- Auto-restart after game over display ---
    if user_data.game_over:
        if user_data.game_over_time and (now - user_data.game_over_time) >= RESTART_DELAY:
            user_data.restart()
            return Gst.FlowReturn.OK
        # Draw game over screen
        output = user_data._get_bg(width, height)
        _draw_game_over(output, user_data)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        user_data.set_frame(output)
        return Gst.FlowReturn.OK

    # --- Check if time's up ---
    if remaining <= 0:
        user_data.game_over = True
        user_data.game_over_time = now
        user_data.current_item = None
        return Gst.FlowReturn.OK

    # --- Spawn item if needed ---
    if user_data.current_item is None:
        user_data.fw = width
        user_data.fh = height
        user_data.spawn_item()
    elif user_data.current_item.expired():
        user_data.spawn_item()

    # --- Collect all wrist positions per player ---
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for detection in detections:
        if detection.get_label() != "person":
            continue

        bbox = detection.get_bbox()

        # Track ID
        track_id = 0
        track_objs = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track_objs) == 1:
            track_id = track_objs[0].get_id()

        player = user_data._get_or_create_player(track_id)

        # Extract wrists
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue
        points = landmarks[0].get_points()
        if len(points) <= RIGHT_WRIST:
            continue

        for idx in (LEFT_WRIST, RIGHT_WRIST):
            pt = points[idx]
            px = int((pt.x() * bbox.width() + bbox.xmin()) * width)
            py = int((pt.y() * bbox.height() + bbox.ymin()) * height)

            # Check catch
            if user_data.current_item is not None:
                item = user_data.current_item
                dist = math.hypot(px - item.x, py - item.y)
                if dist < CATCH_RADIUS:
                    # Catch!
                    pts = item.points
                    player["score"] += pts
                    popup_text = f"+{pts}"
                    popup_colour = (255, 220, 60) if item.kind == "egg" else (210, 180, 100)
                    user_data.popups.append(Popup(popup_text, item.x, item.y, popup_colour))
                    logger.info(
                        "%s caught %s (+%d) — total: %d",
                        player["name"], item.kind, pts, player["score"],
                    )
                    # Spawn next
                    user_data.spawn_item()

    # --- Render ---
    output = user_data._get_bg(width, height)

    # Draw current item
    item = user_data.current_item
    if item is not None:
        if item.kind == "egg":
            draw_easter_egg(output, item.x, item.y, item.color_idx)
        else:
            draw_afikoman(output, item.x, item.y)

        # Timeout progress ring
        age = now - item.spawn_time
        frac = age / ITEM_TIMEOUT
        angle = int(360 * frac)
        ring_color = (200, 200, 200) if frac < 0.66 else (255, 80, 80)
        cv2.ellipse(output, (item.x, item.y - 52), (16, 16), -90, 0, angle, ring_color, 2, cv2.LINE_AA)

    # Draw wrist markers from detections (re-extract for rendering)
    for detection in detections:
        if detection.get_label() != "person":
            continue
        bbox = detection.get_bbox()
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if not landmarks:
            continue
        pts_list = landmarks[0].get_points()
        if len(pts_list) <= RIGHT_WRIST:
            continue
        for idx in (LEFT_WRIST, RIGHT_WRIST):
            pt = pts_list[idx]
            px = int((pt.x() * bbox.width() + bbox.xmin()) * width)
            py = int((pt.y() * bbox.height() + bbox.ymin()) * height)
            # Glowing hand circle
            cv2.circle(output, (px, py), CATCH_RADIUS, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(output, (px, py), 12, (100, 255, 100), -1, cv2.LINE_AA)
            cv2.circle(output, (px, py), 12, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw popups
    user_data.popups = [p for p in user_data.popups if p.alive()]
    for p in user_data.popups:
        alpha = p.alpha()
        cy = p.current_y()
        color = tuple(int(c * alpha) for c in p.colour)
        cv2.putText(output, p.text, (p.x - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

    # Timer bar at top
    _draw_timer(output, remaining, width)

    # Leaderboard on right
    _draw_leaderboard(output, user_data.players, width, height)

    # Convert RGB → BGR for set_frame
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    user_data.set_frame(output)

    return Gst.FlowReturn.OK


# ─── HUD drawing ─────────────────────────────────────────────────────────────
def _draw_timer(img, remaining, width):
    """Draw countdown timer bar at the top."""
    bar_h = 44
    # Semi-transparent dark bar
    overlay = img[:bar_h, :, :].copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_h), (30, 30, 30), -1)
    img[:bar_h, :, :] = cv2.addWeighted(overlay, 0.7, img[:bar_h, :, :], 0.3, 0)

    # Progress bar
    frac = remaining / GAME_DURATION
    bar_w = int((width - 260) * frac)
    bar_color = (80, 220, 80) if remaining > 20 else (255, 80, 80)
    cv2.rectangle(img, (130, 10), (130 + bar_w, 34), bar_color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (130, 10), (width - 130, 34), (180, 180, 180), 1, cv2.LINE_AA)

    # Time text
    mins = int(remaining) // 60
    secs = int(remaining) % 60
    txt = f"{mins}:{secs:02d}"
    cv2.putText(img, txt, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Label
    cv2.putText(img, "Easter Hunt!", (width // 2 - 80, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 60), 2, cv2.LINE_AA)


def _draw_leaderboard(img, players, width, height):
    """Draw leaderboard panel on the right side."""
    panel_w = 210
    x0 = width - panel_w
    # Semi-transparent dark panel
    overlay = img[:, x0:, :].copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, height), (20, 20, 40), -1)
    img[:, x0:, :] = cv2.addWeighted(overlay, 0.75, img[:, x0:, :], 0.25, 0)

    # Title
    cv2.putText(img, "LEADERBOARD", (x0 + 18, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 60), 2, cv2.LINE_AA)
    cv2.line(img, (x0 + 10, 90), (width - 10, 90), (255, 220, 60), 1, cv2.LINE_AA)

    # Sort players by score descending
    sorted_players = sorted(players.values(), key=lambda p: p["score"], reverse=True)

    for i, p in enumerate(sorted_players[:8]):
        y = 120 + i * 40
        # Rank medal
        if i == 0:
            medal_color = (255, 215, 0)
        elif i == 1:
            medal_color = (192, 192, 192)
        elif i == 2:
            medal_color = (205, 127, 50)
        else:
            medal_color = (150, 150, 150)
        cv2.circle(img, (x0 + 20, y - 5), 10, medal_color, -1, cv2.LINE_AA)
        cv2.putText(img, str(i + 1), (x0 + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)

        # Name + score
        cv2.putText(img, p["name"], (x0 + 38, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(p["score"]), (x0 + 155, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 2, cv2.LINE_AA)

    if not players:
        cv2.putText(img, "Waiting for", (x0 + 25, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(img, "players...", (x0 + 35, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def _draw_game_over(img, user_data):
    """Draw game-over screen with final scores."""
    h, w = img.shape[:2]

    # Dark overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

    # Title
    cv2.putText(img, "GAME OVER!", (w // 2 - 160, h // 4),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 220, 60), 4, cv2.LINE_AA)

    # Final scores
    sorted_players = sorted(user_data.players.values(), key=lambda p: p["score"], reverse=True)
    for i, p in enumerate(sorted_players[:6]):
        y = h // 4 + 70 + i * 50
        rank = f"#{i + 1}"
        text = f"{rank}  {p['name']:.<16s} {p['score']:>4d} pts"
        color = (255, 215, 0) if i == 0 else (220, 220, 220)
        cv2.putText(img, text, (w // 2 - 180, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    if not user_data.players:
        cv2.putText(img, "No players joined!", (w // 2 - 140, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)

    # Restart countdown
    if user_data.game_over_time:
        left = max(0, RESTART_DELAY - (time.time() - user_data.game_over_time))
        cv2.putText(img, f"Restarting in {int(left) + 1}s ...",
                    (w // 2 - 140, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2, cv2.LINE_AA)


# ─── App class ───────────────────────────────────────────────────────────────
class EasterEggsGame(GStreamerPoseEstimationApp):
    """Pose-estimation pipeline with Easter game overlay."""

    def __init__(self, app_callback, user_data, parser=None):
        super().__init__(app_callback, user_data, parser)
        self.options_menu.use_frame = False
        user_data.use_frame = True
        self.pipeline_latency = 0
        for name in LEAKY_QUEUES:
            queue = self.pipeline.get_by_name(name)
            if queue is not None:
                queue.set_property("leaky", 2)
                queue.set_property("max-size-buffers", 1)
        appsrc = self.pipeline.get_by_name("app_source")
        if appsrc is not None:
            appsrc.set_property("max-buffers", 1)

    def run(self):
        self.user_data.display_stop.clear()
        display_process = multiprocessing.Process(
            target=display_game_frame, args=(self.user_data,), daemon=True
        )
        display_process.start()
        try:
            super().run()
        finally:
            self.user_data.display_stop.set()
            display_process.join(timeout=3)
            if display_process.is_alive():
                display_process.terminate()
                display_process.join()


def main():
    parser = get_pipeline_parser()
    parser.add_argument(
        "--background",
        type=str,
        default=None,
        help="Path to background image (PNG/JPG).",
    )

    args, _ = parser.parse_known_args()

    # Resolve background: CLI override > downloaded resource > empty (dark fallback)
    bg_path = args.background
    if not bg_path:
        resource = get_resource_path(None, RESOURCES_PHOTOS_DIR_NAME, model="room.png")
        if resource and resource.exists():
            bg_path = str(resource)
        else:
            bg_path = ""
            logger.warning(
                "Default background 'room.png' not found in resources. "
                "Run 'hailo-download-resources' or pass --background."
            )

    user_data = EasterGameCallback(bg_path)
    app = EasterEggsGame(app_callback, user_data, parser)
    app.run()


if __name__ == "__main__":
    main()
