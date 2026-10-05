#!/usr/bin/env python3
"""Generate simple emoji-style PNG sprites with alpha channels for testing
the hailooverlay_community sprite system.

Usage:
    python generate_sprites.py [output_dir]

Default output_dir: /usr/local/hailo/resources/sprites/
"""

import sys
import os
import numpy as np

try:
    import cv2
except ImportError:
    print("Error: opencv-python required. Install with: pip install opencv-python")
    sys.exit(1)

def make_circle_sprite(size, fill_bgr, outline_bgr=(0, 0, 0), outline_w=2):
    """Create a filled circle with outline on transparent background."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = size // 2 - outline_w
    # Fill
    cv2.circle(img, center, radius, (*fill_bgr, 255), -1, cv2.LINE_AA)
    # Outline
    cv2.circle(img, center, radius, (*outline_bgr, 255), outline_w, cv2.LINE_AA)
    return img

def make_star_sprite(size, fill_bgr, n_points=5):
    """Create a star shape on transparent background."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    outer_r = size // 2 - 2
    inner_r = outer_r * 0.4
    pts = []
    for i in range(n_points * 2):
        angle = np.pi * i / n_points - np.pi / 2
        r = outer_r if i % 2 == 0 else inner_r
        pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [pts], (*fill_bgr, 255), cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (0, 0, 0, 255), 1, cv2.LINE_AA)
    return img

def make_emoji_face(size, bg_bgr, expression="smile"):
    """Create a simple emoji face on transparent background."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r = size // 2 - 2
    # Face circle
    cv2.circle(img, (cx, cy), r, (*bg_bgr, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, (0, 0, 0, 255), 2, cv2.LINE_AA)
    # Eyes
    eye_y = cy - r // 4
    eye_lx = cx - r // 3
    eye_rx = cx + r // 3
    eye_r = max(2, r // 8)
    cv2.circle(img, (eye_lx, eye_y), eye_r, (0, 0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (eye_rx, eye_y), eye_r, (0, 0, 0, 255), -1, cv2.LINE_AA)
    # Mouth
    mouth_y = cy + r // 4
    if expression == "smile":
        cv2.ellipse(img, (cx, mouth_y), (r // 3, r // 5), 0, 0, 180,
                    (0, 0, 0, 255), 2, cv2.LINE_AA)
    elif expression == "open":
        cv2.circle(img, (cx, mouth_y + r // 8), r // 5, (0, 0, 0, 255), -1, cv2.LINE_AA)
    elif expression == "sad":
        cv2.ellipse(img, (cx, mouth_y + r // 4), (r // 3, r // 5), 0, 180, 360,
                    (0, 0, 0, 255), 2, cv2.LINE_AA)
    return img

def make_diamond_sprite(size, fill_bgr):
    """Create a diamond shape on transparent background."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    h = size // 2 - 2
    pts = np.array([[cx, cy - h], [cx + h, cy], [cx, cy + h], [cx - h, cy]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (*fill_bgr, 255), cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (0, 0, 0, 255), 1, cv2.LINE_AA)
    return img

def make_heart_sprite(size, fill_bgr):
    """Create a heart shape on transparent background."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    # Approximate heart with two circles + triangle
    r = size // 4
    cv2.circle(img, (cx - r + 1, cy - r // 2), r, (*fill_bgr, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (cx + r - 1, cy - r // 2), r, (*fill_bgr, 255), -1, cv2.LINE_AA)
    pts = np.array([[cx - size // 2 + 4, cy - r // 4],
                    [cx + size // 2 - 4, cy - r // 4],
                    [cx, cy + size // 2 - 4]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (*fill_bgr, 255), cv2.LINE_AA)
    return img


SPRITES = {
    # Keypoint sprites (for eye/joint replacement)
    "star_eye":     lambda sz: make_star_sprite(sz, (0, 255, 255), 5),      # yellow star
    "heart_eye":    lambda sz: make_heart_sprite(sz, (0, 0, 255)),           # red heart
    "fire_eye":     lambda sz: make_diamond_sprite(sz, (0, 100, 255)),       # orange diamond
    "green_dot":    lambda sz: make_circle_sprite(sz, (0, 255, 0)),          # green circle
    "blue_dot":     lambda sz: make_circle_sprite(sz, (255, 128, 0)),        # blue circle
    # Detection/bbox sprites
    "cat_icon":     lambda sz: make_emoji_face(sz, (0, 200, 255), "smile"),  # yellow smiley
    "dog_icon":     lambda sz: make_emoji_face(sz, (200, 200, 0), "open"),   # cyan face
    "thumbs_up":    lambda sz: make_star_sprite(sz, (0, 200, 0), 6),         # green 6-star
    "open_hand":    lambda sz: make_circle_sprite(sz, (255, 200, 0)),        # cyan circle
    "warning":      lambda sz: make_diamond_sprite(sz, (0, 0, 255)),         # red diamond
}


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/hailo/resources/sprites"
    os.makedirs(output_dir, exist_ok=True)

    size = 128  # Generate at 128x128, will be cached at various sizes by SpriteCache
    for name, gen_fn in SPRITES.items():
        path = os.path.join(output_dir, f"{name}.png")
        img = gen_fn(size)
        cv2.imwrite(path, img)
        print(f"  Created {path} ({img.shape[1]}x{img.shape[0]}, 4-channel)")

    print(f"\nGenerated {len(SPRITES)} sprites in {output_dir}")

    # Also update sprites.yaml to point to generated files
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sprites.yaml")
    lines = [
        "# hailooverlay_community sprite configuration",
        "# Auto-generated by generate_sprites.py",
        "",
        "sprites:",
    ]
    for name in SPRITES:
        lines.append(f"  {name}: {output_dir}/{name}.png")
    lines.append("")

    with open(yaml_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Updated {yaml_path}")


if __name__ == "__main__":
    main()
