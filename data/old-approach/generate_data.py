from PIL import Image, ImageDraw, ImageFont
import os

# Configuration
FONT_SIZE = 28
OUTPUT_DIR = "data/train"
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def generate_characters():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create a temporary image to calculate font metrics
    temp_img = Image.new('L', (100, 100), 255)
    draw = ImageDraw.Draw(temp_img)

    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except:
        try:
            font = ImageFont.truetype("Helvetica.ttf", FONT_SIZE)
        except:
            font = ImageFont.load_default()
            print("Using default font (size may vary)")

    for char in CHARS:
        # Create blank image (adjust size based on font)
        img = Image.new('L', (FONT_SIZE, FONT_SIZE), color=255)
        draw = ImageDraw.Draw(img)

        # Draw character centered
        draw.text((5, -2), char, font=font, fill=0)  # Adjust position as needed

        # Save
        img.save(f"{OUTPUT_DIR}/{char}.png")
        print(f"Generated: {char}.png")


if __name__ == "__main__":
    generate_characters()