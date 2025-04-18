import os
from PIL import Image, ImageDraw, ImageFont

# List of special characters you want to generate.
characters = ['.', ',', '(', ')', '-', "'", ':', '#', '%', '[', ']' ]
fonts = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Palatino.ttc",
    "/System/Library/Fonts/Supplemental/PTSerif.ttc",
    "/Library/Fonts/Times New Roman.ttf",
    "/Library/Fonts/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Optima.ttc",
    "/Library/Fonts/Avenir.ttc",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
    "/Library/Fonts/Palatino.ttc"
]


# Image dimensions and font size
IMG_WIDTH = 40
IMG_HEIGHT = 60
FONT_SIZE = 48

def generate_character_images():
    for char in characters:
        # Folder name: "<char>_<char>"
        if char.isupper():
            folder_name = f"U_{char}"
        elif char.islower():
            folder_name = f"L_{char}"
        else:
            folder_name = f"sym_{char}"
        print(char)
        os.makedirs(folder_name, exist_ok=True)
        
        count = 1
        for font_path in fonts:
            try:
                # Try to load the specified font
                font = ImageFont.truetype(font_path, FONT_SIZE)
            except OSError as e:
                print(f"Could not load font: {font_path}. Error: {e}")
                continue  # Skip to the next font if this one fails

            # Create a blank white image
            img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), color="white")
            draw = ImageDraw.Draw(img)

            # Compute the bounding box of the text to center it
            bbox = draw.textbbox((0, 0), char, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            # Center the text
            x = (IMG_WIDTH - w) / 2
            if char == '.' or char == ',':
                y = h / 4 #(IMG_HEIGHT - h) / 5
            else: 
                y = (IMG_HEIGHT - h) / 4
            print(w,h,x,y)
            
            # Draw the character in black
            draw.text((x, y), char, font=font, fill="black")
            
            # Save the image as "<count>.png" in the folder
            output_path = os.path.join(folder_name, f"{count}.png")
            img.save(output_path)
            
            count += 1

if __name__ == "__main__":
    generate_character_images()
    print("Character images generated successfully!")
