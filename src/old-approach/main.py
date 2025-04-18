import os
from os import path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.preprocessing import load_image, binarize, segment_characters
from src.ocr import OCRModel
from src.allergen import AllergenDetector


def get_available_font():
    font_sizes = [28, 24, 20]  # Try different sizes
    font_paths = [
        "arial.ttf",
        "Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "Helvetica.ttf"
    ]

    for size in font_sizes:
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
    print("Warning: Using default font (quality may vary)")
    return ImageFont.load_default()


def evaluate_model(ocr_model, test_dir="data/test_chars"):

    os.makedirs(test_dir, exist_ok=True)
    font = get_available_font()

    # Generate test characters if none exist
    if not os.listdir(test_dir):
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            img = Image.new('L', (28, 28), color=255)
            draw = ImageDraw.Draw(img)
            draw.text((5, 0), char, font=font, fill=0)
            img.save(f"{test_dir}/{char}.png")

    correct = 0
    total = 0

    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        try:
            img = np.array(Image.open(f"{test_dir}/{char}.png").convert('L')) / 255.0
            pred = ocr_model.predict(img)
            if pred == char:
                correct += 1
            total += 1
        except:
            continue

    accuracy = correct / total if total > 0 else 0
    print(f"Model Accuracy: {accuracy:.2%} ({correct}/{total} characters)")
    return accuracy


def process_image(image_path, ocr, detector):
    try:
        # 1. Load and preprocess
        img = load_image(image_path)
        binary_img = binarize(img)

        # 2. Character segmentation
        characters = segment_characters(binary_img)
        if not characters:
            print("Error: No characters detected")
            return

        # 3. OCR
        text = ""
        for char_img in characters:
            # Convert to right format for our model
            char_img_processed = np.array(Image.fromarray(char_img).resize((28, 28))) / 255.0
            text += ocr.predict(char_img_processed)

        print("\nExtracted Text:\n", text)

        # 4. Allergen detection
        allergens = detector.detect(text)
        if allergens:
            print("\n⚠️ Allergens detected:", ", ".join(allergens))
        else:
            print("\n✅ No common allergens detected")

    except Exception as e:
        print(f"Error processing image: {str(e)}")


def main():
    print("=== Food Allergen Detection System ===")

    # Initialize components
    detector = AllergenDetector()

    # Load or train OCR model
    model_path = "models/ocr_model.pkl"
    if path.exists(model_path):
        print("\nLoading pre-trained model...")
        ocr = OCRModel.load(model_path)
    else:
        print("\nTraining new OCR model (this may take a minute)...")
        ocr = OCRModel()
        ocr.train("data/train")
        ocr.save(model_path)
        print("Model trained and saved")

    # Evaluate model
    evaluate_model(ocr)

    # Process test images
    test_images = ["data/test/label1.jpg", "data/test/label2.jpg"]

    for img_path in test_images:
        if path.exists(img_path):
            print(f"\nProcessing {img_path}...")
            process_image(img_path, ocr, detector)
        else:
            print(f"\nTest image not found: {img_path}")
            print("Please add some food label images to data/test/")


if __name__ == "__main__":
    main()