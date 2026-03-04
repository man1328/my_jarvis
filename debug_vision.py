from PIL import Image
from torchvision import transforms


def debug_view(img_path):
    # This matches the ResNet preprocessing exactly
    img = Image.open(img_path).convert('RGB')

    # 1. Resize it to the tiny size the AI uses
    img_tiny = img.resize((32, 32), resample=Image.Resampling.LANCZOS)

    # 2. Save it so YOU can look at it
    img_tiny.save("what_the_ai_sees.png")
    print("✅ Saved 'what_the_ai_sees.png'. Open this file!")


if __name__ == "__main__":
    debug_view("my_test_images/my_pet_photo11.jpeg")