import argparse

import torch
from PIL import Image
from torchvision import models, transforms

from train import EMOTIONS, EmotionCNN, NUM_CLASSES


def load_model(model_path, device):
    model = EmotionCNN(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_emotion(image_path, model, device, transform):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()

    return EMOTIONS[predicted_idx], confidence, probabilities


def main():
    parser = argparse.ArgumentParser(description="Predict emotion from facial image")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument(
        "--model",
        type=str,
        default="best_model.pth",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--show-all", action="store_true", help="Show probabilities for all emotions"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device)

    print(f"Processing image: {args.image_path}")
    emotion, confidence, probabilities = predict_emotion(
        args.image_path, model, device, transform
    )

    print("\n" + "=" * 50)
    print(f"Predicted Emotion: {emotion.upper()}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("=" * 50)

    if args.show_all:
        print("\nAll emotion probabilities:")
        for i, (emo, prob) in enumerate(zip(EMOTIONS, probabilities)):
            bar = "█" * int(prob * 50)
            print(f"{emo:10s}: {prob * 100:5.2f}% {bar}")


if __name__ == "__main__":
    main()
