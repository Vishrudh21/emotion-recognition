from pathlib import Path

import matplotlib.pyplot as plt

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DATA_DIR = Path("data")


def count_images(data_dir):
    stats = {}
    for emotion in EMOTIONS:
        emotion_dir = data_dir / emotion
        if emotion_dir.exists():
            count = len(list(emotion_dir.glob("*.jpg")))
            stats[emotion] = count
        else:
            stats[emotion] = 0
    return stats


def plot_distribution(train_stats, test_stats):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.bar(train_stats.keys(), train_stats.values(), color="steelblue")
    ax1.set_title("Training Set Distribution", fontsize=14)
    ax1.set_xlabel("Emotion")
    ax1.set_ylabel("Number of Images")
    ax1.tick_params(axis="x", rotation=45)
    for i, (emotion, count) in enumerate(train_stats.items()):
        ax1.text(i, count, str(count), ha="center", va="bottom")

    ax2.bar(test_stats.keys(), test_stats.values(), color="coral")
    ax2.set_title("Test Set Distribution", fontsize=14)
    ax2.set_xlabel("Emotion")
    ax2.set_ylabel("Number of Images")
    ax2.tick_params(axis="x", rotation=45)
    for i, (emotion, count) in enumerate(test_stats.items()):
        ax2.text(i, count, str(count), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("dataset_distribution.png", dpi=150)
    print("Distribution plot saved to 'dataset_distribution.png'")
    plt.show()


def main():
    print("=" * 60)
    print("FACIAL EXPRESSION DATASET STATISTICS")
    print("=" * 60)

    train_dir = DATA_DIR / "train"
    train_stats = count_images(train_dir)

    print("\nTRAINING SET:")
    print("-" * 60)
    total_train = 0
    for emotion, count in train_stats.items():
        print(f"{emotion:10s}: {count:5d} images")
        total_train += count
    print("-" * 60)
    print(f"{'TOTAL':10s}: {total_train:5d} images\n")

    test_dir = DATA_DIR / "test"
    test_stats = count_images(test_dir)

    print("TEST SET:")
    print("-" * 60)
    total_test = 0
    for emotion, count in test_stats.items():
        print(f"{emotion:10s}: {count:5d} images")
        total_test += count
    print("-" * 60)
    print(f"{'TOTAL':10s}: {total_test:5d} images\n")

    print("OVERALL STATISTICS:")
    print("-" * 60)
    print(f"Total images:     {total_train + total_test:5d}")
    print(
        f"Training images:  {total_train:5d} ({100 * total_train / (total_train + total_test):.1f}%)"
    )
    print(
        f"Test images:      {total_test:5d} ({100 * total_test / (total_train + total_test):.1f}%)"
    )
    print(f"Number of classes: {len(EMOTIONS)}")
    print("=" * 60)

    plot_distribution(train_stats, test_stats)


if __name__ == "__main__":
    main()
