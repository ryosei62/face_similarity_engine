import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# load_images_from_list 関数は変更なし (省略)
def load_images_from_list(file_list, image_dir, img_size, desc):
    """ファイルリストから画像を読み込むヘルパー関数"""
    images_list = []
    pil_img_size = (img_size[1], img_size[0])

    for file_name in tqdm(file_list, desc=desc):
        img_path = os.path.join(image_dir, file_name)
        try:
            image = Image.open(img_path).convert("RGB")
            image = image.resize(pil_img_size, Image.Resampling.LANCZOS)
            img_array = np.array(image)
            img_array = img_array.astype(np.float32) / 255.0
            images_list.append(img_array)
        except Exception as e:
            print(f"Error loading or processing image {img_path}: {e}. Skipping.")
            continue

    if not images_list:
        return np.array([], dtype=np.float32)  # 型を明示

    return np.stack(images_list, axis=0)


def prepare_dataset(
    image_dir,
    num_total_samples,  # ★ 引数を「使用する総枚数」に変更
    img_size=(64, 64),
    test_size=0.2,
    random_state=42,
):
    """
    データセットを準備する。
    指定された総枚数を、test_sizeの比率で訓練とテストに分け、
    固定化されたリストからスライスして読み込む。
    """

    # 1. 全画像ファイルのリストを取得
    all_files = [
        f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")
    ]
    all_files.sort()

    # 2. ファイルリストを「訓練候補」と「テスト候補」に分割
    # (この分割は random_state で固定される)
    train_files_full, test_files_full = train_test_split(
        all_files, test_size=test_size, random_state=random_state, shuffle=True
    )

    # 3. ★ ユーザーのアイデアを実装 ★
    # num_total_samples を test_size に基づいて分割
    if num_total_samples is None or num_total_samples <= 0:
        # Noneなら候補リスト全体を使う
        train_files_to_load = train_files_full
        test_files_to_load = test_files_full
    else:
        # 比率を計算してスライスする枚数を決める
        test_count = int(num_total_samples * test_size)
        train_count = num_total_samples - test_count

        # ★ それぞれの候補リストからスライス
        # (リストの長さが足りない場合も考慮)
        train_files_to_load = train_files_full[:train_count]
        test_files_to_load = test_files_full[:test_count]

    print(f"Total files found: {len(all_files)}")
    print(f"Full train set (files): {len(train_files_full)}")
    print(f"Full test set (files): {len(test_files_full)}")
    print("-" * 30)
    print(f"Target total samples: {num_total_samples}")
    print(f"Loading {len(train_files_to_load)} images for training...")
    print(f"Loading {len(test_files_to_load)} images for testing...")

    # 4. 画像を読み込む
    train_images = load_images_from_list(
        train_files_to_load, image_dir, img_size, "Loading train images"
    )

    test_images = load_images_from_list(
        test_files_to_load, image_dir, img_size, "Loading test images"
    )

    print(f"\n--- Results ---")
    print(f"Training set shape: {train_images.shape}")
    print(f"Test set shape: {test_images.shape}")

    return train_images, test_images
