# utils.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import math
from tensorflow import keras


# 学習履歴を記録するコールバック
class LossHistoryCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.history = {
            "loss": [],
            "reconstruction_loss": [],
            "kl_loss": [],
            "triplet_loss": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history["loss"].append(logs.get("loss"))
        self.history["reconstruction_loss"].append(logs.get("reconstruction_loss"))
        self.history["kl_loss"].append(logs.get("kl_loss"))
        self.history["triplet_loss"].append(logs.get("triplet_loss"))


# 画像のインデックス指定
def create_arrays_from_indices(all_images, manual_triplets):
    """
    手動で作成したインデックスのリストから、学習用のNumPy配列を作成する。
    """
    # 結果を格納するリスト
    anchor_list = []
    positive_list = []
    negative_list = []

    # 手動リストをループして、インデックスに対応する画像を格納
    for anchor_idx, positive_idx, negative_idx in manual_triplets:
        anchor_list.append(all_images[anchor_idx])
        positive_list.append(all_images[positive_idx])
        negative_list.append(all_images[negative_idx])

    # リストをNumPy配列に変換
    anchors = np.array(anchor_list)
    positives = np.array(positive_list)
    negatives = np.array(negative_list)

    return anchors, positives, negatives


# 入力画像と再構成画像をプロットして保存する関数
def plot_input_and_reconstructed_images(
    model, input_imgs, save_path, num_samples=8, max_cols=8, epoch=None
):
    num_samples = min(num_samples, len(input_imgs))

    # エンコードと再構成
    _, _, z = model.encoder.predict(input_imgs[:num_samples])
    reconstructed_images = model.decoder.predict(z)

    cols = min(num_samples, max_cols)
    rows = 2 * math.ceil(num_samples / cols)

    fig = plt.figure(figsize=(cols * 2, rows * 2))

    for i in range(num_samples):
        # 入力画像
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(np.clip(input_imgs[i], 0, 1))
        ax.set_title("入力画像")
        ax.axis("off")

        # 再構成画像
        ax = fig.add_subplot(rows, cols, i + 1 + (rows // 2) * cols)
        ax.imshow(np.clip(reconstructed_images[i], 0, 1))
        ax.set_title("再構成画像")
        ax.axis("off")

    epoch_str = f" (Epoch {epoch})" if epoch is not None else " (最終)"
    plt.suptitle(f"入力画像と再構成画像{epoch_str}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    print(f"入力画像と再構成画像が {save_path} に保存されました。")


# 損失関数の推移をグラフにして保存する関数
def plot_loss_curve(history, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(history["loss"], label="Total Loss")
    plt.plot(history["reconstruction_loss"], label="Reconstruction Loss")
    plt.plot(history["kl_loss"], label="KL Loss")
    plt.plot(history["triplet_loss"], label="Triplet Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"学習曲線が '{save_path}' に保存されました。")


def plot_similar_images(
    title,
    query_img,
    image_list,
    index_list,
    save_path,
    score_list=None,
    max_items_per_row=6,
):
    """
    クエリ画像と、それに続く画像のリストをグリッド表示して保存するシンプルな汎用関数。
    """
    # --- 表示するアイテムの総数を計算 ---
    num_images_in_list = len(image_list)
    total_items = 1 + num_images_in_list  # クエリ画像(1) + リスト内の画像

    # --- レイアウトを計算 ---
    cols = min(total_items, max_items_per_row)
    rows = math.ceil(total_items / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    fig.suptitle(title, fontsize=16, y=0.98)  # 図全体のタイトル

    if total_items > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # --- 画像を順番にプロット ---
    current_pos = 0

    # 1. クエリ画像を表示
    axes_flat[current_pos].imshow(np.clip(query_img, 0, 1))
    axes_flat[current_pos].set_title("クエリ画像", fontsize=10)
    axes_flat[current_pos].axis("off")
    current_pos += 1

    # 2. リスト内の画像を順番に表示
    for i, img in enumerate(image_list):
        ax = axes_flat[current_pos + i]
        ax.imshow(np.clip(img, 0, 1))

        # スコアがあればタイトルに含める
        if score_list is not None and i < len(score_list):
            title_text = f"idx: {index_list[i]}\nスコア: {score_list[i]:.3f}"
        else:
            title_text = f"idx: {index_list[i]}"

        ax.set_title(title_text, fontsize=9)
        ax.axis("off")

    # 余ったsubplotを非表示にする
    for i in range(total_items, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"画像を {save_path} に保存しました。")


def save_image_grid(images, save_path, items_per_row=20):
    """
    画像のリストを、インデックス付きのグリッド画像として保存する。
    """
    num_images = len(images)
    rows = math.ceil(num_images / items_per_row)
    cols = items_per_row

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    axes_flat = axes.flatten() if num_images > 1 else [axes]

    for i, img in enumerate(images):
        ax = axes_flat[i]
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"idx: {i}", fontsize=9)
        ax.axis("off")

    # 余ったsubplotを非表示
    for i in range(num_images, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)  # 解像度を上げて保存
    plt.close(fig)
    print(f"画像グリッドを {save_path} に保存しました。")
