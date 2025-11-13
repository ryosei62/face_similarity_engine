# 64vae.py
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import ops

# 作成したモジュールをインポート
from dataset import prepare_dataset
from model import build_encoder, build_decoder, VAE
from utils import (
    LossHistoryCallback,
    plot_input_and_reconstructed_images,
    plot_loss_curve,
    plot_similar_images,
    save_image_grid,
    create_arrays_from_indices,
)
from search import (
    find_top_k_similar,
    compare_with_user_judgment,
)
from evaluation import run_evaluation
from evaluation_data import EVALUATION_SETS
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist


def main():
    # --- パラメータ設定 ---
    IMAGE_DIR = "/home/ugrad/23/s2312935/research/archive/images"
    NUM_IMAGES = 1000
    IMG_WIDTH = 64
    IMG_HEIGHT = 64
    CHANNELS = 3
    LATENT_DIM = 64
    EPOCHS = 3
    BATCH_SIZE = 128  # fitメソッドのバッチサイズ

    os.makedirs("grahu", exist_ok=True)
    os.makedirs("selection", exist_ok=True)
    os.makedirs("image_grid", exist_ok=True)

    # --- 1. データセットの準備 ---
    train_images, test_images = prepare_dataset(
        image_dir=IMAGE_DIR,
        num_total_samples=NUM_IMAGES,
        img_size=(IMG_HEIGHT, IMG_WIDTH),
    )

    # --- 2. モデルの構築 ---
    print("\n--- Building VAE model ---")
    encoder = build_encoder(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), latent_dim=LATENT_DIM
    )
    decoder = build_decoder(latent_dim=LATENT_DIM, output_channels=CHANNELS)

    # エンコーダのサマリを表示
    # print("Encoder Summary:")
    # encoder.summary()
    # デコーダのサマリを表示
    # print("\nDecoder Summary:")
    # decoder.summary()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())

    # --- 3. VAEのトレーニング ---
    print("\n--- Training VAE ---")
    all_images = np.concatenate([train_images, test_images], axis=0)
    loss_history_callback = LossHistoryCallback()

    manual_triplets = [
        (330, 390, 438),
        (390, 443, 2),
        (11, 51, 4),
        (5, 11, 6),
        (597, 656, 613),
        (165, 186, 148),
        (138, 159, 139),
        (82, 82, 80),
        (81, 82, 42),
        (60, 61, 65),
        (158, 169, 194),
        (239, 338, 334),
        (139, 310, 276),
        (263, 264, 221),
        (260, 228, 200),
        (386, 387, 367),
        (351, 330, 352),
        (396, 399, 355),
        (394, 371, 358),
        (497, 516, 150),
        (503, 540, 500),
        (630, 629, 609),
        (663, 664, 643),
        (662, 702, 704),
        (703, 700, 660),
        (649, 738, 717),
        (779, 810, 790),
        (893, 878, 849),
        (864, 869, 845),
        (875, 877, 873),
        (594, 597, 595),
        (579, 597, 598),
        (530, 549, 547),
        (655, 656, 657),
        (830, 856, 855),
        (650, 663, 662),
        (524, 583, 584),
        (508, 509, 510),
        (164, 168, 167),
        (10, 32, 16),
        (369, 370, 371),
        (629, 546, 626),
        # ... さらに追加 ...
    ]
    print(f"{len(manual_triplets)}組のTripletを定義しました。")
    print(f"{len(EVALUATION_SETS)}件の評価クエリをインポートしました。")

    anchors, positives, negatives = create_arrays_from_indices(
        all_images, manual_triplets
    )

    # 1. Triplet用データセットを作成
    triplet_dataset = tf.data.Dataset.from_tensor_slices(
        (anchors, positives, negatives)
    )
    # Tripletが少ないので、何度も繰り返し使えるようにする
    triplet_dataset = triplet_dataset.repeat()
    triplet_dataset = triplet_dataset.batch(BATCH_SIZE)

    # 2. VAE（再構成）用データセットを作成
    vae_dataset = tf.data.Dataset.from_tensor_slices(all_images)
    # こちらはシャッフルして、同様に繰り返す
    vae_dataset = vae_dataset.shuffle(len(all_images)).repeat().batch(BATCH_SIZE)

    # 3. 2つのデータセットを結合
    # これで、学習の各ステップで (triplet_batch, vae_batch) の両方がモデルに渡される
    combined_dataset = tf.data.Dataset.zip((triplet_dataset, vae_dataset))

    # 1エポックあたりのステップ数を計算
    steps_per_epoch = len(all_images) // BATCH_SIZE

    # 4. モデルの学習を開始
    print("モデルの学習を開始します...")
    start_time = time.time()

    vae.fit(
        combined_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,  # 1エポックあたりのステップ数を指定
        callbacks=[loss_history_callback],
    )
    end_time = time.time()

    print(f"\nVAE training time: {end_time - start_time:.4f} seconds")

    # --- 4. 結果の保存 ---
    print("\n--- Saving results ---")

    # 再構成画像の保存
    plot_input_and_reconstructed_images(
        model=vae,
        input_imgs=test_images,
        save_path="grahu/64vae_reconstruction.png",
        num_samples=16,
        epoch=EPOCHS,
    )

    # 学習曲線の保存
    plot_loss_curve(
        history=loss_history_callback.history, save_path="grahu/64vae_loss_curve.png"
    )

    # --- 画像グリッドを生成して中身を確認する ---
    save_image_grid(all_images, f"image_grid/selection{NUM_IMAGES}.png")

    # 学習済みモデル(vae)と全画像(all_images)、
    # そして今定義した評価セット(evaluation_sets)を使って評価を実行
    run_evaluation(
        model=vae,
        all_images=all_images,
        evaluation_sets=EVALUATION_SETS,
        k_list=[5, 10, 20],  # P@5, P@10, P@20 を計算
    )

    # top-kを表示する
    # find_top_k_similar(vae, all_images, k=9)
    # find_top_k_similar_euclidean(vae, all_images, k=9)

    # --- ユーザーの主観で精度を評価 ---
    """evaluation_sets = [
        {
            "name": "TestSet1",  # ファイル名に使うための名前
            "query_index": 330,
            "ground_truth_indices": [390, 443],
        },
        {
            "name": "TestSet2",
            "query_index": 481,
            "ground_truth_indices": [445, 499, 755],
        },
        {"name": "TestSet3_...", "query_index": 497, "ground_truth_indices": [516]},
        {
            "name": "TestSet4_...",
            "query_index": 649,
            "ground_truth_indices": [738, 579],
        },
        {"name": "TestSet5_...", "query_index": 629, "ground_truth_indices": [630]},
        # ↑今後、新しいテストセットを追加したい場合は、ここに辞書を追加するだけです。
    ]"""

    # --- topkの画像表示 ---
    # 自分が選んだ画像が検索上位にでるか確かめる
    """
    print("\n--- ユーザー定義の複数セットで評価を開始します ---")
    for test_set in evaluation_sets:
        # 評価セットごとにユニークなファイルパスを作成
        user_selection_path = f"selection/{test_set['name']}_user_selection.png"
        model_result_path = f"selection/{test_set['name']}_model_result.png"

        compare_with_user_judgment(
            vae,
            all_images,
            query_index=test_set["query_index"],
            ground_truth_indices=test_set["ground_truth_indices"],
            k=10,
            user_save_path=user_selection_path,
            model_save_path=model_result_path,  #
        )

    print("\n--- 全ての評価が完了しました ---")
    """


if __name__ == "__main__":
    main()
