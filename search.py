import random
import numpy as np
from PIL import Image,ImageEnhance,ImageFilter
from sklearn.metrics.pairwise import cosine_similarity
from utils import plot_similar_images
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist


def find_top_k_similar(vae, image_database, k=9):
    """
    コサイン類似度を用いて、データベース内から類似画像を検索する。
    """
    print("\n--- 類似画像の検索を開始します ---")

    # 1. データベース内の全画像の潜在ベクトルを計算
    #    z_meanを特徴量として使用します。ノイズの影響を受けないため安定しています。
    print("データベース内の全画像をエンコードしています...")
    z_mean_database, _, _ = vae.encoder.predict(image_database)

    # 2. クエリ画像をランダムに1枚選択
    query_index = random.randint(0, len(image_database) - 1)
    query_img = image_database[query_index]
    query_vector = z_mean_database[query_index].reshape(
        1, -1
    )  # 比較できるように2D配列に変換

    print(f"クエリ画像のインデックス: {query_index}")

    # 3. クエリベクトルと全ベクトルとのコサイン類似度を計算
    print("コサイン類似度を計算しています...")
    similarities = cosine_similarity(query_vector, z_mean_database).flatten()

    # 4. 類似度が高い順にインデックスをソート
    #    自分自身（類似度1.0）がトップに来るので、それ以降のk個を取得
    top_k_indices = np.argsort(similarities)[-k - 1 : -1][::-1]

    # 5. トップkの画像と類似度を取得
    top_k_images = image_database[top_k_indices]
    top_k_similarities = similarities[top_k_indices]

    # 6. 結果をプロットして保存
    plot_similar_images(
        query_img=query_img,
        similar_imgs=top_k_images,
        similarities=top_k_similarities,
        indices=top_k_indices,
        save_path=f"similar_cosin_k{k}.png",
        k=k,
    )

def compare_with_user_judgment(
    vae,
    image_database,
    query_index,
    ground_truth_indices,
    k=10,
    user_save_path="user_selection.png",
    model_save_path="model_result.png",
):
    """
    ユーザーが選んだ正解画像と、モデルのトップk検索結果を、指定されたパスにそれぞれ出力する。
    """
    print("\n--- ユーザー判断とモデル結果の比較 ---")
    print(f"クエリ画像のインデックス: {query_index}")
    print(f"あなたが選んだ正解画像のインデックス: {ground_truth_indices}")

    # --- 共通の準備：検索の実行 ---
    print("データベース全体をエンコードして検索を実行しています...")
    z_mean_database, _, _ = vae.encoder.predict(image_database)
    query_vector = z_mean_database[query_index].reshape(1, -1)
    similarities = cosine_similarity(query_vector, z_mean_database).flatten()

    # --- 処理1: ユーザーが選んだ画像をプロット ---
    user_selected_images = image_database[ground_truth_indices]
    plot_similar_images(
        title=f"あなたが選んだ「似ている」画像 (クエリ idx: {query_index})",
        query_img=image_database[query_index],
        image_list=user_selected_images,
        index_list=ground_truth_indices,
        save_path=user_save_path,  # 引数で受け取ったパスを使用
    )

    # --- 処理2: モデルのトップk検索結果をプロット ---
    sorted_indices = np.argsort(similarities)[::-1]
    model_top_k_indices = [idx for idx in sorted_indices if idx != query_index][:k]
    model_top_k_images = image_database[model_top_k_indices]
    model_top_k_scores = similarities[model_top_k_indices]
    plot_similar_images(
        title=f"モデルの検索結果 Top-{k} (クエリ idx: {query_index})",
        query_img=image_database[query_index],
        image_list=model_top_k_images,
        index_list=model_top_k_indices,
        score_list=model_top_k_scores,
        save_path=model_save_path,  # 引数で受け取ったパスを使用
    )

    # --- 参考情報のテキスト出力 ---
    common_images = set(model_top_k_indices) & set(ground_truth_indices)
    print(
        f"\nモデルのTop-{k}検索結果と、あなたが選んだ画像との間に、{len(common_images)}個の共通画像がありました。"
    )
    if common_images:
        print(f"共通画像のインデックス: {sorted(list(common_images))}")
