# evaluation.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def _calculate_precision_at_k(ranking, ground_truth, k):
    """(内部関数) 1つのクエリのP@kを計算"""
    top_k = ranking[:k]
    ground_truth_set = set(ground_truth)

    hits = 0
    for idx in top_k:
        if idx in ground_truth_set:
            hits += 1

    return hits / k


def _calculate_ap(ranking, ground_truth):
    """(内部関数) 1つのクエリのAP (Average Precision) を計算"""
    ground_truth_set = set(ground_truth)
    num_ground_truth = len(ground_truth_set)

    if num_ground_truth == 0:
        return 0.0

    hits = 0
    sum_precision = 0

    for i, idx in enumerate(ranking):
        if idx in ground_truth_set:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precision += precision_at_i

    return sum_precision / num_ground_truth


def run_evaluation(model, all_images, evaluation_sets, k_list=[5, 10, 20]):
    """
    モデルの検索性能を評価し、mAPとP@kを計算・表示する。

    Args:
        model: 学習済みVAEモデル
        all_images (np.array): 全画像データ (N, H, W, C)
        evaluation_sets (list): 評価セットのリスト
        k_list (list): P@kを計算するkの値のリスト
    """

    print("\n--- 📈 評価プロセスを開始します ---")

    # --- 1. 全データの特徴量（Embedding）を抽出 ---
    # VAEのz_meanを特徴量として使用するのが一般的
    print("全画像から特徴量（z_mean）を抽出中...")
    # model.encoder が (z_mean, z_log_var, z) を返すと仮定
    z_mean, _, _ = model.encoder(all_images, training=False)
    all_embeddings = z_mean.numpy()  # (N, latent_dim)
    print(f"特徴量抽出完了。形状: {all_embeddings.shape}")

    # --- 2. 類似度行列の計算 ---
    print("全画像間のコサイン類似度を計算中...")
    # (N, N) の行列
    similarity_matrix = cosine_similarity(all_embeddings)
    print("類似度行列の計算完了。")

    # --- 3. 全クエリのランキングを作成 ---
    all_rankings = []

    for test_set in tqdm(evaluation_sets, desc="全クエリのランキングを作成中"):
        query_index = test_set["query_index"]

        # 類似度を取得
        query_similarities = similarity_matrix[query_index]

        # 類似度が高い順（降順）にインデックスをソート
        sorted_indices = np.argsort(query_similarities)[::-1]

        # ランキングから自分自身（クエリ）を除外
        ranking = sorted_indices[1:]

        all_rankings.append(
            {
                "query_index": query_index,
                "ground_truth": test_set["ground_truth_indices"],
                "model_ranking": ranking,
            }
        )

    # --- 4. 評価指標（mAP, P@k）の計算 ---

    # mAP
    total_ap = 0
    for item in all_rankings:
        total_ap += _calculate_ap(item["model_ranking"], item["ground_truth"])

    mAP = total_ap / len(all_rankings)

    # P@k
    p_at_k_results = {}
    for k in k_list:
        total_precision = 0
        for item in all_rankings:
            total_precision += _calculate_precision_at_k(
                item["model_ranking"], item["ground_truth"], k
            )

        mean_p_at_k = total_precision / len(all_rankings)
        p_at_k_results[k] = mean_p_at_k

    # --- 5. 結果の表示 ---
    print("\n--- 📊 評価結果 ---")
    print(f"評価クエリ数: {len(all_rankings)}")
    print(f"mAP (Mean Average Precision): {mAP:.4f}")
    for k, value in p_at_k_results.items():
        print(f"Mean Precision@{k} (P@{k}): {value:.4f}")
    print("--------------------")
