# model.py

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from tensorflow import keras
from keras import layers, ops


# サンプリングレイヤー
class Sampling(layers.Layer):
    """(z_mean, z_log_var) を使って潜在ベクトル z をサンプリングする。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


# Encoderの構築
def build_encoder(input_shape=(64, 64, 3), latent_dim=64):
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 4, strides=2, padding="same")(encoder_inputs)
    x = layers.BatchNormalization()(x)  # <--- 追加
    x = layers.LeakyReLU(alpha=0.2)(x)  # <--- reluから変更

    x = layers.Conv2D(64, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)  # <--- 追加
    x = layers.LeakyReLU(alpha=0.2)(x)  # <--- reluから変更

    # 現在の特徴マップの形状を取得: (batch, 16, 16, 64)
    _, h, w, c = x.shape
    # Attentionレイヤーに入力するために形状を変換: (batch, 16*16, 64)
    attention_input = layers.Reshape((h * w, c))(x)

    # Self-Attentionを実行
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=c // 4  # ヘッド数や次元は調整可能
    )(attention_input, attention_input)

    # 元の形状に戻す: (batch, 16, 16, 64)
    attention_output = layers.Reshape((h, w, c))(attention_output)

    # 元の入力にAttentionの結果を足し合わせる (残差接続)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)  # 残差接続の後はLayerNormが相性が良い

    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)  # <--- 追加
    x = layers.LeakyReLU(alpha=0.2)(x)  # <--- reluから変更

    x = layers.Conv2D(256, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)  # <--- 追加
    x = layers.LeakyReLU(alpha=0.2)(x)  # <--- reluから変更

    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


# Decoderの構築
def build_decoder(latent_dim=64, output_channels=3):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(4 * 4 * 256)(latent_inputs)
    x = layers.Reshape((4, 4, 256))(x)

    x = layers.Conv2DTranspose(256, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)  # <--- 追加
    x = layers.LeakyReLU(alpha=0.2)(x)  # <--- reluから変更

    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)  # <--- 追加
    x = layers.LeakyReLU(alpha=0.2)(x)  # <--- reluから変更

    # ▼▼▼【ここからSelf-Attentionブロックを追加】▼▼▼
    # 現在の特徴マップの形状を取得: (batch, 16, 16, 128)
    _, h, w, c = x.shape
    # Attentionレイヤーに入力するために形状を変換: (batch, 16*16, 128)
    attention_input = layers.Reshape((h * w, c))(x)

    # Self-Attentionを実行
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=c // 4  # エンコーダーと設定を合わせるのが一般的
    )(attention_input, attention_input)

    # 元の形状に戻す: (batch, 16, 16, 128)
    attention_output = layers.Reshape((h, w, c))(attention_output)

    # 元の入力にAttentionの結果を足し合わせる (残差接続)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    # ▲▲▲【Self-Attentionブロックここまで】▲▲▲

    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)  # <--- 追加
    x = layers.LeakyReLU(alpha=0.2)(x)  # <--- reluから変更

    x = layers.Conv2DTranspose(32, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)  # <--- 追加
    x = layers.LeakyReLU(alpha=0.2)(x)  # <--- reluから変更
    decoder_outputs = layers.Conv2DTranspose(
        output_channels, 4, activation="sigmoid", padding="same"
    )(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


# VAEモデルクラス
class VAE(keras.Model):
    def __init__(
        self, encoder, decoder, triplet_margin=5.0, triplet_loss_weight=5.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.triplet_margin = triplet_margin
        self.triplet_loss_weight = triplet_loss_weight
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.triplet_loss_tracker = keras.metrics.Mean(name="triplet_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.triplet_loss_tracker,
        ]

    def train_step(self, data):
        triplet_batch, vae_batch = data
        (anchor, positive, negative) = triplet_batch

        with tf.GradientTape() as tape:
            # --- 1. VAE損失 (all_imagesからのバッチで計算) ---
            z_mean_vae, z_log_var_vae, z_vae = self.encoder(vae_batch)
            reconstruction = self.decoder(z_vae)

            reconstruction_loss = ops.mean(
                ops.sum(ops.square(vae_batch - reconstruction), axis=(1, 2, 3))
            )
            kl_loss = ops.mean(
                ops.sum(
                    -0.5
                    * (
                        1
                        + z_log_var_vae
                        - ops.square(z_mean_vae)
                        - ops.exp(z_log_var_vae)
                    ),
                    axis=1,
                )
            )

            z_mean_a, _, _ = self.encoder(anchor)
            z_mean_p, _, _ = self.encoder(positive)
            z_mean_n, _, _ = self.encoder(negative)

            dist_ap = ops.sum(ops.square(z_mean_a - z_mean_p), axis=1)
            dist_an = ops.sum(ops.square(z_mean_a - z_mean_n), axis=1)

            triplet_loss = ops.mean(
                ops.maximum(dist_ap - dist_an + self.triplet_margin, 0.0)
            )
            # --- 3. 合計損失 ---
            total_loss = (
                reconstruction_loss + kl_loss + self.triplet_loss_weight * triplet_loss
            )
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "triplet_loss": self.triplet_loss_tracker.result(),
        }
