import os
import json
import random
import shutil
import re
from itertools import chain
from collections import defaultdict, Counter
from typing import List, Union
from tqdm import tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, num_layers=6, num_heads=8, head_dim=64, dff=1024, dropout=0.1,
                 attention_type="dot_product", maxlen=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = num_heads * head_dim
        self.dropout = dropout
        self.attention_type = attention_type
        self.emb = tf.keras.layers.Embedding(vocab_size, self.d_model)
        self.dec_layers = [
            DecoderLayer(
                num_heads=num_heads,
                head_dim=head_dim,
                dff=dff,
                dropout=dropout,
                attention_type=attention_type,
                maxlen=maxlen
            )
            for _ in range(num_layers)
        ]

        self.pos_encoding = None
        if self.attention_type == "dot_product":
            self._set_positional_encoding()

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        x = self.emb(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if self.attention_type == "dot_product":
            x += self.pos_encoding[:, :seq_len, :]
        x = tf.keras.layers.Dropout(self.dropout)(x, training=training)
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # [seq_len, seq_len]
        for dec in self.dec_layers:
            x = dec(x, training=training, mask=mask)
        return x

    def _set_positional_encoding(self):
        maxlen = 10000
        pos = np.arange(maxlen)[:, None]
        i = np.arange(self.d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[None, :, :]  # [1, maxlen, d_model]
        self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)  # [1, maxlen, d_model]


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_dim, dff, dropout, attention_type="dot_product", maxlen=None):
        super().__init__()
        d_model = num_heads * head_dim
        if attention_type == "dot_product":
            self.mha = MHA(num_heads, head_dim)
        elif attention_type == "relative":
            self.mha = RelativeMHA(num_heads, head_dim, maxlen)
        else:
            raise ValueError(f"expected attention_type in {{dot_product, relative}}, got {attention_type}")
        self.dense_ff = tf.keras.layers.Dense(dff, activation=tf.nn.relu)
        self.dense_model = tf.keras.layers.Dense(d_model)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False, mask=None):
        x1 = self.mha(x, mask)
        x1 = self.dropout1(x1, training=training)
        x = self.ln1(x + x1)
        x1 = self.dense_ff(x)
        x1 = self.dense_model(x1)
        x1 = self.dropout2(x1, training=training)
        x = self.ln2(x + x1)
        return x


class MHA(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dense_input = tf.keras.layers.Dense(num_heads * head_dim * 3)

    def call(self, x, mask=None):
        """
        https://arxiv.org/abs/1706.03762
        :param x: tf.Tensor of shape [N, T, H * D]
        :param mask: tf.Tensor of shape [T, T]
        :return: tf.Tensor of shape [N, T, H * D]
        """
        batch_size = tf.shape(x)[0]
        qkv = self.dense_input(x)  # [N, T, H * D * 3]
        qkv = tf.reshape(qkv, [batch_size, -1, self.num_heads, self.head_dim, 3])  # [N, T, H, D, 3]
        qkv = tf.transpose(qkv, [4, 0, 2, 1, 3])  # [3, N, H, T, D]
        q, k, v = tf.unstack(qkv)  # 3 * [N, H, T, D]

        logits = tf.matmul(q, k, transpose_b=True)  # [N, H, T, T]
        logits /= self.head_dim ** 0.5  # [N, H, T, T]

        if mask is not None:
            mask = mask[None, None, :, :]
            logits += mask * -1e9

        w = tf.nn.softmax(logits, axis=-1)  # [N, H, T, T] (k-axis)
        x = tf.matmul(w, v)  # [N, H, T, D]
        x = tf.transpose(x, [0, 2, 1, 3])  # [N, T, H, D]
        x = tf.reshape(x, [batch_size, -1, self.num_heads * self.head_dim])  # [N, T, D * H]
        return x


class RelativeMHA(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_dim, maxlen):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dense_input = tf.keras.layers.Dense(num_heads * head_dim * 3)
        initializer = tf.random_normal_initializer(stddev=head_dim ** -0.5)
        self.R = tf.Variable(
            name="relative_embeddings",
            initial_value=initializer(shape=[maxlen, maxlen, head_dim]),
            dtype=tf.float32,
        )

    def call(self, x, mask=None):
        """
        :param x: tf.Tensor of shape [N, T, H * D]
        :param mask: tf.Tensor of shape [T, T]
        :return: tf.Tensor of shape [N, T, H * D]
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        qkv = self.dense_input(x)  # [N, T, H * D * 3]
        qkv = tf.reshape(qkv, [batch_size, -1, self.num_heads, self.head_dim, 3])  # [N, T, H, D, 3]
        qkv = tf.transpose(qkv, [4, 0, 2, 1, 3])  # [3, N, H, T, D]
        q, k, v = tf.unstack(qkv)  # 3 * [N, H, T, D]

        dot_product_term = tf.matmul(q, k, transpose_b=True)  # [N, H, T, T]

        q = tf.transpose(q, [2, 0, 1, 3])  # [T, N, H, D]
        q = tf.reshape(q, [seq_len, batch_size * self.num_heads, -1])  # [T, N * H, D]
        x = tf.matmul(q, self.R[:seq_len, :seq_len, :], transpose_b=True)  # [T, N * H, T]
        x = tf.reshape(x, [seq_len, batch_size, self.num_heads, -1])  # [T, N, H, T]
        relative_term = tf.transpose(x, [1, 2, 0, 3])  # [N, H, T, T]

        logits = dot_product_term + relative_term
        logits /= self.head_dim ** 0.5  # [N, H, T, T]

        if mask is not None:
            mask = mask[None, None, :, :]
            logits += mask * -1e9

        w = tf.nn.softmax(logits, axis=-1)  # [N, H, T, T] (k-axis)
        x = tf.matmul(w, v)  # [N, H, T, D]
        x = tf.transpose(x, [0, 2, 1, 3])  # [N, T, H, D]
        x = tf.reshape(x, [batch_size, -1, self.num_heads * self.head_dim])  # [N, T, D * H]
        return x


class MelodyGenerator:
    def __init__(self, predict_fn, vocab):
        self.predict_fn = predict_fn
        self.id2token = dict(enumerate(vocab))
        self.token2id = {v: k for k, v in self.id2token.items()}

    def decode(self, init_tokens: List[str], num_steps=128, max_sequence_for_step=None) -> List[str]:
        sequence = self._tokens2ids(init_tokens)
        score = 0
        ptr = max_sequence_for_step or 0
        for _ in range(num_steps):
            seq_step = sequence[-ptr:]
            probs = self._inference_step([seq_step])
            probs = probs.flatten()
            new_token_id = np.random.choice(list(range(len(probs))), p=probs)
            sequence.append(new_token_id)
            score += np.log(probs[new_token_id])
        sequence = sequence[len(init_tokens):]
        generated_tokens = self._ids2tokens(sequence)
        return generated_tokens

    def beam_search(self, init_tokens: List[str], num_steps=128, max_sequence_for_step=None, beam_width=3) -> List[str]:
        sequences = [self._tokens2ids(init_tokens)]
        scores = [0] * beam_width
        ptr = max_sequence_for_step or 0
        for _ in range(num_steps):
            probs = self._inference_step([x[-ptr:] for x in sequences])  # [1 or beam_width, vocab_size]
            indices = np.argsort(probs)[:, -beam_width:].flatten()  # [beam_width or beam_width ** 2]
            all_ways = [(i // beam_width, j, probs[i // beam_width, j]) for i, j in enumerate(indices)]
            top_ways = sorted(all_ways, key=lambda x: x[-1])[-beam_width:]  # [beam_width]
            new_sequences = []
            new_scores = []
            for i, j, p in top_ways:
                seq = sequences[i].copy()
                seq.append(j)
                new_sequences.append(seq)
                score = scores[i]
                score += np.log(p)
                new_scores.append(score)
            sequences = new_sequences
            scores = new_scores
        i = np.array(scores).argmax()
        sequence = sequences[i][len(init_tokens):]
        generated_tokens = self._ids2tokens(sequence)
        return generated_tokens

    def _inference_step(self, tokens):
        x = np.array(tokens, dtype=np.int32)
        x = tf.constant(x)
        x = self.predict_fn(x)['output_1']
        x = x.numpy()
        x = x[:, -1, :]
        x = self._softmax(x)
        return x

    def _ids2tokens(self, sequence):
        tokens = [self.id2token[x] for x in sequence]
        return tokens

    def _tokens2ids(self, sequence):
        token_ids = [self.token2id[x] for x in sequence]
        return token_ids

    @staticmethod
    def _softmax(x):
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        return probs


class MelodyGeneratorRNN(MelodyGenerator):
    def __init__(self, predict_fn, vocab):
        super().__init__(predict_fn, vocab)

    def decode(self, init_tokens, num_steps=512, max_sequence_for_step=None):
        sequence = self._tokens2ids(init_tokens)
        score = 0.0
        _ = self._inference_step(sequence)
        self.predict_fn.reset_states()
        for _ in range(num_steps):
            token_id = sequence[-1]
            probs = self._inference_step([[token_id]])
            probs = probs.flatten()
            new_token_id = np.random.choice(list(range(len(probs))), p=probs)
            sequence.append(new_token_id)
            score += np.log(probs[new_token_id])
        sequence = sequence[len(init_tokens):]
        generated_tokens = self._ids2tokens(sequence)
        return generated_tokens

    def beam_search(self, init_tokens, num_steps=512, max_sequence_for_step=None, beam_width=3):
        sequence = self._tokens2ids(init_tokens)
        sequences = [sequence] * beam_width
        scores = [0.0] * beam_width
        self.predict_fn.reset_states()
        _ = self._inference_step(sequences)
        for step in range(num_steps):
            last_tokens = [[x[-1]] for x in sequences]
            probs = self.predict_fn(last_tokens)
            if step == 0:
                # потому что в противном случае будут генериться одинаковые последовательности,
                # что будет эквивалентно жадной стратегии
                probs = probs[0, None]
            indices = np.argsort(probs)[:, -beam_width:].flatten()
            all_ways = [(i // beam_width, j, probs[i // beam_width, j]) for i, j in enumerate(indices)]
            top_ways = sorted(all_ways, key=lambda x: x[-1])[-beam_width:]
            new_sequences = []
            new_scores = []
            for i, j, p in top_ways:
                seq = sequences[i].copy()
                seq.append(j)
                new_sequences.append(seq)
                score = scores[i]
                score += np.log(p)
                new_scores.append(score)
            sequences = new_sequences
            scores = new_scores
        i = np.array(scores).argmax()
        sequence = sequences[i][len(init_tokens):]
        generated_tokens = self._ids2tokens(sequence)
        return generated_tokens
