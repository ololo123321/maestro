import os
import json
import random
import shutil
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import List, Union
from tqdm import tqdm, trange
from IPython.display import clear_output
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np


class BaseModel(ABC):
    def __init__(self, sess, config=None):
        self.sess = sess
        self.config = config
        self.name = "maestro"

        self.tokens_ph = None
        self.labels_ph = None
        self.training_ph = None

        self.logits_vocab = None
        self.probs = None
        self.train_op = None
        self.loss = None

    def fit(self, train_sequences, eval_sequences=None, batch_size=64, num_train_steps=5000, num_eval_steps=100,
            eval_step=1000, update_plot_step=10):
        train_loss = []
        eval_loss = []

        def sample(sequences):
            samples = random.sample(sequences, batch_size)
            x = []
            y = []
            for seq in samples:
                x.append(seq[:-1])
                y.append(seq[1:])
            return x, y

        try:
            for i in trange(num_train_steps):
                x_batch, y_batch = sample(train_sequences)
                loss = self._train_step(x_batch, y_batch)
                train_loss.append(loss)

                if eval_sequences:
                    if i != 0 and i % eval_step == 0:
                        eval_loss_tmp = []
                        for _ in range(num_eval_steps):
                            x_batch, y_batch = sample(eval_sequences)
                            loss = self._eval_step(x_batch, y_batch)
                            eval_loss_tmp.append(loss)
                        eval_loss.append(np.mean(eval_loss_tmp))

                if i % update_plot_step == 0:
                    self._plot(train_loss, eval_loss)
            return train_loss, eval_loss
        except KeyboardInterrupt:
            return train_loss, eval_loss

    @property
    def num_trainable_params(self):
        return int(sum(np.prod(x.shape) for x in tf.trainable_variables()))

    def build_model(self):
        vocab_size = self.config["vocab_size"]

        learning_rate = self.config.get("adam", {}).get("learning_rate", 0.001)
        beta1 = self.config.get("adam", {}).get("beta1", 0.9)
        beta2 = self.config.get("adam", {}).get("beta2", 0.999)
        epsilon = self.config.get("adam", {}).get("epsilon", 1e-08)
        print(learning_rate, beta1, beta2, epsilon)

        with tf.variable_scope(self.name):
            self.tokens_ph = tf.placeholder(shape=[None, None], dtype=tf.int32, name="tokens")
            self.labels_ph = tf.placeholder(shape=[None, None], dtype=tf.int32, name="labels")
            self.training_ph = tf.placeholder(shape=None, dtype=tf.bool, name="training")
            logits_model = self.build_decoder()
            logits_vocab = tf.keras.layers.Dense(vocab_size)(logits_model)
            probs = tf.nn.softmax(logits_vocab)
            self.probs = probs[:, -1, :]
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels_ph, logits=logits_vocab)
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
            self.train_op = opt.minimize(self.loss)

    @abstractmethod
    def build_decoder(self):
        pass

    def export(self, model_dir):
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            print("old version of model has been just removed")
        os.makedirs(model_dir)

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        saver = tf.train.Saver(var_list)
        checkpoint_path = os.path.join(model_dir, "model.ckpt")
        saver.save(self.sess, checkpoint_path)

        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)

    def restore(self, model_dir):
        # load config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            self.config = json.load(f)

        # build graph
        self.build_model()

        # load weights
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        saver = tf.train.Saver(var_list)
        checkpoint_path = os.path.join(model_dir, "model.ckpt")
        saver.restore(self.sess, checkpoint_path)

    def init_all_variables(self):
        self.sess.run(tf.global_variables_initializer())

    def export_inference_graph(self, export_dir):
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

        inputs = {
            "tokens": tf.saved_model.utils.build_tensor_info(self.tokens_ph),
            "training": tf.saved_model.utils.build_tensor_info(self.training_ph),
        }

        outputs = {
            "probs": tf.saved_model.utils.build_tensor_info(self.probs)
        }

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.PREDICT_METHOD_NAME
        )

        builder.add_meta_graph_and_variables(
            sess=self.sess,
            tags=[tf.saved_model.SERVING],
            signature_def_map={
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
            }
        )

        builder.save()

    def _train_step(self, x_train, y_train):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.tokens_ph: x_train,
            self.labels_ph: y_train,
            self.training_ph: True
        })
        return loss

    def _eval_step(self, x_train, y_train):
        loss = self.sess.run(self.loss, feed_dict={
            self.tokens_ph: x_train,
            self.labels_ph: y_train,
            self.training_ph: False
        })
        return loss

    @staticmethod
    def _plot(train_loss, eval_loss):
        clear_output(True)
        if eval_loss:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
            ax1.set_title("Train loss")
            ax2.set_title("Eval loss")
            ax1.plot(train_loss)
            ax2.plot(eval_loss, marker='o')
        else:
            plt.title("Train loss")
            plt.plot(train_loss)
        plt.grid()
        plt.show()


class TransformerModel(BaseModel):
    def __init__(self, sess, config=None):
        super().__init__(sess, config)

    def build_decoder(self):
        decoder = Decoder(**self.config)
        x = decoder(self.tokens_ph, training=self.training_ph)
        return x


class RNNModel(BaseModel):
    def __init__(self, sess, config=None):
        super().__init__(sess, config)

    def build_decoder(self):
        vocab_size = self.config["vocab_size"]
        cell_size = self.config["recurrent"]["cell_size"]
        dropout = self.config["dropout"]
        x = tf.keras.layers.Embedding(vocab_size, cell_size)(self.tokens_ph)
        x = tf.keras.layers.Dense(cell_size)(x)
        x = tf.keras.layers.LSTM(cell_size, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(cell_size, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(cell_size, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(dropout)(x, training=self.training_ph)
        return x


class RNNModelEfficient(BaseModel):
    def __init__(self, sess, config=None):
        super().__init__(sess, config)

        self.dense_input = None
        self.dense_vocab = None
        self.cell = None
        self.cell_state = None
        self.next_state = None

    def build_decoder(self):
        vocab_size = self.config["vocab_size"]
        cell_size = self.config["cell_size"]
        num_layers = self.config["num_layers"]

        emb = tf.keras.layers.Embedding(vocab_size, cell_size)
        cell = tf.keras.layers.LSTMCell(cell_size)
        self.cell = tf.keras.layers.StackedRNNCells([cell] * num_layers)

        # common
        batch_size = tf.shape(self.tokens_ph)[0]
        self.cell_state = self.cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)  # [[batch_size, cell_size] x num_layers]
        logits_in = emb(self.tokens_ph)  # [N, T, cell_size]
        first_logits = logits_in[:, 0, :]  # [N, cell_size]

        # for inference
        _, next_state = self.cell(first_logits, self.cell_state)
        self.next_state = next_state

        # for training
        initializer = first_logits, self.cell_state
        logits_model, _ = tf.scan(lambda a, x: self.cell(x, a[1]), tf.transpose(logits_in, [1, 0, 2]), initializer)
        logits_model = tf.transpose(logits_model, [1, 0, 2])
        return logits_model

    def export_inference_graph(self, export_dir):
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

        inputs = {
            "tokens": tf.saved_model.utils.build_tensor_info(self.tokens_ph),
            "training": tf.saved_model.utils.build_tensor_info(self.training_ph),
        }
        for i, (c, h) in enumerate(self.cell_state):
            inputs[f"c_{i}"] = tf.saved_model.utils.build_tensor_info(c)
            inputs[f"h_{i}"] = tf.saved_model.utils.build_tensor_info(h)

        outputs = {
            "probs": tf.saved_model.utils.build_tensor_info(self.probs)
        }
        for i, (c, h) in enumerate(self.next_state):
            outputs[f"c_{i}"] = tf.saved_model.utils.build_tensor_info(c)
            outputs[f"h_{i}"] = tf.saved_model.utils.build_tensor_info(h)

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.PREDICT_METHOD_NAME
        )

        builder.add_meta_graph_and_variables(
            sess=self.sess,
            tags=[tf.saved_model.SERVING],
            signature_def_map={
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
            }
        )

        builder.save()


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, num_layers=6, num_heads=8, head_dim=64, dff=1024, dropout=0.1,
                 attention_type="dot_product", maxlen=None, **kwargs):
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
                maxlen=maxlen,
                name=f"dec_{i}"
            )
            for i in range(num_layers)
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
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # apply sin to even indices in the array; 2i
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # apply cos to odd indices in the array; 2i+1
        pos_encoding = angle_rads[None, :, :]  # [1, maxlen, d_model]
        self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)  # [1, maxlen, d_model]


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_dim, dff, dropout, attention_type="dot_product", maxlen=None, name=None):
        super().__init__()
        d_model = num_heads * head_dim
        if attention_type == "dot_product":
            self.mha = MHA(num_heads, head_dim)
        elif attention_type == "relative":
            self.mha = RelativeMHA(num_heads, head_dim, maxlen, name)
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
    def __init__(self, num_heads, head_dim, maxlen, name):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dense_input = tf.keras.layers.Dense(num_heads * head_dim * 3)
        with tf.variable_scope(name, reuse=False):
            self.R = tf.get_variable("relative_embeddings", [maxlen, maxlen, head_dim], dtype=tf.float32)

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
            probs = self._inference_step([x[-ptr:] for x in sequences])
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

    def _inference_step(self, tokens):
        x = self.predict_fn({"tokens": tokens, "training": False})
        return x["probs"]

    def _ids2tokens(self, sequence):
        tokens = [self.id2token[x] for x in sequence]
        return tokens

    def _tokens2ids(self, sequence):
        token_ids = [self.token2id[x] for x in sequence]
        return token_ids


class MelodyGeneratorRNN(MelodyGenerator):
    def __init__(self, predict_fn, vocab, cell_size, num_layers):
        super().__init__(predict_fn, vocab)
        self.cell_size = cell_size  # для инициализации hidden state
        self.num_layers = num_layers  # для инициализации hidden state

    def decode(self, init_tokens, num_steps=512, max_sequence_for_step=None):
        sequence = self._tokens2ids(init_tokens)
        prev_state = self._get_init_state(sequence[:-1])
        score = 0.0
        for _ in range(num_steps):
            token_id = sequence[-1]
            prev_state = self.predict_fn({
                "tokens": [[token_id]],
                "training": False,
                **prev_state,
            })
            probs = prev_state.pop("probs").flatten()
            new_token_id = np.random.choice(list(range(len(probs))), p=probs)
            sequence.append(new_token_id)
            score += np.log(probs[new_token_id])
        print(score)
        sequence = sequence[len(init_tokens):]
        generated_tokens = self._ids2tokens(sequence)
        return generated_tokens

    def beam_search(self, init_tokens, num_steps=512, max_sequence_for_step=None, beam_width=3):
        sequence = self._tokens2ids(init_tokens)
        sequences = [sequence] * beam_width
        prev_state = self._get_init_state(sequence[:-1])
        prev_state = {k: np.tile(v, [beam_width, 1]) for k, v in prev_state.items()}
        scores = [0.0] * beam_width
        for step in range(num_steps):
            last_tokens = [[x[-1]] for x in sequences]
            prev_state = self.predict_fn({
                "tokens": last_tokens,
                "training": False,
                **prev_state,
            })
            probs = prev_state.pop("probs")
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
        print(scores[i])
        sequence = sequences[i][len(init_tokens):]
        generated_tokens = self._ids2tokens(sequence)
        return generated_tokens

        # generated_tokens = [self._ids2tokens(x) for x in sequences]
        # return generated_tokens, scores

    def _get_init_state(self, sequence):
        state = {f"{i}_{j}": np.zeros((1, self.cell_size)) for i in {"c", "h"} for j in range(self.num_layers)}
        for token_id in sequence:
            state = self.predict_fn({
                "tokens": [[token_id]],
                "training": False,
                **state,
            })
            del state["probs"]
        return state
