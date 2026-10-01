"""A byte decoder with learned positions and block-consumption KV primitives."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tensorflow as tf


@dataclass(frozen=True)
class Config:
    vocabulary: int = 256
    context: int = 96
    width: int = 96
    heads: int = 4
    layers: int = 3
    hidden: int = 192

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("positive integer dimensions required")
        if self.width % self.heads:
            raise ValueError("heads must divide width")


class Attention(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qkv = tf.keras.layers.Dense(3*config.width, use_bias=False)
        self.project = tf.keras.layers.Dense(config.width, use_bias=False)

    def call(self, x, cache=None):
        c = self.config
        batch, time = tf.shape(x)[0], tf.shape(x)[1]
        projection = tf.reshape(self.qkv(x), [batch, time, 3, c.heads, c.width//c.heads])
        q, k, v = tf.unstack(tf.transpose(projection, [2, 0, 3, 1, 4]), axis=0)
        offset = 0 if cache is None else tf.shape(cache[0])[2]
        if cache is not None:
            k, v = tf.concat([cache[0], k], 2), tf.concat([cache[1], v], 2)
        scores = tf.matmul(q, k, transpose_b=True) * ((c.width//c.heads)**-.5)
        mask = tf.range(tf.shape(k)[2])[None, :] <= (tf.range(time)+offset)[:, None]
        scores = tf.where(mask, scores, tf.cast(-1e9, scores.dtype))
        value = tf.matmul(tf.nn.softmax(scores, -1), v)
        value = tf.reshape(tf.transpose(value, [0, 2, 1, 3]), [batch, time, c.width])
        return self.project(value), (k, v)


class Block(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attention = Attention(config)
        self.up = tf.keras.layers.Dense(config.hidden, use_bias=False)
        self.gate = tf.keras.layers.Dense(config.hidden, use_bias=False)
        self.down = tf.keras.layers.Dense(config.width, use_bias=False)

    def call(self, x, cache=None):
        delta, cache = self.attention(self.norm1(x), cache=cache)
        x = x + delta
        normalized = self.norm2(x)
        return x + self.down(self.up(normalized)*tf.nn.silu(self.gate(normalized))), cache


class Decoder(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = tf.keras.layers.Embedding(config.vocabulary, config.width)
        self.position = tf.keras.layers.Embedding(config.context, config.width)
        self.blocks = [Block(config) for _ in range(config.layers)]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def hidden(self, ids, caches=None):
        tf.debugging.assert_rank(ids, 2)
        tf.debugging.assert_positive(tf.shape(ids))
        tf.debugging.assert_greater_equal(ids, 0)
        tf.debugging.assert_less(ids, self.config.vocabulary)
        offset = 0 if caches is None else tf.shape(caches[0][0])[2]
        tf.debugging.assert_less_equal(offset+tf.shape(ids)[1], self.config.context)
        x = self.embedding(ids) + self.position(tf.range(tf.shape(ids)[1])+offset)[None]
        updated = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, cache=None if caches is None else caches[i])
            updated.append(cache)
        return x, tuple(updated)

    def logits(self, x):
        return tf.einsum("btd,vd->btv", self.norm(x), self.embedding.embeddings)

    def call(self, ids, training=False):
        x, _ = self.hidden(ids)
        return self.logits(x)

    def prefill(self, ids):
        if not self.built:
            self(ids)
        x, caches = self.hidden(ids)
        return self.logits(x[:, -1:])[:, 0], caches

    def consume(self, ids, caches):
        if len(caches) != self.config.layers:
            raise ValueError("one KV pair per layer required")
        length = tf.shape(caches[0][0])[2]
        for k, v in caches:
            tf.debugging.assert_rank(k, 4)
            tf.debugging.assert_equal(tf.shape(k), tf.shape(v))
            tf.debugging.assert_equal(tf.shape(k), [tf.shape(ids)[0], self.config.heads, length, self.config.width//self.config.heads])
        x, caches = self.hidden(ids, caches)
        return self.logits(x), caches


class Backend:
    """Batch-one float32 adapter; every returned logit array is materialized."""
    def __init__(self, model):
        self.model = model
        c = model.config
        self.context, self.vocabulary = c.context, c.vocabulary
        if not model.built:
            model(tf.zeros([1, 1], tf.int32))
        if model.compute_dtype != "float32":
            raise ValueError("float32 runtime required")
        ids = tf.TensorSpec([1, None], tf.int32)
        kv = tf.TensorSpec([1, c.heads, None, c.width//c.heads], tf.float32)
        caches = tuple((kv, kv) for _ in model.blocks)
        self._prefill = tf.function(model.prefill, input_signature=[ids])
        self._consume = tf.function(model.consume, input_signature=[ids, caches])

    def prefill(self, ids):
        logits, cache = self._prefill(np_ids(ids))
        return logits.numpy()[0], cache

    def consume(self, ids, caches):
        logits, caches = self._consume(np_ids(ids), caches)
        return logits.numpy()[0], caches

    @staticmethod
    def truncate(caches, length):
        return tuple((k[:, :, :length], v[:, :, :length]) for k, v in caches)

    def traces(self):
        return {"prefill": self._prefill.experimental_get_tracing_count(), "consume": self._consume.experimental_get_tracing_count()}


def np_ids(ids):
    import numpy as np
    return np.asarray(ids, np.int32)[None]


def load_checkpoint(directory):
    directory = Path(directory)
    model = Decoder(Config(**json.loads((directory/"config.json").read_text())))
    model(tf.zeros([1, 1], tf.int32))
    model.load_weights(directory/"model.weights.h5")
    return model
