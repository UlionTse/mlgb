# coding=utf-8
# author=uliontse

import faiss
import tensorflow as tf


seed = 0
n = 100
embedding_width = 8

item_embeddings = tf.keras.initializers.glorot_uniform(seed=seed)(shape=[n, embedding_width], dtype=tf.float32)
user_embeddings = tf.keras.initializers.glorot_uniform(seed=seed+1)(shape=[n, embedding_width], dtype=tf.float32)

item_embeddings = tf.nn.l2_normalize(item_embeddings, axis=1).numpy()
user_embeddings = tf.nn.l2_normalize(user_embeddings, axis=1).numpy()

index = faiss.IndexFlatIP(embedding_width)
index.add(item_embeddings)

# scores, ids = index.search(user_embeddings[:1], k=10)
# print(ids)

user_fid = 0  # faiss_id
top_k = 10

# _, embedding_width = user_embeddings.shape
# user_embeddings, item_embeddings = user_embeddings[:n], item_embeddings[:n]

index = faiss.IndexFlatIP(embedding_width)
index.add(item_embeddings)
scores, ids = index.search(user_embeddings[user_fid:user_fid+1, :], k=top_k)
print(f'matching top@{top_k} item_fid_list of user_fid={user_fid}:')
print(ids[0])
