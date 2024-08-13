# coding=utf-8
# author=uliontse

import numpy
import pandas
import faiss
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay

from mlgb import get_model
from mlgb.tf.components.retrieval import SampledSoftmaxLossLayer
from mlgb.data import get_multiclass_label_data
from mlgb.utils import check_filepath


def get_item_distribution_list(y_true, add_bias=1):
    q_top = tf.reduce_sum(y_true, axis=0, keepdims=True)
    q_bottom = tf.reduce_sum(y_true, axis=None, keepdims=False)
    q = (q_top + add_bias) / (q_bottom + add_bias)  # add 1 avoid log(0) = -inf.
    return q.numpy().reshape(-1).tolist()


def plot_confusion_matrix(y_confusion_matrix, save_dir='.', dpi=800):
    n_classes = y_confusion_matrix.shape[0]
    plt.rcParams.update({'font.size': round(100 / n_classes + 1)})
    _, ax = plt.subplots(figsize=(8.0, 6.0), dpi=dpi)
    display = ConfusionMatrixDisplay(y_confusion_matrix)
    display.plot(ax=ax)
    plt.savefig(f'{save_dir}/plot_confusion_matrix.png', dpi=dpi)
    plt.show()
    return


if __name__ == '__main__':
    lang = 'tf'
    seed = 0
    device = None
    n_classes = 100
    model_name = 'YoutubeDNN'
    sample_mode = 'Sample:batch'
    print(f'model_name: {model_name}')

    tmp_dir = '.tmp'
    model_dir = f'{tmp_dir}/{model_name}_{lang}'
    log_dir = f'{model_dir}/log_dir'
    save_model_dir = f'{model_dir}/save_model'
    check_filepath(tmp_dir, model_dir, log_dir, save_model_dir)

    feature_names, (x_train, y_train), (x_test, y_test) = get_multiclass_label_data(
        n_samples=int(2e4),
        n_classes=n_classes,
        inputs_if_2_groups=True,
        multiclass_item_features_if_only_item_id=True,
        seed=seed,
    )
    y_int_train, y_int_test = y_train, y_test
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)
    print([len(names) for names in feature_names[0]])
    print([len(names) for names in feature_names[1]])

    print('get_item_distribution.')
    global_y_true = numpy.concatenate([y_train, y_test], axis=0)
    global_q_list = get_item_distribution_list(global_y_true)

    model = get_model(
        feature_names=feature_names,
        model_name=model_name,
        task=f'multiclass:{n_classes}',
        aim='matching',
        lang=lang,
        seed=seed,
        device=device,

        model_result_temperature_ratio=None,
        sample_mode=sample_mode,
        sample_num=None,
        sample_item_distribution_list=global_q_list,
        sample_fixed_unigram_frequency_list=global_q_list,
        user_dnn_hidden_units=(256, 128),
    )
    model.compile(
        loss=model.sampled_softmax_loss,  #
        optimizer=tf.optimizers.Nadam(learning_rate=1e-3),
        metrics=[tf.metrics.AUC(multi_label=True, from_logits=True if sample_mode else False)],
    )
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=100,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-3),
        ]
    )

    print(model.summary())
    print(pandas.DataFrame(history.history))

    test_evaluate = model.evaluate(x=x_test, y=y_test, batch_size=64, return_dict=True)
    print(test_evaluate)

    y_pred_prob = model.predict(x_test)
    y_pred_prob = tf.nn.softmax(y_pred_prob, axis=-1).numpy()
    auc = roc_auc_score(
        y_true=y_test,
        y_score=y_pred_prob,
        average='macro',
        multi_class='ovr',
    )
    print('auc:', auc)

    y_int_pred = numpy.argmax(y_pred_prob, axis=-1).reshape(-1)
    print(y_int_pred.shape)
    print(y_int_test.shape)

    print(cm := confusion_matrix(y_int_test, y_int_pred))
    plot_confusion_matrix(cm, save_dir=model_dir)
    print(classification_report(y_int_test, y_int_pred))

    # ranking model to deploy:
    model.save(filepath=save_model_dir)
    local_model = tf.keras.models.load_model(
        filepath=save_model_dir,
        custom_objects={'SampledSoftmaxLossLayer': SampledSoftmaxLossLayer},  #
    )

    y_local_pred = local_model.predict(x_test)
    y_model_pred = model.predict(x_test)
    print('y_local_pred == y_model_pred:', numpy.allclose(y_local_pred, y_model_pred))
    print(numpy.abs(y_local_pred - y_model_pred).max())

    # matching model to explain:
    user_inputs, item_inputs = x_test
    global user_embeddings
    global item_embeddings

    if model_name == 'NCF':
        user_mf_embeddings = local_model.user_mf_embedding_fn(user_inputs)
        item_mf_embeddings = local_model.item_mf_embedding_fn(item_inputs)
        user_dnn_embeddings = local_model.user_dnn_embedding_fn(user_inputs)
        item_dnn_embeddings = local_model.item_dnn_embedding_fn(item_inputs)

        y = local_model.ncf_fn([user_mf_embeddings, item_mf_embeddings, user_dnn_embeddings, item_dnn_embeddings])
    else:
        user_embeddings = local_model.user_embedding_fn(user_inputs)
        item_embeddings = local_model.item_embedding_fn(item_inputs)

        y = user_embeddings * item_embeddings

    y_match_pred = local_model.task_fn(y)
    y_match_pred = y_match_pred.numpy()
    print('y_match_pred == y_model_pred:', numpy.allclose(y_match_pred, y_model_pred))
    print(numpy.abs(y_match_pred - y_model_pred).max())

    # matching model to deploy:
    if model_name != 'NCF':
        user_fid = 0  # faiss_id
        top_k = 10
        n = 100  # limit memory because of testing only.
        user_embeddings, item_embeddings = user_embeddings[:n, :], item_embeddings[:n, :]

        index = faiss.IndexFlatIP(item_embeddings.shape[-1])  # l2_norm first.
        index.add(item_embeddings)
        scores, item_fids = index.search(user_embeddings[user_fid:user_fid+1, :], k=top_k)
        print(f'matching top@{top_k} item_fid_list of user_fid={user_fid}:\n{item_fids[0]}')

    print('done.')
