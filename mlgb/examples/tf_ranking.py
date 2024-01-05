# coding=utf-8
# author=uliontse

import numpy
import pandas
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

from mlgb import get_model, ranking_models
from mlgb.data import get_binary_label_data
from mlgb.utils import check_filepath


if __name__ == '__main__':
    model_name = 'DSIN'
    lang = 'tf'
    seed = 0

    # path of save_model:
    tmp_dir = '.tmp'
    model_dir = f'{tmp_dir}/{model_name}_{lang}'
    log_dir = f'{model_dir}/log_dir'
    save_model_dir = f'{model_dir}/save_model'
    check_filepath(tmp_dir, model_dir, log_dir, save_model_dir)

    # get_data:
    two_inputs_models = ['PLM', 'GRU4Rec', 'Caser', 'SASRec', 'BERT4Rec', 'BST', 'DIN', 'DIEN', 'DSIN']
    feature_names, (x_train, y_train), (x_test, y_test) = get_binary_label_data(
        n_samples=int(1e3),
        negative_class_weight=0.9,
        test_size=0.15,
        inputs_if_2_groups=True if model_name in two_inputs_models else False,
        seed=seed,
    )
    print(f'features: {[len(names) for names in feature_names]}')

    class_weight = dict(enumerate(compute_class_weight(class_weight='balanced', classes=numpy.unique(y_train), y=y_train)))
    print(f'class_weight: {class_weight}')

    # train and evaluate:
    model = get_model(
        feature_names=feature_names,
        model_name=model_name,
        task='binary',
        aim='ranking',
        lang='tf',
        seed=seed,

    )
    model.compile(
        loss=tf.losses.BinaryCrossentropy(),
        optimizer=tf.optimizers.Nadam(learning_rate=1e-3),
        metrics=[
            tfa.metrics.F1Score(num_classes=1, threshold=0.5, average='macro'),
            tf.metrics.AUC(),
        ],
    )
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=32,
        epochs=10,
        validation_split=0.15,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=1e-3),
        ]
    )

    print(model.summary())
    print(pandas.DataFrame(history.history))

    test_evaluate = model.evaluate(x=x_test, y=y_test, batch_size=32, return_dict=True)
    print(test_evaluate)

    y_pred_prob = model.predict(x_test)
    y_pred_prob = numpy.squeeze(y_pred_prob)
    y_pred = numpy.where(y_pred_prob >= 0.5, 1, 0)
    print(y_pred.shape)
    print(y_test.shape)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # ranking model:
    model.save(filepath=save_model_dir, save_format='tf')
    local_model = tf.keras.models.load_model(filepath=save_model_dir)

    y_local_pred = local_model.predict(x_test)
    y_model_pred = model.predict(x_test)
    print('y_local_pred == y_model_pred:', numpy.allclose(y_local_pred, y_model_pred))
    print(numpy.abs(y_local_pred - y_model_pred).max())


