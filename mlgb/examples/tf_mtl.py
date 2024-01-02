# coding=utf-8
# author=uliontse

import numpy
import pandas
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import confusion_matrix, classification_report

from mlgb import get_model, mtl_models
from mlgb.data import get_multitask_label_data
from mlgb.utils import check_filepath


if __name__ == '__main__':
    model_name = mtl_models[3]
    print(f'model_name: {model_name}')

    tmp_dir = '.tmp'
    model_dir = f'{tmp_dir}/{model_name}_tf'
    log_dir = f'{model_dir}/log_dir'
    save_model_dir = f'{model_dir}/save_model'
    check_filepath(tmp_dir, model_dir, log_dir, save_model_dir)

    device = 'cuda' if tf.test.is_built_with_cuda() else 'cpu'
    print(f'device: {device}')

    feature_names, (x_train, y_train), (x_test, y_test) = get_multitask_label_data(n_samples=int(1e4))
    y_train = (y_train[:, 0], y_train[:, 1])
    y_test = (y_test[:, 0], y_test[:, 1])
    print([len(names) for names in feature_names])

    model = get_model(
        feature_names=feature_names,
        model_name=model_name,
        task=('binary', 'binary'),
        aim='mtl',
        lang='tf',
        seed=0,
    )
    model.compile(
        loss=[tf.losses.BinaryCrossentropy(), tf.losses.BinaryCrossentropy()],
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
        class_weight=None,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=10, min_delta=1e-3),
        ]
    )

    print(model.summary())
    print(pandas.DataFrame(history.history))

    test_evaluate = model.evaluate(x=x_test, y=y_test, batch_size=None, return_dict=True)
    print(test_evaluate)

    y_pred_prob = model.predict(x_test)
    y_pred_prob = numpy.squeeze(y_pred_prob)  # flatten
    y_pred = numpy.where(y_pred_prob >= 0.5, 1, 0)
    print(y_pred.shape)

    print('CTR:')
    print(confusion_matrix(y_test[0], y_pred[0]))
    print(classification_report(y_test[0], y_pred[0]))

    print('CVR:')
    print(confusion_matrix(y_test[1], y_pred[1]))
    print(classification_report(y_test[1], y_pred[1]))

    # ranking model:
    model.save(filepath=save_model_dir, save_format='tf')
    local_model = tf.keras.models.load_model(filepath=save_model_dir)

    y_local_pred = local_model.predict(x_test)
    y_model_pred = model.predict(x_test)

    print('y_local_ctr_pred == y_model_ctr_pred:', numpy.allclose(y_local_pred[0], y_model_pred[0]))
    print(numpy.abs(numpy.array(y_local_pred[0]) - numpy.array(y_model_pred[0])).max())

    print('y_local_cvr_pred == y_model_cvr_pred:', numpy.allclose(y_local_pred[1], y_model_pred[1]))
    print(numpy.abs(numpy.array(y_local_pred[1]) - numpy.array(y_model_pred[1])).max())
