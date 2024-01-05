# coding=utf-8
# author=uliontse

import numpy
import torch
from sklearn.metrics import roc_auc_score, f1_score, log_loss
from sklearn.metrics import confusion_matrix, classification_report

from mlgb import get_model, ranking_models
from mlgb.data import get_binary_label_data
from mlgb.utils import check_filepath


def train_and_evaluate(model, optimizer, loss_fn, x_train, y_train, x_test, y_test,
                       epochs=10, train_batch_size=32, test_batch_size=32, valid_ratio=0.15):

    train_batch_num = int(len(y_train) // train_batch_size)  # drop_last=True
    test_batch_num = int(len(y_test) // test_batch_size)  # drop_last=True

    print(end='\n')
    for epoch in range(epochs):
        model.train()
        for i in range(train_batch_num):
            id_l, id_r = int(train_batch_size * i), int(train_batch_size * (i + 1 - valid_ratio))
            if model_name in two_inputs_models:
                x_train_batch = [[m[id_l: id_r] for m in x_t] for x_t in x_train]
            else:
                x_train_batch = [m[id_l: id_r] for m in x_train]
            y_train_batch = y_train[id_l: id_r]

            y_pred_batch = torch.squeeze(model(x_train_batch))
            y_train_batch = torch.as_tensor(y_train_batch, dtype=torch.float32, device=device)

            l1l2_loss = model.l1l2_loss()
            loss = loss_fn(y_pred_batch, y_train_batch) + l1l2_loss  #
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        y_train_epoch = numpy.array([])
        y_valid_epoch = numpy.array([])
        y_test_epoch = numpy.array([])
        y_train_pred_prob_epoch = numpy.array([])
        y_valid_pred_prob_epoch = numpy.array([])
        y_test_pred_prob_epoch = numpy.array([])
        with torch.no_grad():
            for i in range(train_batch_num):
                id_l, id_r = int(train_batch_size * i), int(train_batch_size * (i + 1 - valid_ratio))
                if model_name in two_inputs_models:
                    x_train_batch = [[m[id_l: id_r] for m in x_t] for x_t in x_train]
                else:
                    x_train_batch = [m[id_l: id_r] for m in x_train]
                y_train_batch = y_train[id_l: id_r]

                y_train_output = torch.squeeze(model(x_train_batch)).cpu().numpy()
                y_train_pred_prob_epoch = numpy.concatenate([y_train_pred_prob_epoch, y_train_output], axis=0)
                y_train_epoch = numpy.concatenate([y_train_epoch, y_train_batch], axis=0)

            for i in range(train_batch_num):
                id_l, id_r = int(train_batch_size * (i + 1 - valid_ratio)), int(train_batch_size * (i + 1))
                if model_name in two_inputs_models:
                    x_valid_batch = [[m[id_l: id_r] for m in x_t] for x_t in x_train]
                else:
                    x_valid_batch = [m[id_l: id_r] for m in x_train]
                y_valid_batch = y_train[id_l: id_r]

                y_valid_output = torch.squeeze(model(x_valid_batch)).cpu().numpy()
                y_valid_pred_prob_epoch = numpy.concatenate([y_valid_pred_prob_epoch, y_valid_output], axis=0)
                y_valid_epoch = numpy.concatenate([y_valid_epoch, y_valid_batch], axis=0)

            for i in range(test_batch_num):
                id_l, id_r = int(train_batch_size * i), int(train_batch_size * (i + 1))
                if model_name in two_inputs_models:
                    x_test_batch = [[m[id_l: id_r] for m in x_t] for x_t in x_test]
                else:
                    x_test_batch = [m[id_l: id_r] for m in x_test]
                y_test_batch = y_test[id_l: id_r]

                y_test_output = torch.squeeze(model(x_test_batch)).cpu().numpy()
                y_test_pred_prob_epoch = numpy.concatenate([y_test_pred_prob_epoch, y_test_output], axis=0)
                y_test_epoch = numpy.concatenate([y_test_epoch, y_test_batch], axis=0)

        y_train_pred_epoch = numpy.where(y_train_pred_prob_epoch >= 0.5, 1, 0)
        y_valid_pred_epoch = numpy.where(y_valid_pred_prob_epoch >= 0.5, 1, 0)
        y_test_pred_epoch = numpy.where(y_test_pred_prob_epoch >= 0.5, 1, 0)
        result = {
            'epoch': epoch + 1,
            'bce': round(log_loss(y_train_epoch, y_train_pred_prob_epoch), 4),
            'auc': round(roc_auc_score(y_train_epoch, y_train_pred_prob_epoch), 4),
            'f1': round(f1_score(y_train_epoch, y_train_pred_epoch, average='macro'), 4),

            'val_bce': round(log_loss(y_valid_epoch, y_valid_pred_prob_epoch), 4),
            'val_auc': round(roc_auc_score(y_valid_epoch, y_valid_pred_prob_epoch), 4),
            'val_f1': round(f1_score(y_valid_epoch, y_valid_pred_epoch, average='macro'), 4),

            'test_bce': round(log_loss(y_test_epoch, y_test_pred_prob_epoch), 4),
            'test_auc': round(roc_auc_score(y_test_epoch, y_test_pred_prob_epoch), 4),
            'test_f1': round(f1_score(y_test_epoch, y_test_pred_epoch, average='macro'), 4),
        }
        print('evaluate:', result)
    return model


if __name__ == '__main__':
    # model_name = 'PNN'
    lang = 'torch'
    device = 'cuda'
    seed = 0

    two_inputs_models = ['PLM', 'GRU4Rec', 'Caser', 'SASRec', 'BERT4Rec', 'BST', 'DIN', 'DIEN', 'DSIN']

    data = get_binary_label_data(
        n_samples=int(1e3),
        negative_class_weight=0.9,
        test_size=0.15,
        inputs_if_2_groups=False,
        seed=seed,
    )
    plm_data = get_binary_label_data(
        n_samples=int(1e3),
        negative_class_weight=0.9,
        test_size=0.15,
        inputs_if_2_groups=True,
        seed=seed,
    )

    for i, model_name in enumerate(ranking_models[41:]):
        print(i, model_name, end='\n\n')

        # path of save_model:
        tmp_dir = '.tmp'
        model_dir = f'{tmp_dir}/{model_name}_{lang}'
        log_dir = f'{model_dir}/log_dir'
        save_model_dir = f'{model_dir}/save_model'
        save_model_file = f'{save_model_dir}/model.pth'
        check_filepath(tmp_dir, model_dir, log_dir, save_model_dir,)

        # get_data:
        feature_names, (x_train, y_train), (x_test, y_test) = plm_data if model_name in two_inputs_models else data
        if model_name in two_inputs_models:
            print(f'features: {[[len(names) for names in group] for group in feature_names]}', end='\n\n')
        else:
            print(f'features: {[len(names) for names in feature_names]}', end='\n\n')

        # train and evaluate:
        model = get_model(
            feature_names=feature_names,
            model_name=model_name,
            task='binary',
            aim='ranking',
            lang='torch',
            device=device,
            seed=seed,
            model_l1=0.0,
            model_l2=1e-6,
        )
        print(model, end='\n\n')

        # must init parameters by build() like tf at first run:
        _ = model(x_test)  # shape requires: any batch_size, other shape must be same as x_train and x_test.
        for name, p in model.named_parameters():
            print(name, list(p.shape), sep='\t\t')

        optimizer = torch.optim.NAdam(model.parameters(), lr=1e-3, weight_decay=0.0)  # model_l2 vs weight_decay
        loss_fn = torch.nn.BCELoss()
        model = train_and_evaluate(model, optimizer, loss_fn,
                                   x_train, y_train, x_test, y_test,
                                   epochs=10, train_batch_size=32, test_batch_size=32, valid_ratio=0.15)

        y_pred_prob = numpy.squeeze(model(x_test).detach().cpu().numpy())
        y_pred = numpy.where(y_pred_prob >= 0.5, 1, 0)
        print(y_pred.shape)
        print(y_test.shape)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # ranking model:
        torch.save(obj=model, f=save_model_file)
        local_model = torch.load(f=save_model_file)

        y_local_pred = numpy.squeeze(local_model(x_test).detach().cpu().numpy())
        y_model_pred = numpy.squeeze(model(x_test).detach().cpu().numpy())
        print('y_local_pred == y_model_pred:', numpy.allclose(y_local_pred, y_model_pred))
        print(numpy.abs(y_local_pred - y_model_pred).max())

        print('done', end='\n\n')




