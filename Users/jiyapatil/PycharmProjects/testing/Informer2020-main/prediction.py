"""## Prediction"""

import os

# set saved model path
setting = 'informer_patient121_ftS_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_exp_0'
path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')

# If you already have a trained model, you can set the arguments and model path, then initialize a Experiment and use it to predict
# Prediction is a sequence which is adjacent to the last date of the data, and does not exist in the data
# If you want to get more information about prediction, you can refer to code `exp/exp_informer.py function predict()` and `data/data_loader.py class Dataset_Pred`

exp = Exp(args)

exp.predict(setting, True)

# the prediction will be saved in ./results/{setting}/real_prediction.npy
import numpy as np

prediction = np.load(
    '/Users/jiyapatil/PycharmProjects/testing/Informer2020-main/informer_checkpoints/' + setting + '/real_prediction.npy')

prediction.shape

"""### More details about Prediction - prediction function"""


# here is the detailed code of function predict

def predict(exp, setting, load=False):
    pred_data, pred_loader = exp._get_data(flag='pred')

    if load:
        path = os.path.join(exp.args.checkpoints, setting)
        best_model_path = path + '/' + 'checkpoint.pth'
        exp.model.load_state_dict(torch.load(best_model_path))

    exp.model.eval()

    preds = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(exp.device)
        batch_y_mark = batch_y_mark.float().to(exp.device)

        # decoder input
        if exp.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        elif exp.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        else:
            dec_inp = torch.zeros([batch_y.shape[0], exp.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float().to(exp.device)
        # encoder - decoder
        if exp.args.use_amp:
            with torch.cuda.amp.autocast():
                if exp.args.output_attention:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if exp.args.output_attention:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if exp.args.features == 'MS' else 0
        batch_y = batch_y[:, -exp.args.pred_len:, f_dim:].to(exp.device)

        pred = outputs.detach().cpu().numpy()  # .squeeze()

        preds.append(pred)

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    # result save
    folder_path = '/Users/jiyapatil/PycharmProjects/testing/Informer2020-main/informer_checkpoints/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(folder_path + 'real_prediction.npy', preds)

    return preds


# you can also use this prediction function to get result
prediction = predict(exp, setting, True)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(prediction[0, :, -1])
plt.show()

"""### More details about Prediction - prediction dataset

You can give a `root_path` and `data_path` of the data you want to forecast, and set `seq_len`, `label_len`, `pred_len` and other arguments as other Dataset. The difference is that you can set a more detailed freq such as `15min` or `3h` to generate the timestamp of prediction series.

`Dataset_Pred` only has one sample (including `encoder_input: [1, seq_len, dim]`, `decoder_token: [1, label_len, dim]`, `encoder_input_timestamp: [1, seq_len, date_dim]`, `decoder_input_timstamp: [1, label_len+pred_len, date_dim]`). It will intercept the last sequence of the given data (seq_len data) to forecast the unseen future sequence (pred_len data).
"""

from data.data_loader import Dataset_Pred
from torch.utils.data import DataLoader

Data = Dataset_Pred
timeenc = 0 if args.embed != 'timeF' else 1
flag = 'pred';
shuffle_flag = False;
drop_last = False;
batch_size = 1

freq = args.detail_freq

data_set = Data(
    root_path=args.root_path,
    data_path=args.data_path,
    flag=flag,
    size=[args.seq_len, args.label_len, args.pred_len],
    features=args.features,
    target=args.target,
    timeenc=timeenc,
    freq=freq
)
data_loader = DataLoader(
    data_set,
    batch_size=batch_size,
    shuffle=shuffle_flag,
    num_workers=args.num_workers,
    drop_last=drop_last)

len(data_set), len(data_loader)

# When we finished exp.train(setting) and exp.test(setting), we will get a trained model and the results of test experiment
# The results of test experiment will be saved in ./results/{setting}/pred.npy (prediction of test dataset) and ./results/{setting}/true.npy (groundtruth of test dataset)

preds = np.load(
    '/Users/jiyapatil/PycharmProjects/testing/Informer2020-main/informer_checkpoints/' + setting + '/pred.npy')
trues = np.load(
    '/Users/jiyapatil/PycharmProjects/testing/Informer2020-main/informer_checkpoints/' + setting + '/true.npy')

# [samples, pred_len, dimensions]
preds.shape, trues.shape

import matplotlib.pyplot as plt

# draw OT prediction
plt.figure()
plt.plot(trues[0, :, -1], label='GroundTruth')
plt.plot(preds[0, :, -1], label='Prediction')
plt.legend()
plt.show()
