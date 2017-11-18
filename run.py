from argparse import ArgumentParser
from NasdaqLoader import NasdaqDataset,collate_fn
from torch.utils.data import DataLoader
import BasicRnn
import Encoder_Decoder
import torch
from tqdm import tqdm
from calc_statistics import rmse
import numpy as np




def main(args):
    num_epochs = 10
    max_norm = 0.5
    loadModelFromDisk = False
    saveModelFromDisk = False
    path = '/home/adir/Projects/data/nasdaq100/nasdaq100/small/nasdaq100_padding.csv'
    history = 10
    batch_size = 128
    nasdaq_dataset = NasdaqDataset(path,history,normalization=True,normalize_ys=True,scalingNorm=False)
    train_dl = DataLoader(nasdaq_dataset,batch_size=batch_size ,collate_fn = collate_fn)
    rmse_calc = rmse()
    #model = BasicRnn.create()
    if (loadModelFromDisk):
        with open('/home/adir/Projects/timeSeriesPrediction/models/model.t7', 'rb') as f:
            model = torch.load(f)
    else:
        model = Encoder_Decoder.create()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Reduce LR by 0.1 every 3 epochs
    torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_dl) * 3, gamma=0.1, last_epoch=-1)

    for j in range(num_epochs):
        total_loss = float(0)
        with tqdm(desc='Training', total=len(train_dl)) as pbar:

            for i, (sample, target) in enumerate(train_dl):
                pbar.update()

                model.zero_grad()
                output = model(sample)
                loss = model.criterion(output,target)
                loss.backward()

                #grad_norm(model.parameters())
                torch.nn.utils.clip_grad_norm(model.parameters(),max_norm = max_norm)
                #grad_norm(model.parameters())

                optimizer.step()
                total_loss += loss.data[0]
                pbar.set_description('Training, #epoch={:} loss={:.4}'.format(j+1,np.true_divide(total_loss,i + 1) ))
                unNormOutput = NasdaqDataset.unNormalizedYs(nasdaq_dataset, output.data)
                unNormTarget = NasdaqDataset.unNormalizedYs(nasdaq_dataset, target.data)
                rmse_calc.add(unNormOutput  - unNormTarget  )

        print (unNormOutput[0])
        print(unNormTarget[0])



        total_loss = total_loss / len(train_dl)
        print ('')
        print('RMSE is ', str(rmse_calc.get_rmse()))
        print('MAE is ', str(rmse_calc.get_mae()))
        print ('total_loss = ',str(total_loss))
        print('')

        rmse_calc.reset()
    if (saveModelFromDisk):
        torch.save(model,str('/home/adir/Projects/timeSeriesPrediction/models/model.t7' ))


if __name__ == '__main__':
    parser = ArgumentParser()

    main(parser.parse_args())

