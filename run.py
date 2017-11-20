from argparse import ArgumentParser
from NasdaqLoader import NasdaqDataset,collate_fn
from torch.utils.data import DataLoader
import BasicRnn
import Encoder_Decoder
import torch
from tqdm import tqdm
from calc_statistics import rmse
import numpy as np
import os
from utils import maybe_cuda, grad_norm




def main(args):

    if (args.cuda):
        isCuda = True
    else:
        isCuda = False

    num_epochs = args.epochs
    batch_size = args.bs
    history = 10

    path = '../data/nasdaq100/small/nasdaq100_padding.csv'


    nasdaq_dataset = NasdaqDataset(path,history,normalization=args.normalize,normalize_ys=args.normalize_ys,scalingNorm=args.scaling)
    train_dl = DataLoader(nasdaq_dataset,batch_size=batch_size ,collate_fn = collate_fn)
    rmse_calc = rmse()
    #model = BasicRnn.create()
    if (args.loadModel):
        with open('./models/model.t7', 'rb') as f:
            print ('loaded model ' + os.path.abspath(f.name) )
            model = torch.load(f)
    else:
        model = Encoder_Decoder.create(isCuda)
    model.train()
    model = maybe_cuda(model,args.cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    # Reduce LR by 0.1 every schedulerSteps epochs
    torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_dl) * args.schedulerSteps, gamma=0.1, last_epoch=-1)

    for j in range(num_epochs):
        total_loss = float(0)
        with tqdm(desc='Training', total=len(train_dl)) as pbar:

            for i, (sample, target) in enumerate(train_dl):
                pbar.update()

                model.zero_grad()
                output = model(sample)
                loss = model.criterion(output,maybe_cuda(target,args.cuda))
                loss.backward()

                #grad_norm(model.parameters())
                torch.nn.utils.clip_grad_norm(model.parameters(),max_norm = args.maxNorm)
                #grad_norm(model.parameters())

                optimizer.step()
                total_loss += loss.data[0]
                pbar.set_description('Training, #epoch={:} loss={:.4}'.format(j+1,np.true_divide(total_loss,i + 1) ))
                unNormOutput = NasdaqDataset.unNormalizedYs(nasdaq_dataset, output.data)
                unNormTarget = NasdaqDataset.unNormalizedYs(nasdaq_dataset, target.data)
                rmse_calc.add(unNormOutput.cpu()  - unNormTarget.cpu()  )

        #print a prediction
        #print (unNormOutput[0])
        #print(unNormTarget[0])



        total_loss = total_loss / len(train_dl)
        print ('')
        print ('RMSE: {:.4}, MAE: {:.4}, totalLoss: {:.4} . '.format(rmse_calc.get_rmse(), rmse_calc.get_mae(), total_loss))

        rmse_calc.reset()
    if (args.saveModel):
        torch.save(model,str('/home/adir/Projects/timeSeriesPrediction/models/model.t7' ))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--loadModel', help='load model?', action='store_true')
    parser.add_argument('--saveModel', help='save model?', action='store_true')
    parser.add_argument('--cuda', help='cuda?', action='store_true')
    parser.add_argument('--maxNorm', help='max norm of gradient', default=1)
    parser.add_argument('--lr', help='initial lr', default=1e-3)
    parser.add_argument('--epochs', help='num of epochs', type=int, default=10)
    parser.add_argument('--schedulerSteps', help='lr scheduler steps', type=int, default=3)
    parser.add_argument('--bs', help='Batch size', type=int, default=16)


    parser.add_argument('--scaling', help='scaling or normalization?', action='store_true')
    parser.add_argument('--normalize_ys', help='normalization of labels?', action='store_true')
    parser.add_argument('--normalize', help='use normalization?', action='store_true')










    main(parser.parse_args())

