from argparse import ArgumentParser
from NasdaqLoader import NasdaqDataset
from torch.utils.data import DataLoader
import BasicRnn
import Encoder_Decoder
import torch
from tqdm import tqdm
from calc_statistics import rmse
import numpy as np
import os
from utils import maybe_cuda, grad_norm, softmax, predictions_analysis



def main(args):

    useLabelAsFeatures= False
    useStepLR = False
    useGradClipping = False
    path = '../data/nasdaq100/small/nasdaq100_padding.csv'
    preds_stats = predictions_analysis()
    rmse_calc = rmse()
    nasdaq_dataset = NasdaqDataset(path, args.history,useLabelAsFeatures, normalization=args.normalize, normalize_ys=args.normalize_ys,
                                   convertToBinaryLabel=args.binary)
    train_dl = DataLoader(nasdaq_dataset,batch_size=args.bs ,collate_fn = nasdaq_dataset.collate_fn)


    if (args.loadModel):
        with open('./models/model.t7', 'rb') as f:
            print ('loaded model ' + os.path.abspath(f.name) )
            model = torch.load(f)
    else:
        model = Encoder_Decoder.create(args.cuda,args.binary,encoderInputSize=nasdaq_dataset.get_num_of_features())
        #model = BasicRnn.create()


    model.train()
    model = maybe_cuda(model,args.cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    # Reduce LR by 0.1 every schedulerSteps epochs
    if useStepLR:
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_dl) * args.schedulerSteps, gamma=0.1, last_epoch=-1)


    for j in range(args.epochs):
        total_loss = float(0)
        with tqdm(desc='Training', total=len(train_dl)) as pbar:

            for i, (sample, target) in enumerate(train_dl):
                pbar.update()
                model.zero_grad()
                output = model(sample)
                loss = model.criterion(output,maybe_cuda(target,args.cuda))
                loss.backward()

                if useGradClipping:
                    print('Before clipping grad norm = {:.4}'.format(grad_norm(model.parameters())))
                    torch.nn.utils.clip_grad_norm(model.parameters(),max_norm = args.maxNorm)
                    print('After clipping grad norm = {:.4}'.format(grad_norm(model.parameters())))

                optimizer.step()
                total_loss += loss.data[0]
                pbar.set_description('Training, #epoch={:} loss={:.4}'.format(j+1,np.true_divide(total_loss,i + 1) ))

                if (args.binary):
                    output_prob = softmax(output.data.cpu().numpy())
                    output_preds = output_prob.argmax(axis=1)
                    target_preds = target.data.cpu().numpy()
                    preds_stats.add(output_preds ,target_preds )

                else:
                    unNormOutput = NasdaqDataset.unNormalizedYs(nasdaq_dataset, output.data)
                    unNormTarget = NasdaqDataset.unNormalizedYs(nasdaq_dataset, target.data)
                    rmse_calc.add(unNormOutput.cpu()  - unNormTarget.cpu()  )


        total_loss = total_loss / len(train_dl)

        if (args.binary):
            print ('')
            print('Accuracy={:.4} F1={:.4} Recall={:.4} Precision={:.4}'.format(preds_stats.get_accuracy(),preds_stats.get_f1(),preds_stats.calc_recall(),preds_stats.calc_precision()))
            preds_stats.reset()
        else:
            print ('')
            print ('RMSE: {:.4}, MAE: {:.4}, totalLoss: {:.4} . '.format(rmse_calc.get_rmse(), rmse_calc.get_mae(), total_loss))
            print('')
            rmse_calc.reset()


    if (args.saveModel):
        torch.save(model,str('/home/adir/Projects/timeSeriesPrediction/models/model.t7' ))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--loadModel', help='load model?', action='store_true')
    parser.add_argument('--saveModel', help='save model?', action='store_true')
    parser.add_argument('--binary', help='use binary labels?', action='store_true')
    parser.add_argument('--cuda', help='cuda?', action='store_true')
    parser.add_argument('--maxNorm', help='max norm of gradient', default=1)
    parser.add_argument('--lr', help='initial lr', default=1e-3)
    parser.add_argument('--epochs', help='num of epochs', type=int, default=10)
    parser.add_argument('--history', help='num of epochs', type=int, default=1)
    parser.add_argument('--schedulerSteps', help='lr scheduler steps', type=int, default=3)
    parser.add_argument('--bs', help='Batch size', type=int, default=16)
    parser.add_argument('--scaling', help='scaling or normalization?', action='store_true')
    parser.add_argument('--normalize_ys', help='normalization of labels?', action='store_true')
    parser.add_argument('--normalize', help='use normalization?', action='store_true')


    main(parser.parse_args())

