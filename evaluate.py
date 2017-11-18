from argparse import ArgumentParser
from NasdaqLoader import NasdaqDataset, collate_fn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from calc_statistics import rmse

def main(args):

    model_path = str('/home/adir/Projects/timeSeriesPrediction/models/model.t7' )
    dataset_path = '/home/adir/Projects/data/nasdaq100/nasdaq100/small/nasdaq100_padding.csv'
    rmse_calc = rmse()

    with open(model_path, 'rb') as f:
        model = torch.load(f)
    model.eval()
    history = 5
    nasdaq_dataset = NasdaqDataset(dataset_path, history)
    test_dl = DataLoader(nasdaq_dataset, batch_size=128, collate_fn=collate_fn)
    with tqdm(desc='Testing', total=len(test_dl)) as pbar:
        for i, (sample, target) in enumerate(test_dl):
            pbar.update()
            output = model(sample)
            rmse_calc.add(output.data - target.data)




    print(output.data[10:20])
    print(target.data[10:20])
    print('rmse is ', str(rmse_calc.get_rmse()))


if __name__ == '__main__':
    parser = ArgumentParser()

    main(parser.parse_args())