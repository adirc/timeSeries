import torch
def main():

    x = torch.FloatTensor([1,2,3])
    print (x)
    x = x.view(1,1,3)
    print(x)

if __name__ == '__main__':
    main()
