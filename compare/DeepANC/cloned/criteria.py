import torch

class SquareErrorLoss(torch.nn.Module):
    def __init__(self):
        super(SquareErrorLoss, self).__init__()

    def forward(self, error):
        square_error = error ** 2
        mean_square_error = torch.mean(square_error, dim=1)
        return mean_square_error