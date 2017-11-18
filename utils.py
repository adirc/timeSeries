
def grad_norm(parameters,norm_type = 2):
    total_norm = 0
    for p in parameters:
        if  (p.grad is not  None):
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1/norm_type)
    return total_norm


def maybe_cuda(x,is_cuda = False):

    if (is_cuda):
        return x.cuda()
    return x


