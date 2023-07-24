import torch
import math
import numpy as np
import torch.nn.functional as F


def contact_a(a_hat, m):
    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m
    return a

def sign(x):
    return (x > 0).type(x.dtype)

def soft_sign(x):
    k = 1
    return 1.0/(1.0+torch.exp(-2*k*x))

def postprocess_new(u, m, lr_min, lr_max, num_itr, rho=0.0, with_l1=False,s=math.log(5.0)):
    """
    :param u: utility matrix, u is assumed to be symmetric, in batch
    :param x: RNA sequence, in batch
    :param lr_min: learning rate for minimization step
    :param lr_max: learning rate for maximization step (for lagrangian multiplier)
    :param num_itr: number of iterations
    :param rho: sparsity coefficient
    :param with_l1:
    :return:
    """
    u = sign(u - s) * u
    # initialization
    a_hat = (torch.sigmoid(u)) * soft_sign(u - s).detach()
    lmbd = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1).detach()

    # gradient descent
    for t in range(num_itr):

        grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)).unsqueeze_(-1).expand(u.shape) - u / 2
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        a_hat = a_hat - lr_min * grad
        lr_min = lr_min * 0.99
        if with_l1:
            a_hat = F.relu(torch.abs(a_hat) - rho * lr_min)
        a_hat = 1 - F.relu(1-a_hat)
        lmbd_grad = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1)
        lmbd = lmbd + lr_max * lmbd_grad
        lr_max = lr_max * 0.99

        # print
        # if t % 20 == 19:
        #     n1 = torch.norm(lmbd_grad)
        #     grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)).unsqueeze_(-1).expand(u.shape) - u / 2
        #     grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))
        #     n2 = torch.norm(grad)
        #     print([t, 'norms', n1, n2, aug_lagrangian(u, m, a_hat, lmbd), torch.sum(contact_a(a_hat, u))])

    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m

    return a

