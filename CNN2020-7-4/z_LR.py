import torch
import torch.nn.functional as TF
from itertools import count

W_ = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
b_ = torch.tensor([[5]], dtype=torch.float32)


def f(x, w, b):
    return x.mm(w.T) + b


if __name__ == "__main__":
    x = torch.randn([32, 1])
    # x[0, 0] = 2
    X = torch.cat([x.pow(i) for i in [4, 3, 2, 1]], 1)
    # print(X)
    Y = f(X, W_, b_)
    # print(Y)
    L = torch.nn.Linear(W_.size(1), b_.size(1))
    # print(L)
    for i in count(1):
        loss = TF.smooth_l1_loss(L(X), Y)
        # print(loss)
        loss.backward()
        for p in L.parameters():
            p.data.add_(-0.001 * p.grad.data)
        if loss.data < 0.01:
            break
        L.zero_grad()

    print(loss)

    for param in L.parameters():
        print(param)


#      若是忘记了Pytorch的自动求导机制就研究这个程序
# a = torch.tensor([1], dtype=torch.float32, requires_grad=True)
# b = a * 2
# torch.autograd.backward(b, retain_graph=True)
# print(a.grad.data)
# torch.autograd.backward(b, torch.tensor([0], dtype=torch.float32), retain_graph=True)
# print(a.grad.data)
# b.backward()
# print(a.grad.data)

