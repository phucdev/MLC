import torch
import torch.nn.functional as F


def save_checkpoint(state, filename):
    torch.save(state, filename)


class DummyScheduler(torch.optim.lr_scheduler._LRScheduler):
    def get_lr(self):
        lrs = []
        for param_group in self.optimizer.param_groups:
            lrs.append(param_group['lr'])

        return lrs

    def step(self, epoch=None):
        pass


def tocuda(data):
    if type(data) is list:
        if len(data) == 1:
            return data[0].cuda()
        else:
            return [x.cuda() for x in data]
    else:
        return data.cuda()


'''
def net_grad_norm_max(model, p):
    grad_norms = [x.grad.data.norm(p).item() for x in model.parameters()]
    return max(grad_norms)
'''


def clone_parameters(model):
    assert isinstance(model, torch.nn.Module), 'Wrong model type'

    params = model.named_parameters()

    f_params_dict = {}
    f_params = []
    for idx, (name, param) in enumerate(params):
        new_param = torch.nn.Parameter(param.data.clone())
        f_params_dict[name] = idx
        f_params.append(new_param)

    return f_params, f_params_dict


# target differentiable version of F.cross_entropy
def cross_entropy(input_tensor, target, reduction='mean'):
    # the tensor shape for computer vision or text classification is batch_size x num_classes
    # the tensor shape for token classification is batch_size x seq_length x num_classes, so we need to flatten
    # the batch_size x seq_length dimensions of the input tensor and the pseudo target tensor
    if input_tensor.dim() == 3:     # batch_size x seq_length x num_classes
        flattened_input_tensor = input_tensor.flatten(end_dim=1)
        if target.dim() == 3:   # soft labels
            flattened_target = target.flatten(end_dim=1)
        else:   # hard labels
            flattened_target = target.flatten()
        return F.cross_entropy(flattened_input_tensor, flattened_target, reduction=reduction)
    else:
        return F.cross_entropy(input_tensor, target, reduction=reduction)


# test code for soft_cross_entropy
if __name__ == '__main__':
    K = 100
    for _ in range(10000):
        y = torch.randint(K, (100,))
        y_onehot = F.one_hot(y, K).float()
        logits = torch.randn(100, K)

        l1 = F.cross_entropy(logits, y)
        l2 = cross_entropy(logits, y_onehot)

        print(l1.item(), l2.item(), '%.5f' % (l1 - l2).abs().item())
