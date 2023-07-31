import torch
import torch.nn as nn
import torch.nn.functional as F


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


@torch.no_grad()
def update_params(params, grads, eta, opt, args, deltaonly=False, return_s=False):
    if isinstance(opt, torch.optim.SGD):
        return update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s)
    else:
        raise NotImplementedError('Non-supported main model optimizer type!')


# be aware that the opt state dict returns references, hence take care not to
# modify them
def update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s=False):
    # supports SGD-like optimizers
    ans = []

    if return_s:
        ss = []

    # opt = optimizer
    wdecay = opt.defaults['weight_decay']   # weight decay (L2 penalty) (default: 0)
    momentum = opt.defaults['momentum']    # momentum factor (default: 0)
    dampening = opt.defaults['dampening']   # dampening for momentum (default: 0)
    nesterov = opt.defaults['nesterov']   # enables Nesterov momentum (default: False)

    for i, param in enumerate(params):
        dparam = grads[i] + param * wdecay  # s=1, weight decay is set to 5e-4 per default
        s = 1

        if momentum > 0:    # default is 0.9
            try:
                moment = opt.state[param]['momentum_buffer'] * momentum
            except:
                moment = torch.zeros_like(param)

            moment.add_(dparam, alpha=1. - dampening)  # s=1.-dampening

            if nesterov:    # False per default
                dparam = dparam + momentum * moment  # s= 1+momentum*(1.-dampening)
                s = 1 + momentum * (1. - dampening)
            else:
                dparam = moment  # s=1.-dampening
                s = 1. - dampening

        if deltaonly:   # False per default
            ans.append(- dparam * eta)
        else:
            ans.append(param - dparam * eta)

        if return_s:
            ss.append(s * eta)

    if return_s:
        return ans, ss
    else:
        return ans


# ============== mlc step procedure debug with features (gradient-stopped) from main model ===========
#
# METANET uses the last K-1 steps from main model and imagine one additional step ahead
# to compose a pool of actual K steps from the main model
#
#
def step_hmlc_K(main_net, main_opt, meta_net, meta_opt, loss_f,
                data_s, target_s, data_g, target_g,
                data_c, target_c,
                eta, args):
    # compute gradients (gw) for updating meta_net, logically this does not belong here but is used much later
    logit_g = main_net(data_g)
    loss_g = loss_f(logit_g, target_g)
    gw = torch.autograd.grad(loss_g, main_net.parameters())

    # given current meta net, get corrected label, compute loss with logits and corrected label
    logit_s, x_s_h = main_net(data_s, return_h=True)
    pseudo_target_s = meta_net(x_s_h.detach(), target_s)
    # in order to support ignore_index=-100 for cross entropy loss with soft labels we need to filter before passing
    # the tensors to the loss function
    filtered_pseudo_target_s = pseudo_target_s[target_s != -100]
    filtered_logit_s = logit_s[target_s != -100]
    loss_s = loss_f(filtered_logit_s, filtered_pseudo_target_s)

    # data_c is either one half of the gold data if gold data is available or one half of the silver data
    if data_c is not None:
        # get batch sizes, target_c is half of the original gold data
        bs1 = target_s.size(0)
        bs2 = target_c.size(0)

        logit_c = main_net(data_c)
        loss_s2 = loss_f(logit_c, target_c)
        loss_s = (loss_s * bs1 + loss_s2 * bs2) / (bs1 + bs2)

    # compute gradient of the silver loss w.r.t. the parameter of the classifier
    f_param_grads = torch.autograd.grad(loss_s, main_net.parameters(), create_graph=True)

    # calculate new parameters f_params_new using SGD step
    # dparam_s: s * eta (learning rate) where s is either 1 + momentum * (1 - dampening) if nesterov or 1 - dampening
    # --> scaling factors? learning rates
    f_params_new, dparam_s = update_params(main_net.parameters(), f_param_grads, eta, main_opt, args, return_s=True)
    # 2. set w as w' -> basically get the updated main model
    f_param = []
    for i, param in enumerate(main_net.parameters()):
        f_param.append(param.data.clone())  # make backup of current parameters, then overwrite with new parameters
        param.data = f_params_new[i].data  # use data only as f_params_new has graph

    # training loss Hessian approximation
    Hw = 1  # assume to be identity

    # 3. compute d_w' L_{D}(w'), make prediction on clean data with updated classifier, calculate loss and gradient
    logit_g = main_net(data_g)
    loss_g = loss_f(logit_g, target_g)
    gw_prime = torch.autograd.grad(loss_g, main_net.parameters())

    # 3.5 compute discount factor gw_prime * (I-LH) * gw.t() / |gw|^2
    tmp1 = [(1 - Hw * dparam_s[i]) * gw_prime[i] for i in range(len(dparam_s))]
    gw_norm2 = (_concat(gw).norm()) ** 2
    tmp2 = [gw[i] / gw_norm2 for i in range(len(gw))]
    gamma = torch.dot(_concat(tmp1), _concat(tmp2))

    # because of dparam_s, need to scale up/down f_params_grads_prime for proxy_g/loss_g
    Lgw_prime = [dparam_s[i] * gw_prime[i] for i in range(len(dparam_s))]

    proxy_g = -torch.dot(_concat(f_param_grads), _concat(Lgw_prime))

    # back prop on alphas
    meta_opt.zero_grad()
    proxy_g.backward()

    # accumulate discounted iterative gradient
    for i, param in enumerate(meta_net.parameters()):
        if param.grad is not None:
            param.grad.add_(gamma * args.dw_prev[i])
            args.dw_prev[i] = param.grad.clone()

    # gradient_steps: Number of look-ahead gradient steps for meta-gradient (default: 1)
    if (args.steps + 1) % args.gradient_steps == 0:  # T steps proceeded by main_net
        meta_opt.step()     # optimizer adjusts each parameter by its gradient stored in .grad
        args.dw_prev = [0 for param in meta_net.parameters()]  # 0 to reset

    # modify to w, and then do actual update main_net
    # recover parameters and gradient of the silver loss w.r.t. to those parameters
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]
        param.grad = f_param_grads[i].data
    main_opt.step()     # optimizer adjusts each parameter by its gradient stored in .grad

    return loss_g, loss_s
