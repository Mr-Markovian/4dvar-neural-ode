# train.py
import torch

# def optimize(QG_ODE_model, qg, batch, NIter, alpha_obs, alpha_dyn, delta):
#     X_torch, YObs_torch, M_torch = batch
#     X_torch = torch.autograd.Variable(X_torch, requires_grad=True)
#     X_torch.retain_grad()
#     losses = torch.zeros(NIter, 3)

#     for iter in range(NIter):
#         with torch.set_grad_enabled(True):
#             X_pred = QG_ODE_model(X_torch)
#             sf_pred = qg.get_streamfunction(X_pred)
#             sf_torch = qg.get_streamfunction(X_torch)

#             loss_dyn = torch.sum((sf_torch - sf_pred) ** 2)
#             loss_obs = torch.sum((sf_torch - YObs_torch) ** 2 * M_torch)
#             loss = alpha_obs * loss_obs + alpha_dyn * loss_dyn

#             losses[iter, :] = torch.tensor([loss.item(), loss_dyn.item(), loss_obs.item()])
#             #if iter % 20 == 0:
#             print(f"iter {iter}: loss {loss.item():.3f}, dyn_loss {loss_dyn.item():.3f}, obs_loss {loss_obs.item():.3f}")

#             loss.backward()
#             X_torch = X_torch - delta * X_torch.grad.data
#             X_torch = torch.autograd.Variable(X_torch, requires_grad=True)
    
#     return losses, X_torch

# def train_model(QG_ODE_model, qg, qg_data_vorticity, qg_data_tgt, masks, params, training_cfg):
#     t0 = training_cfg.t0
#     dT = training_cfg.dT
#     NIter = training_cfg.NIter
#     alpha_obs = training_cfg.alpha_obs
#     alpha_dyn = training_cfg.alpha_dyn
#     delta = training_cfg.delta

#     dataset = QGDataset(qg_data_vorticity, qg_data_tgt, masks, t0, dT, params.device)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
#     all_losses = []
#     for batch in dataloader:
#         losses, X_torch = optimize(QG_ODE_model, qg, batch, NIter, alpha_obs, alpha_dyn, delta)
#         all_losses.append(losses)

#     return torch.cat(all_losses), X_torch

def train_model(QG_ODE_model, qg, qg_data_vorticity, qg_data_tgt, masks, params, training_cfg):
    t0 = training_cfg.t0
    dT = training_cfg.dT
    NIter = training_cfg.NIter
    alpha_obs = training_cfg.alpha_obs
    alpha_dyn = training_cfg.alpha_dyn
    delta = training_cfg.delta

    losses = torch.zeros(NIter, 3)

    X_torch = torch.from_numpy(qg_data_vorticity[t0-1:t0-1+dT]).to(params['device'])
    YObs_torch = torch.from_numpy(qg_data_tgt[t0:t0+dT]).to(params['device'])
    M_torch = torch.from_numpy(1 * np.isfinite(masks[t0:t0+dT])).to(params['device'])

    X_torch = torch.autograd.Variable(X_torch, requires_grad=True)
    X_torch = X_torch.to(params['device'])
    X_torch.retain_grad()

    for iter in range(NIter):
        with torch.set_grad_enabled(True):
            X_pred = QG_ODE_model(X_torch[0:dT-1])
            sf_pred = qg.get_streamfunction(X_pred)
            sf_torch = qg.get_streamfunction(X_torch)

            loss_dyn = torch.sum((sf_torch[1:dT, :] - sf_pred) ** 2)
            loss_obs = torch.sum((sf_torch - YObs_torch) ** 2 * M_torch)
            loss = alpha_obs * loss_obs + alpha_dyn * loss_dyn
            losses[iter] = torch.tensor([loss, loss_dyn, loss_obs])

            loss.backward()
            X_torch = X_torch - delta * X_torch.grad.data
            X_torch = torch.autograd.Variable(X_torch, requires_grad=True)

    return losses, X_torch
