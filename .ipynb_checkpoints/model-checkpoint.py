# models.py
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint

class QG_Block(nn.Module):
    def __init__(self, odefunc: nn.Module, integration_cfg):
        super().__init__()
        self.odefunc = odefunc
        self.rtol = integration_cfg.rtol
        self.atol = integration_cfg.atol
        self.step_size = integration_cfg.step_size
        self.solver = integration_cfg.solver
        self.use_adjoint = integration_cfg.adjoint
        self.integration_time = self.step_size * torch.arange(0, 250, dtype=torch.float32)

    @property
    def ode_method(self):
        return odeint_adjoint if self.use_adjoint else odeint

    def forward(self, x: torch.Tensor, adjoint: bool = True, integration_time=None):
        integration_time = self.integration_time if integration_time is None else integration_time
        integration_time = integration_time.to(x.device)
        ode_method = odeint_adjoint if adjoint else odeint
        out = ode_method(
            self.odefunc, x, integration_time, rtol=self.rtol,
            atol=self.atol, method=self.solver, adjoint_params=()
        )
        return out[-1]

