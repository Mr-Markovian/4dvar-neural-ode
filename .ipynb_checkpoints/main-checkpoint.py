# my_script.py
import hydra
from omegaconf import DictConfig
#from config.config import print_config
from data import load_data
from train import train_model
from models import QG_Block
from qg_neural_ode import QG

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    #print_config(cfg)

    qg = QG(cfg.params)
    QG_ODE_model = QG_Block(qg, cfg.integration).to(cfg.params.device)

    qg_data_obs, qg_data_tgt, qg_data_vorticity, masks = load_data(cfg.data.natl_dataset, cfg.data.qg_model_dataset)
    
    losses, X_torch = train_model(QG_ODE_model, qg, qg_data_vorticity, qg_data_tgt, masks, cfg.params, cfg.training)

    np.save('../data/losses.npy', losses.detach().cpu().numpy())
    np.save('../data/optimal_4dvar_solution_.npy', X_torch.detach().cpu().numpy())

if __name__ == "__main__":
    main()
