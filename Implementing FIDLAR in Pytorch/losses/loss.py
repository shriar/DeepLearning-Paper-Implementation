import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gate_loss(y_pred):
    """
    max_gate_opening (ft)
    GATE_S25A     15.64
    GATE_S25B     30.12
    GATE_S25B2    26.44
    GATE_S26_1    30.10
    GATE_S26_2    29.64
    """

    gate_upper_thre = 1
    gate_lower_thre = 0
    
    #### upper threshold: penalize ws_ypred > thre_upper
    mse_upper = torch.relu(y_pred - gate_upper_thre)
    mse_upper_loss = nn.MSELoss()(torch.zeros_like(mse_upper), mse_upper)
    
    #### lower threshold: penalize ws_ypred > thre_lower
    mse_lower = torch.relu(gate_lower_thre - y_pred)
    mse_lower_loss = nn.MSELoss()(torch.zeros_like(mse_lower), mse_lower)
    
    return mse_upper_loss + mse_lower_loss

def water_level_threshold(y_pred):
    """
    ws_max: 5.72 ft
    ws_min: -1.48 ft
    ws_scaled = (ws - ws_min) / (ws_max - ws_min) = (ws+1.48)/7.2
    
    WS_S1       5.59 (#22 > 3.85 ft)
    TWS_S25A    5.41 (#32 > 3.85 ft)
    TWS_S25B    5.72 (#33 > 3.85 ft)
    TWS_S26     5.20 (#34 > 3.85 ft)
    
    WS_S1      -1.20 (#990 < -0.76 ft)
    TWS_S25A   -1.15 (#282 < -0.76 ft)
    TWS_S25B   -1.48 (#454 < -0.76 ft)
    TWS_S26    -1.46 (#492 < -0.76 ft)

    """
    thre_upper = 0.62   # ws = 3.5 ft
    thre_lower = 0.2   # ws = 0.0 ft
    
    #### upper threshold: penalize ws_ypred > thre_upper
    mse_upper = torch.relu(y_pred - thre_upper) * 20
    mse_upper_loss = nn.MSELoss()(torch.zeros_like(mse_upper), mse_upper)
    
    mae_upper = torch.relu(thre_upper - y_pred)
    mae_upper_loss = nn.L1Loss()(torch.zeros_like(mae_upper), mae_upper)
    
    #### lower threshold: penalize ws_ypred > thre_lower
    mse_lower = torch.relu(thre_lower - y_pred)
    mse_lower_loss = nn.MSELoss()(torch.zeros_like(mse_lower), mse_lower)
    
    return mse_upper_loss + mae_upper_loss + mse_lower_loss
