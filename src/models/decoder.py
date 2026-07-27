import torch
import torch.nn as nn


##Physics decoder
class Physics_Decoder(nn.Module):
    def __init__(self)
    
class IVIMPhysicsDecoder(nn.Module):
    def __init__(self, latent_dim, b_values):
        super().__init__()
        self.b_values = b_values  # Tensor of b-values, e.g., [0, 50, 100, ..., 800]
        
        # Network predicts physical parameters: [f, D, D*]
        # We use softplus/sigmoid to enforce physical constraints (positivity, bounds)
        self.param_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # Outputs: [f_raw, D_raw, D*_raw]
        )

    def forward(self, z):
        # 1. Predict raw parameters from latent space
        params = self.param_head(z)
        
        # 2. Enforce Physical Constraints via Activation Functions
        # f (perfusion fraction) must be between 0 and 1
        f = torch.sigmoid(params[:, 0:1]) 
        
        # D and D* must be positive (diffusion coefficients > 0)
        # Softplus ensures > 0; scaling factors adjust typical MRI units (mm²/s)
        D = torch.softplus(params[:, 1:2]) * 1e-3 
        D_star = torch.softplus(params[:, 2:3]) * 1e-2 

        # 3. Apply IVIM Bi-exponential Law (Hard Constraint)
        # S(b)/S0 = (1-f)*exp(-b*D) + f*exp(-b*D*)
        # b_values shape: [1, Num_b_values] to broadcast with batch
        b = self.b_values.unsqueeze(0) 
        
        signal_slow = (1 - f) * torch.exp(-b * D)
        signal_fast = f * torch.exp(-b * D_star)
        
        s_signal = signal_slow + signal_fast
        
        return s_signal, f, D, D_star

# Usage Example
b_vals = torch.tensor([0, 50, 100, 200, 400, 800], dtype=torch.float32)
decoder = IVIMPhysicsDecoder(latent_dim=10, b_values=b_vals)

# Dummy latent vector from encoder
z = torch.randn(4, 10) 
reconstructed_signal, f_pred, d_pred, dstar_pred = decoder(z)
# reconstructed_signal is guaranteed to follow the IVIM bi-exponential decay   