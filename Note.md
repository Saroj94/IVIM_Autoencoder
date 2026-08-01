
# optimizer.zero_grad()          # clear old gradients
# outputs = model(inputs)        # forward pass
# loss = criterion(outputs, targets)
# loss.backward()                # compute new gradients
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # clip if needed
# optimizer.step()               # update weights


# =========================================================================
# STEP 1: INITIAL STACK CONFIGURATIONS
# =========================================================================
import torch
import torch.optim as optim
import numpy as np
# from your_file import PhysicsCVAE

# =========================================================================
# STEP 2 & 3: PHYSIOLOGICAL SETUP & PARAMETER MARKERS
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
b_values_tensor = torch.tensor([...], dtype=torch.float32)

NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# Baseline targets for physical balancing
KL_TARGET = 1e-4
PARAM_REG_WEIGHT = 0.1
WARMUP_EPOCHS = 10
ANNEAL_EPOCHS = 30

# =========================================================================
# STEP 4: DATA PIPE PACKAGING
# =========================================================================
# Assemble your real MRI data arrays here -> Shapes: (N, C, 3, 3) and (N, 2)
# train_loader = DataLoader(TensorDataset(x_train, c_train), batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(TensorDataset(x_val, c_val), batch_size=BATCH_SIZE, shuffle=False)

# =========================================================================
# STEP 5: SEEDING CORES TO CHIPS
# =========================================================================
model = PhysicsCVAE(in_channel=len(b_values_tensor), bvalues=b_values_tensor)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# =========================================================================
# STEP 6: CORE LOOPS PIPELINE
# =========================================================================
for epoch in range(NUM_EPOCHS):
    
    # [A] Compute Annealed KL Dimmer Weight
    if epoch < WARMUP_EPOCHS:
        current_kl_weight = 0.0
    elif epoch < (WARMUP_EPOCHS + ANNEAL_EPOCHS):
        current_kl_weight = ((epoch - WARMUP_EPOCHS) / ANNEAL_EPOCHS) * KL_TARGET
    else:
        current_kl_weight = KL_TARGET
        
    # [B] TRAINING PHASES
    model.train()
    for x_batch, c_batch in train_loader:
        x_batch = x_batch.to(device)
        c_batch = c_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass workflow
        reconstructed, mu, logvar, components, physical_params = model(x_batch, c_batch)
        
        # Calculate loss
        loss, recon_loss, kl_loss = model.cvae_loss(
            recons=reconstructed, x=x_batch, mu=mu, logvar=logvar,
            params=physical_params, kl_weight=current_kl_weight, param_reg_weight=PARAM_REG_WEIGHT
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
    # [C] VALIDATION PHASES
    model.eval()
    with torch.no_grad():
        for x_val, c_val in val_loader:
            x_val = x_val.to(device)
            c_val = c_val.to(device)
            
            recon_v, mu_v, logvar_v, _, params_v = model(x_val, c_val)
            # Calculate validation losses and track metrics here...

    # =========================================================================
    # STEP 7: PRINT REPORT
    # =========================================================================
    print(f"Epoch [{epoch+1:03d}] ... Complete Metrics Log Output Line")
