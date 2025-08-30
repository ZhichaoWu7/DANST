import torch

def lr_scheduler(scheduler, optimizer, ReduceLROnPlateau_factor = 0.1, ReduceLROnPlateau_patience = 5):
    if scheduler =="scheduler_LambdaLR":
        
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                           lr_lambda=lambda epoch: LambdaLR_scheduler_coefficient ** epoch
                                                           )
    elif scheduler =="scheduler_ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                 mode='min',
                                                                 factor=ReduceLROnPlateau_factor,
                                                                 patience=ReduceLROnPlateau_patience,
                                                                 threshold=0.0001,
                                                                 threshold_mode='rel',
                                                                 cooldown=0,
                                                                 min_lr=0)
    else:
        lr_scheduler = None
    return lr_scheduler