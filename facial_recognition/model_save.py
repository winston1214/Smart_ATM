import torch
def save(state, SAVE_DIR, epoch, model, optimizer): 
    with open(SAVE_DIR + state +".pt", "wb") as f:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            f)