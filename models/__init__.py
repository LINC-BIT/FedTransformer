
def cal_weights_num(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000