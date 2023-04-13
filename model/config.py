import model as mdl
from DPT.dpt.models import DPTDepthModel

def get_model(renderer, cfg, device=None, **kwargs):
    depth_estimator = cfg['depth']['type']
    if depth_estimator== 'DPT':
        path = cfg['depth']['path']
        non_negative = cfg['depth']['non_negative']
        scale = cfg['depth']['scale']
        shift = cfg['depth']['shift']
        invert = cfg['depth']['invert']
        freeze = cfg['depth']['freeze']
        depth_estimator = DPTDepthModel(path, non_negative, scale, shift, invert, freeze)
    else:
        depth_estimator = None
    model = mdl.nope_nerf(cfg, renderer, depth_estimator, device)

    return model