import torch
import argparse

from yolact_onnx import Yolact
from utils.functions import SavePath
from utils.augmentations import FastBaseTransform
from data import cfg, set_cfg

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='convert YOLACT .pth to .onnx')
    
    parser.add_argument('--trained_model',
                        type = str,
                        help = 'Trained state_dict file path to convert.'
                        )
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
                        
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_args()

    if args.config is not None:
        set_cfg(args.config)      
    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    with torch.no_grad():
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()

        pred_outs = net(FastBaseTransform()(torch.randn(1, 550, 550,3)))

        torch.onnx.export(net, torch.autograd.Variable(torch.randn(1, 3, 550, 550)), args.trained_model.replace('.pth', '.onnx'), verbose=True,input_names=['input'],
                  output_names=['loc', 'conf', 'mask', 'priors', 'proto'],
                  opset_version=11)
        
