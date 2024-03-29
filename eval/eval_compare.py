###
# Script generates images that have side by side expected and predicted masks for
# each image from DAVIS 2016 dataset.
#

import eval_core
import argparse

## =============================================================================
## ------ Program parser
parser = argparse.ArgumentParser(description='Create comparrison of the model output and expected \
                                 output of DAVIS2016 dataset')
parser.add_argument('--model', type=str, required=True, 
                    help='Input .pth model')
parser.add_argument('--mode', type=str, choices=['train', 'val', 'trainval'], default='val',
                    help='Specifies the DAVIS2016 dataset split: test, val (default), or train.')
parser.add_argument('--dataset-root', type=str, default='./DAVIS', 
                    help='Root directory for DAVIS2016 dataset (default: \'./DAVIS)\'')
parser.add_argument('--output-dir', type=str, default='./eval/comp-out', 
                    help='Path to evaluation output dir (default: \'./eval/comp-out)\'')
args = parser.parse_args()

eval_core.Eval(args).run()
