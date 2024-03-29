###
# Script generates output masks from input model for
# each image from DAVIS 2016 dataset into desired output dir.
#

import eval_core
import argparse

## =============================================================================
## ------ Program parser
parser = argparse.ArgumentParser(description='Generate model output and expected \
                                 output of DAVIS2016 dataset')
parser.add_argument('--model', type=str, required=True, 
                    help='Input .pth model')
parser.add_argument('--mode', type=str, choices=['train', 'val', 'trainval'], default='val',
                    help='Specifies the DAVIS2016 dataset split: test, val (default), or train.')
parser.add_argument('--dataset-root', type=str, default='./DAVIS', 
                    help='Root directory for DAVIS2016 dataset (default: \'./DAVIS)\'')
parser.add_argument('--output-dir', type=str, default='./eval/res', 
                    help='Path to results output dir (default: \'./eval/res)\'')
args = parser.parse_args()

eval_core.Eval(args).run(compare=False)
