import os
import numpy as np
from copy import deepcopy
import torch

from pytorch_lightning import Trainer, seed_everything

from _lightning import rENEModel
from _args import parser


def main():
    print("Starting...")

    # Cleaning up some hparams
    hparams = parser()

    # Set seed
    if hparams.seed is None:
        hparams.seed = np.random.randint(1, 99999)

    # Freeze
    hparams.freeze = True

    # Force CPU usage
    hparams.gpus = 0
    hparams.accelerator = None
    torch.set_default_tensor_type('torch.FloatTensor')
    
    # Verify CUDA is not being used
    if torch.cuda.is_available():
        print("Warning: CUDA is available but we're forcing CPU usage")
        torch.cuda.is_available = lambda: False

    # Input size to tuple
    hparams.input_size = tuple(hparams.input_size)
    print(hparams)

    seed_everything(hparams.seed)
    np.seterr(divide='ignore', invalid='ignore')

    # get slurm version
    slurm_id = os.environ.get("SLURM_JOBID")
    if slurm_id is None:
        version = None
    else:
        version = str(slurm_id)

    base_pred_path = deepcopy(hparams.pred_save_path)
    base_ckpt_path = deepcopy(hparams.ckpt_path)
    for fold_num in range(1, 5):
        hparams.ckpt_path = os.path.join(base_ckpt_path, f"fold_{fold_num}.ckpt")
        for tta in [None, "x+", "x-", "y+", "y-", "z+", "z-"]:
            # set test time augmentation
            hparams.testaug = tta
            hparams.pred_save_path = base_pred_path.replace(".csv", f"_{fold_num}_{tta}.csv")
            
            # init model - load on CPU
            model = rENEModel.load_from_checkpoint(
                hparams.ckpt_path, 
                params=hparams,
                map_location=torch.device('cpu')
            )
            model.eval()
            model.to('cpu')

            # Initialize a trainer for CPU
            trainer = Trainer.from_argparse_args(
                hparams, 
                progress_bar_refresh_rate=2,
                checkpoint_callback=None,
                logger=None,
                gpus=0,
                accelerator=None
            )

            # Train the model ⚡
            trainer.test(model)
        #     break
        # break

    print("We've reached the end...")

if __name__ == '__main__':
    main()