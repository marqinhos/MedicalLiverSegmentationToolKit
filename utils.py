import os
import logging
import subprocess


import numpy as np
import nibabel as nib
import scipy.io
import torch
import torch.distributed as dist
import pdb
import matplotlib.pyplot as plt


LOG_FORMAT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

def configure_logger(rank, log_path=None):
    if log_path:
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)

    # only master process will print & write
    level = logging.INFO if rank in {-1, 0} else logging.WARNING
    handlers = [logging.StreamHandler()]
    if rank in {0, -1} and log_path:
        handlers.append(logging.FileHandler(log_path, "w"))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=handlers,
        force=True,
    )


def save_configure(args):
    if hasattr(args, "distributed"):
        if (args.distributed and is_master(args)) or (not args.distributed):
            with open(f"{args.cp_dir}/config.txt", 'w') as f:
                for name in args.__dict__:
                    f.write(f"{name}: {getattr(args, name)}\n")
    else:
        with open(f"{args.cp_dir}/config.txt", 'w') as f:
            for name in args.__dict__:
                f.write(f"{name}: {getattr(args, name)}\n")


def is_master(args):
    return args.rank % args.ngpus_per_node == 0


def save_nifti(in_image, name, path_out_images='./results'):
    """Function to save a nifti image from a torch tensor

    Args:
        in_image (torch.Tensor): Input image as a torch tensor
        nome (str): name of the file
        path_out_images (str, optional): Path where image going to save. Defaults to './results'.
    """    
    volume = np.array(in_image, dtype=np.float32)
    
    volume = nib.Nifti1Image(volume, np.eye(4))
    
    if not os.path.exists(path_out_images):
        #os.mkdir(path_out_images)
        os.makedirs(path_out_images, exist_ok=True)
    path_data = os.path.join(path_out_images, name)
    nib.save(volume, path_data + ".nii.gz")
    scipy.io.savemat(path_out_images+"vol.mat",  {"data": in_image}) 



def sample_stack(stack, rows=8, cols=8, start_with=20, map='gray', 
                title = 'Visualization', show ='False', path_out_images='./results/'):

    stack = np.array(stack)
    assert stack.ndim == 3
    _min, _max = np.amin(stack), np.amax(stack)
    show_every = (stack.shape[-1]-start_with)//(rows*cols)
    #print(show_every)
    fig,ax = plt.subplots(rows,cols,figsize=[20,20])
    plt.title(title)
    for i in range(rows*cols):
        ind = start_with + i*show_every
        #print(ind)
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[:,:,ind].T, cmap = map, origin="lower",vmin = _min, vmax = _max)
        ax[int(i/rows),int(i % rows)].axis('off')
    
    if show==True:
        plt.show()
    if not os.path.exists(path_out_images):
        #os.mkdir(path_out_images)
        os.makedirs(path_out_images, exist_ok=True)

    fig.savefig(path_out_images + title + '.png')
    plt.close(fig) 


def get_latest_run_version_ckpt_epoch_no(lightning_logs_dir="./logs/lightning_logs", run_version=None):
    """Function to get the latest run version, checkpoint and epoch number

    Args:
        lightning_logs_dir (str, optional): Path for the checkpoints_dir. Defaults to "./logs/lightning_logs".
        run_version (int, optional): Version number. Defaults to None.

    Raises:
        ValueError: Error, no checkpoint found in checkpoints_dir

    Returns:
        _type_: The latest run version, checkpoint
    """    
    if run_version is None:
        run_version = 0
        for dir_name in os.listdir(lightning_logs_dir):
            if "version" in dir_name:
                if int(dir_name.split("_")[1]) > run_version:
                    run_version = int(dir_name.split("_")[1])
                    
    checkpoints_dir = os.path.join(lightning_logs_dir, "version_{}".format(run_version), "checkpoints")
    files = os.listdir(checkpoints_dir)
    ckpt_filename = None
    for file in files:
        if file.endswith(".ckpt"):
            ckpt_filename = file

    if ckpt_filename is not None:
        ckpt_path = os.path.join(checkpoints_dir, ckpt_filename)
    else:
        raise ValueError("Error, no checkpoint found in {}".format(checkpoints_dir))

    return ckpt_path


def check_gpu_memory(self):
    cmd = ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader']
    result = subprocess.check_output(cmd)
    memory_free = [int(x) for x in result.decode('utf-8').strip().split('\n')]
    return memory_free[0]