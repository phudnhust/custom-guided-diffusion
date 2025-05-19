"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import time

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd
import random

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults()) 
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

from noise_refine_model.cross_attention_based import RefineNoiseNet
# from noise_refine_model.mlp_based import RefineNoiseNet

def main():
    start_time = time.perf_counter()
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    print(th.cuda.get_device_properties(dist_util.dev()))
    model.to(dist_util.dev())
    print('device:', dist_util.dev())

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Loading model time: {elapsed_time:.6f} seconds")

    logger.log("sampling...")
    all_images = []
    all_labels = []

    # repo_folder_path = "/mnt/HDD2/phudoan/my_stuff/custom-guided-diffusion/"  # (server 147)
    repo_folder_path = "/mnt/HDD2/phudh/custom-guided-diffusion/"    # (server 118 or 92)
    
    # Generate codebook
    start_time = time.perf_counter()
    print('Loading codebook...')
    K = 32; img_size = 256; T = 1000


    # SHOULD USE: -------- Using torch ---------
    ## th.manual_seed(100)
    # _codebooks = th.randn((T + 1, K, 3, img_size, img_size), dtype=th.float16, device='cpu')
    # _codebooks = _codebooks.numpy()
    # np.save(repo_folder_path + f'models/codebooks_K_{K}.npy', _codebooks)

    _codebooks = np.load(repo_folder_path + f'models/codebooks_K_{K}.npy')
    print('_codebooks.shape:', _codebooks.shape)

    print('Codebook generated!')
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Loading codebooks time: {elapsed_time:.6f} seconds")
    
    start_time = time.perf_counter()

    model_kwargs = {}
    if args.class_cond:
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
    
    #---------------- REFINE NET INITIALIZE -----------------------
    refine_net = RefineNoiseNet(vector_dim=3*256*256, t_embed_dim=128, attn_dim=256).to(dist_util.dev())
    optimizer = th.optim.AdamW(
        refine_net.parameters(),
        lr=1e-7,           # a good starting point
        # weight_decay=1e-2  # small amount of L2 regularization
    )

    # checkpoint = th.load(repo_folder_path + 'refine_net_950.pth')
    # refine_net.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint['epoch'] + 1
    # print('start_epoch:', start_epoch)

    refine_net.train()
    verbose = False

    # load data
    batch_size = 32
    hq_img_folder = '/mnt/HDD2/phudh/custom-guided-diffusion/hq_img/CelebDataProcessed/Barack Obama'
    all_hq_img = [os.path.join(hq_img_folder, f) for f in os.listdir(hq_img_folder) if os.path.isfile(os.path.join(hq_img_folder, f))]
    random.shuffle(all_hq_img)  # Shuffle to ensure randomness before slicing
    hq_img_subset = all_hq_img[:int(0.7 * len(all_hq_img))]
    hq_img_batches = [hq_img_subset[i:i+batch_size] for i in range(0, len(hq_img_subset), batch_size)]

    epoches_loss_list = []
    n_epoch = 1000
    for epoch in range(0, n_epoch):
        print('epoch: ', epoch, end=' ')
        # hq_img_batch = random.sample(hq_img_subset, batch_size)
        epoch_loss_list = []
        for hq_img_batch in hq_img_batches:
            batch_size = len(hq_img_batch)

            timestep = th.randint(1, 1000, (1,)).item()

            sample_fn = diffusion.ddcm_sample_direct
            topk_vectors_list = []
            residuals_list = []
            anchor_vectors_list = []
            for hq_img_path in hq_img_batch:
                sample = sample_fn(
                    model,
                    shape=(args.batch_size, 3, args.image_size, args.image_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    codebook=_codebooks[timestep],
                    hq_img_path=hq_img_path,
                    timestep=timestep,
                    verbose=verbose
                )
                # print('sample: ', sample) 
                if verbose:
                    print('sample[\'retrieved_index\']', sample['retrieved_index'])
                    print('sample[\'top_sim_idx\']', sample['top_sim_idx'])
                    print('-'*20); print()

                topk_vectors_list.append(th.from_numpy(_codebooks[timestep][sample['top_sim_idx']]))
                residuals_list.append(sample['residual'])
                anchor_vectors_list.append(th.from_numpy(_codebooks[timestep][sample['retrieved_index']]).to(dist_util.dev()))


            topk_vectors = th.stack(topk_vectors_list, dim=0).to(dist_util.dev())
            residuals = th.stack(residuals_list, dim=0).to(dist_util.dev())
            anchor_vector = th.stack(anchor_vectors_list, dim=0).to(dist_util.dev())

            timesteps = th.tensor([timestep] * batch_size).to(dist_util.dev())
            t_embed = refine_net.timestep_embedding(timesteps, dim=128).to(dist_util.dev())

            # print('anchor_vector.shape:', anchor_vector.shape)  # 

            # if verbose:
                # print('topk_vectors.shape:', topk_vectors.shape)  # torch.Size([16, 5, 3, 256, 256])
                # print('residuals.shape:', residuals.shape)  # torch.Size([16, 1, 3, 256, 256])
                # print('timesteps.shape:', timesteps.shape)  #  torch.Size([16])

            loss = refine_net.train_refine_step(optimizer=optimizer, topk_vectors=topk_vectors, anchor_vector=anchor_vector, residuals=residuals, t_embed=t_embed)
            epoch_loss_list.append(loss)
        
        if (epoch > 0 and epoch % 100 == 0) or epoch == n_epoch-1:
            refine_net.save_checkpoint(optimizer, epoch, path=repo_folder_path + f'refine_net_{epoch}.pth')
        epoch_loss = np.mean(epoch_loss_list)
        print('epoch_loss:', epoch_loss)
        epoches_loss_list.append(np.mean(epoch_loss))
        
    df = pd.DataFrame({
        'epoch': range(len(epoches_loss_list)),
        'loss': epoches_loss_list
    })
    df.to_csv('learning_curve_' + datetime.now().strftime("_date_%Y%m%d_time_%H%M") +'.csv', index=False)

    dist.barrier()
    logger.log("sampling complete")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Inference time: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()
