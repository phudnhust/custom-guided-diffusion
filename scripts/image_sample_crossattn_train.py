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
from tqdm.auto import tqdm

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

from PIL import Image
from noise_refine_model.crossattn import laplacian_kernel, PixelCrossAttentionRefiner, AlexNetPerceptualLoss
import torch.nn as nn
import torch.optim as optim 

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
    refine_net = PixelCrossAttentionRefiner(feat_dim=3, embed_dim=3, num_heads=3).to(dist_util.dev())
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        refine_net.parameters(), 
        lr=1e-4, weight_decay=1e-5
    )

    perc_loss = AlexNetPerceptualLoss(device=dist_util.dev()).to(dist_util.dev())

    ## checkpoint = th.load(repo_folder_path + 'refine_net_950.pth')
    ## refine_net.load_state_dict(checkpoint['model_state_dict'])
    ## optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ## start_epoch = checkpoint['epoch'] + 1
    ## print('start_epoch:', start_epoch)

    # refine_net.train()
    verbose = False

    # load data
    batch_size = 32
    hq_img_folder = '/mnt/HDD2/phudh/custom-guided-diffusion/hq_img/CelebDataProcessed/Barack Obama'
    all_hq_img = [os.path.join(hq_img_folder, f) for f in os.listdir(hq_img_folder) if os.path.isfile(os.path.join(hq_img_folder, f))]
    # random.shuffle(all_hq_img)  # Shuffle to ensure randomness before slicing
    hq_img_subset = all_hq_img[:int(0.8 * len(all_hq_img))]


    hq_img_subset_copy = hq_img_subset.copy()
    hq_img_subset_copy.sort()
    print('train dataset: ')
    print(hq_img_subset_copy)

    
    hq_img_batches = [hq_img_subset[i:i+batch_size] for i in range(0, len(hq_img_subset), batch_size)]

    dummy_count = 0

    epoches_loss_list = []
    n_epoch = 1000
    for epoch in range(0, n_epoch):
        print('epoch: ', epoch, end=' ')
        hq_img_batch = random.sample(hq_img_subset, batch_size)
        epoch_loss_list = []
        for hq_img_batch in tqdm(hq_img_batches, desc='Batch '):
            batch_size = len(hq_img_batch)

            timestep = th.randint(1, 200, (1,)).item()
            # timestep = 200

            batch_noise_candidate = []
            batch_hf_info         = []
            batch_hf_star         = []
            batch_r_t             = []
            batch_x_start         = []
            batch_x_t             = []
            for hq_img_path in hq_img_batch:
                noise_candidate_list, hf_info_list, hf_star, r_t, x_0_list, x_start, x_t = diffusion.get_5_candidates_for_train(
                    model,
                    shape=(args.batch_size, 3, args.image_size, args.image_size),
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    codebook=_codebooks[timestep],
                    hq_img_path=hq_img_path,
                    timestep=timestep,
                    verbose=verbose
                )       # (noise_candidate_list, x_0_list)


                # print('noise_candidate_list[0].shape: ', noise_candidate_list[0].shape)    # torch.Size([1, 3, 256, 256])
                # print('hf_info_list[0].shape: ', hf_star[0].shape)                         # torch.Size([3, 256, 256])

                noise_candidate = th.stack(noise_candidate_list).squeeze(1)     # torch.Size([5, 3, 256, 256]) 
                hf_info = th.stack(hf_info_list)                                # torch.Size([5, 3, 256, 256])

                # print('hf_star.shape: ', hf_star.shape)                       # torch.Size([3, 256, 256])
                # print('r_t.shape: ', r_t.shape)                               # torch.Size([1, 3, 256, 256])

                batch_noise_candidate.append(noise_candidate)
                batch_hf_info.append(hf_info)
                batch_hf_star.append(hf_star)
                batch_r_t.append(r_t.squeeze(0))
                batch_x_start.append(x_start.squeeze(0))
                batch_x_t.append(x_t)

                # ## ---------- VISUALIZE HF INFO OF X_0|T ----------
                # for high_freq in hf_info_list:
                #     if high_freq.dim() == 3:
                #         # If single-channel, convert to [H, W]
                #         if high_freq.shape[0] == 1:
                #             img_np = high_freq.squeeze(0).cpu().numpy()
                #         # If RGB
                #         elif high_freq.shape[0] == 3:
                #             img_np = high_freq.permute(1, 2, 0).cpu().numpy()
                #     elif high_freq.dim() == 2:
                #         img_np = high_freq.cpu().numpy()

                #     # Normalize to 0-255
                #     img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255.0
                #     img_np = img_np.astype('uint8')

                #     # Convert to PIL Image and save
                #     if img_np.ndim == 2:
                #         img_pil = Image.fromarray(img_np, mode='L')   # Grayscale
                #     else:
                #         img_pil = Image.fromarray(img_np, mode='RGB') # Color

                #     img_pil.save(f'/mnt/HDD2/phudh/custom-guided-diffusion/dummy_x0_t/high_freq_{dummy_count}_t_{timestep}.png')
                #     dummy_count += 1
            
            batch_noise_candidate = th.stack(batch_noise_candidate).to(dist_util.dev())     # torch.Size([32, 5, 3, 256, 256])
            batch_hf_info         = th.stack(batch_hf_info).to(dist_util.dev()).squeeze(2)  # torch.Size([32, 5, 3, 256, 256])
            batch_hf_star         = th.stack(batch_hf_star).to(dist_util.dev()).squeeze(1)  # torch.Size([32, 3, 256, 256])
            batch_r_t             = th.stack(batch_r_t).to(dist_util.dev())                 # torch.Size([32, 3, 256, 256])
            batch_x_start         = th.stack(batch_x_start).to(dist_util.dev())                 # torch.Size([32, 3, 256, 256])
            # batch_x_t             = th.stack(batch_x_t).to(dist_util.dev())                 # torch.Size([32, 3, 256, 256])

            # print('batch_hf_info.shape: ', batch_hf_info.shape)
            # print('batch_hf_star.shape: ', batch_hf_star.shape)

        
            z_hat = refine_net(batch_hf_star, batch_hf_info, batch_noise_candidate)  # → [B, 3, H, W]

            # print('batch_x_start.shape', batch_x_start.shape)
            # print('batch_x_t.shape', batch_x_t.shape)
            # print('z_hat.shape', z_hat.shape)

            # x_0_with_z_hat = diffusion.sample_x_0_t_minus_1_for_lpips(model, batch_x_t, timestep, args.clip_denoised, model_kwargs, z_hat)

            loss = criterion(z_hat, batch_r_t) 
            # loss = criterion(z_hat, batch_r_t) + 0.5*perc_loss(x_0_with_z_hat, batch_x_start)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_list.append(loss.item())

        if (epoch > 0 and epoch % 50 == 0) or epoch == n_epoch-1:
            refine_net.save_checkpoint(optimizer, epoch, path=repo_folder_path + f'new_crossattn_refine_net_{epoch}.pth')
        epoch_loss = np.mean(epoch_loss_list)
        print('epoch_loss:', epoch_loss)
        epoches_loss_list.append(np.mean(epoch_loss))
        
    df = pd.DataFrame({
        'epoch': range(len(epoches_loss_list)),
        'loss': epoches_loss_list
    })
    df.to_csv('learning_curve_crossattn_' + datetime.now().strftime("_date_%Y%m%d_time_%H%M") +'.csv', index=False)

    dist.barrier()
    logger.log("sampling complete")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Inference time: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()
