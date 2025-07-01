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
from PIL import Image
from guided_diffusion.image_util import Transmitter, Receiver

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

from noise_refine_model.crossattn import PixelCrossAttentionRefiner

def main():
    # dummy_user_role = Receiver('/mnt/HDD2/phudh/custom-guided-diffusion/compressed_info/compressed_representation_date_20250630_time_1411.json')
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
    repo_folder_path = "/mnt/HDD2/phu2/custom-guided-diffusion/"  # (server 148)
    # repo_folder_path = "/mnt/HDD2/phudh/custom-guided-diffusion/"    # (server 118 or 92)
    
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
    
    #---------------- REFINE NET INITIALIZE -----------------------
    checkpoint = th.load(repo_folder_path + 'train_with_timestep_from_1_to_400/refine_net_epoch_499.pth')
    refine_net = PixelCrossAttentionRefiner(feat_dim=3, embed_dim=3, num_heads=3).to(dist_util.dev())  
    refine_net.load_state_dict(checkpoint['model_state_dict'])

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        # sample_fn = (
        #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        # )
        sample_fn = diffusion.ddcm_sample_loop
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            codebooks=_codebooks,

            user_role=Transmitter(),

            # user_role=Receiver('/mnt/HDD2/phu2/custom-guided-diffusion/compressed_info/compressed_representation_date_20250701_time_1630.json'),
            # hq_img_path="/mnt/HDD2/phu2/custom-guided-diffusion/hq_img/imagenet-256/academic_gown/000.jpg",

            # user_role=Receiver('/mnt/HDD2/phu2/custom-guided-diffusion/compressed_info/compressed_representation_date_20250701_time_1648.json'),
            # hq_img_path="/mnt/HDD2/phu2/custom-guided-diffusion/hq_img/imagenet-256/academic_gown/004.jpg",

            # -----------------
            # user_role=Receiver('/mnt/HDD2/phu2/custom-guided-diffusion/compressed_info/compressed_representation_date_20250701_time_1703.json'),
            # hq_img_path='/mnt/HDD2/phu2/custom-guided-diffusion/hq_img/CelebDataProcessed/Jennifer Lopez/8.jpg',

            # user_role=Receiver('/mnt/HDD2/phu2/custom-guided-diffusion/compressed_info/compressed_representation_date_20250701_time_1709.json'),
            # hq_img_path='/mnt/HDD2/phu2/custom-guided-diffusion/hq_img/CelebDataProcessed/Leonardo DiCaprio/20.jpg',

            # user_role=Receiver(''),
            hq_img_path='/mnt/HDD2/phu2/custom-guided-diffusion/hq_img/CelebDataProcessed/Barack Obama/15.jpg',

            # -----------------

            # noise_refine_model=refine_net,
            noise_refine_model=None

        )

        # for i, noise in enumerate(diffusion.refine_noise_list):
            # if i < len(diffusion.refine_noise_list) - 1:
                # print('i: ', i, ' difference between noise: ', th.nn.L1Loss()(diffusion.refine_noise_list[i], diffusion.refine_noise_list[i+1]))

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Inference time: {elapsed_time:.6f} seconds")

    start_time = time.perf_counter()

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        out_filename =  f"samples_{shape_str}"
        out_path = os.path.join(repo_folder_path + "npz_output", out_filename + ".npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
        
        # output to png
        data = np.load(out_path)
        images = data['arr_0'][0]
        print('images.shape: ', images.shape)
        images = Image.fromarray(images)
        images.save(repo_folder_path + f"png_output/" + out_filename + datetime.now().strftime("_date_%Y%m%d_time_%H%M") + ".png")

    dist.barrier()
    logger.log("sampling complete")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Export time: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()
