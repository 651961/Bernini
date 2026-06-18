cd /datasets/codes_zsqiao/image_gen/Bernini

NCCL_NET_PLUGIN=none NCCL_DEBUG=WARN FSDP=1 \
torchrun --standalone --nproc-per-node 8 \
    /datasets/codes_zsqiao/image_gen/Bernini/infer_multi_gpu.py \
    --config /models/Bernini-Diffusers \
    --ulysses 8 \
    --num_frames 81 \
    --max_image_size 842 \
    --num_inference_steps 40 \
    --height 1280 --width 720 \
    --guidance_mode vae_txt_vit_wapg \
    --flow_shift 5.0 \
    --seed 42 \
    --fps 16 \
    --omega_txt 4 \
    --omega_tgt 1.5 \
    --omega_img 4.5 \
    --omega_vid 1.25 \
    --omega_scale 0.8 \
    --vit_denoising_step 5 \
    --vit_txt_cfg 1.2 \
    --vit_img_cfg 1.0 \
    --case /datasets/codes_zsqiao/image_gen/Bernini/assets/testcases/ri2v/case4/ri2v.json
