CUDA_DEVICE=0

python -m \
    main \
    --dataset MNIST-nonidd-pat \
    --model CNN \
    --algorithm FedDyn \
    --global_rounds 1000 \
    --batch_size 800 \
    --batch_num_per_client 40 \
    --device_id ${CUDA_DEVICE} \
    --alpha 1.0 \
    --use_wandb True \
    --wandb_project Fed-Lab \
    --wandb_run feddyn-cnn-mnist-pat \