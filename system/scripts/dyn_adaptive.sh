CUDA_DEVICE=1

python -m \
    main \
    --dataset Cifar10-noniid-pat \
    --model ResNet34 \
    --algorithm FedDyn \
    --global_rounds 250 \
    --batch_size 800 \
    --batch_num_per_client 40 \
    --device_id ${CUDA_DEVICE} \
    --adaptive True \
    --alpha 1 \
    --alpha_upper 10 \
    --learning_rate_decay True \
    --learning_rate_decay_gamma 0.98 \
    --use_wandb True \
    --wandb_project Fed-Lab \
    --wandb_run feddyn-resnet-cifar-adaptive-10 \