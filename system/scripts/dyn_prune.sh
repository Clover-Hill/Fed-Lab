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
    --alpha 1 \
    --learning_rate_decay True \
    --learning_rate_decay_gamma 0.98 \
    --prune True \
    --pruning_percentage 0.4 \
    --use_wandb True \
    --wandb_project Fed-Lab \
    --wandb_run feddyn-resnet-cifar-prune-0.4 \