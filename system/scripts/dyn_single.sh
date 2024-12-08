CUDA_DEVICE=1

python -m \
    main \
    --dataset MNIST-noniid-pat \
    --model CNN \
    --algorithm FedDyn \
    --global_rounds 250 \
    --batch_size 800 \
    --batch_num_per_client 40 \
    --device_id ${CUDA_DEVICE} \
    --alpha 1.0 \
    --learning_rate_decay True \
    --learning_rate_decay_gamma 0.98 \
    --use_wandb True \
    --wandb_project Fed-Lab \
    --wandb_run feddyn-cnn-mnist-pat-decay \