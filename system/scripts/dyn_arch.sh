CUDA_DEVICE=0

python -m \
    main \
    --dataset MNIST-iid-200 \
    --model CNN \
    --algorithm FedDyn \
    --global_rounds 2000 \
    --batch_size 800 \
    --num_clients 200 \
    --batch_num_per_client 4 \
    --device_id ${CUDA_DEVICE} \
    --alpha 1 \
    --learning_rate_decay True \
    --learning_rate_decay_gamma 0.99 \
    --use_wandb True \
    --wandb_project Fed-Lab \
    --wandb_run feddyn-cnn-mnist-200-base \