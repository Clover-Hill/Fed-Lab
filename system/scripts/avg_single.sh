CUDA_DEVICE=2

python -m \
    main \
    --dataset MNIST \
    --model CNN \
    --algorithm FedAvg \
    --global_rounds 1000 \
    --batch_size 800 \
    --batch_num_per_client 40 \
    --device_id ${CUDA_DEVICE} \
    --use_wandb True \
    --wandb_project Fed-Lab \
    --wandb_run fedavg-cnn-mnist-iid \