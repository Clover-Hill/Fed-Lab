CUDA_DEVICE=3

python -m \
    main \
    --dataset MNIST \
    --model CNN \
    --algorithm FedAvg \
    --batch_size 1200 \
    --batch_num_per_client 40 \
    --global_rounds 1000 \
    --device_id ${CUDA_DEVICE} \
    --use_wandb True \
    --wandb_project Fed-Lab \
    --wandb_run fedavg-cnn-mnist-iid \