CUDA_DEVICE=0

python -m \
    system.main \
    --dataset MNIST-nonidd-pat \
    --model CNN \
    --algorithm FedAvg \
    --batch_size 200 \
    --global_rounds 1000 \
    --device_id ${CUDA_DEVICE} \