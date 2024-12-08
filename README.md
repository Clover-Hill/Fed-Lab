# Federated Learning - Lab
by Maximus Cao
---
## Features

- Add **wandb** support for better logging
- Add client distributed training for **Multi-GPU** training
- Add **Adaptive Regularization** technique for FedDyn algorithm
- Add **client pruning** for better communication efficiency

---

## Usage

### WandB support

Use the following flags to specify project and run name

```
--use_wandb True \
--wandb_project ... \
--wandb_run ... \
```

### Multi-GPU training

See script *system/scripts/dyn_multi.sh*

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m \
    main_distributed \
    --dataset MNIST-iid-200 \
    --model CNN \
    --algorithm FedDynMulti \
    --global_rounds 20 \
    --batch_size 1200 \
    --batch_num_per_client 6 \
    --num_clients 200 \
    --alpha 1.0 \
    --learning_rate_decay True \
    --learning_rate_decay_gamma 0.98 \
```

### Adaptive Regularization

See script *system/scripts/dyn_adaptive.sh*

```
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
```

### Client Pruning

Use the following flags to use client pruning:

```
--prune True \
--pruning_percentage 0.4 \
```
