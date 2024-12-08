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
    --use_wandb True \
    --wandb_project Fed-Lab \
    --wandb_run feddyn-cnn-mnist-pat-decay \