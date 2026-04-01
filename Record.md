nohup python -m torch.distributed.launch --use-env --nproc_per_node=4 main_dino.py --arch vit_small --data_path /video_ssd/lpm/ImageNet/train --output_dir ./vit_s_224 --epochs=1000 &> vit_s_224.out &

nohup MASTER_PORT=29501 python -m torch.distributed.launch --master_port=0 --use-env --nproc_per_node=1 main_dino.py --arch vit_small --data_path /video_ssd/lpm/ImageNet/train --output_dir ./vit_s_224_debug &> vit_s_224.out &

nohup MASTER_PORT=29501 python -m torch.distributed.launch --master_port=0 --use-env --nproc_per_node=1 main_dino.py --arch vit_small --data_path /video_ssd/lpm/ImageNet/train --output_dir ./vit_s_224_debug &> vit_s_224.out &


python -m torch.distributed.launch --master-port=29501 --use-env --nproc_per_node=1 main_dino.py --arch vit_small --data_path /video_ssd/lpm/ImageNet/train --output_dir ./vit_s_224_debug


MASTER_PORT=29501 nohup python -m torch.distributed.launch --use-env --nproc_per_node=4 main_dino.py --epochs=1000 --batch_size_per_gpu 128 --arch resnet50 --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path /video_ssd/lpm/ImageNet/train --output_dir ./exps/resnet50_vanilla &> resnet50_vanilla.out &


MASTER_PORT=29503 python -m torch.distributed.launch --master_port 29503 --use-env --nproc_per_node=1 main_dino.py --epochs=1000 --batch_size_per_gpu 128 --arch flow --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path /video_ssd/lpm/ImageNet/train --output_dir ./exps/flow_vanilla

nohup python -m torch.distributed.launch --master_port 29503 --use-env --nproc_per_node=4 main_dino.py --epochs=1000 --batch_size_per_gpu 128 --arch flow --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path /video_ssd/lpm/ImageNet/train --output_dir ./exps/flow_vanilla &> flow_vanilla.out &