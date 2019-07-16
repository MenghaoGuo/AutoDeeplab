CUDA_VISIBLE_DEVICES=0,1,2,3 python train_autodeeplab.py --backbone resnet --lr 0.007 --workers 4 --epochs 40 --batch_size 2 --gpu_ids 0,1,2,3 --eval_interval 1 --dataset cityscapes
