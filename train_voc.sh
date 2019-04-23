CUDA_VISIBLE_DEVICES=0,1,2,3 python train_autodeeplab.py --backbone resnet --lr 0.007 
--workers 4 --use-sbd True --epochs 50 --batch-size 2 --gpu-ids 0,1,2,3 --checkname 
deeplab-resnet --eval-interval 1 --dataset pascal
