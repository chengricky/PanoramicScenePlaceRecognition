python PlaceRecognitionTrain.py \
--dataset=highway \
--mode=train \
--resume=checkpoints_res \
--savePath=checkpoints_res_0/ \
--ckpt=latest \
--arch=resnet18 \
--numTrain=2 \
--weightDecay=0.001 \
--cacheBatchSize=32 \
--batchSize=3 \
--threads=4 \
--nEpochs=50 \
--start-epoch=22 \
--cacheRefreshRate=100

