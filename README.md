# Attention-aware-VPR
## Train
run
`python main.py --mode=train --beta=0.0 --lr=0.00001 --batchSize=2 --atten --rect_atten --p_margin=0.5 --loss=impr_triplet --pooling=atten_vlad --mul=3mul --relu=softplus --add_relu --nEpochs=60 --cacheRefreshRate=1000 --margin=0.1 --evalEvery=1 --optim=ADAM --arch=self_define --num_clusters=64 --nGPU=1 --random_crop --dataset=mapillary`

## Evaluation
run 
`python main.py --mode=test --ckpt=best --split=test --resume=./runs/Jun24_05-55-35_-1_3mul --batchSize=2 --atten --rect_atten --p_margin=0.5 --loss=impr_triplet --pooling=atten_vlad --mul=3mul --relu=softplus --add_relu --margin=0.1 --arch=self_define --num_clusters=64 --dataset=mapillary`

to evaluate on beeldbank dataset, modify `--split=beelbank`

# With domain adaptation

run
`python mkmmd_main.py --mode=train --loss=impr_triplet --beta=0.0 --nGPU=1 --lr=0.00001 --alpha=0.99 --batchSize=2 --mul=3muk --DA --atten --rect_atten --pooling=atten_vlad --relu=softplus --add_relu --nEpochs=60 --cacheRefreshRate=1000 --margin=0.1 --evalEvery=1 --optim=ADAM --arch=self_define --num_clusters=64 --random_crop --dataset=mapillary`

## Dataset

The root_dir to dataset is `../data`
