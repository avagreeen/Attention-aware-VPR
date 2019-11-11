# Attention-aware-VPR
run
'python main_casa_loss.py --mode=train --beta=0.0 --lr=0.00001 --batchSize=2 --atten --rect_atten --p_margin=0.5 --loss=impr_triplet --pooling=atten_vlad --mul=3mul --relu=softplus --add_relu --nEpochs=60 --cacheRefreshRate=1000 --margin=0.1 --evalEvery=1 --optim=ADAM --arch=self_define --num_clusters=64 --nGPU=1 --random_crop --dataset=mapillary'
to train a network
