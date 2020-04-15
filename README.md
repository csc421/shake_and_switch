## Switchable-Shake Normalization
For further details you can read our [report](http://www.cs.toronto.edu/~sajadn/sajad_norouzi/CSC2516.pdf)
The model has been added to tensor2tensor models so to run the model you only need to change the model name. You can also turn weight_lower_bound on and off by --weight_lower_bound.

python  tensor2tensor/bin/t2t_trainer.py   --generate_data   --data_dir=~/t2t_data   --output_dir=./outputs/cifar10_switchable_0/   --problem=image_cifar10   --model=switchable_resnet   --hparams_set=resnet_cifar_32   --train_steps=120000 --eval_steps=1000 --eval_throttle_seconds=1 --weight_lower_bound
