# Pytorch ROS

## Comparison of runtime in seconds
| Model | Tensorflow | Pytorch |
| resnet50 | 0.09365 | 0.002186 |
| densenet121 | 0.002448 | 0.0008168 |

## Commands to be run in separate terminals

```sh
roscore

rosrun rug_deep_feature_extraction multi_view_RGBD_object_representation.py resnet50

roslaunch rug_kfold_cross_validation kfold_cross_validation_RGBD_deep_learning_descriptor.launch base_network:="resnet50"
```

## To install
- pip install torch torchvision torchaudio tqdm 

## Notes for the Keras code
- enabling tf.config.optimizer.set_jit(True) makes it a bit faster
- disable all outputs and replacing them with tqdm is also a bit more convinient
- give Prof proper experiment runner file for next year

