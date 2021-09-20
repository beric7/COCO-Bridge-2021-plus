#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:1
#SBATCH -t 5:00:00
#SBATCH -p v100_normal_q
#SBATCH -A infraeval
module load Anaconda
source activate TF_GPU_1.10.0_py_3.6
module load gcc cmake
module load cuda/9.0.176 
module load cudnn/7.1

export PYTHONPATH=$PYTHONPATH:/home/beric7/new_code_obj_detector/tensorflow/models/research:/home/beric7/new_code_obj_detector/tensorflow/models/research/slim

cd $PBS_O_WORKDIR

python ~/COCO-Bridge-2020/MODELS/BridgeDetailModels/COCO_VDOT_1470_Sel_Aug/ssd_train.py -SAVE_TRAINING_DATA_HERE '/training_data/var1_original_parameters/' -PATH_TO_MODEL '/home/beric7/COCO-Bridge-2020/MODELS/BridgeDetailModels/COCO_VDOT_1470_Sel_Aug' -VAR 'var1'
