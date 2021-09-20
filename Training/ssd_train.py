# original source from Google:
# https://github.com/tensorflow/models/blob/master/research/object_detection/train.py

import functools
import json
import os
import tensorflow as tf
import sys
import argparse

from object_detection import trainer
from object_detection.builders import dataset_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import dataset_util

# module-level variables ##############################################################################################

#######################################################################################################################
# this next comment line is necessary to suppress a false PyCharm warning
# noinspection PyUnresolvedReferences
def train(*args):
    
    args = args[0]
    print(args)
    
    # this is the big (pipeline).config file that contains various directory locations and many tunable parameters
    PIPELINE_CONFIG_PATH = args.PATH_TO_MODEL + "/" + args.VAR + "_ssd_inception_v2_coco.config"
    
    SAVE_TRAINING_DATA_HERE = args.PATH_TO_MODEL + '/' + args.SAVE_TRAINING_DATA_HERE
    
    # number of clones to deploy per worker
    NUM_CLONES = 1
    
    # Force clones to be deployed on CPU.  Note that even if set to False (allowing ops to run on gpu),
    # some ops may still be run on the CPU if they have no GPU kernel
    CLONE_ON_CPU = False
    
    print("starting program . . .")
    
    print(sys.path)

    # show info to std out during the training process
    tf.logging.set_verbosity(tf.logging.INFO)

    configs = config_util.get_configs_from_pipeline_file(PIPELINE_CONFIG_PATH)
    tf.gfile.Copy(PIPELINE_CONFIG_PATH, os.path.join(SAVE_TRAINING_DATA_HERE, 'ssd_pipeline.config'), overwrite=True)

    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']

    model_fn = functools.partial(model_builder.build, model_config=model_config, is_training=True)

    # ToDo: this nested function seems odd, factor this out eventually ??
    # nested function
    def get_next(config):
        return dataset_util.make_initializable_iterator(dataset_builder.build(config)).get_next()
    # end nested function
    print ("1")
    create_input_dict_fn = functools.partial(get_next, input_config)

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)
    print ("2")
    # parameters for a single worker
    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''
    print ("3")
    if cluster_data and 'worker' in cluster_data:
        # number of total worker replicas include "worker"s and the "master".
        worker_replicas = len(cluster_data['worker']) + 1
    # end if
    print ("4")
    if cluster_data and 'ps' in cluster_data:
        ps_tasks = len(cluster_data['ps'])
    # end if
    print ("5")
    if worker_replicas > 1 and ps_tasks < 1:
        raise ValueError('At least 1 ps task is needed for distributed training.')
    # end if
    print ("6")
    if worker_replicas >= 1 and ps_tasks > 0:
        # set up distributed training
        server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc', job_name=task_info.type, task_index=task_info.index)
        if task_info.type == 'ps':
            server.join()
            return
        # end if

        worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
        task = task_info.index
        is_chief = (task_info.type == 'master')
        master = server.target
    # end if
    print ("7")
    trainer.train(create_input_dict_fn, model_fn, train_config, master, task, NUM_CLONES, worker_replicas,
                  CLONE_ON_CPU, ps_tasks, worker_job_name, is_chief, SAVE_TRAINING_DATA_HERE)

#######################################################################################################################
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-PATH_TO_MODEL', type=str,  required=True, help='model path directory')
    parser.add_argument('-SAVE_TRAINING_DATA_HERE', type=str, required=True, help='save to this directory')
    parser.add_argument('-VAR', type=str, required=True, help='variation')
    args = parser.parse_args()
    train(args)
######################################################################################################################
