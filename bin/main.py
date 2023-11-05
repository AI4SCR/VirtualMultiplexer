import argparse
import yaml
import torch
import mlflow
import time

from train import trainer
from test import Test
from i2iTranslation.constant import define_constants
from i2iTranslation.utils.util import read_mlflow, robust_mlflow, flatten, set_seed, start_train_experiment, start_test_experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parameters
    parser.add_argument("--base_path",
                        type=str,
                        help='base directory of images, tissue masks, and model outcomes')
    parser.add_argument("--i2i_config_path",
                        type=str,
                        help='full path to config yaml file. sample configs are available in ./configs')
    parser.add_argument("--src_marker",
                        type=str,
                        help="name of the source marker",
                        default='HE')
    parser.add_argument("--dst_marker",
                        type=str,
                        help="name of the destination/target marker",
                        default='NKX3')
    parser.add_argument("--is_train",
                        type=eval,
                        default=False)
    parser.add_argument("--is_test",
                        type=eval,
                        default=False)
    parser.add_argument("--mlflow_experiment",
                        type=str,
                        help="name of the mlflow experiment")

    args = parser.parse_args()
    args = vars(args)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #### Train I2I: GAN loss + StyleContent loss
    if args['is_train']:
        # start experiment
        start_time = time.time()
        experiment_id, run_id = start_train_experiment(args)

        # define paths and constants
        constants = define_constants(**args)
        for key, val in constants.items():
            if key not in args.keys():
                args[key] = val

        # load i2i configs
        with open(constants['config_path']) as f:
            config = yaml.safe_load(f)
            config['test']['experiment_id'] = experiment_id
            config['test']['run_id'] = run_id

        # finalize args dict
        args.update(config)
        args = flatten(args)

        # log parameters
        robust_mlflow(mlflow.log_params, args)

        # set device seed
        set_seed(device, args['train.params.seed'])

        # run trainer
        trainer(args, device)

        print('\n\nTraining completed \n\n\n')
        robust_mlflow(mlflow.log_metric, 'total_training_time', round(time.time() - start_time, 2))

    #### Test
    if args['is_test']:
        start_time = time.time()

        if mlflow.active_run() is None:
            # load test i2i config
            with open(args['i2i_config_path']) as f:
                config = yaml.safe_load(f)

            # read train experiment information from mlflow
            run_params = read_mlflow(args, mlflow_run_id=config['test']['run_id'])

            # merge parameters
            args.update(run_params)
            # override mlflow test default params with test config params
            for key, val in flatten(config).items():
                args[key] = val

            # start test experiment
            start_test_experiment(args)

        args['is_train'] = False
        args['is_test'] = True

        # run tester
        test_obj = Test(args, device)
        test_obj.tester()

        print('\n\nTesting completed \n\n\n\n')
        robust_mlflow(mlflow.log_metric, 'total_testing_time', round(time.time() - start_time, 2))