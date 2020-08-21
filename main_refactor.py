import argparse
import os
from run_decagon_toy import RunDecagonToy
from run_decagon_real import RunDecagonReal
from constants import PARAMS, INPUT_FILE_PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=False,
                        action='store_true',
                        help='Whether to log run or nor, default True')
    parser.add_argument('--real', default=False,
                        action='store_true',
                        help='Run on real data or toy example')
    parser.add_argument('--batch-size', default=PARAMS['batch_size'], type=int,
                        help='Batch size')
    parser.add_argument('--hidden1', default=PARAMS['hidden1'], type=int,
                        help="Number of neurons on first layer")
    parser.add_argument('--hidden2', default=PARAMS['hidden2'], type=int,
                        help="Number of neurons on second layer")
    parser.add_argument('--epoch', default=PARAMS['epoch'], type=int,
                        help="Number of neurons on second layer")
    parser.add_argument('--cpu', default=False,
                        action='store_true',
                        help='Run on cpu instead of gpu')

    args = parser.parse_args()

    if args.log:
        import neptune
        neptune.init('Pollutants/sandbox')

    PARAMS['epoch'] = args.epoch
    PARAMS['hidden1'] = args.hidden1
    PARAMS['hidden2'] = args.hidden2
    PARAMS['batch_size'] = args.batch_size

    val_test_size = 0.1
    if args.log:
        neptune.create_experiment(name='example_with_parameters',
                                  params=PARAMS,
                                  upload_stdout=True,
                                  upload_stderr=True,
                                  send_hardware_metrics=True,
                                  upload_source_files='**/*.py')

        neptune.set_property("val_test_size", val_test_size)

    if not args.real:
        run = RunDecagonToy()
        run.run(adj_path=None, path_to_split=f'data/split/toy/{PARAMS["batch_size"]}',
                val_test_size=val_test_size,
                batch_size=PARAMS['batch_size'], num_epochs=PARAMS['epoch'],
                dropout=PARAMS['dropout'], max_margin=PARAMS['max_margin'],
                print_progress_every=150, log=args.log, on_cpu=args.cpu)
    else:
        run = RunDecagonReal(combo_path=f'{INPUT_FILE_PATH}/bio-decagon-combo.csv',
                             ppi_path=f'{INPUT_FILE_PATH}/bio-decagon-ppi.csv',
                             mono_path=f'{INPUT_FILE_PATH}/bio-decagon-mono.csv',
                             targets_path=f'{INPUT_FILE_PATH}/bio-decagon-targets-all.csv',
                             min_se_freq=500, min_se_freq_mono=40)
        run.run(path_to_split=f'data/split/real/{PARAMS["batch_size"]}',
                val_test_size=val_test_size, batch_size=PARAMS['batch_size'],
                num_epochs=PARAMS['epoch'], dropout=PARAMS['dropout'],
                max_margin=PARAMS['max_margin'],
                print_progress_every=150, adj_path='data/adj/real',
                log=args.log, on_cpu=args.cpu)
    if args.log:
        neptune.stop()
