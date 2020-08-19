import argparse
import os
from run_decagon_toy import RunDecagonToy
from run_decagon_real import RunDecagonReal
from constants import PARAMS, INPUT_FILE_PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess data downloaded from Phantasus')
    parser.add_argument('--no-log', default=False,
                        action='store_true',
                        help='Whether to log run or nor, default True')
    parser.add_argument('--real', default=False,
                        action='store_true',
                        help='Run on real data or toy example')
    parser.add_argument('--batch-size', default=4, type=int,
                        help='Batch size (default is 4)')
    parser.add_argument('--hidden1', default=64, type=int,
                        help="Number of neurons on first layer")
    parser.add_argument('--hidden2', default=32, type=int,
                        help="Number of neurons on second layer")
    parser.add_argument('--epoch', default=50, type=int,
                        help="Number of neurons on second layer")

    args = parser.parse_args()

    if args.no_log:
        import neptune
        neptune.init('Pollutants/sandbox')


    PARAMS['epochs'] = args.epoch
    PARAMS['hidden1'] = args.hidden1
    PARAMS['hidden2'] = args.hidden2
    PARAMS['batch_size'] = args.batch_size

    val_test_size = 0.1
    if args.no_log:
        neptune.create_experiment(name='example_with_parameters',
                                  params=PARAMS,
                                  upload_stdout=True,
                                  upload_stderr=True,
                                  send_hardware_metrics=True,
                                  upload_source_files='**/*.py')

        neptune.set_property("val_test_size", val_test_size)

    if not args.real:
        run = RunDecagonToy()
        run.run(adj_path=None, path_to_split='data/split/toy', val_test_size=val_test_size,
                batch_size=PARAMS['batch_size'], num_epochs=PARAMS['epochs'],
                dropout=PARAMS['dropout'], max_margin=PARAMS['max_margin'],
                print_progress_every=150, no_log=args.no_log)
    else:
        run = RunDecagonReal(combo_path=f'{INPUT_FILE_PATH}/bio-decagon-combo.csv',
                             ppi_path=f'{INPUT_FILE_PATH}/bio-decagon-ppi.csv',
                             mono_path=f'{INPUT_FILE_PATH}/bio-decagon-mono.csv',
                             targets_path=f'{INPUT_FILE_PATH}/bio-decagon-targets-all.csv',
                             min_se_freq=500)
        run.run(path_to_split='data/split/real', val_test_size=val_test_size, batch_size=PARAMS['batch_size'],
                num_epochs=PARAMS['epochs'], dropout=PARAMS['dropout'], max_margin=PARAMS['max_margin'],
                print_progress_every=150, adj_path='data/adj/real', no_log=args.no_log)
    if args.no_log:
        neptune.stop()
