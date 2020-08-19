import argparse
from run_decagon_toy import RunDecagonToy
from run_decagon_real import RunDecagonReal
import neptune

neptune.init('Pollutants/sandbox')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess data downloaded from Phantasus')
    parser.add_argument('--no_log', default=True,
                        action='store_true',
                        help='Whether to log run or nor, default True')
    parser.add_argument('--real', default=False,
                        help='Run on real data or toy example')

    args = parser.parse_args()
    decagon_data_file_directory = 'data/input'

    PARAMS = {'neg_sample_size': 1,
              'learning_rate': 0.001,
              'epochs': 50,
              'hidden1': 64,
              'hidden2': 32,
              'weight_decay': 0,
              'dropout': 0.1,
              'max_margin': 0.1,
              'batch_size': 512,
              'bias': True}
    val_test_size = 0.1
    if not args.no_log:
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
        run = RunDecagonReal(combo_path=f'{decagon_data_file_directory}/bio-decagon-combo.csv',
                             ppi_path=f'{decagon_data_file_directory}/bio-decagon-ppi.csv',
                             mono_path=f'{decagon_data_file_directory}/bio-decagon-mono.csv',
                             targets_path=f'{decagon_data_file_directory}/bio-decagon-targets-all.csv',
                             min_se_freq=500)
        run.run(path_to_split='data/split/real', val_test_size=val_test_size, batch_size=PARAMS['batch_size'],
                num_epochs=PARAMS['epochs'], dropout=PARAMS['dropout'], max_margin=PARAMS['max_margin'],
                print_progress_every=150, adj_path='data/adj/real', no_log=args.no_log)
    if not args.no_log:
        neptune.stop()
