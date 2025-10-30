import os
import argparse
import pprint
import utils
from typing import Optional

# python .\show_run_config.py --options "e:\vscode\HiDDeN\experiments\combined-noise\options-and-config.pickle"
def resolve_options_path(run_folder: Optional[str], options_file: Optional[str], checkpoint_file: Optional[str]) -> str:
    if options_file:
        return options_file
    if run_folder:
        return os.path.join(run_folder, 'options-and-config.pickle')
    if checkpoint_file:
        run = os.path.dirname(os.path.dirname(checkpoint_file))
        return os.path.join(run, 'options-and-config.pickle')
    raise ValueError('One of --run-folder, --options, or --checkpoint must be provided')


def main() -> None:
    parser = argparse.ArgumentParser(description='Show HiDDeN run configuration from options-and-config.pickle')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-folder', '-r', type=str, help='Run folder containing options-and-config.pickle')
    group.add_argument('--options', '-o', type=str, help='Path to options-and-config.pickle')
    group.add_argument('--checkpoint', '-c', type=str, help='Path to a checkpoint .pyt file under the run folder')
    args = parser.parse_args()

    options_path = resolve_options_path(args.run_folder, args.options, args.checkpoint)
    if not os.path.exists(options_path):
        raise FileNotFoundError(f'Options file not found: {options_path}')

    train_options, hidden_config, noise_config = utils.load_options(options_path)

    print('== TrainingOptions ==')
    print(pprint.pformat(vars(train_options)))
    print('\n== HiDDenConfiguration ==')
    print(pprint.pformat(vars(hidden_config)))
    print('\n== noise_config ==')
    print(pprint.pformat(noise_config))


if __name__ == '__main__':
    main()
