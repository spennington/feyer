import argparse

import trainer


def get_args():
    """Define the task arguments with the default values.
    Returns:
        experiment parameters
    """
    args_parser = argparse.ArgumentParser()


    # Experiment arguments
    args_parser.add_argument(
        '--batch-size',
        help='Batch size for each training and evaluation step.',
        type=int,
        default=256)
    args_parser.add_argument(
        '--num-epochs',
        help="""\
        Maximum number of training data epochs on which to train.
        If both --train-size and --num-epochs are specified,
        --train-steps will be: (train-size/train-batch-size) * num-epochs.\
        """,
        default=10,
        type=int,
    )
    args_parser.add_argument(
        '--device',
        help='Device to run on. Values are cuda, mps, cpu, auto. Behavior of auto is not optimized.',
        default='auto',
        type=str,
    )

    # Estimator arguments
    args_parser.add_argument(
        '--learning-rate',
        help='Learning rate value for the optimizers.',
        default=1e-3,
        type=float)
    args_parser.add_argument(
        '--weight-decay',
        help='Optimizer L2 decay.',
        default=1e-2,
        type=float)

    # Network size arguments
    args_parser.add_argument(
        '--embedding-dimensions',
        help='Number of dimensions in the embedding layer',
        default=16,
        type=int)
    args_parser.add_argument(
        '--hidden-layer-size',
        help='Number of neurons in the hidden layer.',
        default=32,
        type=int)


    # Saved model arguments
    args_parser.add_argument(
        '--output-folder',
        required=True,
        help='The name of the folder where to save the model and training results in.')
    args_parser.add_argument(
        '--model-name',
        required=True,
        help='The name of your saved model')

    return args_parser.parse_args()


def main():
    """Setup / Start the experiment
    """
    args = get_args()
    print(args)
    trainer.run(args)


if __name__ == '__main__':
    main()