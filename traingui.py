# noinspection PyPackageRequirements
from gooey import GooeyParser, Gooey
import subprocess


@Gooey(clear_before_run=True, default_size=(993, 519))
def main():
    parser = GooeyParser(description="Train yolo v5")
    weights_group = parser.add_argument_group("Weights")
    weights_group.add_argument(
        "--weights",
        type=str,
        help="initial weights path",
        widget="FileChooser",
    )
    weights_group.add_argument(
        "--resume",
        type=str,
        help="resume most recent training",
        widget="FileChooser",
    )
    parser.add_argument(
        "data",
        type=str,
        default="data/coco128.yaml",
        help="data.yaml path",
        widget="FileChooser",
    )
    train_group = parser.add_argument_group("Train")
    train_group.add_argument(
        "batch", type=int, default=16, help="total batch size for all GPUs"
    )
    train_group.add_argument("epochs", type=int, default=300)
    parser.add_argument(
        "--nosave", action="store_true", help="only save final checkpoint"
    )
    parser.add_argument("--notest", action="store_true", help="only test final epoch")
    parser.add_argument("--evolve", action="store_true", help="evolve hyperparameters")

    args = parser.parse_args()
    result = "python train.py"
    result += " --data " + args.data
    result += " --batch " + str(args.batch)
    result += " --epochs " + str(args.epochs)

    if args.nosave:
        result += " --nosave"

    if args.notest:
        result += " --notest"

    if args.evolve:
        result += " --evolve"

    if args.resume is not None:
        result += " --resume " + args.resume

    if args.weights is not None:
        result += " --weights " + args.weights

    print(args)
    process = subprocess.Popen(result.split(), stdout=subprocess.PIPE)
    process.communicate()


if __name__ == "__main__":
    main()
