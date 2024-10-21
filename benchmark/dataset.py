import os
import random
import pyperf
from docetl.dataset import Dataset


def add_cmdline_args(cmd, args):
    if args.input_file:
        cmd.extend(("--input-file", args.input_file))

    return cmd


def run(file_path: str):
    t0 = pyperf.perf_counter()
    dataset = Dataset(None, "file", file_path)
    data = dataset.load()
    dataset.sample(random.randint(1, len(data)), random=False)
    t1 = pyperf.perf_counter()
    return t1 - t0


def main():
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.argparser.add_argument(
        "--input-file",
        required=True,
        help="File path to benchmark",
    )
    args = runner.argparser.parse_args()

    # ensure file path exists
    if not os.path.exists(args.input_file):
        raise ValueError(f"File path {args.input_file} does not exist")

    benchmark_name = os.path.basename(args.input_file)
    runner.bench_func(
        f"dataset-class-benchmark::{benchmark_name}",
        run,
        args.input_file,
    )


if __name__ == "__main__":
    main()
