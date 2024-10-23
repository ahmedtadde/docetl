import os
import random
import tempfile
from typing import Optional
from diskcache import Cache, FanoutCache
import pyperf
from docetl.dataset import Dataset


def add_cmdline_args(cmd, args):
    if args.dir:
        cmd.extend(("--dir", args.dir))
    if args.samples:
        cmd.extend(("--samples", args.samples))
    return cmd


def run(dir_path: str, samples: str, cache_store: Optional[Cache] = None):
    files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
        and f.lower().endswith((".csv", ".json"))
    ]
    t0 = pyperf.perf_counter()
    for file in files:
        dataset = Dataset(None, "file", file, cache_store=cache_store)
        sample_size = len(dataset.load())
        for _ in range(int(samples)):
            dataset.sample(random.randint(1, sample_size), random=False)
    return pyperf.perf_counter() - t0


def main():
    try:
        runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
        runner.argparser.add_argument(
            "--dir",
            required=True,
            help="Directory with files to benchmark",
        )
        runner.argparser.add_argument(
            "--samples",
            required=True,
            help="Number of samples to take from each file",
        )
        args = runner.parse_args()

        # ensure dir path exists
        if not os.path.isdir(args.dir):
            raise ValueError(f"Directory {args.dir} does not exist")

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_cache_dir = os.path.join(temp_dir, "cache")
            os.makedirs(dataset_cache_dir, exist_ok=True)
            with FanoutCache(directory=dataset_cache_dir) as datasets_disk_cache:
                benchmark_name = os.path.basename(args.dir)
                runner.bench_func(
                    f"dataset-class-benchmark::{benchmark_name}",
                    run,
                    args.dir,
                    args.samples,
                    datasets_disk_cache,
                )

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
