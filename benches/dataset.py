import os
import json
import csv
import random
import string
import shutil
import argparse
from typing import List, Dict
import numpy as np
from line_profiler import LineProfiler
from docetl.dataset import Dataset


def generate_random_data(target_size: int) -> List[Dict[str, str]]:
    data = []
    current_size = 0

    while current_size < target_size:
        text_length = random.randint(10, 100)
        text = "".join(random.choices(string.ascii_lowercase, k=text_length))

        item = {"id": str(len(data)), "text": text}

        # Each ASCII character is 1 byte
        item_size = len(text)
        data.append(item)
        current_size += item_size

    return data


def create_file(path: str, data: List[Dict[str, str]], file_type: str):
    if file_type == "json":
        with open(path, "w") as f:
            json.dump(data, f)
    elif file_type == "csv":
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)


def generate_input_files(folder: str, min_size: int, max_size: int, file_count: int):
    print(f"Generating input files in {folder}...")
    if not os.path.exists(folder):
        os.makedirs(folder)

    sizes = np.random.normal(
        loc=(min_size + max_size) / 2, scale=(max_size - min_size) / 6, size=file_count
    )
    sizes = np.clip(sizes, min_size, max_size).astype(int)

    for i, size in enumerate(sizes):
        file_type = random.choice(["json", "csv"])
        data = generate_random_data(size)
        file_path = os.path.join(folder, f"file_{i}.{file_type}")
        create_file(file_path, data, file_type)

    print(f"Generated {file_count} files in {folder}.")


def benchmark_load(file_path: str, iterations: int):
    def load_wrapper():
        for _ in range(iterations):
            dataset = Dataset(None, "file", file_path)
            dataset.load()

    profiler = LineProfiler()
    profiler.add_function(Dataset.load)
    profiler_wrapper = profiler(load_wrapper)
    profiler_wrapper()

    return profiler


def run_benchmark(folder: str, iterations: int):
    stats = {}

    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)

        profiler = benchmark_load(file_path, iterations)

        stats[file_name] = profiler.get_stats()

        print(f"File: {file_name}")
        print("Profile:")
        profiler.print_stats()

        # Debug information
        print("\nDebug Info:")
        print(f"Timings type: {type(stats[file_name].timings)}")
        print(f"Timings keys: {stats[file_name].timings.keys()}")
        print(
            f"Timings values type: {type(next(iter(stats[file_name].timings.values())))}"
        )

        # Attempt to calculate total time
        try:
            total_time = sum(stat[2] for stat in stats[file_name].timings.values())

            print(f"Total time: {total_time:.4f}s")
        except (AttributeError, IndexError, TypeError) as e:
            print(
                f"Unable to calculate total time due to unexpected structure of timing data: {e}"
            )

        print("\n" + "=" * 50 + "\n")

    return stats


def save_benchmark_summary(stats, output_file):
    with open(output_file, "w") as f:
        f.write("Benchmark Summary\n")
        f.write("=================\n\n")

        for file_name in stats.keys():
            f.write(f"File: {file_name}\n")
            f.write("-" * (len(file_name) + 6) + "\n")

            try:
                total_time = sum(stat[2] for stat in stats[file_name].timings.values())

                f.write(f"Total time: {total_time:.4f}s\n\n")
            except (AttributeError, IndexError, TypeError) as e:
                f.write(
                    f"Unable to calculate total time due to unexpected structure of timing data: {e}\n\n"
                )

        f.write(
            "Note: Detailed timing information is available in the console output.\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Dataset load method")
    parser.add_argument(
        "--folder",
        default="tmp/dataset_class_benchmark_files",
        required=True,
        help="Folder to store generated files",
    )
    parser.add_argument(
        "--output-file",
        default="dataset_class_benchmark_summary.txt",
        help="File to store benchmark summary",
    )
    parser.add_argument(
        "--delete-folder-and-generate",
        action="store_true",
        help="Delete and regenerate input files",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=4 * 1024,  # 4 KB
        help="Minimum file size in bytes",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=64 * 1024 * 1024,  # 64 MB
        help="Maximum file size in bytes",
    )
    parser.add_argument(
        "--file-count",
        type=int,
        default=10,  # 10 files
        help="Number of files to generate",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,  # 10 iterations
        help="Number of iterations for each load operation",
    )

    args = parser.parse_args()

    if args.delete_folder_and_generate or not os.path.exists(args.folder):
        if os.path.exists(args.folder):
            shutil.rmtree(args.folder)
            print(
                f"Deleted folder: {args.folder} since it already exists and --delete-folder-and-generate was passed."
            )
        generate_input_files(args.folder, args.min_size, args.max_size, args.file_count)

    # set the cache dir to the folder
    os.environ["BENCHMARK_CACHE_DIR"] = args.folder
    stats = run_benchmark(args.folder, args.iterations)
    save_benchmark_summary(stats, args.output_file)
