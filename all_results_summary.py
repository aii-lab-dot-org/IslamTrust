import os
import json
import csv
from collections import defaultdict, OrderedDict

def compute_accuracy_per_category_per_subdir(root_dir):
    """
    Computes accuracy per category and total accuracy for each subdirectory path under `root_dir`.
    Expected JSON structure:
        {
            "is_correct": true/false,
            "category": "some_category"
        }
    """
    results = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    all_categories = set()

    # Traverse all directories and collect stats
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        category = data.get("category")
                        is_correct = data.get("is_correct")

                        if category is not None and isinstance(is_correct, bool):
                            rel_path = os.path.relpath(dirpath, root_dir)
                            results[rel_path][category]["total"] += 1
                            if is_correct:
                                results[rel_path][category]["correct"] += 1
                            all_categories.add(category)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    all_categories = sorted(all_categories)  # consistent column order

    # Compute accuracies
    accuracies = OrderedDict()
    for subdir, cat_stats in sorted(results.items()):
        subdir_acc = OrderedDict()
        total_correct = total_total = 0
        for cat in all_categories:
            correct = cat_stats[cat]["correct"]
            total = cat_stats[cat]["total"]
            acc = correct / total if total > 0 else None
            subdir_acc[cat] = acc
            total_correct += correct
            total_total += total
        subdir_acc["TOTAL"] = total_correct / total_total if total_total > 0 else None
        accuracies[subdir] = subdir_acc

    return accuracies, all_categories


def save_accuracies_to_csv(accuracies, all_categories, output_file):
    """
    Saves the accuracy results into a CSV file.
    """
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["Subdirectory"] + all_categories + ["TOTAL"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for subdir, cat_accs in accuracies.items():
            row = {"Subdirectory": subdir}
            for cat in all_categories + ["TOTAL"]:
                val = cat_accs[cat]
                row[cat] = f"{val*100:.2f}%" if val is not None else "N/A"
            writer.writerow(row)


if __name__ == "__main__":
    root_dir = "results"  # your root directory
    output_csv = "accuracy_per_subdir.csv"

    accuracies, all_categories = compute_accuracy_per_category_per_subdir(root_dir)
    save_accuracies_to_csv(accuracies, all_categories, output_csv)

    print(f"\n✅ Accuracy results saved to '{output_csv}'")

    # Optional: print summary to console
    print("\nAccuracy per directory/subdirectory:")
    header = " | ".join(all_categories + ["TOTAL"])
    print(f"\n{'Subdirectory':<40} | {header}")
    print("-" * (42 + 10 * len(all_categories)))

    for subdir, cat_accs in accuracies.items():
        acc_values = [
            f"{(cat_accs[cat]*100):5.2f}%" if cat_accs[cat] is not None else "  N/A "
            for cat in all_categories + ["TOTAL"]
        ]
        print(f"{subdir:<40} | " + " | ".join(acc_values))
