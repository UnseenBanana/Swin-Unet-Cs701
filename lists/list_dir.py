import os
import re


def natural_sort_key(s):
    # Split the string into case number and slice number
    case_match = re.match(r"case(\d+)_slice(\d+)", s)
    if case_match:
        case_num = int(case_match.group(1))
        slice_num = int(case_match.group(2))
        # Return a tuple that will be used for sorting
        return (case_num, slice_num)
    return s


dir_name = "./datasets/cs701_224/val_npz"
file_name = "./lists/lists_cs701/val.txt"

slices = []

for f in os.listdir(dir_name):
    # remove .npz extension
    f = f.split(".")[0]
    slices.append(f)

# Sort using the natural sort key function
slices.sort(key=natural_sort_key)

with open(file_name, "w") as file:
    for s in slices:
        file.write(f"{s}\n")
