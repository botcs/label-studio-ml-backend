"""
Generate LabelStudio tasks for a given directory by mapping
all images in the directory to a single task per image.
"""
import glob
import json
import sys
import os

all_tasks = []

root_dir = sys.argv[1]

# recursively find all images in the directory
for filename in glob.iglob(root_dir + './**/*.jpg', recursive=True):
    # remove the root directory from the filename
    filename = filename.replace(root_dir, "")
    task = {
        "data": {
            "image":  "/data/local-files/?d=" + filename,
        }
    }
    all_tasks.append(task)

with open("tasks.json", "w") as f:
    json.dump(all_tasks, f, indent=2)

print(f"Generated {len(all_tasks)} tasks")


