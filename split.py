import os
import pandas as pd
from sklearn.model_selection import train_test_split

folder_path = "folder"
mask_folder_path = "sam_masks"


files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
print("first names:", files[:5])

#group the image
image_groups = {f: f.split(".rf.")[0] for f in files}
unique_groups = sorted(set(image_groups.values()))

#splitting the group
dev_groups, eval_groups = train_test_split(unique_groups, test_size=0.2, random_state=42)

# 3. each image follows its group
dev_files  = [f for f in files if image_groups[f] in dev_groups]
eval_files = [f for f in files if image_groups[f] in eval_groups]


# 4. save and check
df = pd.DataFrame({"filename": dev_files + eval_files,
                   "split": ["dev"] * len(dev_files) + ["eval"] * len(eval_files)})
df.to_csv("split.csv", index=False)

print(df["split"].value_counts())


# 5. every image should have a SAM mask
missing = [f for f in files
           if not os.path.exists(os.path.join(mask_folder_path, os.path.splitext(f)[0] + ".png"))]
print("images without masks:", missing)