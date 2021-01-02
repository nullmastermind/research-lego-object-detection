import os

labels = []
dict_labels = {}

with open("labels.txt", "r") as f:
    lines = f.read().split("\n")
    for line in lines:
        label = str(line.strip())
        labels.append(label)
        dict_labels[label] = True
    f.close()


def save_label(label, save_dir):
    copy_label = False

    if not (label in dict_labels):
        copy_label = True
        labels.append(label)
        dict_labels[label] = True
        with open("labels.txt", "w") as f:
            f.write("\n".join(labels))
            f.close()

    copy_label_to = "{}/labels.txt".format(save_dir)
    if (not os.path.isfile(copy_label_to)) or copy_label:
        with open(copy_label_to, "w") as f:
            f.write("\n".join(labels))
            f.close()
        print("updated labels")
