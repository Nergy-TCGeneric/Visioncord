def load_cocos_class_names(coco_file_name: str) -> "list[str]":
    class_names = []
    with open(coco_file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)

    return class_names