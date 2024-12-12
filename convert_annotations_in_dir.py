import os
from argparse import ArgumentParser
from typing import List, Tuple

from PIL import Image


def resize_bbox(
    bbox: List[int], original_size: Tuple[int, int], target_size: Tuple[int, int]
):
    """
    Converts and resizes from
    Args:
        bbox:
        original_size:
        target_size:

    Returns:

    """
    l = bbox[0]
    t = bbox[1]
    a = bbox[2]
    b = bbox[3]
    return [
        max((l / original_size[0]) * target_size[0], 0),
        max((t / original_size[1]) * target_size[1], 0),
        min((a / original_size[0]) * target_size[0], target_size[0]),
        min((b / original_size[1]) * target_size[1], target_size[1]),
    ]


def process_image(
    img_path: str,
    annotation_path_in: str,
    annotation_path_out: str,
    target_size: Tuple[int, int] = (640, 640),
) -> None:
    with Image.open(img_path) as img:
        img_orig_size = img.size
    new_annotation_lines = []
    print(f"read {annotation_path_in}")
    with open(annotation_path_in) as fp:
        split_lines = [line.split(",") for line in fp.readlines()]
        for elements in split_lines:
            bbox = [int(x) for x in elements[1:5]]
            resized_bbox = resize_bbox(bbox, img_orig_size, target_size)
            updated_elements = (
                elements[:1] + [str(x) for x in resized_bbox] + elements[5:]
            )
            new_annotation_lines.append(",".join(updated_elements))
    with open(annotation_path_out, "w") as fp:
        fp.writelines(new_annotation_lines)
    print(f"saved {annotation_path_out}")


def process_dir(input_dir: str, output_dir: str) -> None:
    img_paths = [
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.split(".")[-1].lower().lower() in ["jpg", "jpeg", "png", "ppm"]
    ]
    for img_path in img_paths:
        annotation_file_name = f"{os.path.basename(img_path).lower()}.txt"
        process_image(
            img_path,
            os.path.join(input_dir, annotation_file_name),
            os.path.join(output_dir, annotation_file_name)
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dir_in")
    parser.add_argument("dir_out")
    args = parser.parse_args()

    process_dir(args.dir_in, args.dir_out)
