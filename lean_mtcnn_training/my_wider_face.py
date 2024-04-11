from pathlib import Path
import tensorflow_datasets as tfds
import os
from PIL import Image


class MyWiderFace(tfds.core.GeneratorBasedBuilder):
    """WIDER FACE dataset."""

    VERSION = tfds.core.Version('1.0.0')

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("WIDER FACE dataset."),
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),
                "bbox": tfds.features.BBoxFeature(),
            }),
            supervised_keys=("image", "bbox"),
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Replace 'your_custom_url_for_training_data' and 'your_custom_url_for_test_data'
        # with actual URLs where the WIDER FACE dataset can be downloaded.
        urls_to_download = {
            "train": "https://drive.google.com/file/d/15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M/view?pli=1",
            # "validaton": "https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view?usp=sharing",
            "test": "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_test.zip?download=true",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        local_annotations_path = os.path.join(downloaded_files["test"], "wider_face_split")
        print(downloaded_files)

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"images_dir": os.path.join(downloaded_files["train"], "WIDER_train", "images"),
                "annotations_dir": os.path.join(local_annotations_path, "wider_face_train_bbx_gt.txt"),},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={"images_dir": os.path.join(downloaded_files["test"], "WIDER_test", "images"),
                "annotations_dir": os.path.join(local_annotations_path, "wider_face_test_bbx_gt.txt"),},
            ),
        ]


    def _generate_examples(self, images_dir, annotations_dir):
        """Yields examples."""
        with open(annotations_dir, "r") as annotations_file:
            while True:
                image_path_line = annotations_file.readline().strip()
                image_path_line = Path(image_path_line)
                if not image_path_line:  # End of file
                    break
                images_dir = Path(images_dir)
                print(f"Image dir: {images_dir}")
                image_path = os.path.join(images_dir, image_path_line)
                print(f"Processing {image_path}...")

                # Read the number of faces and ensure it's an integer
                num_faces_line = annotations_file.readline().strip()
                num_faces = int(num_faces_line)

                bboxes = []
                for _ in range(num_faces):
                    bbox_line = annotations_file.readline().strip()
                    parts = bbox_line.split()

                    # Convert strings to integers
                    x_min, y_min, width, height = map(int, parts[:4])
                    x_max = x_min + width
                    y_max = y_min + height

                    # Open the image to get its dimensions for normalization
                    with Image.open(image_path) as img:
                        img_width, img_height = img.size

                    # Normalize bbox coordinates
                    bboxes.append(tfds.features.BBox(
                        ymin=y_min / img_height, xmin=x_min / img_width,
                        ymax=y_max / img_height, xmax=x_max / img_width
                    ))

                # Adjust for multiple bboxes if necessary; here's a simplified approach
                if bboxes:
                    yield image_path, {
                        "image": image_path,
                        "bbox": bboxes[0],  # This example takes the first bbox; adjust as needed
                    }
