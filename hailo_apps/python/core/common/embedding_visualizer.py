# region imports
# Standard library imports
import json
import os
import shutil
from datetime import datetime

# Third-party library imports
import numpy as np

try:
    import fiftyone as fo
    import fiftyone.brain as fob
except ImportError:
    print(
        "The 'fiftyone' library is not installed. Please see installation guide here: https://docs.voxel51.com/getting_started/install.html"
    )
    exit(1)

# Local application/library imports
try:
    from hailo_apps.python.core.common.db_handler import DatabaseHandler, Record
    from hailo_apps.python.core.common.defines import (
        FACE_RECON_DATABASE_DIR_NAME,
        FACE_RECON_SAMPLES_DIR_NAME
    )
except ImportError:
    from .db_handler import DatabaseHandler, Record
    from .defines import (
        FACE_RECON_DATABASE_DIR_NAME,
        FACE_RECON_SAMPLES_DIR_NAME
    )


def visualize_embeddings(db_handler):
    """Visualize embeddings from LanceDB using Voxel51's fiftyone.

    Args:
        db_handler: DatabaseHandler instance for database operations
    """
    records = db_handler.tbl_records.to_pandas()

    # Create a FiftyOne dataset
    dataset_name = f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset = fo.Dataset(dataset_name)

    # Create dummy directory for placeholder files
    dummy_dir = "/tmp/dummy_images"
    os.makedirs(dummy_dir, exist_ok=True)

    # Add samples to dataset with embeddings
    for _idx, record in records.iterrows():
        pictures = json.loads(record["samples_json"]) if record["samples_json"] else []
        if pictures and len(pictures) > 0:
            # Create a directory for this person
            person_dir = os.path.join(dummy_dir, str(record["global_id"]))
            os.makedirs(person_dir, exist_ok=True)
            # Process each picture for this person
            for picture_idx, picture in enumerate(pictures):
                image_path = os.path.join(person_dir, f"picture_{picture_idx}.jpg")
                try:
                    if os.path.exists(picture["sample_path"]):
                        shutil.copy2(picture["sample_path"], image_path)
                    else:
                        with open(
                            image_path, "w"
                        ) as f:  # File doesn't exist, write the path as fallback
                            f.write(str(picture["sample_path"]))
                except (TypeError, ValueError):
                    with open(image_path, "w") as f:  # If there's any issue, write whatever we have
                        f.write(str(picture["sample_path"]))
                # Create sample with the actual picture image
                sample = fo.Sample(
                    filepath=image_path,
                    global_id=record["global_id"],
                    name=record["name"],
                    classificaiton_confidence_threshold=str(
                        record["classificaiton_confidence_threshold"]
                    ),
                    picture_id=picture["id"],
                    embedding=np.array(picture["embedding"]),
                )
                dataset.add_sample(sample)

    # Compute the embedding visualization with adjusted parameters
    num_samples = len(dataset)
    if num_samples < 10:  # For small datasets, always use PCA to avoid t-SNE issues
        fob.compute_visualization(
            dataset, embeddings="embedding", method="pca", brain_key="embeddings_viz", pca_dims=2
        )
    else:  # Only use t-SNE for larger datasets
        perplexity = min(num_samples // 3, 30)  # Very conservative perplexity
        fob.compute_visualization(
            dataset,
            embeddings="embedding",
            method="tsne",
            brain_key="embeddings_viz",
            pca_dims=min(num_samples - 1, 10),
            tsne_perplexity=perplexity,
        )

    # Launch the FiftyOne App to visualize
    try:
        session = fo.launch_app(dataset)
        print(
            "You can edit the following fields in the app:\n1. Name - In case name is 'Uknown'\n2. Confidence threshold\nChanges will be saved back to the database when you close the app."
        )
        session.wait()
    except Exception as e:
        print(f"Error launching FiftyOne: {e}")
    except KeyboardInterrupt:
        print("\nSaving changes & closing session.")
    finally:  # Ctrl + c
        save_fiftyone_changes_to_lancedb(dataset, db_handler)  # Save changes back to LanceDB
        shutil.rmtree(dummy_dir, ignore_errors=True)
    return dataset


def save_fiftyone_changes_to_lancedb(dataset, db_handler):
    """Save changes made in FiftyOne back to LanceDB: classificaiton_confidence_threshold and name only if the previous name was "Unknown".
    Strong assumption: the name and threshold must be changed to same value for all global_id samples.

    Args:
        dataset: FiftyOne dataset with changes
        db_handler: DatabaseHandler instance for database operations
    """
    original_samples = {sample.id: sample.to_dict() for sample in dataset}
    for _sample_id, sample_data in original_samples.items():
        global_id = sample_data.get("global_id")
        db_handler.update_record_classificaiton_confidence_threshold(
            global_id, float(sample_data["classificaiton_confidence_threshold"])
        )
        if db_handler.get_record_by_id(global_id)["name"] == "Unknown":
            db_handler.update_record_label(global_id, sample_data["name"])


if __name__ == "__main__":
    db_handler = DatabaseHandler(
        db_name="persons.db",
        table_name="persons",
        schema=Record,
        threshold=0.35,
        database_dir=os.path.join('../../pipeline_apps/', FACE_RECON_DATABASE_DIR_NAME),
        samples_dir=os.path.join('../../pipeline_apps/', FACE_RECON_SAMPLES_DIR_NAME)
    )
    visualize_embeddings(db_handler)
