# region imports
# Standard library imports
import json
import os
import time
import uuid
from typing import Any

# Third-party imports
import numpy as np
import lancedb
from lancedb.pydantic import LanceModel, Vector

# Local application-specific imports
try:
    from hailo_apps.python.core.common.defines import (
        FACE_RECON_DATABASE_DIR_NAME,
        FACE_RECON_SAMPLES_DIR_NAME
    )
except ImportError:
    from .defines import (
        FACE_RECON_DATABASE_DIR_NAME,
        FACE_RECON_SAMPLES_DIR_NAME
    )
# endregion


class Record(LanceModel):
    # mandatory fields
    global_id: str  # unique id
    label: str  # unique (but same IRL record might have multiple e.g., "Bob", "Bob glasses" etc.) with default "None" value
    avg_embedding: Vector(512)  # type: ignore the warning
    last_sample_recieved_time: int  # epoch timestamp: In case the last sample removed - not maintend to previous sample time...
    samples_json: str  # Store samples as a JSON string  # [{"embedding", "sample_path", "id"}] (path to sample)
    classificaiton_confidence_threshold: float
    # optional fields, but default values are set
    value: float = 0.0  # in some cases numeric value might be relevant


class DatabaseHandler:
    def __init__(self, db_name, table_name, schema, threshold, database_dir, samples_dir):
        self.db = self.__init_database(
            db_name=db_name, database_dir=database_dir, samples_dir=samples_dir
        )
        self.tbl_records = self.__init_table(
            self.db,
            table_name=table_name,
            schema=schema,
            indexes=[("global_id", "BTREE"), ("label", "BTREE")],
        )
        self.classificaiton_confidence_threshold = (
            threshold  # Default classification confidence threshold
        )

    def __init_database(self, db_name: str, database_dir: str, samples_dir: str):
        """Initializes the LanceDB database.

        Args:
            db_name (str): The name of the database file.
            database_dir (str): The directory where the database file will be stored.
            samples_dir (str): The directory where sample files will be stored.

        Returns:
            lancedb.LanceDB: The connected database instance.
        """
        os.makedirs(database_dir, exist_ok=True)  # Create the directory if it doesn't exist
        db = lancedb.connect(uri=os.path.join(database_dir, db_name))  # Connect to the database
        self.samples_dir = samples_dir
        return db

    def __init_table(self, db, table_name: str, schema=None, indexes=None):
        """Ensures that a table exists in the database. If it doesn't exist, creates it.

        Args:
            db (lancedb.LanceDB): The LanceDB database instance.
            table_name (str): The name of the table to check or create.
            schema (Any): The schema to use when creating the table (if needed).
            indexes (List[Tuple[str, str]]): A list of indexes to create, where each tuple contains
                                            the column name and index type.

        Returns:
            lancedb.Table: The table instance.
        """
        try:
            table = db.open_table(table_name)
        except Exception as err:
            if schema is None:
                raise ValueError("Schema must be provided to create a new table.") from err
            table = db.create_table(table_name, schema=schema)
            if indexes:
                for column, index_type in indexes:
                    table.create_scalar_index(column, index_type=index_type)
        return table

    def create_record(
        self, embedding: np.ndarray, sample: str, timestamp: int, label: str = "Unknown"
    ) -> dict[str, Any]:
        """Creates a record in the LanceDB table and generates a global ID.

        Args:
            embedding (np.ndarray) (required): The (first) sample embedding vector.
            label (str) (optional): The label (e.g., name) associated with the record.
            sample (str) (required): The sample sample path.
            timestamp (int) (required): The timestamp of the sample.

        Returns:
            record: The newly created record record as dict.

        Note: sample file path id != image id

        In case after this insertion there are more than 256 records in the table, the table will be indexed by the embedding column.
        """
        record = Record(
            global_id=str(uuid.uuid4()),
            label=label,
            avg_embedding=embedding.tolist(),
            last_sample_recieved_time=timestamp,
            samples_json=json.dumps(
                [{"embedding": embedding.tolist(), "sample_path": sample, "id": str(uuid.uuid4())}]
            ),
            classificaiton_confidence_threshold=self.classificaiton_confidence_threshold,
        )
        self.tbl_records.add([record])
        if len(self.tbl_records.search().to_list()) > 256:
            self.tbl_records.create_index(
                vector_column_name="embedding", metric="cosine", replace=True
            )
        return record.model_dump()

    def insert_new_sample(
        self, record: dict[str, Any], embedding: np.ndarray, sample: str, timestamp: int
    ) -> None:
        """Adds a new sample to a record, creates for the sample id and recalculates the average embedding.

        Args:
            record (Dict[str, Any]): The record to insert the sample into.
            embedding (np.ndarray): The sample embedding vector.
            sample (str): The sample sample path.
            timestamp (int): The timestamp of the sample.
        """
        samples = record["samples_json"]
        samples.append(
            {"embedding": embedding.tolist(), "sample_path": sample, "id": str(uuid.uuid4())}
        )
        all_embeddings = [
            np.array(sample["embedding"]) for sample in samples
        ]  # Recalculate the average embedding
        avg_embedding = np.mean(all_embeddings, axis=0)
        self.tbl_records.update(
            where=f"global_id = '{record['global_id']}'",
            values={
                "avg_embedding": avg_embedding,
                "samples_json": json.dumps(samples),
                "last_sample_recieved_time": timestamp,
            },
        )

    def remove_sample_by_id(self, global_id: str, sample_id: str) -> bool:
        """Removes a sample from a record & recalculates the average embedding.

        Args:
            global_id (str): The global ID of the record to remove from.
            sample_id (str): The ID of the sample to remove.

        Returns:
            bool: True if the record was removed, False otherwise.
        """
        record = self.tbl_records.search().where(f"global_id = '{global_id}'").to_list()[0]
        samples = json.loads(record["samples_json"])
        sample_to_delete = next(sample for sample in samples if sample["id"] == sample_id)
        self.delete_record_sample(sample_to_delete)
        new_samples = [sample for sample in samples if sample["id"] != sample_id]
        if not new_samples:
            self.tbl_records.delete(where=f"global_id = '{global_id}'")
            return True
        else:
            self.tbl_records.update(
                where=f"global_id = '{global_id}'",
                values={
                    "avg_embedding": np.mean(
                        [np.array(sample["embedding"]) for sample in new_samples], axis=0
                    ).tolist(),
                    "samples_json": json.dumps(new_samples),
                },
            )
            return False

    def search_record(
        self, embedding: np.ndarray, top_k: int = 1, metric_type: str = "cosine"
    ) -> dict[str, Any]:
        """Searches for a record in the LanceDB table by embedding vector similarity.

        Args:
            embedding (np.ndarray): The sample embedding vector to search for.
            top_k (int): The number of top results to return.
            metric_type (str): The similarity metric to use (e.g., "cosine").

        Returns:
            Dict[str, Any]: The search result with classification confidence.
        """
        search_result = (
            self.tbl_records.search(embedding.tolist(), vector_column_name="avg_embedding")
            .metric(metric_type)
            .limit(top_k)
            .to_list()
        )
        if search_result:
            search_result[0]["samples_json"] = json.loads(search_result[0]["samples_json"])
            if (
                (1 - search_result[0]["_distance"])
                > search_result[0]["classificaiton_confidence_threshold"]
            ):  # if search_result[0]['_distance']>1 the condition is false by default (1-1.1=-0.1) because default value if 0.3
                return search_result[0]
        # No match from DB
        return {
            "global_id": str(uuid.uuid4()),
            "label": "Unknown",
            "avg_embedding": None,
            "last_sample_recieved_time": None,
            "samples_json": None,
            "classificaiton_confidence_threshold": None,
            "_distance": 0.0,
        }

    def update_record_label(self, global_id: str, label: str = "Unknown") -> None:
        """Updates the label associated with a record in the LanceDB table.

        Args:
            global_id (str): The global ID of the record to update.
            label (str): The new label to associate with the record.
        """
        self.tbl_records.update(where=f"global_id = '{global_id}'", values={"label": label})

    def update_record_classificaiton_confidence_threshold(
        self, global_id: str, classificaiton_confidence_threshold: float
    ) -> None:
        """Updates the classificaiton confidence threshold associated with a record in the LanceDB table.

        Args:
            global_id (str): The global ID of the record to update.
            classificaiton_confidence_threshold (str): The new classificaiton confidence threshold to associate with the record.
        """
        self.tbl_records.update(
            where=f"global_id = '{global_id}'",
            values={"classificaiton_confidence_threshold": classificaiton_confidence_threshold},
        )

    def update_classification_confidence_threshold_for_all(self, new_threshold: float) -> None:
        """Updates the classification_confidence_threshold for all records in the LanceDB table.

        Args:
            new_threshold (float): The new confidence threshold value to set for all records.
        """
        # Get all records from the database
        records = self.get_all_records()
        # Iterate through each record and update the classification_confidence_threshold
        for record in records:
            global_id = record["global_id"]  # Assuming each record has a unique global_id
            self.tbl_records.update(
                where=f"global_id = '{global_id}'",
                values={"classificaiton_confidence_threshold": new_threshold},
            )

    def delete_record(self, global_id: str) -> None:
        """Deletes a record from the LanceDB table.

        Args:
            global_id (str): The global ID of the record to delete.
        """
        record = self.get_record_by_id(global_id)
        for sample in record["samples_json"]:
            self.delete_record_sample(sample)
        self.tbl_records.delete('global_id = "' + global_id + '"')

    def clear_table(self) -> None:
        """Deletes all records from the LanceDB table."""
        to_delete = ", ".join(
            [f"'{record['global_id']}'" for record in self.tbl_records.search().to_list()]
        )  # Get all records
        self.tbl_records.delete(f"global_id IN ({to_delete})")
        # Clear all files from the self.samples_dir folder
        if os.path.exists(self.samples_dir):
            for filename in os.listdir(self.samples_dir):
                file_path = os.path.join(self.samples_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        print("All records deleted from the database")

    def clear_unknown_labels(self) -> None:
        """Deletes all records from the LanceDB table with the label 'Unknown'."""
        records = self.tbl_records.search().where("label = 'Unknown'").to_list()
        if len(records) > 0:
            for record in records:
                for sample in json.loads(record["samples_json"]):
                    self.delete_record_sample(sample)
            to_delete = ", ".join([f"'{record['global_id']}'" for record in records])
            self.tbl_records.delete(f"global_id IN ({to_delete})")

    def clear_unknown_labels_keep_latest(self) -> None:
        """Deletes all records from the LanceDB table with the label 'Unknown',
        except the latest one based on the last sample received time.
        """
        # Fetch all records with the label 'Unknown'
        records = self.tbl_records.search().where("label = 'Unknown'").to_list()
        current_time = int(time.time())
        if len(records) > 1:  # Only proceed if there are 'Unknown' records
            # Find the record with the latest 'last_sample_recieved_time'
            latest_record = max(
                records,
                key=lambda record: self.get_records_last_sample_recieved_time(record["global_id"]),
            )
            latest_global_id = latest_record["global_id"]

            # if latest timestamp is older than 10 seconds - delete him also: this is the case when start button clicked but no person in front of the camera - so the latest might be from previous run
            if current_time - latest_record["last_sample_recieved_time"] < 10:
                # Filter out the latest record and prepare the rest for deletion
                records_to_delete = [
                    record for record in records if record["global_id"] != latest_global_id
                ]
            else:
                records_to_delete = records

            for record in records_to_delete:
                for sample in json.loads(record["samples_json"]):
                    self.delete_record_sample(sample)

            to_delete = ", ".join([f"'{record['global_id']}'" for record in records_to_delete])
            self.tbl_records.delete(f"global_id IN ({to_delete})")
            self.keep_only_last_sample(latest_global_id)

        elif len(records) == 1:
            latest_record = records[0]
            if current_time - latest_record["last_sample_recieved_time"] < 10:
                # If there's only one 'Unknown' record, keep it as is
                latest_global_id = latest_record["global_id"]
                self.keep_only_last_sample(latest_global_id)
            else:
                self.delete_record(
                    latest_global_id
                )  # Delete the only record if it's older than 10 seconds

    def keep_only_last_sample(self, global_id: str) -> None:
        """Updates the record with the given global_id to retain only the last sample
        in the samples_json field.
        Assumption - samples are added in order of time - the last one is the latest one.

        Args:
            global_id (str): The global ID of the record to update.
        """
        # Fetch the record by global_id
        record = self.get_record_by_id(global_id)
        if not record:
            return

        # Parse the samples_json field
        samples = record.get("samples_json", [])
        if len(samples) > 1:
            # Keep only the last sample
            for sample in samples[:-1]:
                self.remove_sample_by_id(
                    global_id, sample["id"]
                )  # safe way incl. recalculation of the average embedding

    def get_all_records(self, only_unknowns=False) -> dict[str, Any]:
        """Gets all records from the LanceDB table.

        Args:
            only_unknowns (bool): If True, return only records with the label 'Unknown'.

        Returns:
            List[Dict[str, Any]]: All the records.
        """
        if only_unknowns:
            records = self.tbl_records.search().where("label = 'Unknown'").to_list()
        else:
            records = self.tbl_records.search().to_list()

        for record in records:
            record["samples_json"] = json.loads(record["samples_json"])
        return records

    def get_record_by_id(self, global_id: str) -> dict[str, Any]:
        """Gets a record record from the LanceDB table by global ID.

        Args:
            global_id (str): The global ID of the record to retrieve.

        Returns:
            Dict[str, Any]: The record record.
        """
        result = self.tbl_records.search().where(f"global_id = '{global_id}'").to_list()[0]
        if result:
            result["samples_json"] = json.loads(result["samples_json"])
            return result
        return None

    def get_record_by_label(self, label: str = "Unknown") -> dict[str, Any]:
        """Gets a record record from the LanceDB table by label.

        Args:
            label (str): The label of the record to retrieve.

        Returns:
            Dict[str, Any]: The record record.
        """
        results = self.tbl_records.search().where(f"label = '{label}'").to_list()
        if not results:
            return None  # Return None if no records are found
        return results[0]

    def get_records_num_samples(self, global_id: str) -> int:
        """Gets the number of samples associated with a record.

        Args:
            global_id (str): The global ID of the record to retrieve.

        Returns:
            int: The number of samples.
        """
        return len(self.get_record_by_id(global_id)["samples_json"])

    def get_records_classificaiton_confidence_threshold(self, global_id: str) -> float:
        """Gets the classificaiton confidence threshold associated with a record.

        Args:
            global_id (str): The global ID of the record to
            retrieve.

        Returns:
            float: The classificaiton confidence threshold.
        """
        return self.get_record_by_id(global_id)["classificaiton_confidence_threshold"]

    def get_records_last_sample_recieved_time(self, global_id: str) -> int:
        """Gets the last sample recieved time associated with a record.

        Args:
            global_id (str): The global ID of the record to retrieve.

        Returns:
            int: The last sample recieved time.
        """
        return self.get_record_by_id(global_id)["last_sample_recieved_time"]

    def delete_record_sample(self, sample: tuple[str, str, str]):
        """Deletes the sample sample file.

        Args:
            sample (Dict[str, Any]): The sample record containing the sample file path.
        """
        sample_path = sample["sample_path"]
        if sample_path and os.path.exists(sample_path):
            os.remove(sample_path)

    def calibrate_classification_confidence_threshold(self):
        """Calibrates the classification confidence threshold based on confidence circles area.
        Smaller areas result in a smaller classification confidence threshold.
        """
        records = self.tbl_records.search().to_list()
        areas = []

        for record in records:
            # Get embeddings for the record
            samples = json.loads(record["samples_json"])
            embeddings = [np.array(sample["embedding"]) for sample in samples]

            if len(embeddings) < 2:
                # Skip calibration if there are not enough embeddings to calculate variance
                areas.append(0)
                continue

            # Perform PCA to reduce embeddings to 2D
            reduced_embeddings, principal_components, mean = self.perform_pca(
                embeddings, n_components=2
            )

            # Calculate standard deviations (semi-major and semi-minor axes)
            std_dev = np.std(reduced_embeddings, axis=0)
            semi_major_axis, semi_minor_axis = std_dev[0], std_dev[1]

            area = np.pi * semi_major_axis * semi_minor_axis
            areas.append(area)

        # Normalize areas for threshold calibration
        areas = np.array(areas)
        if len(areas) > 0 and np.max(areas) != np.min(areas):
            norm_areas = (areas - np.min(areas)) / (np.max(areas) - np.min(areas))
        else:
            norm_areas = np.zeros_like(areas)

        for i, record in enumerate(records):
            new_threshold = 0.1 + (0.9 - 0.1) * (1 - norm_areas[i])
            self.update_record_classificaiton_confidence_threshold(
                record["global_id"], new_threshold
            )

    def perform_pca(self, embeddings, n_components=2):
        """Perform PCA to reduce the dimensionality of embeddings.

        Args:
            embeddings (np.ndarray): The input data matrix of shape (n_samples, n_features).
            n_components (int): The number of principal components to retain.

        Returns:
            np.ndarray: The reduced embeddings of shape (n_samples, n_components).
            np.ndarray: The principal components (eigenvectors).
            np.ndarray: The mean of the embeddings.
        """
        # Step 1: Center the data (subtract the mean)
        mean = np.mean(embeddings, axis=0)
        centered_embeddings = embeddings - mean

        # Step 2: Compute the covariance matrix
        covariance_matrix = np.cov(centered_embeddings, rowvar=False)

        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Step 4: Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Select the top n_components eigenvectors
        principal_components = eigenvectors[:, :n_components]

        # Step 6: Project the data onto the principal components
        reduced_embeddings = np.dot(centered_embeddings, principal_components)

        return reduced_embeddings, principal_components, mean


if __name__ == "__main__":
    # usage example
    database_handler = DatabaseHandler(
        db_name="persons.db",
        table_name="persons",
        schema=Record,
        threshold=0.5,
        database_dir=os.path.join('../../pipeline_apps/', FACE_RECON_DATABASE_DIR_NAME),
        samples_dir=os.path.join('../../pipeline_apps/', FACE_RECON_SAMPLES_DIR_NAME)
    )
    all_records = database_handler.get_all_records()
    alice = database_handler.get_record_by_label(label="Alice")
