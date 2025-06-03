import json
from typing import List, Dict, Any
from agno.tools import tool
from utils.text_processing import bytes2human
import numpy as np
from datasets import load_dataset, load_dataset_builder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from agno.utils.log import logger


class HuggingFaceDatasetSearch:
    def __init__(self):
        self.ds = None
        self.description_vectors = None
        self.vectorizer = None
        logger.info("Loading metadata dataset 'nkasmanoff/huggingface-datasets'...")
        try:
            self.metadata_ds = load_dataset("nkasmanoff/huggingface-datasets")["train"]
            logger.info("Metadata dataset loaded.")
        except Exception as e:
            logger.error(f"Error loading metadata dataset: {e}")
            self.metadata_ds = None

    def setup_search_space(self, min_likes=5, max_size_gb=10, exclude_disabled=True):
        if self.metadata_ds is None:
            logger.error("Failed to load metadata dataset. No search possible.")
            return

        metadata_list = []
        for item in self.metadata_ds:
            if exclude_disabled and item.get("disabled", False):
                continue

            likes = item.get("likes", 0)
            if likes < min_likes:
                continue

            size_bytes = item.get("size", 0)
            if size_bytes > max_size_gb * 1024**3:
                logger.debug(
                    f"Skipping {item.get('id', 'Unknown')} - too large: {bytes2human(size_bytes)}"
                )
                continue

            metadata_list.append(item)

        if not metadata_list:
            logger.warning("No datasets meet the specified criteria.")
            return

        # Store filtered dataset
        self.ds = metadata_list
        logger.info(f"Filtered dataset created with {len(self.ds)} entries.")

        # Prepare for vector search
        self.setup_vectorization()

    def setup_vectorization(self):
        if not self.ds:
            return

        logger.info("Normalizing likes and downloads...")
        # Simple normalization to [0, 1] range for ranking
        likes = [item.get("likes", 0) for item in self.ds]
        max_likes = max(likes) if likes else 1
        for item in self.ds:
            item["normalized_likes"] = item.get("likes", 0) / max_likes
        logger.info("Normalization complete.")

        logger.info("Vectorizing descriptions using TF-IDF...")
        try:
            descriptions = [item.get("description", "") or "" for item in self.ds]
            self.vectorizer = TfidfVectorizer(
                max_features=5000, stop_words="english", ngram_range=(1, 2)
            )
            self.description_vectors = self.vectorizer.fit_transform(descriptions)
            logger.info("TF-IDF vectorization complete.")
            logger.info(f"Description vectors shape: {self.description_vectors.shape}")
        except Exception as e:
            logger.error(f"Error during TF-IDF vectorization: {e}")
            self.description_vectors = None
            logger.error("TF-IDF vectorization failed. No search possible.")

    def search_datasets(
        self, query: str, top_k: int = 10, include_live_info=True
    ) -> List[Dict[str, Any]]:
        if not self.ds or self.description_vectors is None:
            return []

        try:
            # Transform query
            query_vector = self.vectorizer.transform([query])
        except Exception as e:
            logger.error(f"Error transforming query: {e}")
            return []

        # Compute similarities
        similarities = linear_kernel(query_vector, self.description_vectors).flatten()

        # Score = 0.7 * similarity + 0.3 * normalized_likes
        scores = 0.7 * similarities + 0.3 * np.array(
            [item["normalized_likes"] for item in self.ds]
        )

        # Get top results
        top_indices = scores.argsort()[-top_k:][::-1]
        top_datasets = [self.ds[i] for i in top_indices]

        if include_live_info:
            top_datasets = self.add_live_split_info(top_datasets)

        return top_datasets

    def add_live_split_info(
        self, datasets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        logger.info(f"Fetching live split info for top {len(datasets)} datasets...")

        for item in datasets:
            dataset_id = item.get("id")
            if not dataset_id:
                continue

            try:
                dbuilder_info = load_dataset_builder(
                    dataset_id, trust_remote_code=True
                ).info

                splits_info = {}
                if dbuilder_info.splits:
                    for split_name, split_info in dbuilder_info.splits.items():
                        splits_info[split_name] = {
                            "num_examples": split_info.num_examples,
                            "num_bytes": split_info.num_bytes,
                        }

                item["live_splits"] = splits_info

                if dbuilder_info.features:
                    item["live_features"] = str(dbuilder_info.features)

            except Exception as e:
                logger.warning(f"Could not fetch live info for {dataset_id}: {e}")
                item["live_splits"] = {}
                item["live_features"] = "Unable to fetch"

        logger.info("Live split info fetching complete.")
        return datasets
