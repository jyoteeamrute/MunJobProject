import os
import re
import numpy as np
from nltk.corpus import stopwords
from sentence_transformers import util, SentenceTransformer
from constants import path_data, skill_type_threshold, DEFAULT_MAX_NUMBER_OF_SKILLS_FOR_GPT, THRESHOLD_STEP


class EmbeddingManager:
    stop_words = set(stopwords.words('english'))

    def __init__(self):
        self.model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")

    @staticmethod
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = " ".join(word for word in text.split() if word not in EmbeddingManager.stop_words)
        return text

    def pad_or_truncate_embedding(self, embedding, target_size):
        current_size = embedding.shape[0]
        if current_size < target_size:
            padding = np.zeros((target_size - current_size,))
            embedding = np.concatenate((embedding, padding), axis=0)
        elif current_size > target_size:
            embedding = embedding[:target_size]
        return embedding

    def add_to_embeddings(self, title, description, path_key, warnings_fn=None):
        if path_key not in path_data:
            message = "Invalid path key provided."
            if warnings_fn:
                warnings_fn(message)
            return

        processed_title = self.preprocess_text(title)
        processed_description = self.preprocess_text(description)
        embeddings_file_path, processed_lines_file_path, title_lines_file_path = path_data[path_key]

        os.makedirs(os.path.dirname(embeddings_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(processed_lines_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(title_lines_file_path), exist_ok=True)

        title_description = f"{processed_title or ''}::{processed_description or ''}"

        embedding = self.model.encode(title_description)

        if all(os.path.exists(file_path) for file_path in
               [embeddings_file_path, processed_lines_file_path, title_lines_file_path]):
            try:
                embeddings = np.load(embeddings_file_path, allow_pickle=True)
                processed_lines = np.load(processed_lines_file_path, allow_pickle=True).tolist()
                title_lines = np.load(title_lines_file_path, allow_pickle=True).tolist()
            except Exception as e:
                if warnings_fn:
                    warnings_fn(f"Error loading files: {e}")
                return

            if title_description in processed_lines:
                message = f"'{title_description}' already exists in embeddings."
                if warnings_fn:
                    warnings_fn(message)
                return
            else:
                if embeddings.size == 0:
                    # If embeddings is empty, initialize it with the new embedding
                    embeddings = embedding[np.newaxis, :]
                elif embeddings.ndim == 1:
                    # If embeddings is 1D, make it 2D
                    embeddings = embeddings[np.newaxis, :]
                target_size = embeddings.shape[1]  # Get target size from existing embeddings

                embedding = self.pad_or_truncate_embedding(embedding, target_size)
                embeddings = np.vstack((embeddings, embedding))
                processed_lines.append(title_description)
                title_lines.append(title)
        else:
            # Handle case where embeddings might be empty or 1D
            target_size = embedding.shape[0]
            embeddings = embedding[np.newaxis, :]  # Ensure it's a 2D array
            processed_lines = [title_description]
            title_lines = [title]

        try:
            np.save(embeddings_file_path, embeddings, allow_pickle=True)
            np.save(processed_lines_file_path, processed_lines, allow_pickle=True)
            np.save(title_lines_file_path, title_lines, allow_pickle=True)
            if warnings_fn:
                warnings_fn(f"'{title_description}' added to embeddings.")

        except Exception as e:
            if warnings_fn:
                warnings_fn(f"Error saving files: {e}")

        return True

    def delete_from_embeddings(self, title, description, path_key, warnings_fn=None):
        if path_key not in path_data:
            message = "Invalid path key provided."
            if warnings_fn:
                warnings_fn(message)
            return

        processed_title = self.preprocess_text(title)
        processed_description = self.preprocess_text(description)

        title_description = f"{processed_title or ''}::{processed_description or ''}"

        embeddings_file_path, processed_lines_file_path, title_lines_file_path = path_data[path_key]

        if all(os.path.exists(file_path) for file_path in
               [embeddings_file_path, processed_lines_file_path, title_lines_file_path]):
            try:
                embeddings = np.load(embeddings_file_path, allow_pickle=True)
                processed_lines = np.load(processed_lines_file_path, allow_pickle=True).tolist()
                title_lines = np.load(title_lines_file_path, allow_pickle=True).tolist()

            except Exception as e:
                if warnings_fn:
                    warnings_fn(f"Error loading files: {e}")
                return

            if title_description not in processed_lines:
                message = f"'{title_description}' does not exist in embeddings."
                if warnings_fn:
                    warnings_fn(message)
                return

            index = processed_lines.index(title_description)
            embeddings = np.delete(embeddings, index, axis=0)
            processed_lines.pop(index)
            title_lines.pop(index)

            try:
                np.save(embeddings_file_path, embeddings)
                np.save(processed_lines_file_path, processed_lines, allow_pickle=True)
                np.save(title_lines_file_path, title_lines, allow_pickle=True)

            except Exception as e:
                if warnings_fn:
                    warnings_fn(f"Error saving files: {e}")

            if warnings_fn:
                warnings_fn(f"'{title_description}' deleted from embeddings.")
        else:
            message = "Embeddings file or processed lines file does not exist."
            if warnings_fn:
                warnings_fn(message)

    def update_embeddings(self, old_title, old_description, new_title, new_description, old_path_key, new_path_key=None,
                          warnings_fn=None):
        self.delete_from_embeddings(old_title, old_description, old_path_key, warnings_fn)
        if new_path_key is None:
            new_path_key = old_path_key
        self.add_to_embeddings(new_title, new_description, new_path_key, warnings_fn)

    # for each skill_type
    def filter_skills(self, title, description, warnings_fn=None):
        try:
            processed_title = self.preprocess_text(title)
            processed_description = self.preprocess_text(description)
            processed_title_description = f"{processed_title or ''}::{processed_description or ''}"
            target_embedding = self.model.encode(processed_title_description)

            retrieved_skills = {}
            all_title_similarity_pairs = {}

            # First, load and compute similarities once
            for path_key, (embeddings_file_path, processed_lines_file_path, title_lines_file_path) in path_data.items():
                # if path_key not in ['Course', 'Professions']:
                if all(os.path.exists(file_path) for file_path in
                       [embeddings_file_path, processed_lines_file_path, title_lines_file_path]):
                    print(
                        f"Loading data from {embeddings_file_path}, {processed_lines_file_path}, and {title_lines_file_path}")
                    skill_embeddings = np.load(embeddings_file_path, allow_pickle=True)
                    title_lines = np.load(title_lines_file_path, allow_pickle=True).tolist()

                    similarities = util.dot_score(target_embedding, skill_embeddings)[0].cpu().tolist()

                    # Combine docs & scores and sort by decreasing score
                    title_similarity_pairs = sorted(list(zip(title_lines, similarities)), key=lambda x: x[1],
                                                    reverse=True)
                    all_title_similarity_pairs[path_key] = title_similarity_pairs
                    retrieved_skills[path_key] = title_similarity_pairs

            # Initialize the threshold increment
            threshold_increment = THRESHOLD_STEP

            # Copy original thresholds to modify
            current_thresholds = skill_type_threshold.copy()

            while True:
                total_pairs_count = sum(
                    len([pair for pair in all_title_similarity_pairs[path_key] if
                         pair[1] > current_thresholds[path_key]])
                    for path_key in all_title_similarity_pairs)

                if total_pairs_count <= DEFAULT_MAX_NUMBER_OF_SKILLS_FOR_GPT:
                    break

                # Increment the thresholds
                for key in current_thresholds:
                    current_thresholds[key] += threshold_increment

            # Filter the retrieved_skills with the updated thresholds
            filtered_titles = []
            for path_key in retrieved_skills:
                filtered_titles.extend(
                    [pair[0] for pair in retrieved_skills[path_key] if pair[1] > current_thresholds[path_key]])

            return filtered_titles

        except Exception as e:
            message = f"Failed to filter skills: {e}"
            if warnings_fn:
                warnings_fn(message)
            print(message)
            return []

    def find_top_similar_skills(self, title, description, top_similar, warnings_fn=None):
        try:
            processed_title = self.preprocess_text(title)
            processed_description = self.preprocess_text(description)
            processed_title_description = f"{processed_title or ''}::{processed_description or ''}"
            target_embedding = self.model.encode(processed_title_description)

            all_title_similarity_pairs = []

            # Load and compute similarities for all path keys
            for path_key, (embeddings_file_path, processed_lines_file_path, title_lines_file_path) in path_data.items():
                if all(os.path.exists(file_path) for file_path in
                       [embeddings_file_path, processed_lines_file_path, title_lines_file_path]):
                    print(
                        f"Loading data from {embeddings_file_path}, {processed_lines_file_path}, and {title_lines_file_path}")
                    skill_embeddings = np.load(embeddings_file_path, allow_pickle=True)
                    title_lines = np.load(title_lines_file_path, allow_pickle=True).tolist()

                    similarities = util.dot_score(target_embedding, skill_embeddings)[0].cpu().tolist()

                    # Combine docs & scores and sort by decreasing score
                    title_similarity_pairs = list(zip(title_lines, similarities))
                    all_title_similarity_pairs.extend(title_similarity_pairs)

            # Sort all pairs by similarity score in descending order
            all_title_similarity_pairs.sort(key=lambda x: x[1], reverse=True)

            # Get the top N similar pairs (title, similarity)
            top_similar_skills = all_title_similarity_pairs[:top_similar]

            return top_similar_skills

        except Exception as e:
            message = f"Failed to find top similar skills: {e}"
            if warnings_fn:
                warnings_fn(message)
            print(message)
            return []
