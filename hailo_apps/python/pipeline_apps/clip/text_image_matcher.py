import json
import os
import sys
import argparse
import numpy as np

# Prevent importing local clip.py file by ensuring it's not in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_original_path = sys.path.copy()

# Remove all instances of current directory from sys.path
while _current_dir in sys.path:
    sys.path.remove(_current_dir)

# Also remove empty string which represents current directory
while '' in sys.path:
    sys.path.remove('')

# Also remove '.' which represents current directory
while '.' in sys.path:
    sys.path.remove('.')

"""
This class is used to store the text embeddings and match them to image embeddings
This class should be used as a singleton!
An instance of this class is created in the end of this file.
import text_image_matcher from this file to make sure that only one instance of the TextImageMatcher class is created.
Example: from TextImageMatcher import text_image_matcher
"""

# Set up global variables. Only required imports are done in the init functions
openai_clip = None
torch = None

class TextEmbeddingEntry:
    def __init__(self, text="", embedding=None, negative=False, ensemble=False):
        self.text = text
        self.embedding = embedding if embedding is not None else np.array([])
        self.negative = negative
        self.ensemble = ensemble
        self.probability = 0.0
        self.tracked_probability = 0.0

    def to_dict(self):
        return {
            "text": self.text,
            "embedding": self.embedding.tolist(),
            "negative": self.negative,
            "ensemble": self.ensemble
        }

class Match:
    def __init__(self, row_idx, text, similarity, entry_index, negative, passed_threshold):
        self.row_idx = row_idx  # row index in the image embedding
        self.text = text  # best matching text
        self.similarity = similarity  # similarity between the image and best text embeddings
        self.entry_index = entry_index  # index of the entry in TextImageMatcher.entries
        self.negative = negative  # True if the best match is a negative entry
        self.passed_threshold = passed_threshold  # True if the similarity is above the threshold

    def to_dict(self):
        return {
            "row_idx": self.row_idx,
            "text": self.text,
            "similarity": self.similarity,
            "entry_index": self.entry_index,
            "negative": self.negative,
            "passed_threshold": self.passed_threshold
        }

class TextImageMatcher:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextImageMatcher, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name="RN50x4", threshold=0.8, max_entries=6):
        self.model = None  # model is initialized in init_clip
        self.preprocess = None  # preprocess is initialized in init_clip
        self.model_runtime = None
        self.model_name = model_name
        self.threshold = threshold
        self.run_softmax = True
        self.device = "cpu"

        self.max_entries = max_entries
        self.entries = [TextEmbeddingEntry() for _ in range(max_entries)]
        self.user_data = None  # user data can be used to store additional information
        self.text_prefix = "A photo of a "
        self.ensemble_template = [
            'a photo of a {}.',
            'a photo of the {}.',
            'a photo of my {}.',
            'a photo of a big {}.',
            'a photo of a small {}.',
        ]
        self.track_id_focus = None  # Used to focus on specific track id when showing confidence

    def init_clip(self):
        """Initialize the CLIP model."""
        global openai_clip, torch
        
        # Force fresh import of the external clip package
        import importlib
        if 'clip' in sys.modules:
            del sys.modules['clip']
        
        import clip as openai_clip
        import torch
        
        self.model, self.preprocess = openai_clip.load(self.model_name, device=self.device)
        self.model_runtime = "clip"

    def set_threshold(self, new_threshold):
        self.threshold = new_threshold

    def set_text_prefix(self, new_text_prefix):
        self.text_prefix = new_text_prefix

    def set_ensemble_template(self, new_ensemble_template):
        self.ensemble_template = new_ensemble_template

    def update_text_entries(self, new_entry, index=None):
        if index is None:
            for i, entry in enumerate(self.entries):
                if entry.text == "":
                    self.entries[i] = new_entry
                    return
            self.entries.append(new_entry)
        elif 0 <= index < len(self.entries):
            self.entries[index] = new_entry
        else:
            print(f"Index {index} is out of bounds for entries list.")

    def add_text(self, text, index=None, negative=False, ensemble=False):
        if self.model_runtime is None:
            print("No model is loaded. Please call init_clip before calling add_text.")
            return
        text_entries = [template.format(text) for template in self.ensemble_template] if ensemble else [self.text_prefix + text]

        global openai_clip, torch
        text_tokens = openai_clip.tokenize(text_entries).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            ensemble_embedding = torch.mean(text_features, dim=0).cpu().numpy().flatten()
        new_entry = TextEmbeddingEntry(text, ensemble_embedding, negative, ensemble)
        self.update_text_entries(new_entry, index)

    def get_embeddings(self):
        """Return a list of indexes to self.entries if entry.text != ""."""
        return [i for i, entry in enumerate(self.entries) if entry.text != ""]

    def get_texts(self):
        """Return all entries' text (not only valid ones)."""
        return [entry.text for entry in self.entries]

    def save_embeddings(self, filename):
        data_to_save = {
            "threshold": self.threshold,
            "text_prefix": self.text_prefix,
            "ensemble_template": self.ensemble_template,
            "entries": [entry.to_dict() for entry in self.entries]
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f)

    def load_embeddings(self, filename):
        if not os.path.isfile(filename):
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('')  # Create an empty file or initialize with some data
        else:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.threshold = data['threshold']
                    self.text_prefix = data['text_prefix']
                    self.ensemble_template = data['ensemble_template']
                    self.entries = [TextEmbeddingEntry(text=entry['text'],
                                                       embedding=np.array(entry['embedding']),
                                                       negative=entry['negative'],
                                                       ensemble=entry['ensemble'])
                                    for entry in data['entries']]
            except Exception as e:
                print(f"Error while loading file {filename}: {e}")

    def get_image_embedding(self, image):
        if self.model_runtime is None:
            print("No model is loaded. Please call init_clip before calling get_image_embedding.")
            return None
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_input)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        return image_embedding.cpu().numpy().flatten()

    def match(self, image_embedding_np, report_all=False, update_tracked_probability=None):
        """
        This function is used to match an image embedding to a text embedding
        Returns a list of tuples: (row_idx, text, similarity, entry_index)
        row_idx is the index of the row in the image embedding
        text is the best matching text
        similarity is the similarity between the image and text embeddings
        entry_index is the index of the entry in self.entries
        If the best match is a negative entry, or if the similarity is below the threshold, the tuple is not returned
        If no match is found, an empty list is returned
        If report_all is True, the function returns a list of all matches,
        including negative entries and entries below the threshold.
        """
        if len(image_embedding_np.shape) == 1:
            image_embedding_np = image_embedding_np.reshape(1, -1)
        results = []
        all_dot_products = None
        valid_entries = self.get_embeddings()
        if len(valid_entries) == 0:
            return []
        text_embeddings_np = np.array([self.entries[i].embedding for i in valid_entries])
        for row_idx, image_embedding_1d in enumerate(image_embedding_np):
            dot_products = np.dot(text_embeddings_np, image_embedding_1d)
            all_dot_products = dot_products[np.newaxis, :] if all_dot_products is None else np.vstack((all_dot_products, dot_products))

            if self.run_softmax:
                similarities = np.exp(100 * dot_products)
                similarities /= np.sum(similarities)
            else:
                # These magic numbers were collected by running actual inferences and measureing statistics.
		        # stats min: 0.27013595659637846, max: 0.4043235050452188, avg: 0.33676838831786493
                # map to [0,1]
                similarities = (dot_products - 0.27) / (0.41 - 0.27)
                similarities = np.clip(similarities, 0, 1)

            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            for i, _ in enumerate(similarities):
                self.entries[valid_entries[i]].probability = similarities[i]
                if update_tracked_probability is None or update_tracked_probability == row_idx:
                    self.entries[valid_entries[i]].tracked_probability = similarities[i]
            new_match = Match(row_idx,
                              self.entries[valid_entries[best_idx]].text,
                              best_similarity, valid_entries[best_idx],
                              self.entries[valid_entries[best_idx]].negative,
                              best_similarity > self.threshold)
            if not report_all and new_match.negative:
                continue
            if report_all or new_match.passed_threshold:
                results.append(new_match)
        return results

text_image_matcher = TextImageMatcher()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="text_embeddings.json", help="output file name default=text_embeddings.json")
    parser.add_argument("--interactive", action="store_true", help="input text from interactive shell")
    parser.add_argument("--image-path", type=str, default=None, help="Optional, path to image file to match. Note image embeddings are not running on Hailo here.")
    parser.add_argument('--texts-list', nargs='+', help='A list of texts to add to the matcher, the first one will be the searched text, the others will be considered negative prompts.\n Example: --texts-list "cat" "dog" "yellow car"')
    parser.add_argument('--texts-json', type=str, help='A json of texts to add to the matcher, the json will include 2 keys negative and positive, the values are going to be lists of texts\n Example: --texts-json resources/texts_json_example.json')
    args = parser.parse_args()

    matcher = TextImageMatcher()
    matcher.init_clip()
    texts = []
    if args.interactive:
        while True:
            text = input(f'Enter text (leave empty to finish) {matcher.text_prefix}: ')
            if text == "":
                break
            texts.append(text)
    elif args.texts_list:
        texts = args.texts_list
    elif args.texts_json:
        with open(args.texts_json, 'r') as file:
            data = json.load(file)
            texts_positive =  data['positive']
            texts_negative = data['negative']
    else:
        texts = ["birthday cake", "person", "landscape"]
    
    if not args.texts_json:
        texts_positive = [texts[0]]
        texts_negative = texts[1:]

    for text in texts_positive:
        matcher.add_text(text, negative=False)
    for text in texts_negative:
        matcher.add_text(text, negative=True)

    matcher.save_embeddings(args.output)

    if args.image_path is None:
        sys.exit()

if __name__ == "__main__":
    main()