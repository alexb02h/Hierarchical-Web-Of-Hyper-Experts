import pandas as pd
import torch

class PhaseManager:
	def __init__(self, csv_path):
		df = pd.read_csv(csv_path, nrows=0)
		self.all_labels = df.columns[1:].tolist()
		self.labels_to_idx = {label: i for i, label in enumerate(self.all_labels)}

	def get_indices(self, label_list): return [self.label_to_idx[l] for l in label_list if l in self.labels_to_idx]

manager = PhaseManager("Dataset/annotations_final.csv")
