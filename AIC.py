import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Subset
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from typing import List, Dict, Tuple
import librosa
import pandas as pd
import os
import numpy as np
import networkx as nx
import copy
import random

#1D Convolution -> Batch-Norm -> ReLU -> max_pool
#Changes channels from in to out, while reducing temporal length by pool_size

class SampleCNNBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3,pool_size=3):
		super(SampleCNNBlock, self).__init__()
		self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False) #added bias = False
		self.bn = nn.BatchNorm1d(out_channels)
		self.relu = nn.ReLU(inplace = True)
		self.pool = nn.MaxPool1d(pool_size)

	def forward(self, x): 
		'''

		Input : (N, in_channels, L_in) 

		After Conv1D : (N, out_channels, L_in)

		After BatchNorm1D and Relu: shape unchanged

		After MaxPool1D: (N, out_channels, L_out) where L_out = floor(L_in / pool_size)

		'''

		x = self.pool(self.relu(self.bn(self.conv(x))))
		return x



#using a ResidualBlock to use the previous layer in the current layer. Essentially using the output of the SampleCNNBlock and adding to to a 1D convolution that also uses a 1D max pooling function.

class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=3):
		super(ResidualBlock, self).__init__()
		self.main = SampleCNNBlock(in_channels, out_channels, kernel_size, pool_size)
		self.residual = nn.Conv1d(in_channels, out_channels, kernel_size = 1) if in_channels != out_channels else nn.Identity()



	def forward(self,x):
		out = self.main(x)
		res = self.residual(F.max_pool1d(x,kernel_size=3))
		return out + res



class SqueezeExcite(nn.Module):
	def __init__(self,channels, reduction=16):
		super(SqueezeExcite, self).__init__()
		self.fc1 = nn.Linear(channels, channels // reduction)
		self.fc2 = nn.Linear(channels // reduction, channels)

	def forward(self,x):
		z = x.mean(dim=2)
		s = torch.sigmoid(self.fc2(F.relu(self.fc1(z))))
		s = s.unsqueeze(2)
		return x * s



class ReSEBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=3):
		super(ReSEBlock, self).__init__()
		self.resblock = ResidualBlock(in_channels, out_channels, kernel_size, pool_size)
		self.se = SqueezeExcite(out_channels)

	def forward(self, x):
		x = self.resblock(x)
		x = self.se(x)
		return x

class AudioSpecialist(nn.Module):
	def __init__(self, specialist_id, in_channels, out_channels):
		super(AudioSpecialist,self).__init__()
		self.id = specialist_id
		self.in_channels = in_channels
		self.out_channels = out_channels
  
		self.processor = ReSEBlock(in_channels,out_channels)
		self.pool = nn.AdaptiveAvgPool1d(1)
  
		self.activation_history = []
  
	def forward(self,x):
		features = self.processor(x)
		pooled = self.pool(features).squeeze(2)
  
		self.activation_history.append(pooled.detach().mean(dim=0))
		if len(self.activation_history) > 100: self.activation_history.pop(0)
  
		return features, pooled

	def get_behavioral_signature(self):
		if not self.activation_history: return torch.zeros(self.out_channels)
		return torch.stack(self.activation_history).mean(dim=0)

class ControllerExpert(nn.Module):
	def __init__(self, context_dim=128, hidden_dim=256):
		super(ControllerExpert,self).__init__()

		self.network = nn.Sequential(
			nn.Linear(context_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim)
		)
	
	def forward(self, context): return self.network(context)
 
	def generate_specialist_config(self, context, in_channels, out_channels):
		conditioning = self(context)
		return {'in_channels' : in_channels, 'out_channels' : out_channels}

class EvolutionaryAudioGNN(nn.Module):
	def __init__(self, in_channels=1, num_classes=50, initial_specialists=4):
		super(EvolutionaryAudioGNN,self).__init__()
        
		self.num_classes = num_classes
		self.specialist_id_counter = 0
        
		self.controllers = nn.ModuleList([ControllerExpert() for _ in range(3)])
        
		self.graph = nx.DiGraph()
		self.specialists = nn.ModuleDict()
		self.coupling_matrix = {}
        
		channels = [64, 128, 256, 512]
		prev_c = in_channels
        
		for c in channels: 
			self.add_specialist(prev_c, c)
			prev_c = c
   
		spec_ids = list(self.specialists.keys())
		for i in range(len(spec_ids) - 1):  self.graph.add_edge(spec_ids[i], spec_ids[i+1], weight=1.0)
  
		total_features = sum(channels)
		self.classifier = nn.Linear(total_features, num_classes)
	
	def add_specialist(self, in_channels, out_channels):
		spec_id = f"spec_{self.specialist_id_counter}"
		self.specialist_id_counter += 1
  
		specialist = AudioSpecialist(spec_id, in_channels, out_channels) 
		print("Womp Womp")
		self.specialists[spec_id] = specialist
		self.graph.add_node(spec_id, channels=out_channels) 
  
		return spec_id

	def forward(self,x):
		specialist_outputs = {}
		pooled_features = []
  
		for spec_id in nx.topological_sort(self.graph):
			specialist = self.specialists[spec_id]
   
			predecessors = list(self.graph.predecessors(spec_id))

			if not predecessors: spec_input = x
			else:
				pred_outputs = [specialist_outputs[pred] for pred in predecessors]
				spec_input = pred_outputs[-1]
    
			features, pooled = specialist(spec_input)
			specialist_outputs[spec_id] = features
			pooled_features.append(pooled)
   
		combined = torch.cat(pooled_features, dim=1)
		return self.classifier(combined)

	def measure_specialist_similarity(self, spec_id1, spec_id2):
		sig1 = self.specialists[spec_id1].get_behavioral_signature()
		sig2 = self.specialists[spec_id2].get_behavioral_signature()

		if sig1.shape != sig2.shape: return 0.0

		similarity = F.cosine_similarity(sig1.unsqueeze(0), sig2.unsqueeze(0))
		return similarity.item()

	def should_couple(self, spec_id1, spec_id2, threshold=0.8):
		similarity = self.measure_specialist_similarity(spec_id1,spec_id2)
		graph_distance = nx.shortest_path_length(self.graph.to_undirected(), spec_id1, spec_id2) if nx.has_path(self.graph.to_undirected(), spec_id1, spec_id2) else float('inf')
   
		return similarity > threshold and graph_distance < 3

	def update_coupling(self):
		spec_ids = list(self.specialists.keys())
		for i, id1 in enumerate(spec_ids):
			for id2 in spec_ids[i+1:]:
				if self.should_couple(id1,id2):
					current_coupling = self.coupling_matrix.get((id1,id2), 0.0)
					self.coupling_matrix[(id1,id2)] = min(1.0 , current_coupling + 0.1)
				else:
					if (id1, id2) in self.coupling_matrix: self.coupling_matrix[(id1, id2)] = max(0.0, self.coupling_matrix[(id1,id2)] - 0.05)

	def get_architecture_stats(self):
		return{
			'num_specialists' : len(self.specialists),
			'num_edges' : self.graph.number_of_edges(),
			'num_coupled_pairs' : sum(1 for v in self.coupling_matrix.values() if v > 0.5),
			'avg_coupling' : np.mean(list(self.coupling_matrix.values())) if self.coupling_matrix else 0.0 
		}

class MagnaTagATuneDataset(Dataset):
	def __init__(self, audio_dir, annotations_file, clip_info_file, sample_rate=16000, window_size=3.0, hop_size=1.5, transform=None):
		self.audio_dir = audio_dir
		self.annotations = pd.read_csv(annotations_file, delimiter='\t')
		self.clip_info = pd.read_csv(clip_info_file)
		self.sample_rate = sample_rate
		self.window_size = int(sample_rate * window_size)
		self.hop_size = int(sample_rate * hop_size)
		self.transform = transform
		self.tag_columns = [col for col in self.annotations.columns if col not in ['clip_id','mp3_path']]
		self.entries = []

		for idx, row in self.annotations.iterrows():
			mp3_path = row['mp3_path']
			wav_path = mp3_path.replace('.mp3','.wav')

			full_path = os.path.join(audio_dir,wav_path)

			if os.path.exists(full_path):

				labels = row[self.tag_columns].values.astype(np.float32)

				self.entries.append((full_path, labels))

	def __len__(self): return len(self.entries)

	def __getitem__(self, idx):
		path, labels = self.entries[idx]
		waveform, sr = torchaudio.load(path)

		if sr != self.sample_rate:
			resampler = torchaudio.transforms.Resample(orig_freq=sr,new_freq=self.sample_rate)
			waveform = resampler(waveform)
		if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
		if waveform.shape[1] < self.window_size:
			pad_amount = self.window_size - waveform.shape[1]
			if pad_amount > 0: waveform = F.pad(waveform,(0,pad_amount), mode='constant', value=0)
		else:
			max_offset = waveform.shape[1] - self.window_size
			start = random.randint(0, max_offset)
			waveform = waveform[:, start:start + self.window_size]

		if self.transform: waveform = self.transform(waveform)

		return waveform, torch.tensor(labels, dtype=torch.float32)

class EvolutionaryTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.performance_history = []
        self.stagnation_counter = 0
        self.diversity_pressure = 0.5
        
    def is_stuck(self, patience=5, threshold=0.01):
        if len(self.performance_history) < patience: return False
        
        recent = self.performance_history[-patience:]
        improvement = max(recent) - min(recent)
        
        return improvement < threshold
    
    def mutate_add_specialist(self, out_channels):
        existing = list(self.model.specialists.keys())
        
        if not existing: 
            in_c = 1
            parent = None
        else:
            parent = np.random.choice(existing)
            in_c = self.model.specialists[parent].out_channels
        
        new_id = self.model.add_specialist(in_c, out_channels)
        
        if parent: self.model.graph.add_edge(parent, new_id, weight=1.0)
            
        print(f"Specialist {new_id} Added (Input: {in_c}, Output: {out_channels})")
        return new_id
    
    def mutate_add_connection(self):
        spec_ids = list(self.model.specialists.keys())
        if len(spec_ids) < 2: return
        
        for _ in range(10):
            src, dst = np.random.choice(spec_ids, 2, replace=False)
            if not self.model.graph.has_edge(src,dst):
                self.model.graph.add_edge(src, dst, weight=1.0)
                if not nx.is_directed_acyclic_graph(self.model.graph): self.model.graph.remove_edge(src,dst)
                else:
                    print(f"Added edge {src} -> {dst}")
                    return
                
    def mutate_remove_specialist(self):
        if len(self.model.specialists) <= 2: return
        
        spec_id = np.random.choice(list(self.model.specialists.keys()))
        self.model.graph.remove_node(spec_id)
        del self.model.specialists[spec_id]
        print(f"Specialists {spec_id} Removed")
        
    def adapt_diversity_pressure(self, current_f1):
        self.performance_history.append(current_f1)
        
        if self.is_stuck():
            self.stagnation_counter += 1
            self.diversity_pressure = min(1.0, self.diversity_pressure + 0.1)
        else:
            self.stagnation_counter = 0
            self.diversity_pressure = max(0.2, self.diversity_pressure - 0.05)
            
    def evolve(self):
        mutation_prob = self.diversity_pressure
        
        if np.random.random() < mutation_prob * 0.3:
            channels = [64, 128, 256, 512]
            out_c = np.random.choice(channels)
            self.mutate_add_specialist(out_channels=out_c)
            
        if np.random.random() < mutation_prob * 0.5: self.mutate_add_connection()
        
        if np.random.random() < mutation_prob * 0.2: self.mutate_remove_specialist()
        
        self.model.update_coupling()

dataset = MagnaTagATuneDataset(
	audio_dir='Dataset/wav_combined',
	annotations_file='Dataset/annotations_final.csv',
	clip_info_file='Dataset/clip_info.csv',
	sample_rate=16000,
	window_size=3.0
)

waveform, labels = dataset[0]
print("Waveform shape:", waveform.shape)
print("Labels:", labels)

val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size

indices = np.arange(len(dataset))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

train_dataset = Subset(dataset, train_indices.tolist())
val_dataset = Subset(dataset, val_indices.tolist())

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print(f"Train set size: {len(train_dataset)}")
print(f"Val set size:   {len(val_dataset)}")
print(f"Train batches:  {len(train_loader)}")
print(f"Val batches:    {len(val_loader)}")

for batch_waveforms, batch_labels in train_loader:
	print("Batch waveforms shape:", batch_waveforms.shape)
	print("Batch labels shape:", batch_labels.shape)
	break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EvolutionaryAudioGNN(in_channels=1, num_classes=dataset[0][1].shape[0]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def compute_pos_weights(dataset):
	labels = [label for _, label in dataset]
	label_matrix = torch.stack(labels)
	pos_counts = label_matrix.sum(dim=0)
	neg_counts = label_matrix.shape[0] - pos_counts
	weights = torch.log1p(neg_counts / (pos_counts + 1e-5)) * 2.0 #added multiplication of the function by 2
	weights /= weights.mean()
	return weights

def find_optimal_thresholds(y_true, y_scores,n_splits=3 ,num_thresholds=50):
	n_tags = y_true.shape[1]
	optimal_thresholds = np.full(n_tags,0.5)

	kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

	for tag_idx in range(n_tags):
		tag_true = y_true[:, tag_idx]
		tag_scores = y_scores[:, tag_idx]
		if tag_true.sum() == 0: continue

		best_threshold = 0.5
		best_f1 = 0.0

		thresholds = np.linspace(0.1,0.9,num_thresholds)

		for threshold in thresholds:
			f1_scores = []
			for train_idx, val_idx in kf.split(tag_true):
				train_preds = (tag_scores[train_idx] >= threshold).astype(int)
				train_f1 = f1_score(tag_true[train_idx], train_preds, zero_division=0)
				f1_scores.append(train_f1)
			avg_f1 = np.mean(f1_scores)
			if avg_f1 > best_f1:
				best_f1 = avg_f1
				best_threshold = threshold
		optimal_thresholds[tag_idx] = best_threshold
	return optimal_thresholds

pos_weights = compute_pos_weights(train_dataset).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

evolution_trainer = EvolutionaryTrainer(model,device)

num_epochs = 30 
best_f1 = 0.0

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

for epoch in range(num_epochs):
	model.train()

	total_loss = 0
	for waveforms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
		waveforms = waveforms.to(device)
		labels = labels.to(device)

		outputs = model(waveforms)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		
	avg_train_loss = total_loss / len(train_loader)
	
	model.eval()
	val_loss = 0
	all_preds, all_targets = [], []

	with torch.no_grad():
		for waveforms, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
			waveforms = waveforms.to(device)
			labels = labels.to(device)

			outputs = model(waveforms)
			loss = criterion(outputs, labels)
			val_loss += loss.item()

			preds = torch.sigmoid(outputs).cpu().numpy()
			targets = labels.cpu().numpy()
			all_preds.append(preds)
			all_targets.append(targets)
	avg_val_loss = val_loss / len(val_loader)

	y_true = np.concatenate(all_targets)
	y_scores = np.concatenate(all_preds)
	optimal_thresholds = find_optimal_thresholds(y_true, y_scores, n_splits=3)
	y_pred = np.zeros_like(y_scores)

	for tag_idx in range(y_scores.shape[1]): y_pred[:,tag_idx] = (y_scores[:,tag_idx] >= optimal_thresholds[tag_idx]).astype(int)
	
	f1 = f1_score(y_true, y_pred, average='micro')
	evolution_trainer.adapt_diversity_pressure(f1)
 
	if epoch > 1 and epoch % 3 == 0:
		print("\n Evolution Step")
		evolution_trainer.evolve()
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

		stats = model.get_architecture_stats()
		print(f"Architecture: {stats['num_specialists']} specialists, "
              f"{stats['num_edges']} edges, "
              f"{stats['num_coupled_pairs']} coupled pairs")
  
	print(f"Epoch [{epoch + 1}/{num_epochs} |  Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | F1 Score: {f1:.4f} | Diversity: {evolution_trainer.diversity_pressure:.2f}")
 
	if f1 > best_f1:
		best_f1 = f1
		torch.save({
			'epoch' : epoch + 1,
			'model_state_dict' : model.state_dict(),
			'optimizer_state_dict' : optimizer.state_dict(),
			'f1': f1,
			'architecture' : model.get_architecture_stats()
		}, 'evolutionary_checkpoint.pth')
		print(f"Model saved with F1: {f1:.4f}")