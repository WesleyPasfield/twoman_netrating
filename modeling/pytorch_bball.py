import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.optim.optimizer import Optimizer
from tqdm import tqdm,trange

from typing import List, Any, Union, Dict, Optional, Tuple, Generator, Collection
from pathlib import PosixPath

use_cuda = torch.cuda.is_available()

def set_method(method:str):
	if method =='regression':
		return None, F.mse_loss
	if method =='logistic':
		return torch.sigmoid, F.binary_cross_entropy
	if method=='multiclass':
		return F.softmax, F.cross_entropy


def set_optimizer(model_params:Generator[Tensor,Tensor,Tensor], opt_params:List[Union[str,float]]):

	'''
	Gives option to pass in learning rate & momentum avlues
	Pass LR as second parameter, momentum as third parameter
	'''

	try:
		opt, lr, m = opt_params
	except:
		opt, lr = opt_params
	if opt == "Adam":
		return torch.optim.Adam(model_params, lr=lr)
	if opt == "Adagrad":
		return torch.optim.Adam(model_params, lr=lr)
	if opt == "RMSprop":
		return torch.optim.RMSprop(model_params, lr=lr, momentum=m)
	if opt == "SGD":
		return torch.optim.SGD(model_params, lr=lr, momentum=m)


def set_scheduler(optimizer:Optimizer, sch_params:List[Any]):

	'''
	Learning rate schedule
	'StepLR', 3, 0.1
	'''

	sch,s,g = sch_params
	if sch == "StepLR":
		return StepLR(optimizer, step_size=s, gamma=g)
	if sch == "MultiStepLR":
		return MultiStepLR(optimizer, milestones=s, gamma=g)
	if sch == "CosineAnnealing":
		return CosineAnnealing(optimizer, )

def dl(inputlen:int, outputlen:int, dropout:float):
	return nn.Sequential(
		nn.Linear(inputlen, outputlen),
		nn.LeakyReLU(inplace=True),
		nn.Dropout(dropout)
		)

class DeepDense(nn.Module):
	def __init__(self,
				 embeddings_input:List[Tuple[str,int,int]],
				 embeddings_encoding_dict:Dict[str,Any],
				 continuous_cols:List[str],
				 deep_column_idx:Dict[str,int],
				 hidden_layers:List[int],
				 dropout:List[float],
				 output_dim:int):
		"""
		Model consisting in a series of Dense Layers that receive continous
		features concatenated with categorical features represented with
		embeddings
		Parameters:
		embeddings_input: List
			List of Tuples with the column name, number of unique values and
			embedding dimension. e.g. [(education, 11, 32), ...]
		embeddings_encoding_dict: Dict
			Dict containing the encoding mappings
		continuous_cols: List
			List with the name of the continuous cols
		deep_column_idx: Dict
			Dict containing the index of the embedding columns. Required to
			slice the tensors.
		hidden_layers: List
			List with the number of neurons per dense layer. e.g: [64,32]
		dropout: List
			List with the dropout between the dense layers. We do not apply dropout
			between Embeddings and first dense or last dense and output. Therefore
			this list must contain len(hidden_layers)-1 elements. e.g: [0.5]
		output_dim: int
			1 for logistic regression or regression, N-classes for multiclass
		"""
		super(DeepDense, self).__init__()

		self.embeddings_input = embeddings_input
		self.embeddings_encoding_dict = embeddings_encoding_dict
		self.continuous_cols = continuous_cols
		self.deep_column_idx = deep_column_idx

		for col,val,dim in embeddings_input:
			setattr(self, 'emb_layer_'+col, nn.Embedding(val, dim))
		input_emb_dim = np.sum([emb[2] for emb in embeddings_input])+len(continuous_cols)
		print(input_emb_dim)
		hidden_layers = [input_emb_dim] + hidden_layers
		print(hidden_layers)
		dropout = [0.0] + dropout
		self.dense = nn.Sequential()
		for i in range(1, len(hidden_layers)):
			self.dense.add_module(
				'dense_layer_{}'.format(i-1),
				dl( hidden_layers[i-1], hidden_layers[i], dropout[i-1])
				)
		self.dense.add_module('last_linear', nn.Linear(hidden_layers[-1], output_dim))

	def forward(self, X:Tensor)->Tensor:
		emb = [getattr(self, 'emb_layer_'+col)(X[:,self.deep_column_idx[col]].long()) for col,_,_ in self.embeddings_input]
		if self.continuous_cols:
			cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
			cont = [X[:, cont_idx].float()]
			inp = torch.cat(emb+cont, 1)
		else:
			inp = torch.cat(emb, 1)
		out = self.dense(inp)
		return out

	def model_compile(self,
				method:str,
				optimizer,
				lr_scheduler:Optional[Dict[str,List[Any]]]=None):
		"""Set the activation, loss and the optimizer.
		Parameters:
		----------
		method: str
			'regression', 'logistic' or 'multiclass'
		optimizer: str or Dict
			if str one of the following: 'SGD', 'Adam', or 'RMSprop'
			if Dict must contain  elements for different models
			e.g. optimizer = {'wide: ['SGD', 0.001, 0.3]', 'deep':['Adam', 0.001]}
		"""

		self.method = method
		self.activation, self.criterion = set_method(method)

		deep_dense_opt = set_optimizer(self.parameters(), optimizer)
		deep_dense_sch = set_scheduler(deep_dense_opt, lr_scheduler) if lr_scheduler else None
		self.optimizer = deep_dense_opt
		self.lr_scheduler = deep_dense_sch

	def fit(self, n_epochs:int, train_loader:DataLoader, eval_loader:Optional[DataLoader]=None):

		train_steps =  (len(train_loader.dataset) // train_loader.batch_size) + 1
		if eval_loader:
			eval_steps =  (len(eval_loader.dataset) // eval_loader.batch_size) + 1
		for epoch in range(n_epochs):
			if self.lr_scheduler: self.lr_scheduler.step()
			net = self.train() ## Explicitly states this is training
			total, correct, running_loss = 0,0,0
			with trange(train_steps) as t: ## trange a function within tqdm progress package
				for i, (data,target) in zip(t, train_loader):
					t.set_description('epoch %i' % (epoch+1))
					X = tuple(x.cuda() for x in data) if use_cuda else data
					y = target.float() if self.method != 'multiclass' else target
					y = y.cuda() if use_cuda else y

					self.optimizer.zero_grad() ## Zeros out gradient. For RNN can skip this for BPTT. Weights are accumlated.
					y_pred =  net(X) ## Executes forward pass
					if(self.criterion == F.cross_entropy): ## for classification
						loss = self.criterion(y_pred, y)
					else:
						loss = self.criterion(y_pred, y.view(-1, 1)) ## View rearranges dimensions of Tensor
					loss.backward() ##
					self.optimizer.step() ## Parameter update based on current gradient

					running_loss += loss.item()
					avg_loss = running_loss/(i+1)

					if self.method != "regression":
						total+= y.size(0)
						if self.method == 'logistic':
							y_pred_cat = (y_pred > 0.5).squeeze(1).float()
						if self.method == "multiclass":
							_, y_pred_cat = torch.max(y_pred, 1)
						correct+= float((y_pred_cat == y).sum().item())
						t.set_postfix(acc=correct/total, loss=avg_loss)
					else:
						t.set_postfix(loss=np.sqrt(avg_loss)) ## For printing out progress

			if eval_loader:
				total, correct, running_loss = 0,0,0
				net = self.eval()
				with torch.no_grad():
					with trange(eval_steps) as v:
						for i, (data,target) in zip(v, eval_loader):
							v.set_description('valid')
							X = tuple(x.cuda() for x in data) if use_cuda else data
							y = target.float() if self.method != 'multiclass' else target
							y = y.cuda() if use_cuda else y
							y_pred =  net(X)
							if(self.criterion == F.cross_entropy):
								loss = self.criterion(y_pred, y)
							else:
								loss = self.criterion(y_pred, y.view(-1, 1))
							running_loss += loss.item()
							avg_loss = running_loss/(i+1)
							if self.method != "regression":
								total+= y.size(0)
								if self.method == 'logistic':
									y_pred_cat = (y_pred > 0.5).squeeze(1).float()
								if self.method == "multiclass":
									_, y_pred_cat = torch.max(y_pred, 1)
								correct+= float((y_pred_cat == y).sum().item())
								v.set_postfix(acc=correct/total, loss=avg_loss)
							else:
								v.set_postfix(loss=np.sqrt(avg_loss))

	def predict(self, dataloader:DataLoader)->np.ndarray:
		test_steps =  (len(dataloader.dataset) // dataloader.batch_size) + 1
		net = self.eval()
		preds_l = []
		with torch.no_grad():
			with trange(test_steps) as t:
				for i, data in zip(t, dataloader):
					t.set_description('predict')
					X = tuple(x.cuda() for x in data) if use_cuda else data
					# This operations is cheap in terms of computing time, but
					# would be more efficient to append Tensors and then cat
					preds_l.append(net(X).cpu().data.numpy())
			return np.vstack(preds_l).squeeze(1)

	def get_embeddings(self, col_name:str) -> Dict[str,np.ndarray]:

		## Make this work

		params = list(self.named_parameters())
		emb_layers = [p for p in params if 'emb_layer' in p[0]]
		emb_layer  = [layer for layer in emb_layers if col_name in layer[0]][0]
		embeddings = emb_layer[1].cpu().data.numpy()
		col_label_encoding = self.embeddings_encoding_dict[col_name]
		inv_dict = {v:k for k,v in col_label_encoding.items()}
		embeddings_dict = {}
		for idx,value in inv_dict.items():
			embeddings_dict[value] = embeddings[idx]
		return embeddings_dict

class model_dataloader(Dataset):

	"""
	Loads in dataframe and converts to prep for modeling
	"""

	def __init__(self, xvals, yvals, mode:str='train', uhlog=False):

		### Need to figure out how to convert test properly

		self.mode = mode
		self.xdata = xvals
		if uhlog is True:
			self.ydata = np.log(yvals + 35.0)
		else:
			self.ydata = yvals


	def __getitem__(self, idx:int):

		X = self.xdata[idx]
		if self.mode == 'test':
			return X
		if self.mode == 'train':
			y  = self.ydata[idx]
			return X, y

	def __len__(self):
		return len(self.xdata)