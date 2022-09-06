import torch 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE





def visualize_data(data, model, device):
	''''''

	model = model.to(device)
	for images, labels in data:
		if device:
			images = images.to(device)
			labels = labels.to(device)
		
		output = model(images)
	
	output = output.cpu().detach().numpy()
	labels = labels.to('cpu').numpy()

	tsne = TSNE(n_components = 2)
	embeddings = tsne.fit_transform(output)

	plt.figure(figsize = (10, 10))
	plt.title('The embeddings learned by ViT')
	plt.scatter(embeddings[:, 0], embeddings[:, 1], c = labels, s = 50, cmap = 'Paired')
	plt.colorbar()
	plt.show()




def clip_gradient(model, clip = 2.0):
	"""Rescales norm of computed gradients.

	Parameters
	----------
	model: nn.Module 
		Module.

	clip: float
		Maximum norm.

	"""

	for p in model.parameters():
		if p.grad is not None:
			param_norm = p.grad.data.norm()
			clip_coef = clip / (param_norm + 1e-6)
			if clip_coef < 1:
				p.grad.data.mul_(clip_coef)




def save_checkpoint(path, model_state, optimizer_state, loss):
	''''''

	torch.save({
		'model_state_dict': model_state,
		'optimizer_state_dict' : optimizer_state, 
		}, path)



def load_checkpoint(path, model, optimizer):
	checkpoint = torch.load(path)

	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	
	



def count_params(model):
	'''Returns the number of parameters of a model'''

	return sum([params.numel() for params in model.parameters() if params.requires_grad == True])

