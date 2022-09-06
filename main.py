import torch 
import torch.nn as nn 
from utils import clip_gradient, count_params
from torchvision import transforms as T
from torchvision import datasets
import warmup_scheduler
from model import MLPMixer



def train_func(train_loader, model, optimizer, loss_func, max_epochs = 100, validation_loader = None, batch_size = 128, 
				scheduler = None, device = None, test_loader = None, train_loader_plain = None, clip_grad = 2.0):

	''' 
	This function takes raw data as input and converst it to data loader itself.
	Also, it does apply the model on the test data if test data is given. 

	'''


	n_batches_train = len(train_loader)
	n_batches_val = len(validation_loader)
	n_samples_train = batch_size * n_batches_train
	n_samples_val = batch_size * n_batches_val


	losses = []
	accuracy = []
	validation_loss = []
	validation_accuracy = []


	for epoch in range(max_epochs):
		running_loss, correct = 0, 0
		for images, labels in train_loader:
			if device:
				images = images.to(device)
				labels = labels.to(device)

			#================= Training =============================
			model.train()
			outputs = model(images)
			loss = loss_func(outputs, labels)
			predictions = outputs.argmax(1)
			correct += int(sum(predictions == labels))
			running_loss += loss.item()


			#======================== BACKWARD AND OPTIMZIE  =================================
			optimizer.zero_grad()
			loss.backward()
			clip_gradient(model, clip_grad)
			optimizer.step()


		loss_epoch = running_loss / n_batches_train
		accuracy_epoch = correct / n_samples_train
		scheduler.step()

		losses.append(loss_epoch)
		accuracy.append(accuracy_epoch)

		print('Epoch [{}/{}], Training Accuracy [{:.4f}], Training Loss: {:.4f}'
			.format(epoch + 1, max_epochs, accuracy_epoch, loss_epoch), end = '  ')


		#====================== Validation ============================
		if validation_loader:
			model.eval()

			val_loss, val_corr = 0, 0
			for val_images, val_labels in validation_loader:
				if device:
					val_images = val_images.to(device)
					val_labels = val_labels.to(device)

				outputs = model(val_images)
				loss = loss_func(outputs, val_labels)
				_, predictions = outputs.max(1)
				val_corr += int(sum(predictions == val_labels))
				val_loss += loss.item()


			loss_val = val_loss / n_batches_val
			accuracy_val = val_corr / n_samples_val

			validation_loss.append(loss_val)
			validation_accuracy.append(accuracy_val)


			print('Validation accuracy [{:.4f}], Validation Loss: {:.4f}'
				.format(accuracy_val, loss_val))


	if test:
		
		correct = 0
		total = 0

		for images, labels in test_loader:
			if device:
				images = images.to(device)
				labels = labels.to(device)

			n_data = images[0]
			total += n_data
			outputs = model(images)
			predictions = outputs.argmax(1)
			correct += int(sum(predictions == labels))

		accuracy = correct / total 
		print('Test Accuracy: {}'.format(accuracy))

	
	#====================== Saving the Model ============================  

	model_save_name = 'mlp_mixer.pt'
	path = F"/content/gdrive/My Drive/{model_save_name}" 
	torch.save({
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict' : optimizer.state_dict(), 
		}, path)



	return {'loss': losses, 'accuracy': accuracy, 
			'val_loss': validation_loss, 'val_accuracy': validation_accuracy}



	
def main(parameters):

	path = r"/content/gdrive/MyDrive/mlp_mixer_shahrivar"
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('Starting MLP-Mixer....')
	print(device)

	plain_tranformation = T.Compose([T.ToTensor(),
								    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

	transformed_cifar10 = datasets.CIFAR10(path, download = True, train = True, transform = plain_tranformation)
	transformed_cifar10_test = datasets.CIFAR10(path, download = True, train = False, transform = plain_tranformation)


	validation, test = torch.utils.data.random_split(transformed_cifar10_test, [5000, 5000])


	train_loader = torch.utils.data.DataLoader(transformed_cifar10, batch_size = parameters['batch_size'], shuffle = True, drop_last = True)
	test_loader = torch.utils.data.DataLoader(test, batch_size = parameters['batch_size'], shuffle= True, drop_last = True)   
	val_loader = torch.utils.data.DataLoader(validation, batch_size = parameters['batch_size'], shuffle = True, drop_last = True) 


	model = MLPMixer(in_channels = 3, img_size = parameters['img_size'], dim = parameters['dim'], num_classes = parameters['n_classes'],
					depth = parameters['depth'], patch_size = parameters['patch_size'], token_dim = parameters['token_dim'], 
					channel_dim = parameters['channel_dim']).to(device)

	n_parameters = count_params(model)
	print('The number of trainable parameters is : {}'.format(n_parameters))
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr = parameters['lr'], weight_decay = parameters['weight_decay'])
	base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min = 1e-6)
	scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=5, after_scheduler = base_scheduler)
	

	history = train_func(train_loader, model = model, optimizer = optimizer, loss_func = criterion, max_epochs = 100,  
						validation_loader = val_loader, batch_size = parameters['batch_size'], scheduler = scheduler, device = device,
						clip_grad = parameters['clip_grad'])


	return history, model 





if __name__ == '__main__': 

	parameters = {'batch_size': 512, 'lr': 0.0005, 'weight_decay': 0.05, 'img_size': 32,'depth' : 8, 
	  			  'patch_size' : 8, 'n_classes' : 10, 'max_epochs' : 100, 'clip_grad': 2.0, 'drop': 0., 
	  			  'dim': 384, 'token_dim': 256, 'channel_dim': 2048}


	history, model = main(parameters) 
