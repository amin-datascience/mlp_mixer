import torch 
import torch.nn as nn 



class PatchEmbedding(nn.Module):

	def __init__(self, img_size, patch_size, embed_dim, in_channels = 3):
		super(PatchEmbedding, self).__init__() 
		self.img_size = img_size 
		self.patch_size = patch_size 
		self.n_patches = (img_size // patch_size) ** 2

		self.transform = nn.Conv2d(in_channels = in_channels, out_channels = embed_dim, 
								   kernel_size = patch_size, stride = patch_size) 


	def forward(self, x):
		x = self.transform(x) #(n_samples, embed_dim, width, height)
		x = x.flatten(2)
		x = x.transpose(1, 2) #(n_samples, n_patches, embed_dim)

		return x



class MLP(nn.Module):

	def __init__(self, dim, hidden_dim, dropout):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, dim),
			nn.Dropout(dropout)
			)


	def forward(self, x):
		return self.mlp(x)



class MixerBlock(nn.Module):

	def __init__(self, dim, n_patches, token_dim, channel_dim, dropout = 0.):
		super().__init__()

		self.norm1 = nn.LayerNorm(dim)
		self.norm2 = nn.LayerNorm(dim)
		self.token_mixer = MLP(dim = n_patches, hidden_dim = token_dim, dropout = dropout)	
		self.channel_mixer = MLP(dim = dim, hidden_dim = channel_dim, dropout = dropout)


	def forward(self, x):

		x = x + self.token_mixer(self.norm1(x).transpose(-1, -2)).transpose(-1, -2 )
		x = x + self.channel_mixer(self.norm2(x))

		return x



class MLPMixer(nn.Module):

	def __init__(self, in_channels, img_size , dim, num_classes, depth, patch_size, token_dim, channel_dim):
		super().__init__()

		self.patch_embed = PatchEmbedding(img_size = img_size, patch_size = patch_size, embed_dim = dim, in_channels = 3)
		self.n_patches = self.patch_embed.n_patches

		self.blocks = nn.ModuleList([
			MixerBlock(dim = dim, n_patches = self.n_patches, token_dim = token_dim, channel_dim = channel_dim)
			for i in range(depth)
			])

		self.layer_norm = nn.LayerNorm(dim)
		self.head = nn.Linear(dim, num_classes)


	def forward(self, x):

		x = self.patch_embed(x)
		for block in self.blocks:
			x = block(x)

		x = self.layer_norm(x)
		x = x.mean(dim = 1)
		
		return self.head(x)


	


if __name__ == '__main__':

	x = torch.randn(1, 3, 32, 32)
	model = MLPMixer(in_channels = 3, img_size = 32, dim = 384, num_classes = 10, depth = 8, 
		patch_size = 8, token_dim = 256, channel_dim = 2048)

	out = model(x)
	print(out.shape)
