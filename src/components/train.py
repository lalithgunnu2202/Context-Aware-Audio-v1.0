# Pseudo-code for the CVAE Forward Pass
import torch
# import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCVAE(nn.Module):
    def __init__(self, clip_dim=768, latent_dim=128, input_shape=(1, 64, 64)):
        super(AudioCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape # (Channels, Height, Width) of Mel Spec
        
        # --- Encoder ---
        # We flatten the Mel Spec and concat with CLIP embedding
        flat_size = input_shape[0] * input_shape[1] * input_shape[2]
        self.encoder_net = nn.Sequential(
            nn.Linear(flat_size + clip_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # --- Decoder ---
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim + clip_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, flat_size),
            nn.Sigmoid() # Use Sigmoid if Mel Specs are normalized [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, mel_spec, clip_emb):
        # Flatten Mel Spec: [Batch, 1, 64, 64] -> [Batch, 4096]
        x = mel_spec.view(mel_spec.size(0), -1)
        
        # ENCODE: Concat(Audio, CLIP)
        enc_input = torch.cat([x, clip_emb], dim=1)
        h = self.encoder_net(enc_input)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        
        # SAMPLE
        z = self.reparameterize(mu, logvar)
        
        # DECODE: Concat(Z, CLIP)
        dec_input = torch.cat([z, clip_emb], dim=1)
        recon_x = self.decoder_net(dec_input)
        
        return recon_x.view(-1, *self.input_shape), mu, logvar

    def generate(self, clip_emb):
        """Inference Phase: Only needs the CLIP embedding"""
        z = torch.randn(clip_emb.size(0), self.latent_dim).to(clip_emb.device)
        dec_input = torch.cat([z, clip_emb], dim=1)
        recon_x = self.decoder_net(dec_input)
        return recon_x.view(-1, *self.input_shape)
    

def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (how close is the generated audio to real?)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence (regularization for latent space)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kld_loss

# Example Training Step
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_step(mel_data, image_features):
    optimizer.zero_grad()
    
    # image_features is your [1, 768] CLIP vector
    recon_mel, mu, logvar = model(mel_data, image_features)
    
    loss = loss_function(recon_mel, mel_data, mu, logvar)
    loss.backward()
    optimizer.step()
    return loss.item()

from transformers import CLIPProcessor, CLIPVisionModel
from PIL import Image

# 1. Load Pretrained CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Load your Trained VAE
vae = AudioCVAE().to(device) 
# vae.load_state_dict(torch.load("vae_weights.pt"))

# 3. RUN INFERENCE
def generate_audio_from_image(img_path):
    # Prepare Image
    image = Image.open(img_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Get CLIP Context Vector (Pooled Output)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        # Using the pooled output [1, 768]
        scene_context = outputs.pooler_output 
        
    # Generate Mel Spectrogram from VAE Decoder
    with torch.no_grad():
        generated_mel = vae.generate(scene_context)
        
    return generated_mel # This is your Audio Representation!

# Result: [1, 1, 64, 64] tensor representing audio frequency over time
result_spec = generate_audio_from_image("forest.jpg")