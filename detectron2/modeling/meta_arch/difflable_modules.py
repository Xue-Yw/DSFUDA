import torch
import torch.nn as nn


class NoiseAdapter(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size == 3:
            self.feat = nn.Sequential(
                Bottleneck(channels, channels, reduction=8),
                nn.AdaptiveAvgPool2d(1)
            )
        else:
            self.feat = nn.Sequential(
                nn.Conv2d(channels, channels * 2, 1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels * 2, channels, 1),
                nn.BatchNorm2d(channels),
            )
        self.pred = nn.Linear(channels, 2)

    def forward(self, x):
        x = self.feat(x).flatten(1)
        x = self.pred(x).softmax(1)[:, 0]
        return x

    
class DiffusionModel(nn.Module):
    def __init__(self, channels_in, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.time_embedding = nn.Embedding(1280, channels_in)

        if kernel_size == 3:
            self.pred = nn.Sequential(
                Bottleneck(channels_in, channels_in),
                Bottleneck(channels_in, channels_in),
                nn.Conv2d(channels_in, channels_in, 1),
                nn.BatchNorm2d(channels_in),
                #diffusion
                # nn.Linear(2069 * 7 * 7, 21),
                nn.Conv2d(channels_in, 21, kernel_size=1)
            )
            self.fc = nn.Linear(21 * 7 * 7, 21)
        else:
            self.pred = nn.Sequential(
                nn.Conv2d(channels_in, channels_in * 4, 1),
                nn.BatchNorm2d(channels_in * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in * 4, channels_in, 1),
                nn.BatchNorm2d(channels_in),
                nn.Conv2d(channels_in, channels_in * 4, 1),
                nn.BatchNorm2d(channels_in * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in * 4, channels_in, 1)
            )

    def forward(self, noisy_image, t):#[512, 2069, 7, 7]
        if t.dtype != torch.long:
            t = t.type(torch.long)
        feat = noisy_image#[512, 2069, 7, 7]
        feat = feat + self.time_embedding(t)[..., None, None]
        ret = self.pred(feat)#[512, 21, 7, 7]
        ret = ret.view(ret.size(0), -1)  # 展平操作[512, 1029]
        ret = self.fc(ret)#512, 21]
        return ret


class AutoEncoder(nn.Module):
    def __init__(self, channels, latent_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, latent_channels, 1, padding=0),
            nn.BatchNorm2d(latent_channels)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, channels, 1, padding=0),
        )

    def forward(self, x):
        hidden = self.encoder(x)
        out = self.decoder(hidden)
        return hidden, out

    def forward_encoder(self, x):
        return self.encoder(x)
    

class DDIMPipeline:
    '''
    Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddim/pipeline_ddim.py
    '''

    def __init__(self, model, scheduler, noise_adapter=None, solver='ddim'):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.noise_adapter = noise_adapter
        self._iter = 0
        self.solver = solver

    def __call__(
            self,
            batch_size,#512
            device,
            dtype,#torch.float32
            shape,#shape=gt_feat.shape[1:] #[2069, 7, 7]
            feat,#feat=gt_feat,#[512, 2069, 7, 7] grad_fn=<CopySlices>
            generator = None,
            eta: float = 0.0,
            num_inference_steps: int = 50,#5
            proj = None
    ):

        # Sample gaussian noise to begin loop
        image_shape = (batch_size, *shape)#(512, 2069, 7, 7)
        
        # print(f"Before operation, version: {feat._version}")
        if self.noise_adapter is not None:
            noise = torch.randn([image_shape[0],21], device=device, dtype=dtype)#只产生标注的noise
            noise = noise.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)#[512, 21, 7, 7]
            # grad_fn=<SelectBackward0>
            timesteps = self.noise_adapter(feat)#512
            # feat加噪
            # feat[:,-21:,:,:] = self.scheduler.add_noise_diff2(feat[:,-21:,:,:], noise, timesteps)
            feat = torch.cat((feat[:, :-21, :, :], self.scheduler.add_noise_diff2(feat[:,-21:,:,:], noise, timesteps)), dim=1)
            image = feat#[512, 2069, 7, 7]
        else:
            image = feat
        # print(f"Before operation, version: {feat._version}")
        # image = feat

        # set step values
        self.scheduler.set_timesteps(num_inference_steps*2)

        # import pdb
        # pdb.set_trace()

        for t in self.scheduler.timesteps[len(self.scheduler.timesteps)//2:]:#[400, 300, 200, 100,   0]
            noise_pred = self.model(image, t.to(device))#[512, 21]

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            # 翻译如下：

# 预测图像 \( x_{t-1} \) 的前一个均值，并根据 \(\eta\) 添加方差
#             # \(\eta\) 对应论文中的 \(\eta\)，取值范围应在 [0, 1] 之间
#             # 从 \( x_t \) 到 \( x_{t-1} \)
            noised_label = image[:,-21:,:,:][:, :, 0, 0]
            denoised_label= self.scheduler.step(
                noise_pred, t, noised_label, eta=eta, use_clipped_model_output=True, generator=generator
            )['prev_sample'] 
            image[:,-21:,:,:] = denoised_label.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)
        
        # import pdb
        # pdb.set_trace()

        self._iter += 1        
        return image


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        return out + x
