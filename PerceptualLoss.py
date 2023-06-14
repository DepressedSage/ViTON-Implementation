import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F 
import utils.config as config

class Perceptual_Loss(nn.Module):
    def __init__(self, lambdas):
        super(Perceptual_Loss, self).__init__()
        self.lambdas = lambdas
        self.mse = torch.nn.MSELoss()
        self.vgg = models.vgg16(pretrained=True).eval()
        self.vgg = self.vgg.half().to(config.DEVICE)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, I_prime, I, M, M0):
        loss = 0.0
        
        def vgg_features(model, input_tensor):
            feature_maps = list()
            def hook_fn(module, input, output):
                feature_maps.append(output)

            target_layers = [model.features[layer_idx] for layer_idx in [2, 7, 12, 19, 26]]
            hooks = [layer.register_forward_hook(hook_fn) for layer in target_layers]

            model(input_tensor)
            return feature_maps, hooks

        phi_I_prime_features, I_prime_hooks = vgg_features(self.vgg, I_prime)
        phi_I_features, I_hooks = vgg_features(self.vgg, I)

        # Compute perceptual loss for each layer
        for i, lambda_i in enumerate(self.lambdas):
            if i == 0:
                phi_I_prime = I_prime
                phi_I = I
            else:
                phi_I_prime = phi_I_prime_features[i-1]
                phi_I = phi_I_features[i-1]
            loss += lambda_i * F.smooth_l1_loss(phi_I_prime.to(torch.float32), phi_I)

        # Compute L1 loss for mask
        loss += F.smooth_l1_loss(M.to(torch.float32), M0)

        for idx in range(len(I_prime_hooks)):
            I_prime_hooks[idx].remove()
            I_hooks[idx].remove()

        return loss

    def normalize_input(self, x):
        return (x - 0.5) * 2.0