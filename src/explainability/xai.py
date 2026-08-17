"""Interprétabilité : Grad-CAM (CNN) et attention rollout (ViT)."""
import numpy as np
import torch
import torch.nn.functional as F

from src import config


def denormalize(tensor):
    """Inverse la normalisation ImageNet pour l'affichage."""
    mean = np.array(config.IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(config.IMAGENET_STD).reshape(3, 1, 1)
    img = tensor.cpu().numpy() * std + mean
    return np.clip(img.transpose(1, 2, 0), 0, 1)


def gradcam_cnn(model, image_tensor, target_class=None):
    """Grad-CAM sur la dernière couche convolutive de ResNet18.

    Implémentation directe par hooks : évite une dépendance externe et
    rend le mécanisme explicite (utile pour la soutenance).

    IMPORTANT : le backbone étant gelé, on réactive temporairement les
    gradients — sans quoi la carte serait nulle.
    """
    model.eval()
    saved = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad = True

    activations, gradients = {}, {}
    target_layer = model.layer4[-1]

    h1 = target_layer.register_forward_hook(
        lambda m, i, o: activations.__setitem__("value", o))
    h2 = target_layer.register_full_backward_hook(
        lambda m, gi, go: gradients.__setitem__("value", go[0]))

    try:
        x = image_tensor.unsqueeze(0).clone().requires_grad_(True)
        logits = model(x)
        if target_class is None:
            target_class = logits.argmax(1).item()

        model.zero_grad()
        logits[0, target_class].backward()

        acts = activations["value"][0]          # (C, 7, 7)
        grads = gradients["value"][0]           # (C, 7, 7)
        weights = grads.mean(dim=(1, 2))        # importance par canal

        cam = F.relu((weights[:, None, None] * acts).sum(0))
        cam = F.interpolate(cam[None, None], size=(config.IMG_SIZE, config.IMG_SIZE),
                            mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cam.detach().cpu().numpy()
    finally:
        h1.remove(); h2.remove()
        for p, flag in zip(model.parameters(), saved):
            p.requires_grad = flag

    return cam, target_class


@torch.no_grad()
def attention_rollout_vit(model, image_tensor, discard_ratio=0.9):
    """Attention rollout (Abnar & Zuidema, 2020) sur ViT-Tiny.

    Multiplie les matrices d'attention de tous les blocs (identité ajoutée
    pour les connexions résiduelles), puis extrait la ligne du token [CLS].

    discard_ratio écarte les liens d'attention les plus faibles, qui
    ajoutent surtout du bruit au produit cumulé.
    """
    model.eval()
    attentions = []

    def hook(module, inputs, output):
        """Recalcule les poids d'attention.

        timm utilise une attention fusionnée qui n'expose pas la matrice ;
        on la reconstruit depuis la projection qkv du bloc.
        """
        x = inputs[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads,
                                    C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attentions.append(attn.softmax(dim=-1).detach().cpu())

    handles = [blk.attn.register_forward_hook(hook) for blk in model.blocks]
    try:
        logits = model(image_tensor.unsqueeze(0))
        pred = logits.argmax(1).item()
    finally:
        for h in handles:
            h.remove()

    # Produit cumulé des attentions, tête moyennée
    result = torch.eye(attentions[0].size(-1))
    for attn in attentions:
        a = attn.mean(dim=1)[0]                     # moyenne sur les têtes

        flat = a.flatten()
        n_discard = int(flat.numel() * discard_ratio)
        if n_discard > 0:
            _, idx = flat.topk(n_discard, largest=False)
            flat[idx] = 0
        a = flat.reshape(a.shape)

        a = a + torch.eye(a.size(-1))               # connexion résiduelle
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result

    n_patches = result.size(-1) - 1                 # hors token [CLS]
    grid = int(n_patches ** 0.5)
    mask = result[0, 1:].reshape(grid, grid)        # ligne du [CLS]

    mask = F.interpolate(mask[None, None], size=(config.IMG_SIZE, config.IMG_SIZE),
                         mode="bilinear", align_corners=False)[0, 0]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask.numpy(), pred


def overlay(image_np, heatmap, alpha=0.5):
    """Superpose une carte de chaleur (jet) à l'image dénormalisée."""
    import matplotlib.cm as cm
    colored = cm.jet(heatmap)[..., :3]
    return np.clip((1 - alpha) * image_np + alpha * colored, 0, 1)