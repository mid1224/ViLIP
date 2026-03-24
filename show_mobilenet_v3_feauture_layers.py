import torchvision.models as models

pretrained_weights = models.MobileNet_V3_Small_Weights.DEFAULT
mobilenet_v3_small_model = models.mobilenet_v3_small(weights=pretrained_weights)

print(mobilenet_v3_small_model)

print("\nFEATURES LAYERS")
for idx, layer in enumerate(mobilenet_v3_small_model.features):
    print(f"features.{idx}: {layer}")