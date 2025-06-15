import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Same architecture as used during training
class KeypointClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(21 * 2, 20),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class KeyPointClassifier:
    def __init__(self, model_path='model/keypoint_classifier/keypoint_classifier_weights.pth', device='cpu'):
        self.device = torch.device(device)
        self.model = KeypointClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, landmark_list):
        # Convert input to tensor and add batch dimension
        input_tensor = torch.tensor([landmark_list], dtype=torch.float32).to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze()
            result_index = torch.argmax(probs).item()

        return result_index
