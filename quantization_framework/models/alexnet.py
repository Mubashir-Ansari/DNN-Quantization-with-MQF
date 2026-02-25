import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        # EXACT architecture from the user's p2q0-7ber00001.py script
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # Reset Global MQF Scale Transceiver for new batch
        from quantization.register_aware_executor import MQF_GLOBAL_CONTEXT
        MQF_GLOBAL_CONTEXT['last_scale'] = 1.0
        MQF_GLOBAL_CONTEXT['is_first_layer'] = True
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = out.view(out.size(0), -1)

        out = self.relu_fc1(self.fc1(out))
        out = self.dropout1(out)
        
        out = self.relu_fc2(self.fc2(out))
        out = self.dropout2(out)
        
        out = self.fc3(out)
        return torch.nn.functional.log_softmax(out, dim=1)

    def get_activation_mapping(self):
        """
        Maps weight-bearing modules to their final activation output modules
        for register packing simulation.
        """
        return {
            "conv1.0": "conv1",
            "conv2.0": "conv2",
            "conv3.0": "conv3",
            "conv4.0": "conv4",
            "conv5.0": "conv5",
            "fc1": "relu_fc1",
            "fc2": "relu_fc2",
            "fc3": "fc3"
        }

# Alias for checkpoints saved with this name
fasion_mnist_alexnet = AlexNet

def alexnet(num_classes=10, **kwargs):
    return AlexNet(num_classes=num_classes)
