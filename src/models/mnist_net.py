import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN architecture for MNIST / FashionMNIST classification.
#
# use_dropout 開關：
#   True  (預設) = 兩層 Dropout 皆啟用；attack_defense_pipeline 直接呼叫 mnistNet()
#                  取得此行為（訓練時開 dropout、eval 時自動關）。
#   False        = 完全關閉 Dropout；mnist/main.ipynb 以此模式訓練
#                  *_cnn_no_dropout.pt 乾淨模型。
# 兩種模式的 state_dict 鍵相同（Dropout 無參數），因此權重檔可互通載入。
class mnistNet(nn.Module):
    def __init__(self, use_dropout=True):
        super(mnistNet, self).__init__()
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)  # 26x26x32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)  # 24x24x64
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(in_features=9216, out_features=128)  # 12*12*64
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        if self.use_dropout:
            x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)

        if self.use_dropout:
            x = self.dropout2(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output
