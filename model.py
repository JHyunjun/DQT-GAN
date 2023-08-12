import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.act_1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout_1 = nn.Dropout(p=0.3)
        self.conv2d_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.act_2 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.conv2d_3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.act_3 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout_3 = nn.Dropout(p=0.3)
        self.conv2d_4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.act_4 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout_4 = nn.Dropout(p=0.3)
        self.conv2d_5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.act_5 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout_5 = nn.Dropout(p=0.3)
        self.conv2d_6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.act_6 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout_6 = nn.Dropout(p=0.3)
        self.conv2d_7 = nn.Conv2d(256, 512, 3, 1, 1)
        self.act_7 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout_7 = nn.Dropout(p=0.3)
        self.conv2d_8 = nn.Conv2d(512, 512, 3, 1, 1)
        self.act_8 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout_8 = nn.Dropout(p=0.3)
        self.trans_conv2d_1 = nn.ConvTranspose2d(512, 1, 3, 1, 1)

    def forward(self, x):
        x = self.dropout_1(self.act_1(self.conv2d_1(x)))
        x = self.dropout_2(self.act_2(self.conv2d_2(x))) + x
        x = self.dropout_3(self.act_3(self.conv2d_3(x)))
        x = self.dropout_4(self.act_4(self.conv2d_4(x))) + x
        x = self.dropout_5(self.act_5(self.conv2d_5(x)))
        x = self.dropout_6(self.act_6(self.conv2d_6(x))) + x
        x = self.dropout_7(self.act_7(self.conv2d_7(x)))
        x = self.dropout_8(self.act_8(self.conv2d_8(x))) + x
        x = self.trans_conv2d_1(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(16384, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x
