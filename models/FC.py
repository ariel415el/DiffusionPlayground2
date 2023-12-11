import torch
import torch.nn as nn


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n, self.groups, c // self.groups, h, w)  # group
        x = x.transpose(1, 2).contiguous().view(n, -1, h, w)  # shuffle

        return x


class ConvBnSiLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.module = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                                    nn.BatchNorm2d(out_channels),
                                    nn.SiLU(inplace=True))

    def forward(self, x):
        return self.module(x)




class TimeMLP(nn.Module):
    '''
    naive introduce timestep information to feature maps with mlp and add shortcut
    '''

    def __init__(self, embedding_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                 nn.SiLU(),
                                 nn.Linear(hidden_dim, out_dim))
        self.act = nn.SiLU()

    def forward(self, x, t):
        t_emb = self.mlp(t)
        x = x + t_emb

        return self.act(x)


def FC_block(in_feat, out_feat, normalize='in'):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize == "bn":
        layers.append(nn.BatchNorm1d(out_feat))
    elif normalize == "in":
        layers.append(nn.InstanceNorm1d(out_feat))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return torch.nn.Sequential(*layers)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.conv0 = FC_block(in_channels, out_channels // 2)

        self.time_mlp = TimeMLP(embedding_dim=time_embedding_dim, hidden_dim=out_channels, out_dim=out_channels // 2)
        self.conv1 = FC_block(out_channels // 2, out_channels)

    def forward(self, x, t=None):
        x_shortcut = self.conv0(x)
        if t is not None:
            x = self.time_mlp(x_shortcut, t)
        x = self.conv1(x)

        return [x, x_shortcut]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.conv0 = FC_block(in_channels, in_channels // 2)

        self.time_mlp = TimeMLP(embedding_dim=time_embedding_dim, hidden_dim=in_channels, out_dim=in_channels // 2)
        self.conv1 = FC_block(in_channels // 2, out_channels // 2)

    def forward(self, x, x_shortcut, t=None):
        x = torch.cat([x, x_shortcut], dim=1)
        x = self.conv0(x)
        if t is not None:
            x = self.time_mlp(x, t)
        x = self.conv1(x)

        return x


class FC(nn.Module):
    '''
    simple unet design without attention
    '''

    def __init__(self, im_size, timesteps, time_embedding_dim, in_channels=3, out_channels=2, bottleneck_dim=32, n_layers=2):
        super().__init__()

        in_dim = in_channels * im_size**2
        channels = self._cal_channels(bottleneck_dim, n_layers)

        self.init_FC = FC_block(in_dim, channels[0][0])
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(c[0], c[1], time_embedding_dim) for c in channels])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(c[1], c[0], time_embedding_dim) for c in channels[::-1]])

        self.mid_block = FC_block(channels[-1][1], channels[-1][1] // 2)

        self.final_FC = FC_block(channels[0][0] // 2, out_channels * im_size**2)

    def forward(self, x, t=None):
        b,c,h,w = x.shape
        x = self.init_FC(x.reshape(b, -1))
        if t is not None:
            t = self.time_embedding(t)
        encoder_shortcuts = []
        for encoder_block in self.encoder_blocks:
            x, x_shortcut = encoder_block(x, t)
            encoder_shortcuts.append(x_shortcut)
        x = self.mid_block(x)
        encoder_shortcuts.reverse()
        for decoder_block, shortcut in zip(self.decoder_blocks, encoder_shortcuts):
            x = decoder_block(x, shortcut, t)
        x = self.final_FC(x).reshape(b,-1,h,w)

        return x

    def _cal_channels(self, bottleneck_dim, n_layers):
        dims = [bottleneck_dim * 2**i for i in range(n_layers+1)][::-1]
        # dims.insert(0, dims[-1])
        channels = []
        for i in range(len(dims) - 1):
            channels.append((dims[i], dims[i + 1]))  # in_channel, out_channel

        return channels


if __name__ == "__main__":
    x = torch.randn(15, 3, 64, 64)
    t = torch.randint(0, 1000, (15,))
    model = Unet(1000, 128)
    y = model(x, t)
    print(y.shape)
