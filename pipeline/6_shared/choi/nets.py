"""The networks in PyTorch: a U-Net close to the published design, a U-Net with residual blocks, squeeze-and-excitation gates and dropout, and a per-pixel network, with helpers for widths, dropout and counts of parameters and multiply-accumulate operations."""
import torch
import torch.nn as nn
BASE_CHANNELS = (64, 128, 256, 512)
SE_REDUCTION = 8

def width_channels(c, w):
    x = c * float(w)
    n = int(round(x))
    if n < 1 or abs(x - n) > 1e-09:
        raise ValueError('width %r gives %r channels for %d' % (w, x, c))
    return n

def conv_bn_relu(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())

def up_bn_lrelu(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))

class BaselineUNet(nn.Module):

    def __init__(self, in_ch=18, w=1.0):
        super().__init__()
        c64, c128, c256, c512 = (width_channels(c, w) for c in BASE_CHANNELS)
        self.in_ch = in_ch
        self.w = w
        self.enc1a = conv_bn_relu(in_ch, c64)
        self.enc1b = conv_bn_relu(c64, c128)
        self.down1 = conv_bn_relu(c128, c256, stride=2)
        self.enc2 = conv_bn_relu(c256, c256)
        self.down2 = conv_bn_relu(c256, c512, stride=2)
        self.enc3 = conv_bn_relu(c512, c512)
        self.up1 = up_bn_lrelu(c512, c512)
        self.dec1a = conv_bn_relu(c512 + c256, c512)
        self.dec1b = conv_bn_relu(c512, c256)
        self.up2 = up_bn_lrelu(c256, c256)
        self.dec2a = conv_bn_relu(c256 + c128, c128)
        self.dec2b = conv_bn_relu(c128, c64)
        self.head = nn.Conv2d(c64, 1, kernel_size=1)

    def forward(self, x):
        s128 = self.enc1b(self.enc1a(x))
        s256 = self.enc2(self.down1(s128))
        b = self.enc3(self.down2(s256))
        d = self.dec1b(self.dec1a(torch.cat([self.up1(b), s256], dim=1)))
        d = self.dec2b(self.dec2a(torch.cat([self.up2(d), s128], dim=1)))
        return self.head(d)

class ResBlock(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.act2 = nn.ReLU()

    def forward(self, x):
        y = self.act1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act2(y + x)

class ResBlockProj(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn_s = nn.BatchNorm2d(cout)
        self.act2 = nn.ReLU()

    def forward(self, x):
        y = self.act1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act2(y + self.bn_s(self.short(x)))

class SEGate(nn.Module):

    def __init__(self, c, reduction=SE_REDUCTION):
        super().__init__()
        r = c // reduction
        if r < 1 or r * reduction != c:
            raise ValueError('SE gate: %d channels, reduction %d' % (c, reduction))
        self.fc1 = nn.Linear(c, r)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(r, c)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=(2, 3))
        s = self.gate(self.fc2(self.act(self.fc1(s))))
        return x * s[:, :, None, None]

class ResUNet(nn.Module):

    def __init__(self, in_ch=24, w=1.0, dropout=0.1):
        super().__init__()
        c1, c2, c3, c4 = (width_channels(c, w) for c in BASE_CHANNELS)
        self.in_ch = in_ch
        self.w = w
        self.stem = conv_bn_relu(in_ch, c1)
        self.enc1 = ResBlock(c1)
        self.down1 = conv_bn_relu(c1, c2, stride=2)
        self.enc2 = ResBlock(c2)
        self.down2 = conv_bn_relu(c2, c3, stride=2)
        self.enc3 = ResBlock(c3)
        self.down3 = conv_bn_relu(c3, c4, stride=2)
        self.enc4 = ResBlock(c4)
        self.drop = nn.Dropout(dropout)
        self.up3 = up_bn_lrelu(c4, c3)
        self.dec3 = ResBlockProj(2 * c3, c3)
        self.se3 = SEGate(c3)
        self.up2 = up_bn_lrelu(c3, c2)
        self.dec2 = ResBlockProj(2 * c2, c2)
        self.se2 = SEGate(c2)
        self.up1 = up_bn_lrelu(c2, c1)
        self.dec1 = ResBlockProj(2 * c1, c1)
        self.se1 = SEGate(c1)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(self.stem(x))
        s2 = self.enc2(self.down1(s1))
        s3 = self.enc3(self.down2(s2))
        b = self.drop(self.enc4(self.down3(s3)))
        d = self.se3(self.dec3(torch.cat([self.up3(b), s3], dim=1)))
        d = self.se2(self.dec2(torch.cat([self.up2(d), s2], dim=1)))
        d = self.se1(self.dec1(torch.cat([self.up1(d), s1], dim=1)))
        return self.head(d)

class PixMLP(nn.Module):

    def __init__(self, in_ch=24):
        super().__init__()
        self.in_ch = in_ch
        self.body = nn.Sequential(nn.Conv2d(in_ch, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        return self.body(x)

def set_dropout(model, p):
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.p = float(p)
            n += 1
    return n

def count_params(model):
    p = sum((int(t.numel()) for t in model.parameters()))
    b = sum((int(t.numel()) for t in model.buffers() if t.dtype.is_floating_point))
    return {'parameters': p, 'float_buffer_elements': b}

def count_macs(model, in_ch, height=64, width=64):
    rows = []
    hooks = []

    def hook_conv(name):

        def f(mod, inp, out):
            k = mod.kernel_size[0] * mod.kernel_size[1]
            if isinstance(mod, nn.ConvTranspose2d):
                hw = inp[0].shape[2] * inp[0].shape[3]
                macs = hw * mod.in_channels * (mod.out_channels // mod.groups) * k
                kind = 'convtranspose2d'
            else:
                hw = out.shape[2] * out.shape[3]
                macs = hw * mod.out_channels * (mod.in_channels // mod.groups) * k
                kind = 'conv2d'
            rows.append({'layer': name, 'kind': kind, 'in': int(mod.in_channels), 'out': int(mod.out_channels), 'kernel': '%dx%d' % tuple(mod.kernel_size), 'stride': int(mod.stride[0]), 'in_hw': '%dx%d' % tuple(inp[0].shape[2:4]), 'out_hw': '%dx%d' % tuple(out.shape[2:4]), 'macs': int(macs)})
        return f

    def hook_linear(name):

        def f(mod, inp, out):
            rows.append({'layer': name, 'kind': 'linear', 'in': int(mod.in_features), 'out': int(mod.out_features), 'kernel': '-', 'stride': 0, 'in_hw': '-', 'out_hw': '-', 'macs': int(mod.in_features * mod.out_features)})
        return f
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(mod.register_forward_hook(hook_conv(name)))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(hook_linear(name)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            x = torch.zeros(1, in_ch, height, width)
            y = model(x)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    tot = {'conv2d': 0, 'convtranspose2d': 0, 'linear': 0}
    for r in rows:
        tot[r['kind']] += r['macs']
    conv_all = tot['conv2d'] + tot['convtranspose2d']
    return {'conv_and_transposed': conv_all, 'linear': tot['linear'], 'all_counted': conv_all + tot['linear'], 'by_kind': tot, 'output_shape': list(y.shape), 'layers': rows}
