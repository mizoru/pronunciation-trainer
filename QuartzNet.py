class NonSeparableQuartzSubblock(Module):
    r"A basic building sub-block of Quartznet with non-separable Conv layers"
    
    def __init__(self, in_channels, out_channels, kernel, stride, drop = 0.2, **kwargs):
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride, (kernel-1)//2, **kwargs)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)
        
    def forward(self, x, res = 0):
        x = self.conv(x)
        x = self.norm(x)
        if not isinstance(res, int): x += res
        x = self.act(x)
        return self.drop(x)
    
    
class QuartzSubblock(Module):
    r"A basic building sub-block of QuartzNet with separable convolutions"
    
    def __init__(self, in_channels, out_channels, kernel, stride, drop = 0.2, **kwargs):
        self.conv = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel, stride, (kernel-1)//2, groups=in_channels, **kwargs),
                                 nn.Conv1d(in_channels, out_channels, 1, 1))
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)
        
    def forward(self, x, res = 0):
        x = self.conv(x)
        x = self.norm(x)
        if not isinstance(res, int): x += res
        x = self.act(x)
        return self.drop(x)
    
class QuartzBlock(Module):
    
    def __init__(self, in_channels, out_channels, kernel, num_subblocks=3, drop = 0.2):
        self.subblocks = nn.Sequential(*[QuartzSubblock(in_channels, out_channels, kernel, 1, drop = 0.2)] \
        + [QuartzSubblock(out_channels, out_channels, kernel, 1, drop = 0.2) for i in range(num_subblocks-2)])
        self.res_subblock = QuartzSubblock(out_channels, out_channels, kernel, 1, drop = 0.2)
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.res_norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x, res = self.subblocks(x), self.res_conv(x)
        x = self.res_subblock(x, self.res_norm(x))
        return x
    
class QuartzNet(Module):
    
    def __init__(self, n_mels, n_vocab, num_blocks_factor=2, num_subblocks = 5):
        KERNELS = [33, 39, 51, 63, 75]
        OUTS = [256, 256, 512, 512, 512, 256]
        DROPS = [0.2]*3 + [0.3]*2
        
        self.model = nn.Sequential(*[NonSeparableQuartzSubblock(n_mels, 256, 33, 2)] \
        + [QuartzBlock((OUTS[i], OUTS[i-1])[j==0], OUTS[i], KERNELS[i], num_subblocks, DROPS[i])
           for i in range(5) for j in range(num_blocks_factor)] \
        + [QuartzSubblock(OUTS[-2], 512, 87, 1, 0.4)],
          NonSeparableQuartzSubblock(512, 1024, 1, 1, 0.4), nn.Conv1d(1024, n_vocab, 1, dilation=2), nn.Softmax(1)
        )
        
    def forward(self, x):
        return self.model(x)
    
    
    
    
    
    
class NonSeparableQuartzSubblock(Module):
    r"A basic building sub-block of Quartznet with non-separable Conv layers"
    
    def __init__(self, in_channels, out_channels, kernel, stride, drop = 0, **kwargs):
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride, (kernel-1)//2, **kwargs)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)
        
    def forward(self, x, res = 0):
        x = self.conv(x)
        x = self.norm(x)
        if not isinstance(res, int): x += res
        x = self.act(x)
        return self.drop(x)
    
    
class QuartzSubblock(Module):
    r"A basic building sub-block of QuartzNet with separable convolutions"
    
    def __init__(self, in_channels, out_channels, kernel, stride, drop = 0, **kwargs):
        self.conv = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel, stride, (kernel-1)//2, groups=in_channels, **kwargs),
                                 nn.Conv1d(in_channels, out_channels, 1, 1))
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)
        
    def forward(self, x, res = 0):
        x = self.conv(x)
        x = self.norm(x)
        if not isinstance(res, int): x += res
        x = self.act(x)
        return self.drop(x)
    
class QuartzBlock(Module):
    
    def __init__(self, in_channels, out_channels, kernel, num_subblocks=3, drop = 0):
        self.subblocks = nn.Sequential(*[QuartzSubblock(in_channels, out_channels, kernel, 1, drop = 0)] \
        + [QuartzSubblock(out_channels, out_channels, kernel, 1, drop = 0.) for i in range(num_subblocks-2)])
        self.res_subblock = QuartzSubblock(out_channels, out_channels, kernel, 1, drop = 0)
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.res_norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x, res = self.subblocks(x), self.res_conv(x)
        x = self.res_subblock(x, self.res_norm(x))
        return x
    
class QuartzNet(Module):
    
    def __init__(self, n_mels, n_vocab, num_blocks_factor=2, num_subblocks = 5):
        KERNELS = [33, 39, 51, 63, 75]
        OUTS = [256, 256, 512, 512, 512, 256]
        # DROPS = [0.2]*3 + [0.3]*2
        DROPS = [0.] * 5
        self.model = nn.Sequential(*[NonSeparableQuartzSubblock(n_mels, 256, 33, 2)] \
        + [QuartzBlock((OUTS[i], OUTS[i-1])[j==0], OUTS[i], KERNELS[i], num_subblocks, DROPS[i])
           for i in range(5) for j in range(num_blocks_factor)] \
        + [NonSeparableQuartzSubblock(OUTS[-2], 512, 87, 1, 0.4)],
          NonSeparableQuartzSubblock(512, 1024, 1, 1, 0.4, dilation=2),
        nn.Conv1d(1024, n_vocab, 1),
                                   nn.LogSoftmax(1)
        )
        
    def forward(self, x):
        return self.model(x)