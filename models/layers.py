from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride = 1):
        super(ResidualBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride = stride, padding=1, padding_mode = 'reflect'),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, padding_mode = 'reflect')
        )
        self.idt_conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, stride = stride)

    def forward(self, x):
        x_idt = self.idt_conv(x)
        x = self.res_block(x)

        return x + x_idt
    

