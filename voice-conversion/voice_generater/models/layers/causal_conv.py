import torch


class CausalConv1d(torch.nn.Module):
    """CausalConv1d module with customized initialization."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation=1, bias=True, pad="ConstantPad1d", pad_params={"value": 0.0}):
        """Initialize CausalConv1d module."""
        super(CausalConv1d, self).__init__()
        self.pad = getattr(torch.nn, pad)((kernel_size - 1) * dilation, **pad_params)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size,
                                    dilation=dilation, bias=bias)

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T).
        Returns:
            Tensor: Output tensor (B, out_channels, T).
        """
        return self.conv(self.pad(x))[:, :, :x.size(2)]


class CausalConvTranspose1d(torch.nn.Module):
    """CausalConvTranspose1d module with customized initialization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        """Initialize CausalConvTranspose1d module."""
        super(CausalConvTranspose1d, self).__init__()
        self.deconv = torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, bias=bias)
        self.stride = stride

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T_in).
        Returns:
            Tensor: Output tensor (B, out_channels, T_out).
        """
        return self.deconv(x)[:, :, :-self.stride]


class ResidualStack(torch.nn.Module):
    """Residual stack module introduced in MelGAN."""

    def __init__(self,
                 kernel_size=3,
                 channels=32,
                 dilation=1,
                 bias=True,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={},
                 use_causal_conv=False,
                 ):
        """Initialize ResidualStack module.
        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels of convolution layers.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_causal_conv (bool): Whether to use causal convolution.
        """
        super(ResidualStack, self).__init__()

        # defile residual stack part
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
            self.stack = torch.nn.Sequential(
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                getattr(torch.nn, pad)((kernel_size - 1) // 2 * dilation, **pad_params), 
                torch.nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=bias),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Conv1d(channels, channels, 1, bias=bias),
            )
        else:
            self.stack = torch.nn.Sequential(
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                CausalConv1d(channels, channels, kernel_size, dilation=dilation,
                             bias=bias, pad=pad, pad_params=pad_params),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Conv1d(channels, channels, 1, bias=bias),
            )

        # defile extra layer for skip connection
        self.skip_layer = torch.nn.Conv1d(channels, channels, 1, bias=bias)

    def forward(self, c):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, channels, T).
        Returns:
            Tensor: Output tensor (B, chennels, T).
        """
        return self.stack(c) + self.skip_layer(c)        