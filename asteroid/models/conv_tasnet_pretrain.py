from asteroid_filterbanks import make_enc_dec
from ..masknn import TDConvNet
from .base_models import BaseEncoderMaskerDecoder
import warnings
import hubconf
import torch
from upstream.interfaces import Featurizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PretrainEncoder(torch.nn.Module):
    def __init__(
        self,
        upstream,
        upstream_ckpt,
        upstream_model_config,
        upstream_feature_selection,
        sample_rate,
        n_filters
    ):
        super(PretrainEncoder, self).__init__()
        self.upstream = upstream
        self.sample_rate = sample_rate
        self.upstream_feature_selection = upstream_feature_selection
        assert sample_rate == 16000
        Upstream = getattr(hubconf, self.upstream)
        self.upstream = Upstream(
            ckpt = upstream_ckpt,
            model_config = upstream_model_config,
            refresh = False,
        ).to(device)
        self.featurizer = Featurizer(
            self.upstream,
            upstream_feature_selection
        ).to(device)
        self.conv = torch.nn.Conv1d(in_channels=self.featurizer.output_dim, 
                out_channels=n_filters, 
                kernel_size=1)
        self.downsample_rate = self.featurizer.downsample_rate
        self.n_feats_out = n_filters

    def forward(self, x):
        wavs = [x[i, 0, :] for i in range(len(x))]
        #print("wavs", wavs[0].size())
        #print("wavs", len(wavs))
        with torch.no_grad():
            self.upstream.eval()
            features = self.upstream(wavs)
        #print("features", len(features))
        #print("features", features.keys())
        features = self.featurizer(wavs, features)
        #print("features", len(features))
        #print("features", features[0].size())
        features = torch.transpose(torch.stack(features), 1, 2)
        #print("features", features.size())
        features = self.conv(features)
        #print("features", features.size())
        return features

class PretrainDecoder(torch.nn.Module):
    def __init__(
        self,
        downsample_rate,
        n_filters,
    ):
        super(PretrainDecoder, self).__init__()
        self.transconv = torch.nn.ConvTranspose1d(
                in_channels=n_filters,
                out_channels=1,
                kernel_size=downsample_rate * 2,
                stride=downsample_rate,
        )
        
    def forward(self, x):
        bs, n_spk = x.size(0), x.size(1)
        x = x.view(-1, x.size(2), x.size(3))
        x = self.transconv(x)
        x = x.view(bs, n_spk, -1)
        return x

class ConvTasNetPretrain(BaseEncoderMaskerDecoder):
    """ConvTasNet separation model, as described in [1].

    Args:
        n_src (int): Number of sources in the input mixtures.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        conv_kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        causal (bool, optional) : Whether or not the convolutions are causal.
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sampling rate of the model.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References
        - [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
          for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
          https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        n_src,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="gLN",
        mask_act="sigmoid",
        in_chan=None,
        causal=False,
        fb_name="free",
        kernel_size=16,
        n_filters=512,
        stride=8,
        encoder_activation=None,
        sample_rate=8000,
        upstream=None,
        upstream_ckpt=None, 
        upstream_model_config=None,
        upstream_feature_selection='hidden_states',
        **fb_kwargs,
    ):
        encoder = PretrainEncoder(
                upstream,
                upstream_ckpt,
                upstream_model_config,
                upstream_feature_selection,
                sample_rate,
                n_filters
            )
        decoder = PretrainDecoder(
                downsample_rate = encoder.downsample_rate,
                n_filters = n_filters
        )
        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )
        if causal and norm_type not in ["cgLN", "cLN"]:
            norm_type = "cLN"
            warnings.warn(
                "In causal configuration cumulative layer normalization (cgLN)"
                "or channel-wise layer normalization (chanLN)  "
                f"must be used. Changing {norm_type} to cLN"
            )
        # Update in_chan
        masker = TDConvNet(
            n_feats,
            n_src,
            out_chan=out_chan,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            bn_chan=bn_chan,
            hid_chan=hid_chan,
            skip_chan=skip_chan,
            conv_kernel_size=conv_kernel_size,
            norm_type=norm_type,
            mask_act=mask_act,
            causal=causal,
        )
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)
