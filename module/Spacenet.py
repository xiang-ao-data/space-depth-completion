import torch
from module.basic_model import *
from module.common import *
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dFunction
import torch
import torch.nn as nn

class backbone_with_dream(nn.Module):
    def __init__(self, args):
        super(backbone_with_dream, self).__init__()
        self.args = args
        self.cbam_64 = SAMMAFB(64)
        self.cbam_128 = SAMMAFB(128)
        self.cbam_256 = SAMMAFB(256)
        self.cbam_512 = SAMMAFB(512)
        self.cbam_1024 = SAMMAFB(1024)
        self.cbam_192 = SAMMAFB(192)
        self.cbam_448 = SAMMAFB(448)
        self.cbam_960 = SAMMAFB(960)
        self.cbam_1984 = SAMMAFB(1984)
        self.rgb_conv_init = convbnrelu(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.rgb_encoder_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2)
        self.rgb_encoder_layer3 = BasicBlockGeo(inplanes=64, planes=128, stride=2)
        self.rgb_encoder_layer5 = BasicBlockGeo(inplanes=128, planes=256, stride=2)
        self.rgb_encoder_layer7 = BasicBlockGeo(inplanes=256, planes=512, stride=2)
        self.rgb_encoder_layer9 = BasicBlockGeo(inplanes=512, planes=1024, stride=2)

        self.rgb_decoder_layer8 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2,
                                               output_padding=1)
        self.rgb_decoder_layer6 = deconvbnrelu(in_channels=1024, out_channels=256, kernel_size=5, stride=2, padding=2,
                                               output_padding=1)
        self.rgb_decoder_layer4 = deconvbnrelu(in_channels=512, out_channels=128, kernel_size=5, stride=2, padding=2,
                                               output_padding=1)
        self.rgb_decoder_layer2 = deconvbnrelu(in_channels=256, out_channels=64, kernel_size=5, stride=2, padding=2,
                                               output_padding=1)
        self.rgb_decoder_layer0 = deconvbnrelu(in_channels=128, out_channels=32, kernel_size=5, stride=2, padding=2,
                                               output_padding=1)
        self.rgb_decoder_output = deconvbnrelu(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1,
                                               output_padding=0)
        
        self.rgb_reconstructed_layer8 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2,
                                               output_padding=1)
        self.rgb_reconstructed_layer6 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2,
                                               output_padding=1)
        self.rgb_reconstructed_layer4 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2,
                                               output_padding=1)
        self.rgb_reconstructed_layer2 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2,
                                               output_padding=1)
        self.rgb_reconstructed_layer0 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2,
                                               output_padding=1)
        self.rgb_reconstructed_output = deconvbnrelu(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1,
                                               output_padding=0)
        
        # depth encoder
        self.depth_conv_init = convbnrelu(in_channels=2, out_channels=32, kernel_size=5, stride=1,
                                          padding=2)  # rgb confidence and segment confidence
        self.depth_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2)
        self.depth_layer3 = BasicBlockGeo(inplanes=192, planes=192, stride=2)
        self.depth_layer5 = BasicBlockGeo(inplanes=448, planes=448, stride=2)
        self.depth_layer7 = BasicBlockGeo(inplanes=960, planes=960, stride=2)
        self.depth_layer9 = BasicBlockGeo(inplanes=1984, planes=1984, stride=2)

        # decoder
        self.decoder_layer0 = deconvbnrelu(in_channels=1984, out_channels=1024, kernel_size=1, stride=1, padding=0,
                                           output_padding=0)
        self.decoder_layer1 = deconvbnrelu(in_channels=1024, out_channels=960, kernel_size=5, stride=2, padding=2,
                                           output_padding=1)
        self.decoder_layer2 = deconvbnrelu(in_channels=960, out_channels=448, kernel_size=5, stride=2, padding=2,
                                           output_padding=1)
        self.decoder_layer3 = deconvbnrelu(in_channels=448, out_channels=192, kernel_size=5, stride=2, padding=2,
                                           output_padding=1)
        self.decoder_layer4 = deconvbnrelu(in_channels=192, out_channels=64, kernel_size=5, stride=2, padding=2,
                                           output_padding=1)
        self.decoder_layer5 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2,
                                           output_padding=1)
        self.decoder_layer6 = convbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=1)
        weights_init(self)
    
    def forward(self,input):
        rgb = input['rgb']
        d = input['d']
        plane = input['plane']
        rgb_feature = self.rgb_conv_init(torch.cat((rgb, d), dim=1))
        rgb_feature1 = self.rgb_encoder_layer1(rgb_feature)
        rgb_feature3 = self.rgb_encoder_layer3(rgb_feature1)
        rgb_feature5 = self.rgb_encoder_layer5(rgb_feature3)
        rgb_feature7 = self.rgb_encoder_layer7(rgb_feature5)
        rgb_feature9 = self.rgb_encoder_layer9(rgb_feature7)
        rgb_feature10 = rgb_feature9

        rgb_feature_reconstructed8 = self.rgb_reconstructed_layer8(rgb_feature10)
        rgb_recons8_plus = rgb_feature_reconstructed8 + rgb_feature7
        rgb_feature_reconstructed6 = self.rgb_reconstructed_layer6(rgb_recons8_plus)
        rgb_recons6_plus = rgb_feature_reconstructed6 + rgb_feature5
        rgb_feature_reconstructed4 = self.rgb_reconstructed_layer4(rgb_recons6_plus)
        rgb_recons4_plus = rgb_feature_reconstructed4 + rgb_feature3
        rgb_feature_reconstructed2 = self.rgb_reconstructed_layer2(rgb_recons4_plus)
        rgb_recons2_plus = rgb_feature_reconstructed2 + rgb_feature1
        rgb_feature_reconstructed0 = self.rgb_reconstructed_layer0(rgb_recons2_plus)
        rgb_recons0_plus = rgb_feature_reconstructed0 + rgb_feature
        rgb_reconstruction = self.rgb_reconstructed_output(rgb_recons0_plus)


        rgb_feature_decoder8 = self.rgb_decoder_layer8(rgb_feature10)
        rgb_feature8_plus = torch.cat((rgb_feature_decoder8, rgb_recons8_plus), dim=1)
        rgb_feature8_plus = self.cbam_1024(rgb_feature8_plus)
        rgb_feature_decoder6 = self.rgb_decoder_layer6(rgb_feature8_plus)
        rgb_feature6_plus = torch.cat((rgb_feature_decoder6, rgb_recons6_plus), dim=1)
        rgb_feature6_plus = self.cbam_512(rgb_feature6_plus)
        rgb_feature_decoder4 = self.rgb_decoder_layer4(rgb_feature6_plus)
        rgb_feature4_plus = torch.cat((rgb_feature_decoder4, rgb_recons4_plus), dim=1)
        rgb_feature4_plus = self.cbam_256(rgb_feature4_plus)
        rgb_feature_decoder2 = self.rgb_decoder_layer2(rgb_feature4_plus)
        rgb_feature2_plus = torch.cat((rgb_feature_decoder2, rgb_recons2_plus), dim=1)
        rgb_feature2_plus = self.cbam_128(rgb_feature2_plus)
        rgb_feature_decoder0 = self.rgb_decoder_layer0(rgb_feature2_plus)
        rgb_feature0_plus = torch.cat((rgb_feature_decoder0, rgb_recons0_plus), dim=1)
        rgb_feature0_plus = self.cbam_64(rgb_feature0_plus)
        rgb_output = self.rgb_decoder_output(rgb_feature0_plus)
        rgb_depth = rgb_output[:, 0:1, :, :]
        rgb_conf = rgb_output[:, 1:2, :, :]

        sparsed_feature = self.depth_conv_init(torch.cat((d, rgb_depth), dim=1))
        sparsed_feature1 = self.depth_layer1(sparsed_feature)
        sparsed_feature2_plus = torch.cat([rgb_feature2_plus, sparsed_feature1], 1)
        sparsed_feature2_plus = self.cbam_192(sparsed_feature2_plus)
        sparsed_feature3 = self.depth_layer3(sparsed_feature2_plus)
        sparsed_feature4_plus = torch.cat([rgb_feature4_plus, sparsed_feature3], 1)
        sparsed_feature4_plus = self.cbam_448(sparsed_feature4_plus)
        sparsed_feature5 = self.depth_layer5(sparsed_feature4_plus)
        sparsed_feature6_plus = torch.cat([rgb_feature6_plus, sparsed_feature5], 1)
        sparsed_feature6_plus = self.cbam_960(sparsed_feature6_plus)
        sparsed_feature7 = self.depth_layer7(sparsed_feature6_plus)
        sparsed_feature8_plus = torch.cat([rgb_feature8_plus, sparsed_feature7], 1)
        sparsed_feature8_plus = self.cbam_1984(sparsed_feature8_plus)
        sparsed_feature9 = self.depth_layer9(sparsed_feature8_plus)

        if self.args.useplane == True:
            plane_feature = self.depth_conv_init(torch.cat((plane, rgb_depth), dim=1))
            plane_feature1 = self.depth_layer1(plane_feature)
            plane_feature2_plus = torch.cat([rgb_feature2_plus, plane_feature1], 1)
            plane_feature2_plus = self.cbam_192(plane_feature2_plus)
            plane_feature3 = self.depth_layer3(plane_feature2_plus)
            plane_feature4_plus = torch.cat([rgb_feature4_plus, plane_feature3], 1)
            plane_feature4_plus = self.cbam_448(plane_feature4_plus)
            plane_feature5 = self.depth_layer5(plane_feature4_plus)
            plane_feature6_plus = torch.cat([rgb_feature6_plus, plane_feature5], 1)
            plane_feature6_plus = self.cbam_960(plane_feature6_plus)
            plane_feature7 = self.depth_layer7(plane_feature6_plus)
            plane_feature8_plus = torch.cat([rgb_feature8_plus, plane_feature7], 1)
            plane_feature8_plus = self.cbam_1984(plane_feature8_plus)
            plane_feature9 = self.depth_layer9(plane_feature8_plus)
            sparsed_feature9 = plane_feature9+sparsed_feature9
            sparsed_feature7 = plane_feature7+sparsed_feature7
            sparsed_feature5 = plane_feature5+sparsed_feature5
            sparsed_feature3 = plane_feature3+sparsed_feature3
            sparsed_feature1 = plane_feature1+sparsed_feature1
        
        decoder_feature0 = self.decoder_layer0(sparsed_feature9)
        fusion1 = decoder_feature0
        decoder_feature1 = self.decoder_layer1(fusion1)
        fusion2 = sparsed_feature7 + decoder_feature1
        decoder_feature2 = self.decoder_layer2(fusion2)
        fusion3 = sparsed_feature5 + decoder_feature2
        decoder_feature3 = self.decoder_layer3(fusion3)
        fusion4 = sparsed_feature3 + decoder_feature3
        decoder_feature4 = self.decoder_layer4(fusion4)
        fusion5 = sparsed_feature1 + decoder_feature4
        decoder_feature5 = self.decoder_layer5(fusion5)

        depth_output = self.decoder_layer6(decoder_feature5)
        d_depth, d_conf = torch.chunk(depth_output, 2, dim=1)
    
        rgb_conf, d_conf = torch.chunk(self.softmax(torch.cat((rgb_conf, d_conf), dim=1)),
                                                      2, dim=1)
        output = rgb_conf * rgb_depth + d_conf * d_depth + d


        if self.args.model_name == 'NLSPN':
            return torch.cat((rgb_feature0_plus, decoder_feature5),1), \
                rgb_depth, rgb_reconstruction, d_depth, output

        elif self.args.model_name == 'CSPN':
            return torch.cat((rgb_feature0_plus, decoder_feature5),1), torch.cat([decoder_feature4,rgb_feature2_plus],1), rgb_depth, rgb_reconstruction, d_depth, output
        else:
            return rgb_depth, rgb_reconstruction,d_depth, output


class NLSPN(nn.Module):
    def __init__(self, args, ch_g, ch_f, k_g, k_f):
        super(NLSPN, self).__init__()

        # Guidance : [B x ch_g x H x W]
        # Feature : [B x ch_f x H x W]

        # Currently only support ch_f == 1
        assert ch_f == 1, 'only tested with ch_f == 1 but {}'.format(ch_f)

        assert (k_g % 2) == 1, \
            'only odd kernel is supported but k_g = {}'.format(k_g)
        pad_g = int((k_g - 1) / 2)
        assert (k_f % 2) == 1, \
            'only odd kernel is supported but k_f = {}'.format(k_f)
        pad_f = int((k_f - 1) / 2)

        self.args = args
        self.prop_time = self.args.prop_time
        self.affinity = self.args.affinity

        self.ch_g = ch_g
        self.ch_f = ch_f
        self.k_g = k_g
        self.k_f = k_f
        # Assume zero offset for center pixels
        self.num = self.k_f * self.k_f - 1
        self.idx_ref = self.num // 2

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            self.conv_offset_aff = nn.Conv2d(
                self.ch_g, 3 * self.num, kernel_size=self.k_g, stride=1,
                padding=pad_g, bias=True
            )
            self.conv_offset_aff.weight.data.zero_()
            self.conv_offset_aff.bias.data.zero_()

            if self.affinity == 'TC':
                self.aff_scale_const = nn.Parameter(self.num * torch.ones(1))
                self.aff_scale_const.requires_grad = False
            elif self.affinity == 'TGASS':
                self.aff_scale_const = nn.Parameter(
                    self.args.affinity_gamma * self.num * torch.ones(1))
            else:
                self.aff_scale_const = nn.Parameter(torch.ones(1))
                self.aff_scale_const.requires_grad = False
        else:
            raise NotImplementedError

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = pad_f
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64

    def _get_offset_affinity(self, guidance, confidence=None, rgb=None):
        B, _, H, W = guidance.shape

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            offset_aff = self.conv_offset_aff(guidance)
            o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)

            # Add zero reference offset
            offset = torch.cat((o1, o2), dim=1).view(B, self.num, 2, H, W)
            list_offset = list(torch.chunk(offset, self.num, dim=1))
            list_offset.insert(self.idx_ref,
                               torch.zeros((B, 1, 2, H, W)).type_as(offset))
            offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

            if self.affinity in ['AS', 'ASS']:
                pass
            elif self.affinity == 'TC':
                aff = torch.tanh(aff) / self.aff_scale_const
            elif self.affinity == 'TGASS':
                aff = torch.tanh(aff) / (self.aff_scale_const + 1e-8)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Apply confidence
        if self.args.conf_prop:
            list_conf = []
            offset_each = torch.chunk(offset, self.num + 1, dim=1)

            modulation_dummy = torch.ones((B, 1, H, W)).type_as(offset).detach()

            for idx_off in range(0, self.num + 1):
                ww = idx_off % self.k_f
                hh = idx_off // self.k_f

                if ww == (self.k_f - 1) / 2 and hh == (self.k_f - 1) / 2:
                    continue

                offset_tmp = offset_each[idx_off].detach()
                if self.args.legacy:
                    offset_tmp[:, 0, :, :] = \
                        offset_tmp[:, 0, :, :] + hh - (self.k_f - 1) / 2
                    offset_tmp[:, 1, :, :] = \
                        offset_tmp[:, 1, :, :] + ww - (self.k_f - 1) / 2

                # conf_tmp = ModulatedDeformConvFunction.apply(
                #     confidence, offset_tmp, modulation_dummy, self.w_conf,
                #     self.b, self.stride, 0, self.dilation, self.groups,
                #     self.deformable_groups, self.im2col_step)
                conf_tmp = ModulatedDeformConv2dFunction.apply(confidence, offset_tmp, modulation_dummy,
                                                               self.w_conf, self.b, self.stride, 0, self.dilation,
                                                               self.groups, self.deformable_groups)
                list_conf.append(conf_tmp)

            conf_aff = torch.cat(list_conf, dim=1)
            aff = aff * conf_aff.contiguous()

        # Affinity normalization
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4

        if self.affinity in ['ASS', 'TGASS']:
            aff_abs_sum[aff_abs_sum < 1.0] = 1.0

        if self.affinity in ['AS', 'ASS', 'TGASS']:
            aff = aff / aff_abs_sum

        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum

        list_aff = list(torch.chunk(aff, self.num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)

        return offset, aff

    def _propagate_once(self, feat, offset, aff):
        # feat = ModulatedDeformConvFunction.apply(
        #     feat, offset, aff, self.w, self.b, self.stride, self.padding,
        #     self.dilation, self.groups, self.deformable_groups, self.im2col_step
        # )
        feat = ModulatedDeformConv2dFunction.apply(feat, offset, aff,
                                                   self.w, self.b, self.stride,
                                                   self.padding, self.dilation,
                                                   self.groups, self.deformable_groups)

        return feat

    def forward(self, feat_init, guidance, confidence=None, feat_fix=None,
                rgb=None):
        assert self.ch_g == guidance.shape[1]
        assert self.ch_f == feat_init.shape[1]

        if self.args.conf_prop:
            assert confidence is not None

        if self.args.conf_prop:
            offset, aff = self._get_offset_affinity(guidance, confidence, rgb)
        else:
            offset, aff = self._get_offset_affinity(guidance, None, rgb)

        # Propagation
        if self.args.preserve_input:
            assert feat_init.shape == feat_fix.shape
            mask_fix = torch.sum(feat_fix > 0.0, dim=1, keepdim=True).detach()
            mask_fix = (mask_fix > 0.0).type_as(feat_fix)

        feat_result = feat_init

        list_feat = []

        for k in range(1, self.prop_time + 1):
            # Input preservation for each iteration
            if self.args.preserve_input:
                feat_result = (1.0 - mask_fix) * feat_result \
                              + mask_fix * feat_fix

            feat_result = self._propagate_once(feat_result, offset, aff)

            list_feat.append(feat_result)

        return feat_result, list_feat, offset, aff, self.aff_scale_const.data


class NLSPNModel(nn.Module):
    def __init__(self, args):
        super(NLSPNModel, self).__init__()

        self.backbone = backbone_with_dream(args)
        self.args = args
        self.num_neighbors = self.args.prop_kernel * self.args.prop_kernel - 1

        # Encoder
        self.conv1_rgb = conv_bn_relu(1, 48, kernel=3, stride=1, padding=1,
                                      bn=False)
        self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                      bn=False)


        # Guidance Branch
        # 1/1
        self.gd_dec1 = conv_bn_relu(96, 32, kernel=3, stride=1,
                                    padding=1)
        self.gd_dec0 = conv_bn_relu(64 + 32, self.num_neighbors, kernel=3, stride=1,
                                    padding=1, bn=False, relu=False)

        if self.args.conf_prop:
            # Confidence Branch
            # Confidence is shared for propagation and mask generation
            # 1/1
            self.cf_dec1 = conv_bn_relu(96, 32, kernel=3, stride=1,
                                        padding=1)
            self.cf_dec0 = nn.Sequential(
                nn.Conv2d(32 + 64, 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )

        self.prop_layer = NLSPN(args, self.num_neighbors, 1, 3,
                                self.args.prop_kernel)

        # Set parameter groups
        params = []
        for param in self.named_parameters():
            if param[1].requires_grad:
                params.append(param[1])

        params = nn.ParameterList(params)

        self.param_groups = [
            {'params': params, 'lr': self.args.lr}
        ]

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, sample):

        rgb = sample['rgb']
        feature1, rgb_depth, semantic_depth, d_depth, dep = self.backbone(sample)
        pred_init = dep

        # Encoding
        fe1_rgb = self.conv1_rgb(rgb)
        fe1_dep = self.conv1_dep(dep)
        fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)

        # Guidance Decoding
        gd_fd1 = self.gd_dec1(feature1)
        guide = self.gd_dec0(self._concat(gd_fd1, fe1))

        if self.args.conf_prop:
            # Confidence Decoding
            cf_fd1 = self.cf_dec1(feature1)
            confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
        else:
            confidence = None

        # Diffusion
        y, y_inter, offset, aff, aff_const = \
            self.prop_layer(pred_init, guide, confidence, dep, rgb)

        # Remove negative depth
        y = torch.clamp(y, min=0)

        output = {'pred': y, 'pred_init': pred_init, 'pred_inter': y_inter,
                  'guidance': guide, 'offset': offset, 'aff': aff,
                  'gamma': aff_const, 'confidence': confidence}

        return output

class A_CSPN_plus_plus(nn.Module):
    def __init__(self, args):
        super(A_CSPN_plus_plus, self).__init__()

        self.backbone = backbone_with_dream(args)
        self.kernel_conf_layer = convbn(96, 3)
        self.mask_layer = convbn(96, 1)
        self.iter_guide_layer3 = CSPNGenerate(96, 3)
        self.iter_guide_layer5 = CSPNGenerate(96, 5)
        self.iter_guide_layer7 = CSPNGenerate(96, 7)

        self.kernel_conf_layer_s2 = convbn(192, 3)
        self.mask_layer_s2 = convbn(192, 1)
        self.iter_guide_layer3_s2 = CSPNGenerate(192, 3)
        self.iter_guide_layer5_s2 = CSPNGenerate(192, 5)
        self.iter_guide_layer7_s2 = CSPNGenerate(192, 7)

        self.dimhalf_s2 = convbnrelu(192, 96, 1, 1, 0)
        self.att_12 = convbnrelu(192, 2)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.downsample = SparseDownSampleClose(stride=2)
        self.softmax = nn.Softmax(dim=1)
        self.CSPN3 = CSPN(3)
        self.CSPN5 = CSPN(5)
        self.CSPN7 = CSPN(7)

        weights_init(self)

    def forward(self, input):
        d = input['d']
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
        feature_s1, feature_s2, rgb_depth, rgb_reconstruction, d_depth, output = self.backbone(input)
        #feature_s1, feature_s2, coarse_depth = self.backbone(input)
        depth = output

        d_s2, valid_mask_s2 = self.downsample(d, valid_mask)
        mask_s2 = self.mask_layer_s2(feature_s2)
        mask_s2 = torch.sigmoid(mask_s2)
        mask_s2 = mask_s2*valid_mask_s2

        kernel_conf_s2 = self.kernel_conf_layer_s2(feature_s2)
        kernel_conf_s2 = self.softmax(kernel_conf_s2)
        kernel_conf3_s2 = kernel_conf_s2[:, 0:1, :, :]
        kernel_conf5_s2 = kernel_conf_s2[:, 1:2, :, :]
        kernel_conf7_s2 = kernel_conf_s2[:, 2:3, :, :]

        mask = self.mask_layer(feature_s1)
        mask = torch.sigmoid(mask)
        mask = mask*valid_mask

        kernel_conf = self.kernel_conf_layer(feature_s1)
        kernel_conf = self.softmax(kernel_conf)
        kernel_conf3 = kernel_conf[:, 0:1, :, :]
        kernel_conf5 = kernel_conf[:, 1:2, :, :]
        kernel_conf7 = kernel_conf[:, 2:3, :, :]

        feature_12 = torch.cat((feature_s1, self.upsample(self.dimhalf_s2(feature_s2))), 1)
        att_map_12 = self.softmax(self.att_12(feature_12))

        guide3_s2 = self.iter_guide_layer3_s2(feature_s2)
        guide5_s2 = self.iter_guide_layer5_s2(feature_s2)
        guide7_s2 = self.iter_guide_layer7_s2(feature_s2)
        guide3 = self.iter_guide_layer3(feature_s1)
        guide5 = self.iter_guide_layer5(feature_s1)
        guide7 = self.iter_guide_layer7(feature_s1)

        depth_s2 = depth
        depth_s2_00 = depth_s2[:, :, 0::2, 0::2]
        depth_s2_01 = depth_s2[:, :, 0::2, 1::2]
        depth_s2_10 = depth_s2[:, :, 1::2, 0::2]
        depth_s2_11 = depth_s2[:, :, 1::2, 1::2]

        depth_s2_00_h0 = depth3_s2_00 = depth5_s2_00 = depth7_s2_00 = depth_s2_00
        depth_s2_01_h0 = depth3_s2_01 = depth5_s2_01 = depth7_s2_01 = depth_s2_01
        depth_s2_10_h0 = depth3_s2_10 = depth5_s2_10 = depth7_s2_10 = depth_s2_10
        depth_s2_11_h0 = depth3_s2_11 = depth5_s2_11 = depth7_s2_11 = depth_s2_11

        for i in range(6):
            depth3_s2_00 = self.CSPN3(guide3_s2, depth3_s2_00, depth_s2_00_h0)
            depth3_s2_00 = mask_s2*d_s2 + (1-mask_s2)*depth3_s2_00
            depth5_s2_00 = self.CSPN5(guide5_s2, depth5_s2_00, depth_s2_00_h0)
            depth5_s2_00 = mask_s2*d_s2 + (1-mask_s2)*depth5_s2_00
            depth7_s2_00 = self.CSPN7(guide7_s2, depth7_s2_00, depth_s2_00_h0)
            depth7_s2_00 = mask_s2*d_s2 + (1-mask_s2)*depth7_s2_00

            depth3_s2_01 = self.CSPN3(guide3_s2, depth3_s2_01, depth_s2_01_h0)
            depth3_s2_01 = mask_s2*d_s2 + (1-mask_s2)*depth3_s2_01
            depth5_s2_01 = self.CSPN5(guide5_s2, depth5_s2_01, depth_s2_01_h0)
            depth5_s2_01 = mask_s2*d_s2 + (1-mask_s2)*depth5_s2_01
            depth7_s2_01 = self.CSPN7(guide7_s2, depth7_s2_01, depth_s2_01_h0)
            depth7_s2_01 = mask_s2*d_s2 + (1-mask_s2)*depth7_s2_01

            depth3_s2_10 = self.CSPN3(guide3_s2, depth3_s2_10, depth_s2_10_h0)
            depth3_s2_10 = mask_s2*d_s2 + (1-mask_s2)*depth3_s2_10
            depth5_s2_10 = self.CSPN5(guide5_s2, depth5_s2_10, depth_s2_10_h0)
            depth5_s2_10 = mask_s2*d_s2 + (1-mask_s2)*depth5_s2_10
            depth7_s2_10 = self.CSPN7(guide7_s2, depth7_s2_10, depth_s2_10_h0)
            depth7_s2_10 = mask_s2*d_s2 + (1-mask_s2)*depth7_s2_10

            depth3_s2_11 = self.CSPN3(guide3_s2, depth3_s2_11, depth_s2_11_h0)
            depth3_s2_11 = mask_s2*d_s2 + (1-mask_s2)*depth3_s2_11
            depth5_s2_11 = self.CSPN5(guide5_s2, depth5_s2_11, depth_s2_11_h0)
            depth5_s2_11 = mask_s2*d_s2 + (1-mask_s2)*depth5_s2_11
            depth7_s2_11 = self.CSPN7(guide7_s2, depth7_s2_11, depth_s2_11_h0)
            depth7_s2_11 = mask_s2*d_s2 + (1-mask_s2)*depth7_s2_11

        depth_s2_00 = kernel_conf3_s2*depth3_s2_00 + kernel_conf5_s2*depth5_s2_00 + kernel_conf7_s2*depth7_s2_00
        depth_s2_01 = kernel_conf3_s2*depth3_s2_01 + kernel_conf5_s2*depth5_s2_01 + kernel_conf7_s2*depth7_s2_01
        depth_s2_10 = kernel_conf3_s2*depth3_s2_10 + kernel_conf5_s2*depth5_s2_10 + kernel_conf7_s2*depth7_s2_10
        depth_s2_11 = kernel_conf3_s2*depth3_s2_11 + kernel_conf5_s2*depth5_s2_11 + kernel_conf7_s2*depth7_s2_11

        depth_s2[:, :, 0::2, 0::2] = depth_s2_00
        depth_s2[:, :, 0::2, 1::2] = depth_s2_01
        depth_s2[:, :, 1::2, 0::2] = depth_s2_10
        depth_s2[:, :, 1::2, 1::2] = depth_s2_11

        #feature_12 = torch.cat((feature_s1, self.upsample(self.dimhalf_s2(feature_s2))), 1)
        #att_map_12 = self.softmax(self.att_12(feature_12))
        refined_depth_s2 = depth*att_map_12[:, 0:1, :, :] + depth_s2*att_map_12[:, 1:2, :, :]
        #refined_depth_s2 = depth

        depth3 = depth5 = depth7 = refined_depth_s2

        #prop
        for i in range(6):
            depth3 = self.CSPN3(guide3, depth3, depth)
            depth3 = mask*d + (1-mask)*depth3
            depth5 = self.CSPN5(guide5, depth5, depth)
            depth5 = mask*d + (1-mask)*depth5
            depth7 = self.CSPN7(guide7, depth7, depth)
            depth7 = mask*d + (1-mask)*depth7

        refined_depth = kernel_conf3*depth3 + kernel_conf5*depth5 + kernel_conf7*depth7
       
        return  rgb_depth, d_depth,refined_depth