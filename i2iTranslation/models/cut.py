"""
Implementation of Contrastive Unpaired Translation (CUT)
with neighborhood, global and local consistency objectives
for unpaired image-to-image translation.
"""

import numpy as np
import mlflow
import torch

from i2iTranslation.models.base_model import BaseModel
from i2iTranslation.models import networks
from i2iTranslation.models.neighborhood_objectives import GANLoss, PatchNCELoss
from i2iTranslation.models.global_objectives import SCLossCriterion
from i2iTranslation.models.local_objectives import DiscriminatorROI, ClassifierROI
from i2iTranslation.utils.util import robust_mlflow


class CUTModel(BaseModel):
    def __init__(self, args, device):
        """
        Parameters:
            args -- stores all the experiment information
        """
        BaseModel.__init__(self, args, device)

        self.set_params(args)

        # define generator network
        self.netG = networks.define_G(args=args)
        self.netF = networks.define_F(args=args, device=device)
        robust_mlflow(mlflow.log_param, 'netG_params',
                      sum(p.numel() for p in self.netG.parameters() if p.requires_grad))
        robust_mlflow(mlflow.log_param, 'netF_params',
                      sum(p.numel() for p in self.netF.parameters() if p.requires_grad))

        if self.is_train:
            # define discriminator network
            self.netD = networks.define_D(args=args)
            robust_mlflow(mlflow.log_param, 'netD_params',
                          sum(p.numel() for p in self.netD.parameters() if p.requires_grad))

            # define loss functions
            self.criterionGAN = GANLoss(self.gan_mode, device=device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionNCE = PatchNCELoss(args, self.device)
            if self.is_roi_discriminator:
                self.criterionBCE = torch.nn.BCELoss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=self.lr_G,
                betas=(self.beta1, self.beta2)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=self.lr_D,
                betas=(self.beta1, self.beta2)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # image-level style and content modules
            if self.is_image_match:
                self.criterionImage = SCLossCriterion(args, device)
                self.optimizer_I = torch.optim.Adam(
                    self.netG.parameters(),
                    lr=self.lr_I,
                    betas=(self.beta1, self.beta2)
                )
                self.optimizers.append(self.optimizer_I)

            # bounding box discriminator
            if self.is_roi_discriminator:
                # ROI discriminator
                self.netD_ROI = DiscriminatorROI(base_filters=self.roi_d_base_filters)
                robust_mlflow(mlflow.log_param, 'netD_ROI_params',
                              sum(p.numel() for p in self.netD_ROI.parameters() if p.requires_grad))

                self.optimizer_D_ROI = torch.optim.Adam(
                    self.netD_ROI.parameters(),
                    lr=self.lr_D_ROI,
                    betas=(self.beta1, self.beta2)
                )
                self.optimizers.append(self.optimizer_D_ROI)

                # ROI classifier
                self.netC_ROI = ClassifierROI(base_filters=self.roi_c_base_filters, roi_patch_size=self.roi_patch_size)
                robust_mlflow(mlflow.log_param, 'netC_ROI_params',
                              sum(p.numel() for p in self.netC_ROI.parameters() if p.requires_grad))

                self.optimizer_C_ROI = torch.optim.Adam(
                    self.netC_ROI.parameters(),
                    lr=self.lr_C_ROI,
                    betas=(self.beta1, self.beta2)
                )
                self.optimizers.append(self.optimizer_C_ROI)


        # load data to device
        for model_name in self.model_names:
            net = getattr(self, 'net' + model_name)
            setattr(self, 'net' + model_name, net.to(self.device))

    def set_params(self, args):
        self.input_nc = args['train.data.input_nc']
        self.output_nc = args['train.data.output_nc']

        # Model
        self.netF_mode = args['train.model.projector.netF']

        # Neighborhood objective (GAN + NCE loss) params
        self.gan_mode = args['train.params.loss.gan_loss.gan_mode']
        self.nce_idt = args['train.params.loss.gan_loss.nce_idt']
        self.lambda_NCE = args['train.params.loss.gan_loss.lambda_NCE']
        self.nce_layers = [int(i) for i in args['train.params.loss.gan_loss.nce_layers'].split(',')]
        self.nce_num_patches = args['train.params.loss.gan_loss.nce_num_patches']
        self.flip_equivariance = args['train.params.loss.gan_loss.flip_equivariance']

        # Optimizer params
        self.lr_G = args['train.params.optimizer.lr.lr_G']
        self.lr_D = args['train.params.optimizer.lr.lr_D']
        self.beta1 = args['train.params.optimizer.params.beta1']
        self.beta2 = args['train.params.optimizer.params.beta2']

        # Global objective (Style & Content loss) params
        self.is_image_match = args['train.params.is_image_match']
        if self.is_image_match:
            self.lambda_con = args['train.params.loss.sc_loss.lambda_con']
            self.lambda_sty = args['train.params.loss.sc_loss.lambda_sty']
            self.lr_I = args['train.params.optimizer.lr.lr_I']

        # Local objective (ROI discriminator + Cell classification loss) params
        self.is_roi_discriminator = args['train.params.is_roi_discriminator']
        if self.is_roi_discriminator:
            # ROI discriminator params
            self.roi_d_base_filters = args['train.model.roi_discriminator.base_filters']
            self.lambda_d_roi = args['train.params.loss.roi_loss.lambda_d_roi']
            self.lr_D_ROI = args['train.params.optimizer.lr.lr_D_ROI']

            # ROI classifier params
            self.roi_c_base_filters = args['train.model.roi_classifier.base_filters']
            self.roi_patch_size = args['train.model.roi_classifier.roi_patch_size']
            self.lambda_c_roi = args['train.params.loss.roi_loss.lambda_c_roi']
            self.lr_C_ROI = args['train.params.optimizer.lr.lr_C_ROI']

        # models to save/load to the disk
        if self.is_train:
            self.model_names = ['G', 'F', 'D']
            if self.is_roi_discriminator:
                self.model_names += ['D_ROI', 'C_ROI']
        else:
            self.model_names = ['G']

        # specify the training losses to print out. training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_GAN', 'NCE_A', 'NCE_B', 'D_real', 'D_fake']
        if self.is_roi_discriminator:
            self.loss_names += ['D_roi_gen', 'D_roi', 'C_roi_gen', 'C_roi']

        # specify the images to save/display. training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B']
        if self.is_train:
            self.visual_names += ['real_B']
            if self.nce_idt:
                self.visual_names += ['idt_B']

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.real_A = input['src'].to(self.device)
        if self.is_train:
            self.real_B = input['dst'].to(self.device)

            if self.is_roi_discriminator:
                self.condition_A = input['condition_src'].to(self.device)
                self.bboxes_A = input['bboxes_src'].to(self.device)
                self.condition_B = input['condition_dst'].to(self.device)
                self.bboxes_B = input['bboxes_dst'].to(self.device)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        self.forward()                     # compute fake images: G(A)
        if self.is_train:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                  # calculate gradients for G
            if self.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(
                    self.netF.parameters(),
                    lr=self.lr_G,
                    betas=(self.beta1, self.beta2)
                )
                self.optimizers.append(self.optimizer_F)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.nce_idt and self.is_train else self.real_A
        if self.flip_equivariance:
            self.flipped_for_equivariance = self.is_train and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def forward_with_anchor(self, y_anchor, x_anchor, mode):
        """Run forward pass; called by functions <test>."""
        assert not self.is_train
        self.fake_B = self.netG.forward_with_anchor(
            self.real_A, y_anchor=y_anchor, x_anchor=x_anchor, mode=mode
        )

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""

        # Real
        self.loss_D_real = self.criterionGAN(self.netD(self.real_B), True)
        # Fake; stop backprop to the generator by detaching fake_B
        self.loss_D_fake = self.criterionGAN(self.netD(self.fake_B.detach()), False)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""

        # GAN loss: D(G(A))
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True)

        # NCE loss: NCE(A, G(A))
        if self.lambda_NCE > 0.0:
            self.loss_NCE_A = self.compute_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE_A = 0.0

        # NCE-IDT loss: NCE(B, G(B))
        if self.nce_idt and self.lambda_NCE > 0.0:
            self.loss_NCE_B = self.compute_NCE_loss(self.real_B, self.idt_B)
        else:
            self.loss_NCE_B = 0
        loss_NCE = self.lambda_NCE * (self.loss_NCE_A + self.loss_NCE_B) * 0.5

        # ROI loss
        if self.is_roi_discriminator:
            validity_roi = self.netD_ROI(self.fake_B, self.condition_A, self.bboxes_A)
            self.loss_D_roi_gen = self.criterionGAN(validity_roi, True) * self.lambda_d_roi

            labels_roi, logits_roi = self.netC_ROI(self.fake_B, self.bboxes_A)
            self.loss_C_roi_gen = self.criterionBCE(logits_roi, labels_roi) * self.lambda_c_roi
        else:
            self.loss_D_roi_gen = 0
            self.loss_C_roi_gen = 0

        self.loss_G = self.loss_G_GAN + loss_NCE + self.loss_D_roi_gen + self.loss_C_roi_gen
        return self.loss_G

    def compute_NCE_loss(self, src, tgt):
        feat_q, patch_ids_q = self.netG(tgt, num_patches=self.nce_num_patches, encode_only=True)

        if self.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k, _ = self.netG(src, num_patches=self.nce_num_patches, encode_only=True, patch_ids=patch_ids_q)
        feat_k_pool = self.netF(feat_k)
        feat_q_pool = self.netF(feat_q)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.criterionNCE(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / len(self.nce_layers)


    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # Discriminator ROIs
        if self.is_roi_discriminator:
            # ROI discriminator
            self.set_requires_grad(self.netD_ROI, True)
            self.optimizer_D_ROI.zero_grad()
            validity_roi = self.netD_ROI(self.real_B, self.condition_B, self.bboxes_B)
            real_loss = self.criterionGAN(validity_roi, True)
            validity_roi = self.netD_ROI(self.fake_B.detach(), self.condition_A, self.bboxes_A)
            fake_loss = self.criterionGAN(validity_roi, False)
            self.loss_D_roi = (real_loss + fake_loss) * 0.5
            self.loss_D_roi.backward()
            self.optimizer_D_ROI.step()

            # ROI classification
            self.set_requires_grad(self.netC_ROI, True)
            self.optimizer_C_ROI.zero_grad()
            labels_roi, logits_roi = self.netC_ROI(self.real_B, self.bboxes_B)
            real_loss = self.criterionBCE(logits_roi, labels_roi)
            labels_roi, logits_roi = self.netC_ROI(self.fake_B.detach(), self.bboxes_A)
            fake_loss = self.criterionBCE(logits_roi, labels_roi)
            self.loss_C_roi = (real_loss + fake_loss) * 0.5
            self.loss_C_roi.backward()
            self.optimizer_C_ROI.step()

        # update G and F
        self.set_requires_grad(self.netD, False)
        if self.is_roi_discriminator:
            self.set_requires_grad(self.netD_ROI, False)
            self.set_requires_grad(self.netC_ROI, False)

        self.optimizer_G.zero_grad()
        if self.netF_mode == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.netF_mode == 'mlp_sample':
            self.optimizer_F.step()

    def set_input_image(self, input):
        if self.is_train:
            # move images to device
            self.real_A_img = input['src_real'].to(self.device)
            self.real_B_img = input['dst_real'].to(self.device)
            self.fake_B_img = input['dst_fake'].to(self.device)

    def optimize_parameters_image(self):
        self.set_requires_grad(self.netD, False)                   # D require no gradients when optimizing G
        if self.is_roi_discriminator:
            self.set_requires_grad(self.netD_ROI, False)
            self.set_requires_grad(self.netC_ROI, False)
        self.optimizer_I.zero_grad()                                           # set G gradients to zero

        # compute style and content losses at image-level
        loss_content_B_fake, loss_style_B_fake = self.criterionImage(self.real_A_img, self.real_B_img, self.fake_B_img)
        self.loss_content_B_fake = self.lambda_con * loss_content_B_fake
        self.loss_style_B_fake = self.lambda_sty * loss_style_B_fake

        # calculate gradients for G_A
        self.loss_I = 0.5 * (self.loss_content_B_fake + self.loss_style_B_fake)
        self.loss_I.backward()                                     # back propagate for G_A
        self.optimizer_I.step()                                    # update G_A's weights

        return {'image' : self.loss_I.detach().item(),
                'content_B_fake' : self.loss_content_B_fake.detach().item(),
                'style_B_fake': self.loss_style_B_fake.detach().item()}
