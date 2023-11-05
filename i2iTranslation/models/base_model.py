import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from i2iTranslation.models.networks import get_scheduler


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, args, device):
        """Initialize the BaseModel class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cyclegan_model.py for an example.
        """
        self.args = args
        self.is_train = args['is_train']
        self.device = device
        self.loss_names = []
        self.optimizer_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.save_dir = args['i2i_checkpoints_path']
        os.makedirs(self.save_dir, exist_ok=True)

        self.is_i2i_stage2 = True if 'i2i_stage2' in args.keys() else False

        if self.is_i2i_stage2:
            if self.is_train:
                self.load_dir = args['i2i_checkpoints_path1']
            else:
                self.load_dir = args['i2i_checkpoints_path']
        else:
            self.load_dir = args['i2i_checkpoints_path']


    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def forward_with_anchor(self, y_anchor, x_anchor, mode):
        """Run forward pass for KIN; called by <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, args):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.is_train:
            self.schedulers = [get_scheduler(optimizer, args) for optimizer in self.optimizers]
        if not self.is_train or args['train.params.save.continue_train']:
            self.load_networks(args['test.model_epoch'])

    def data_dependent_initialize(self, data):
        pass

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def inference(self):
        """Forward function used in inference time.
        """
        self.eval()
        with torch.no_grad():
            self.forward()

    def inference_with_anchor(self, y_anchor, x_anchor, mode):
        self.eval()
        with torch.no_grad():
            self.forward_with_anchor(y_anchor, x_anchor, mode)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.args['train.params.optimizer.lr.lr_policy'] == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def get_current_learning_rates(self):
        """Return traning learning rates"""
        learning_rates = OrderedDict()
        for name in self.optimizer_names:
            if isinstance(name, str):
                optim = getattr(self, 'optimizer_' + name)
                for param_group in optim.param_groups:
                    learning_rates[name] = param_group['lr']
        return learning_rates

    def save_networks(self, epoch=None):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                if epoch is not None:
                    save_path = os.path.join(self.save_dir, f'{epoch}_net_{name}.pth')
                else:
                    save_path = os.path.join(self.save_dir, f'best_net_{name}.pth')
                net = getattr(self, 'net' + name)

                if torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.to(self.device)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            # ignore roi discriminators
            if 'ROI' in name:
                continue

            if isinstance(name, str):
                load_path = os.path.join(self.load_dir, f'{epoch}_net_{name}.pth')
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))

                net = getattr(self, 'net' + name)
                model_dict = net.state_dict()

                if len(model_dict.keys()) == 0:
                    continue

                if model_dict.keys() == state_dict.keys():
                    model_dict.update(state_dict)
                    net.load_state_dict(model_dict)
                else:
                    if len(model_dict.keys()) == len(state_dict.keys()):
                        model_dict_keys = list(model_dict.keys())
                        for i, (key, value) in enumerate(state_dict.items()):
                            key_ = model_dict_keys[i]
                            model_dict[key_] = value
                        net.load_state_dict(model_dict)
                    else:
                        raise ValueError('Unmatched models')

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad