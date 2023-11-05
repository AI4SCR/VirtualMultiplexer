import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from i2iTranslation.dataloader import ImagePatchDataset
from i2iTranslation.models import create_model
from i2iTranslation.postprocess import PostProcess

class Test:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.i2i_checkpoints_path = args['i2i_checkpoints_path']
        self.i2i_prediction_path = args['i2i_prediction_path']
        self.patch_batch_size = args['train.params.patch_batch_size']
        self.is_visualize = args['test.is_visualize']

        # setup post-processor
        self.post_processor = PostProcess(args)

        # setup testing
        self.test_setup()

    def test_setup(self):
        if self.is_visualize:
            os.makedirs(self.i2i_prediction_path, exist_ok=True)

        # test for the best checkpoint
        checkpoints = glob.glob(f'{self.i2i_checkpoints_path}/*_G*')
        model_epochs = [os.path.basename(x).split('_')[0] for x in checkpoints]

        if 'best' in model_epochs:
            self.args['test.model_epoch'] = 'best'
        else:
            model_epochs = [int(x) for x in model_epochs]
            self.args['test.model_epoch'] = str(max(model_epochs))

    def tester(self):
        # test dataset
        test_dataset = ImagePatchDataset(args=self.args, **self.args)
        print(f'number of test images: {len(test_dataset)} \n')

        # load model
        model = create_model(self.args, self.device)
        model.setup(self.args)
        print('model loaded')

        with torch.no_grad():
            for i, data in tqdm(enumerate(test_dataset)):
                src_patches, src_patch_coords, src_image_path, dst_image_path = data
                print(f'{i}/{len(test_dataset)}: {os.path.basename(src_image_path)}')

                # patch-wise prediction
                dst_patches_fake = []
                for j in range(0, len(src_patches), self.patch_batch_size):
                    model.set_input({'src': src_patches[j: j + self.patch_batch_size]})
                    model.inference()

                    # get image results
                    visuals = model.get_current_visuals()
                    dst_patches_fake.append(visuals['fake_B'])
                dst_patches_fake = torch.cat(dst_patches_fake)
                del src_patches

                # post-processing
                dst_image_fake = self.post_processor.postprocess(
                    src_image_path=src_image_path,
                    dst_image_path=dst_image_path,
                    dst_patches_fake=dst_patches_fake,
                    src_patch_coords=src_patch_coords
                )

                if self.is_visualize:
                    Image.fromarray(dst_image_fake.astype(np.uint8)).save(
                        f'{self.i2i_prediction_path}/{os.path.basename(dst_image_path)}'
                    )