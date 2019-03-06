# Author : Bayu Aditya

from .aug_image_bbox_util import display_augmentation_result
from .aug_image_bbox_util import convert_dict, invert_dict

from .aug_image_bbox_util import image_augmentation_blur_with_bbox
from .aug_image_bbox_util import image_augmentation_intensity_with_bbox
from .aug_image_bbox_util import image_augmentation_noise_with_bbox
from .aug_image_bbox_util import image_augmentation_rotate_with_bbox
from .aug_image_bbox_util import image_augmentation_scale_with_bbox
from .aug_image_bbox_util import image_augmentation_shear_with_bbox
from .aug_image_bbox_util import image_augmentation_translate_with_bbox
from .aug_image_bbox_util import image_augmentation_rotate_shear_with_bbox

def create_initialization_dict_for_dataset(parameter_mode):
    dict_param = {}

    for subclass in parameter_mode:
        dict_param.update({subclass : {}})
        for mode in parameter_mode[subclass]:
            dict_param[subclass].update({mode : {}})
            dict_param[subclass][mode].update({'images' : []})
            dict_param[subclass][mode].update({'bbox_char' : []})
    return dict_param

class augmented_image_class():
    def __init__(self, img_array, dict_bbox = None):
        self._image = img_array
        self._dict_bbox = dict_bbox
        self.parameter_mode = {
            'scale' : ['small', 'big'],
            'shear' : ['left', 'right'],
            'rotate' : ['left', 'right'],
            'translate' : ['up', 'down', 'left', 'right'],
            'intensity' : ['glow', 'dark'],
            'noise' : ['gaussian', 's&p', 'speckle'],
            'blur' : ['very_low', 'low', 'med', 'high'],
            'rotate_shear' : ['left', 'right']}
        #    'crop' : ['very_low', 'low', 'med', 'high']    # Not yet
        #    }
        
        self.dict_result = create_initialization_dict_for_dataset(self.parameter_mode)
        # how to fill : self.dict_result
        # untuk image     ---> self.dict_result[subclass][mode]['images'] = img_
        # untuk bbox_char ---> self.dict_result[subclass][mode]['bbox_char] = dict_bbox_index_['bbox_char']

        # Fill self.dict_result
        for subclass in self.parameter_mode:
            for mode in self.parameter_mode[subclass]:
                func = eval('image_augmentation_' + subclass + '_with_bbox')
                img_, dict_bbox_index_ = func(self._image, self._dict_bbox, mode)

                self.dict_result[subclass][mode]['images'] = img_
                self.dict_result[subclass][mode]['bbox_char'] = dict_bbox_index_['bbox_char']

    def get_result(self):
        return self.dict_result
        
    def view_result(self):
        for subclass in self.dict_result:
            for mode in self.dict_result[subclass]:
                img_ = self.dict_result[subclass][mode]['images']
                dict_bbox_index_ = self.dict_result[subclass][mode]
                title = subclass + ' : ' + mode
                display_augmentation_result(img_, dict_bbox_index_, title)