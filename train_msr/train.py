import argparse
import functools

from macls.trainer import MAClsTrainer
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/resnet.yml',        'Configuration file')
add_arg("local_rank",       int,    0,                          'The parameters required for multi-GPU training')
add_arg("use_gpu",          bool,   True,                       'Whether to use GPU to evaluate the model')
add_arg('save_model_path',  str,    'models/',                  'The path to save the mixing matrix')
add_arg('resume_model',     str,    None,                       'Resume training, set to None if not using a pre-trained model')
add_arg('pretrained_model', str,    None,                       'The path to the pre-trained model. Set to None if not using a pre-trained model')
args = parser.parse_args()
print_arguments(args=args)

# Retrieve the trainer
trainer = MAClsTrainer(configs=args.configs, use_gpu=args.use_gpu)

trainer.train(save_model_path=args.save_model_path,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model)
