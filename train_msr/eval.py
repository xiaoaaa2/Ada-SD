import argparse
import functools
import time

from macls.trainer import MAClsTrainer
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/resnet.yml',         "Configuration file")
add_arg("use_gpu",          bool,  True,                        "Whether to use GPU to evaluate the model")
add_arg('save_matrix_path', str,   'output/images/',            "The path to save the mixing matrix")
add_arg('resume_model',     str,   'models/CAMPPlus_Fbank/best_model/',  "The path of the model")
args = parser.parse_args()
print_arguments(args=args)

# Retrieve the trainer
trainer = MAClsTrainer(configs=args.configs, use_gpu=args.use_gpu)

# Evaluate the model
start = time.time()
loss, accuracy = trainer.evaluate(resume_model=args.resume_model,
                                  save_matrix_path=args.save_matrix_path)
end = time.time()
print('Evaluation time consumption：{}s，loss：{:.5f}，accuracy：{:.5f}'.format(int(end - start), loss, accuracy))
