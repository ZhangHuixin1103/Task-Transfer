export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/data1/tzz/huixin/Task-Transfer/.cache"

# python eval.py --use_qwen_for_prompt
# python eval.py --fixed_prompt "This is a visual in-context learning task. The first two images are an input and output of Task A. The third image is the input for Task B. The goal is to perform Task B on the third image and generate output image, learning from Task A."

# python eval_1.py
# python eval_2.py --use_qwen_for_prompt
# python eval_2.py --fixed_prompt "This is a visual in-context learning task. The first two images are an input and output of an image processing task. The third image is the input for the SAME task. The goal is to perform this task on the third image and generate output image, learning from the pair before."

# python eval_seedream.py --use_qwen_for_prompt
# python eval_seedream.py --fixed_prompt "This is a visual in-context learning task. The first two images are an input and output of Task A. The third image is the input for Task B. The goal is to perform Task B on the third image and generate output image, learning from Task A."

# python eval_lumina.py --use_qwen_for_prompt
# python eval_lumina.py --fixed_prompt "This is a visual in-context learning task. The first two images are an input and output of Task A. The third image is the input for Task B. The goal is to perform Task B on the third image and generate output image, learning from Task A."

# python eval_flux.py --use_qwen_for_prompt
# python eval_flux.py --fixed_prompt "This is a visual in-context learning task. The first two images are an input and output of Task A. The third image is the input for Task B. The goal is to perform Task B on the third image and generate output image, learning from Task A."

# python eval_omnigen.py --use_qwen_for_prompt
# python eval_omnigen.py --fixed_prompt "This is a visual in-context learning task. The first two images are an input and output of Task A. The third image is the input for Task B. The goal is to perform Task B on the third image and generate output image, learning from Task A."

python eval_qwen.py --use_qwen_for_prompt
# python eval_qwen.py --fixed_prompt "This is a visual in-context learning task. The first two images are an input and output of Task A. The third image is the input for Task B. The goal is to perform Task B on the third image and generate output image, learning from Task A."

# python eval_firered.py --use_qwen_for_prompt
# python eval_firered.py --fixed_prompt "This is a visual in-context learning task. The first two images are an input and output of Task A. The third image is the input for Task B. The goal is to perform Task B on the third image and generate output image, learning from Task A."
