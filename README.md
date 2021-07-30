autogif
-------

GIF generation pipeline in JAX/Flax. The pipeline generates GIF-like responses by:

1. generating a text reply that taking into account context and previous turns in the dialogue.
2. generating images with VQGAN that match the generated text.
3. filtering the generated images on semantics and coherence with [CLIP](https://github.com/openai/CLIP).
4. generating an animation sequence from the selected image.

#### Model implementations (in JAX)

1. Dialogue response generation: [this repository](models/response.py).
2. Image generation: [VGQAN implementation by Patil Suraj](https://github.com/patil-suraj/vqgan-jax).
3. Image selection: [Transformers implementation of CLIP](https://huggingface.co/transformers/model_doc/clip.html).
4. Image sequence generation: [this repository](models/animation.py).

#### Reference papers

- VGQAN: https://compvis.github.io/taming-transformers/
- CLIP: https://openai.com/blog/clip/

#### Datasets:

- https://sites.google.com/view/emotiongif-2020/shared-task/dataset
- https://github.com/bshmueli/ReactionGIF
- https://huggingface.co/datasets/julien-c/reactiongif
- https://github.com/google-research-datasets/conceptual-captions

##### Similar projects

- DALL-E Pytorch: https://github.com/lucidrains/DALLE-pytorch
- DALL-E mini: https://huggingface.co/spaces/flax-community/dalle-mini
