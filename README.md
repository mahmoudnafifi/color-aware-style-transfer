# CAMS: Color-Aware Multi-Style Transfer
[Mahmoud Afifi](https://sites.google.com/view/mafifi)<sup>1</sup>, [Abdullah Abuolaim](https://sites.google.com/view/abdullah-abuolaim/)\*<sup>1</sup>, [Mostafa Hussien](https://www.linkedin.com/in/mostafakorashy/)\*<sup>2</sup>, [Marcus A. Brubaker](https://mbrubake.github.io/)<sup>1</sup>,  [Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)<sup>1</sup>

<sup>1</sup>York University  
<sup>2</sup>École de technologie supérieure

\* denotes equal contribution

Reference code for the paper [CAMS: Color-Aware Multi-Style Transfer.](https://arxiv.org/abs/2106.13920) Mahmoud Afifi, Abdullah Abuolaim, Mostafa Hussien, Marcus A. Brubaker, and Michael S. Brown. arXiv preprint, 2021. If you use this code, please cite our paper:
```
@article{afifi2021coloraware,
  title={CAMS: Color-Aware Multi-Style Transfer},
  author={Afifi, Mahmoud and Abuolaim, Abdullah and Hussien, Mostafa and Brubaker, Marcus A. and Brown, Michael S.},
  journal={arXiv preprint arXiv:2106.13920},
  year={2021}
}
```

![github](https://user-images.githubusercontent.com/37669469/122465812-8478ab00-cf86-11eb-86ba-8f98dc1d76ba.jpg)


### Get Started
Run `color_aware_st.py` or check the Colab link from [here](https://colab.research.google.com/drive/1_unMZ4zUqKwnSmMVZ1KknZQ74CXJzfvg?usp=sharing). 

### Manual Selection
Our method allows the user to manually select the color correspondences between palettes or ignore some colors when optimizing. 
![user_selection](https://user-images.githubusercontent.com/37669469/122466000-bd188480-cf86-11eb-92e2-f7ad46d07140.jpg)

To enable this mode, use `SELECT_MATCHES = True`.

### Other useful parameters:
* `SMOOTH`: smooth generated mask before optimizing.
* `SHOW_MASKS`: to visualize the generated masks during optimization.
* `SIGMA`: to control the fall off in the radial basis function when generating the masks. Play with its value to get different results; generally, 0.25 and 0.3 work well in most cases.
* `PALETTE_SIZE`: number of colors in each palette.
* `ADD_BLACK_WHITE`: to append black and white colors to the final palette before optimizing.
* `STYLE_LOSS_WEIGHT`: weight of style loss
* `CONTENT_LOSS_WEIGHT`: weight of content loss.
* `COLOR_DISTANCE`: similarity metric when computing the mask. Options include: `'chroma_L2'` (L2 on chroma space) or `'L2'` (L2 on RGB space).
* `STYLE_FEATURE_DISTANCE`: similarity metric for style loss. Options include: `'L2'` or `'COSINE'` (for cosine similarity). 
* `CONTENT_FEATURE_DISTANCE`: = similarity metric for content loss. Options include: `'L2'` or `'COSINE'` (for cosine similarity). 
* `OPTIMIZER`: optimization algorithm. Options include: `'LBFGS'`, `'Adam'`, `'Adagrad'`.


### MIT License
