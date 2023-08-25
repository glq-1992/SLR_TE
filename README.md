# Comb_CSLR
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/visual-alignment-constraint-for-continuous/sign-language-recognition-on-rwth-phoenix)](https://paperswithcode.com/sota/sign-language-recognition-on-rwth-phoenix?p=visual-alignment-constraint-for-continuous)

This repo holds codes of the paper: Visual Alignment Constraint for Continuous Sign Language Recognition.(ICCV 2021) [[paper]](https://arxiv.org/abs/2104.02330)

---
### Prerequisites

- This project is implemented in Pytorch (>1.8). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.

- [Optional] sclite [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), install kaldi tool to get sclite for evaluation. After installation, create a soft link toward the sclite:    
  `ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite`
  We also provide a python version evaluation tool for convenience, but sclite can provide more detailed statistics.

- [Optional] [SeanNaren/warp-ctc](https://github.com/SeanNaren/warp-ctc) At the beginning of this research, we adopt warp-ctc for supervision, and we recently find that pytorch version CTC can reach similar results.

### Data Preparation

1. Download the CCUL Dataset [[download link]](https://github.com/glq-1992/CCUL_all/tree/main).

### Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model on CCUL, run the command below:

`python main_EinT_2d.py --device 0


### To Do List

- [x] Pure python implemented evaluation tools.
- [x] WAR and WER calculation scripts.

### Citation

If you find this repo useful in your research works, please consider citing:

```latex
@InProceedings{Min_2021_ICCV,
    author    = {Min, Yuecong and Hao, Aiming and Chai, Xiujuan and Chen, Xilin},
    title     = {Visual Alignment Constraint for Continuous Sign Language Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {11542-11551}
}
```

### Relevant paper

Self-Mutual Distillation Learning for Continuous Sign Language Recognition[[paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Hao_Self-Mutual_Distillation_Learning_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html)

```latex
@InProceedings{Hao_2021_ICCV,
    author    = {Hao, Aiming and Min, Yuecong and Chen, Xilin},
    title     = {Self-Mutual Distillation Learning for Continuous Sign Language Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {11303-11312}
}
```

### Acknowledge

We appreciate the help from Runpeng Cui, Hao Zhou@[Rhythmblue](https://github.com/Rhythmblue) and Xinzhe Han@[GeraldHan](https://github.com/GeraldHan) :)
