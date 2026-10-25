# Gemino reproduction · 30500-step model

天津大学尹鹏飞整理的Gemino研究复现包。**不是作者发布的官方权重**。已完成30轮训练；在已评估的检查点中，30500步具有较好的感知指标，因此作为首个发布候选。

## 下载与安装（已验证目标：Linux / Python3.10 / NVIDIA CUDA）

最方便：在[Releases](https://github.com/yp-fly/gemino-reproduction/releases/tag/v0.1.0)下载完整推理包ZIP，解压后已有权重；也可克隆仓库后单独下载：

```bash
git clone https://github.com/yp-fly/gemino-reproduction.git
cd gemino-reproduction
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python download_weights.py
```

如果仅使用已经解码的低清视频和高清参考，直接运行下文low模式，不需要编译编解码器。要模拟论文的高分辨率→低清VP8→重建链路，再执行：

```bash
sudo apt-get update
sudo apt-get install -y build-essential libvpx-dev
python tools/build_codec.py
```

## 推理

已有256×256、30fps的低清视频和1024×1024参考图：

```bash
python infer.py --input low_video.mp4 --input-mode low --reference reference.png --output outputs/reconstructed.mp4 --device cuda
```

从1024×1024、30fps原视频模拟完整编解码重建：

```bash
python infer.py --input input_1024.mp4 --input-mode high --output outputs/reconstructed.mp4 --device cuda
```

若原视频大于1024方形，可显式加`--crop center`；这会裁剪视野，**不是完整1080p重建**。high模式取首帧作为唯一高清参考，后续不使用目标高清图辅助重建。low模式只解码输入，不重新压缩；若低清编码失真与训练不同，质量需另行验证。输出固定1024×1024、30fps，无音频；CPU模式可运行但不承诺实时。

快速链路检查（合成图案，不作为人脸质量演示）：

```bash
python infer.py --input examples/synthetic.mp4 --output outputs/smoke.mp4 --device cuda --max-frames 4
```

## 必须了解的适用范围

- 这是对公开清单中`needle_drop`同一人物的个性化模型，不是通用人物模型。其他人的脸、多人场景、遮挡和大幅运动均不保证效果。
- 输入256×256、输出1024×1024。不同输入尺寸、宽高比和完整1080p需单独适配。
- 官方配置的45 Kbps标签会映射到内部200 Kbps；本项目两段验证的低清载荷约196 Kbps，**不宣称实际45 Kbps达到这些画质**。高清参考、协议及重传开销另外计算。
- 已评估的30500步低头关键帧LPIPS0.2743；6个补充片段平均LPIPS0.2611。它们来自同两个验证源，不是独立人物泛化测试或论文25测试视频的复现。
- 未证明长视频完全无重影或闪烁；未复现完整WebRTC通话或NetAdapt剪枝。源码与依赖需要安装，完整包不包含PyTorch/CUDA运行时。
- 使用范围和许可见[模型卡](MODEL_CARD.md)、[许可证说明](LICENSE)及第三方通知。不要把本包当作已获授权的商用人像服务。

## 包含什么

独立推理入口、官方网络和VP8相关源码、30500步推理权重（Release）、固定训练配置、数据来源清单、训练/评估适配代码、评估指标、许可证和校验文件。没有原始训练视频、SSH凭据、私人邮件、Conda环境或GB级优化器状态。

`reproduction/`是训练适配代码归档，不是已附带完整训练数据的一键重训包。运行`python tools/prepare_reproduction.py`准备代码布局，再按provenance清单获取并审查视频，准备校验过的FOMM和VGG权重后训练。推理不依赖这些步骤。

## 来源

[Gemino论文](https://www.usenix.org/system/files/nsdi24-sivaraman.pdf) · [官方model](https://github.com/geminovc/model) · [官方scripts](https://github.com/geminovc/scripts) · [官方aiortc](https://github.com/geminovc/aiortc)

本项目对官方代码做运行适配、数据重新准备和自行训练，不代表作者维护或认可。论文的码率优势不能直接作为本检查点的结果。
