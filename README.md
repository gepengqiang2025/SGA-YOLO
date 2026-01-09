# SGA-YOLO: A Lightweight Real-Time Object Detection Network for UAV Infrared Images

We provide the HIT-UAV dataset used in our work. You can download it by the following link: [dataset](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset)

# Overview of SGA-YOLO

The performance of existing object detection algorithms significantly degrades when applied to low-resolution infrared (IR) images captured by unmanned aerial vehicles (UAVs), which suffers from slow inference speed, low detection precision, and redundant network parameters. To tackle these issues, this paper proposes a lightweight real-time object detection network for UAV IR images, termed SGA-YOLO, which is designed based on the you only look at once version 8n (YOLOv8n) framework. First of all, the efficient SENetV2-neck enhances the correlation between different channels, which realizes efficient multi-scale feature fusion and improves detection precision. Subsequently, the lightweight S2GM backbone combines ShuffleNetV2-stride2 and C2f\_Ghost modules, which significantly reduces the network parameters and increases inference speed. Finally, the adaptive fine-grained channel (AFGC) attention mechanism is coupled to further enhance detection precision and effectively mitigate background interference. Compared with the YOLOv8n, SGA-YOLO achieves a 27\% reduction in network parameters, a 4.7\% increment in precision, a 2.5\% increment in recall rate, a 30.86\% reduction in GFLOPs, a 16.3\% increment in FPS, a 3.50\% increment in mAP@0.5, and a 1.3\% increment in mAP@0.5:0.95. In addition, it supports deployment on resource-constrained embedded development boards, offering a new perspective on designing lightweight UAV IR object detection networks for real-world applications in intelligent transportation systems.


# Publication
```
If you want to use this work, please consider citing the following paper.
@article{ge2025sga,
  title={SGA-YOLO: A Lightweight Real-Time Object Detection Network for UAV Infrared Images},
  author={Ge, Pengqiang and Wan, Minjie and Qian, Weixian and Xu, Yunkai and Kong, Xiaofang and Gu, Guohua and Chen, Qian},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
```
