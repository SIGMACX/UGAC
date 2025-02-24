
# (UGAC）Uncertainty-Guided Adaptive Correction for Semi-Supervised Medical Image Segmentation
## The code will be released later.

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/SIGMACX/UGAC">
    <img src="logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">"UGAC</h3>
  <p align="center">
    Uncertainty-Guided Adaptive Correction for Semi-Supervised Medical Image Segmentation
    <br />
    <a href="https://github.com/shaojintian/Best_README_template"><strong>EXPLORE THE DOCUMENTATION FOR THIS PROJECT »</strong></a>
    <br />
    <br />
<!--     <a href="https://github.com/shaojintian/Best_README_template">查看Demo</a> -->
    ·
<!--     <a href="https://github.com/shaojintian/Best_README_template/issues">报告Bug</a>
    ·
    <a href="https://github.com/shaojintian/Best_README_template/issues">提出新特性</a> -->
  </p>

</p>

【Abstract】
Consistent perturbation strategies have emerged as a dominant paradigm in semi-supervised medical image segmentation. Nevertheless, prevailing approaches inadequately address two critical challenges: (1) prediction errors induced by data uncertainty from distribution shifts, and (2) loss instability caused by model uncertainty in parameter generalization. To overcome these limitations, we propose an Uncertainty-Guided Adaptive Correction (UGAC) framework with three key innovations. First, we develop a dual-path uncertainty rectification mechanism that employs normalized entropy measures to detect error-prone regions in unlabeled predictions, followed by bilateral correction through confidence-weighted fusion. Second, we introduce adversarial consistency constraints that leverage labeled data to discriminate authentic segmentation patterns, effectively regularizing uncertainty propagation in unlabeled predictions through spectral normalization. Third, we architect a frequency-aware segmentation backbone through our novel Freqfusion module, which performs adaptive spectral decomposition during feature decoding to explicitly disentangle high-frequency (boundary-aware) and low-frequency (structural) components, thereby enhancing anatomical boundary sensitivity. Comprehensive evaluations on MM-WHS, BUSI, and PROMISE12 datasets demonstrate UGAC's superior performance. The proposed framework exhibits robust generalizability across CT, MRI, and ultrasound modalities while maintaining computational efficiency comparable to baseline UNet implementations.
 
![image](https://github.com/SIGMACX/UGAC/blob/main/fig_UGAC_main_7.pdf)

<!-- links -->
[your-project-path]:shaojintian/Best_README_template
[contributors-shield]: https://img.shields.io/github/contributors/shaojintian/Best_README_template.svg?style=flat-square
[contributors-url]: https://github.com/shaojintian/Best_README_template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/shaojintian/Best_README_template.svg?style=flat-square
[forks-url]: https://github.com/shaojintian/Best_README_template/network/members
[stars-shield]: https://img.shields.io/github/stars/shaojintian/Best_README_template.svg?style=flat-square
[stars-url]: https://github.com/shaojintian/Best_README_template/stargazers
[issues-shield]: https://img.shields.io/github/issues/shaojintian/Best_README_template.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/shaojintian/Best_README_template.svg
[license-shield]: https://img.shields.io/github/license/shaojintian/Best_README_template.svg?style=flat-square
[license-url]: https://github.com/shaojintian/Best_README_template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian




