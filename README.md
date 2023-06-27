# PatchHybrid
> **A generalized, architecture-agnostic defense against adversarial patches.**  

By [Priya Naphade](https://github.com/pnaphade) and [Sean Wang](https://github.com/see-ann)

---

## Table of Contents
- [About](#about)
- [Technical Overview](#technical-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## About
Adversarial patches represent a distinctive form of threat to machine learning models, specifically convolutional neural networks (CNNs). These visually innocuous but carefully crafted stickers can be applied to images with the intent to mislead the model into making incorrect classifications. Given the increasing reliance on CNNs for tasks ranging from facial recognition to autonomous driving, the implications of such attacks can be severe.

The PatchHybrid project has been designed to counter these patch attacks by providing a robust, generalized defense mechanism. It strategically combines the methodologies of two existing approaches, PatchGuard and PatchCleanser, in order to enhance the resilience of CNNs against adversarial patches.

PatchHybrid was initially forked from the PatchGuard project. PatchGuard's primary mechanism of defense is by employing an algorithm that isolates and nullifies adversarial patches without impeding the model's performance on non-adversarial images. It has shown promising results in the field but has its limitations, particularly when dealing with complex and diverse patch attacks.

To complement PatchGuard's methodology, PatchHybrid integrates strategies from PatchCleanser. PatchCleanser focuses on cleansing the input data from adversarial influence before it reaches the CNN. It applies a pre-processing step which is specifically designed to neutralize adversarial patches, providing an additional layer of defense.

---

## Technical Overview
# Defense Strategy: PatchHybrid

PatchHybrid offers a robust defense mechanism against adversarial patches via a two-step process:

## Step 1: Estimate Patch Size

The first phase of our approach aims to estimate the size of the adversarial patch affecting an image, without requiring any prior knowledge of it. 

This process involves utilizing convolutional neural networks (CNNs) with different receptive field (RF) sizes to classify the image. A critical insight is that smaller RFs result in a fewer number of highly corrupted features, leading to a right-skewed feature distribution. As the RF size increases and more features get corrupted, the skewness of the feature distribution decreases.

To estimate the patch size, we follow these steps:

- Classify the image with a CNN with a large RF and note the skewness of the feature distribution.
- Repeat the process with a smaller RF CNN. As the RF size decreases, we expect the skewness to increase. We can identify the patch size when we observe a significant change in skewness, indicating that the RF size matches the patch size.

## Step 2: Apply PatchCleanser

In the second phase of our defense mechanism, we apply the PatchCleanser strategy. PatchCleanser effectively cleanses the input data from adversarial influence before it is processed by the CNN, thereby offering a layer of defense against adversarial patches.

## Architecture-Agnostic Defense Mechanism

PatchHybrid offers the significant advantage of being architecture-agnostic. This characteristic allows clients the flexibility to classify using their preferred model, without being constrained by the specific requirements of the defense strategy.

## Inspiration from Existing Defenses: PatchGuard

Our strategy draws inspiration from the existing defense mechanism, PatchGuard. Developed by Xiang et al. 2021, PatchGuard employs small RFs in CNNs to limit the number of corrupted features, thus providing a solid defense against adversarial patches. However, it requires a CNN with small RFs to limit the number of corrupted features and is not architecture-agnostic.

Our PatchHybrid project aims to transcend these limitations by integrating the strengths of PatchGuard and PatchCleanser, delivering a more flexible and effective defense against adversarial patches.


For a deeper understanding of our approach, please review the slides provided [here](https://docs.google.com/presentation/d/192jbgCVT67bNAeFHziwDVyU2CTjsPfjIsE-Ui7IVt9I/edit?usp=sharing).

---

## Getting Started

### Prerequisites
This project uses Python 3.8+. You need the following libraries installed.
- numpy
- matplotlib
- pytorch
- torchvision
- sklearn

### Installation
1. Clone this repository
git clone https://github.com/your_username_/PatchHybrid.git](https://github.com/see-ann/PatchHybrid.git

2. Install the requirements
pip install -r requirements.txt

--

---

## Contributing
Any contributions you make are **greatly appreciated**. Please follow the steps below:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

---

## License
Distributed under the MIT License. See `LICENSE` for more information.

---

## Acknowledgements
- [PatchGuard](https://github.com/inspire-group/PatchGuard)
- [PatchCleanser](https://github.com/inspire-group/PatchCleanser)
- Special thanks to our faculty mentor: Professor Prateek Mittal
- Intel research mentor: Abhishek Chakraborty
- Graduate mentors: Tong Wu, Saeed Mahloujifar, Chong Xiang

---
