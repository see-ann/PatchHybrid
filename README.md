# PatchHybrid
> **A generalized, architecture-agnostic defense against adversarial patches.**  

By [Priya Naphade](https://github.com/pnaphade) and [Sean Wang](https://github.com/see-ann)

![Banner](https://user-images.githubusercontent.com/placeholder/banner.png)

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
Adversarial patches are stickers that can be applied to images to induce misclassification in convolutional neural networks. `PatchHybrid` aims to provide a generalized defense against such patch attacks, by combining strategies from [PatchGuard](https://github.com/inspire-group/PatchGuard) and [PatchCleanser](https://github.com/inspire-group/PatchCleanser). This project was initially forked from PatchGuard and has been subsequently modified to implement our novel defense strategy.

---

## Technical Overview
Our defense strategy works in two steps:
1. Estimate patch size
2. Apply PatchCleanser

Our method does not require prior knowledge of the patch size. Hence, a client can eventually classify with the model of their choice, providing an architecture-agnostic defense mechanism.

For a deeper understanding of our approach, please review the detailed technical report [here](LINK-TO-REPORT).

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
git clone https://github.com/your_username_/PatchHybrid.git

2. Install the requirements
pip install -r requirements.txt

---

## Usage
Please refer to our [wiki](LINK-TO-WIKI) for detailed usage instructions.

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
