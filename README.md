# Citation
This code refers to the following publication:
'''
@InProceedings{Puligandla_2025_ICCV,
    author    = {Puligandla, Venkata Anirudh and Ceperic, Vladimir and Knezevic, Tihomir},
    title     = {Scalable Optical Convolutional Neural Networks For Edge Applications},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2025},
    pages     = {1724-1733}
}
'''
Please cite the paper if you find this useful.

# Optical Fourier CNNs
Simulation and Validation of Optical Fourier Convolution Neural Networks (OFCNN). Simple package contianing only 4 code files:
- models.py => defines the different CNNs defined for Photonic Integrated Circuits (PIC) simulations and standard CNNs implemented using Pytorch
- data_loader.py => Load MNIST, FMNIST, or CIFAR10 dataset with corresponding transformations
- mnist_tests.py => (main file) Train various optical and non-optical CNNs
    - Train the CNN of choice on the dataset of choice by uncommenting aprropriate code lines at the beginning of the file
- result_plots.py => Plot the training and validation losses and accuracies
    - Also loads the trained optical CNN models and plots their performance for different levels of phase-encoding errors

 #### Dependencies
- matplotlib==3.10.3
- numpy==2.2.5
- torch==2.5.1
- torcheval==0.0.7
- git+https://github.com/SPOTLIGHT-24-26/pytorch-onn.git
- torchonn_pyutils==0.0.3.2
- torchvision==0.20.1

#### Notes
- Use the pytorch-onn repository mentioned in dependencies.
    - It supports complex-valued weights for certain layers which is crucial for the functioning of Fourier CNN using complex-valued data
    - This package also provides the Fourier convolution layers and complex-valued activations necessary for PIC simulations
- The trained models and the training progress (loss and accuracy per epoch) are stored into files in folders the current working directory
   - Make sure these directories exist
- result_plot.py also refers to these directories to plot the results
