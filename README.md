# Architecture or Augmentation? Reproducing Affine-Invariant CNNs for Photometric Robustness

Portfolio project in IDATG2208 Machine Learning (NTNU, autumn 2026), project 8. Scaled-down reproduction of Messaoudi et al., *Designing Affine-Invariant Neural Networks for Photometric Corruption Robustness and Generalization* (ICLR 2026), on CIFAR-10 and Fashion-MNIST.

## Google Colab

All results in the report were produced in Google Colab on a single Tesla T4 GPU. To run there, upload the three `.py` files, select a GPU runtime and run:

```python
import run_experiments as RE, analysis as AN
RE.run_groups(['all'], out_dir='results', data_root='./data')
AN.make_all(run_dir='results/runs', out_dir='results')
```
