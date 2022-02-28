# Contrastive-Machine-Learning-reveals-the-structure-of-individual-variation-in-ASD
Code and materials for paper "Contrastive machine learning reveals the structure of neuroanatomical variation within Autism"



* [Notebooks/](Notebooks/)
  * [01-Train-AutoEncoders.ipynb](Notebooks/01-Train-AutoEncoders.ipynb) Trains VAE and CVAE autoencoders
  * [02-Extract-Latent-Features.ipynb](Notebooks/02-Extract-Latent-Features.ipynb) Extracts latent features (shared, ASD-specific & VAE) using trained autoencoder models
  * [03-Analysis-RSA.ipynb](Notebooks/03-Analysis-RSA.ipynb) RSA analyses (Figure 1B/1C)
  * [04-Analysis-Clustering-Results.ipynb](Notebooks/04-Analysis-Clustering-Results.ipynb) Clustering analyses (Figure 1D)
  * [05-Jacobian-Make-Jacobians.ipynb](Notebooks/05-Jacobian-Make-Jacobians.ipynb) Generates the Jacobian Determinant maps using synthethic "TC-Twins"
  * [06-Jacobian-Jacobian-Analysis.ipynb](Notebooks/06-Jacobian-Jacobian-Analysis.ipynb) Calculates LOSO PCA and plots neuroanatomical associations (Figure 2)
  * [helper_funcs.py](Notebooks/helper_funcs.py) helper functions called by analysis notebooks
  * [make_models2.py](Notebooks/make_models2.py) Code defining VAE and CVAE Tensorflow models


* [Data/](Data/)
  * Generated data and necessary files  

* [tf_weights/](tf_weights/) Trained weights for VAE and CVAE models
  * [CVAE_weights/](tf_weights/CVAE_weights/)
    * CVAE_weights.z01
    * CVAE_weights.z02
    * ...
  * [VAE_weights/](tf_weights/VAE_weights/)
    * VAE_weights.z01 
    * VAE_weights.z02
    * ...


_N.B the trained weights are zipped into a multi-part zip file (.z01,.z02,.z03 etc.) and need to be unzipped before use._
