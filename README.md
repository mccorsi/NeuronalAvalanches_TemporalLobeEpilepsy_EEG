# Automated classification of temporal lobe epilepsy using neuronal avalanches: a noninvasive investigation via resting state EEG


---
This repository contains the code and supporting documents associated with the following manuscript:

XXXXXXXXX, XXXXXX, XXXXXX and XXXXXX (2023). Automated classification of temporal lobe epilepsy using neuronal avalanches: a noninvasive investigation via resting state EEG. Biorxiv. LINK DA AGGIUNGERE

 
---
## Authors:
* Gian Marco Duma
* Emahnuel Troisi Lopez
* [Marie-Constance Corsi](https://marieconstance-corsi.netlify.app), Inria Paris, Paris Brain Institute, Sorbonne Université
* [Pierpaolo Sorrentino](https://scholar.google.nl/citations?user=T1k8qBsAAAAJ&hl=en), Institut de Neuroscience des Systèmes, Aix-Marseille University
* 


---
## Abstract
TODO


## Code
This repository contains the code used to run the analysis performed and to plot the figures.
Analysis was performed with the following Matlab version: 9.10.0.1649659 (R2021a) Update 1.
For a complete list of the Matlab packages used for the analysis, please refer to the 'Start_SessionMatlab.m' file.
In 'requirements.txt' a list of all the Python dependencies is proposed to plot the connectomes.


---
## Figures

### Figure 1 - Analysis pipeline 
![Fig. 1](./Figures_paper/Fig1.tiff)


### Figure 2 - Accuracy and receiver operating characteristic curves
![Fig. 2](./Figures_paper/Fig2.tiff)

*A. Full distribution of the classification accuracy of the support vector machine (SVM) model across the cross-validation splits when trained with avalanche transition matrix (ATM) and imaginary coherence (ImCoh). On the left the accuracy classification using narrow-band filtered signal (3-14 Hz) and on right broadband signal (3-40 Hz). B. Mean receiver operating characteristic (ROC) curves across cross-validation splits and the corresponding area under the curve (AUC), both for ATM (orange line) and ImCoh (purple line). *


### Figure 3 - Feature importance for model interpretability
![Fig. 2](./Figures_paper/Fig3.tiff)

*A. histogram of the probability of the feature importance value in ATM and ImCoh. The histogram shows a narrow distribution for ImCoh and broader distribution for ATM, suggesting that in ATM that certain edges drive the majority of information necessary for differentiation of the two groups. B. Edge representation in a chord plot, showing the importance of each edge in the classification. C. Mean importance value of each edge of a specific brain region. This representation highlights which regions mainly impact in the classification. *


### Figure 4 - Time dependence of classification accuracy
![Fig. 2](./Figures_paper/Fig4.tiff)




