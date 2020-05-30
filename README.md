# Steel Defect Detection

In this project we used the Severstal steel defect image dataset from [Kaggle](https://www.kaggle.com/c/severstal-steel-defect-detection/data) to build three convolutional neural networks models. The dataset contains images of steel surface some with no defects and others with at least one type of detect. In total there is four types of defects with a small amount of images with more than one type of defects. 

First we explored the data (EDA) and found class imbalance for the images that have at least one type of defect. Second, we built a binary classifier to predict if a steel furface has a defect. Third, we built a multi-class model to predict the type of defect on the steel suyrface. Finally, the Unet model colors the defect based on the type.

# Tasks and corresponding file/folder names:

EDA - EDA, segm, data_augmentation.ipynb\
Binary Classification - Binary_2_save.ipynb\
Multi-class Classification - Multiclass-classification\
Defect Detection - Defect_Detection-Unet.ipynb

# Proposal, Report and Final Presentation:

Proposal - Steel Defect Detection_proposal.docx\
Report - Steel Defect Detection report-final.Pdf\
Final Presentation - Steel_Defect_Detection.ppt
