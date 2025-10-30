## GeosciencePlusAI

This repository is part of the teaching material for *Geoscience with Integrated Data Science* at *Heriot-Watt University*.

It includes applications of **Machine Learning** (ML) techniques - specifically **clustering**, **classification** and **regression** - to well log data. While these methods can be applied to any well log dataset, this repository uses well log data from seven wells in the **Volve** field dataset as an example. 

### Overview

The notebooks in this repository follow a structured workflow:

- **Visualisation & Exploration**  
   - Examine well log data and visualise lithology distributions.  
   - Determine *training*, *validation*, and *testing* splits for well data. 

- **Clustering with K-means**  
   - Apply K-Means clustering, an *unsupervised* learning algorithm, to group data points based on similarities in well log responses.
   - Analyse resulting clusters, and investigate how well they align with known lithologies, providing insights into natural groupings within the dataset.

- **Classification with ML Algorithms**
   
   To classify lithology, we employ two *supervised* learning algorithms:
   - **K-Nearest Neighbours** (KNN): A distance-based classification method that predicts lithology labels by considering the majority class among the nearest neighbors of a data point.
   - **Random Forest** (RF): An ensemble learning method that builds multiple decision trees to improve classification performance by aggregating predictions, enhancing accuracy, and reducing overfitting.
   - **Support Vector Machine** (SVM): A powerful classification algorithm that finds the optimal hyperplane to separate classes in the feature space. It is particularly effective for high-dimensional datasets and can model complex decision boundaries using different kernel functions (e.g., linear, RBF, polynomial).
 
- **Permeability Prediction with Neural Networks**  
   - Use a feedforward *neural network* to predict permeability (KLOGH) from well log measurements.
   - Analyse network predictions to evaluate how well trends in permeability are captured.

### Data

The dataset consists of well log data from seven wells in the **Volve** field dataset, with their paths visualised in 3D below. 
![3D Visualization of Volve Field Wells](https://github.com/farah-rabie/GeosciencePlusAI/blob/main/Data/Images/3D%20Visualisation%20of%20Volve%20Field%20Wells.png)
For details on data extraction, log descriptions, and file formats, see the README file [here]([Data/README.md](https://github.com/farah-rabie/GeosciencePlusAI/blob/main/Data/README.md)).

### Usage

1. Start with **visualisation** to explore well log distributions.  
2. Use **clustering methods** to segment data based on feature similarities.  
3. Train and evaluate **classification models**, namely KNN and RF.
4. Use **neural networks** to predict permeability from well log measurements.

### Licence

This repository is licensed under the **MIT Licence**. See `LICENSE` for details.

### Who Can I Contact?
Farah Rabie (f.rabie@hw.ac.uk)

Prof. Vasily Demyanov (v.demyanov@hw.ac.uk)
