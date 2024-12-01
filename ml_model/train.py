import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import joblib

def train_model():  
    # Load the dataset
    data = pd.read_csv('data/Mall_Customers.csv')

    # Preprocess the dataset
    x = data.iloc[:, [3, 4]].values

    # Split the data into training and test sets
    wcss_list=[]#initailsing for values of k

    #using the for loops for iterations from 1 to 10
    for i in range(1,11):
        kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
        kmeans.fit(x)
        wcss_list.append(kmeans.inertia_)

    plt.plot(range(1,11),wcss_list)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    # Train the model
    model = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)

    # fit the model
    model.fit(x)

    # Save the model
    joblib.dump(model, 'ml_model/kmeans.pkl')
    return model
if __name__ == '__main__':
    train_model()