import pandas as pd
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
import matplotlib.pyplot as plt
import scipy

def analyze_kmeans(df, features):
    kmeans = KMeans(n_clusters=3, random_state=69)

    kmeans.fit(df[features])
    df['labels'] = kmeans.labels_

    df_label0 = df[df['labels']==0]
    
    print("Number of samples with label 0 for kmeans: ", len(df_label0))
    print("Number of Iris-setosa samples labeled 0 for kmeans: ", len(df_label0[df_label0['class']=='Iris-setosa']))
    print("Number of Iris-versicolor samples labeled 0 for kmeans: ", len(df_label0[df_label0['class']=='Iris-versicolor']))
    print("Number of Iris-virginica samples labeled 0 for kmeans: ", len(df_label0[df_label0['class']=='Iris-virginica']))
    print("")

    df_label1 = df[df['labels']==1]
    print("Number of samples with label 1 for kmeans: ", len(df_label1))
    print("Number of Iris-setosa samples labeled 1 for kmeans: ", len(df_label1[df_label1['class']=='Iris-setosa']))
    print("Number of Iris-versicolor samples labeled 1 for kmeans: ", len(df_label1[df_label1['class']=='Iris-versicolor']))
    print("Number of Iris-virginica samples labeled 1 for kmeans: ", len(df_label1[df_label1['class']=='Iris-virginica']))
    print("")

    df_label2 = df[df['labels']==2]
    print("Number of samples with label 2 for kmeans: ", len(df_label2))
    print("Number of Iris-setosa samples labeled 2 for kmeans: ", len(df_label2[df_label2['class']=='Iris-setosa']))
    print("Number of Iris-versicolor samples labeled 2 for kmeans: ", len(df_label2[df_label2['class']=='Iris-versicolor']))
    print("Number of Iris-virginica samples labeled 2 for kmeans: ", len(df_label2[df_label2['class']=='Iris-virginica']))
    print("")

    df.drop(columns=['labels'], inplace=True)

def analyze_em(df, features):
    em = GaussianMixture(n_components=3, random_state=69)
    predictions = em.fit_predict(df[features])
    df['labels'] = predictions

    df_label0 = df[df['labels']==0]
    
    print("Number of samples with label 0 for EM: ", len(df_label0))
    print("Number of Iris-setosa samples labeled 0 for EM: ", len(df_label0[df_label0['class']=='Iris-setosa']))
    print("Number of Iris-versicolor samples labeled 0 for EM: ", len(df_label0[df_label0['class']=='Iris-versicolor']))
    print("Number of Iris-virginica samples labeled 0 for EM: ", len(df_label0[df_label0['class']=='Iris-virginica']))
    print("")

    df_label1 = df[df['labels']==1]
    print("Number of samples with label 1 for EM: ", len(df_label1))
    print("Number of Iris-setosa samples labeled 1 for EM: ", len(df_label1[df_label1['class']=='Iris-setosa']))
    print("Number of Iris-versicolor samples labeled 1 for EM: ", len(df_label1[df_label1['class']=='Iris-versicolor']))
    print("Number of Iris-virginica samples labeled 1 for EM: ", len(df_label1[df_label1['class']=='Iris-virginica']))
    print("")

    df_label2 = df[df['labels']==2]
    print("Number of samples with label 2 for EM: ", len(df_label2))
    print("Number of Iris-setosa samples labeled 2 for EM: ", len(df_label2[df_label2['class']=='Iris-setosa']))
    print("Number of Iris-versicolor samples labeled 2 for EM: ", len(df_label2[df_label2['class']=='Iris-versicolor']))
    print("Number of Iris-virginica samples labeled 2 for EM: ", len(df_label2[df_label2['class']=='Iris-virginica']))
    print("")

    df.drop(columns=['labels'], inplace=True)

def analyze_PCA(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    pca = PCA(n_components=2, random_state=69)
    principalComponents = pca.fit_transform(df_scaled)

    print(pca.singular_values_)

    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

    new_df = pd.concat([principalDf, df[['class']]], axis=1)

    plt.xlabel('Principal Component 1', fontsize = 15)
    plt.ylabel('Principal Component 2', fontsize = 15)
    plt.title('2 component PCA', fontsize = 20)

    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'g', 'b']
    for iris_class, color in zip(classes,colors):
        indicesToKeep = new_df['class'] == iris_class
        plt.scatter(new_df.loc[indicesToKeep, 'principal component 1']
               , new_df.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)

    plt.legend(classes)
    plt.savefig('figures/PCA_iris.png')
    plt.clf()

    pca = PCA(n_components=4, random_state=69)
    principalComponents = pca.fit_transform(df_scaled)

    print(pca.singular_values_)

    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4'])

    new_df = pd.concat([principalDf, df[['class']]], axis=1)

    analyze_kmeans(new_df, ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4'])
    analyze_em(new_df, ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4'])

def analyze_ICA(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    ica = FastICA(n_components=2, random_state=69)
    independentComponents = ica.fit_transform(df_scaled)

    print(scipy.stats.kurtosis(independentComponents))

    independentDf = pd.DataFrame(data = independentComponents
             , columns = ['independent component 1', 'independent component 2'])

    new_df = pd.concat([independentDf, df[['class']]], axis=1)

    plt.xlabel('Independent Component 1', fontsize = 15)
    plt.ylabel('Independent Component 2', fontsize = 15)
    plt.title('2 component ICA', fontsize = 20)

    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'g', 'b']
    for iris_class, color in zip(classes,colors):
        indicesToKeep = new_df['class'] == iris_class
        plt.scatter(new_df.loc[indicesToKeep, 'independent component 1']
               , new_df.loc[indicesToKeep, 'independent component 2']
               , c = color
               , s = 50)

    plt.legend(classes)
    plt.savefig('figures/ICA_iris.png')
    plt.clf()

    ica = FastICA(n_components=4, random_state=69)
    independentComponents = ica.fit_transform(df_scaled)

    print(scipy.stats.kurtosis(independentComponents))

    independentDf = pd.DataFrame(data = independentComponents
             , columns = ['independent component 1', 'independent component 2', 'independent component 3', 'independent component 4'])

    new_df = pd.concat([independentDf, df[['class']]], axis=1)

    analyze_kmeans(new_df, ['independent component 1', 'independent component 2', 'independent component 3', 'independent component 4'])
    analyze_em(new_df, ['independent component 1', 'independent component 2', 'independent component 3', 'independent component 4'])

def analyze_random_projection(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    transformer = GaussianRandomProjection(n_components=2)
    randomComponents = transformer.fit_transform(df_scaled)

    randomDf = pd.DataFrame(data = randomComponents
             , columns = ['random component 1', 'random component 2'])

    new_df = pd.concat([randomDf, df[['class']]], axis=1)

    plt.xlabel('Random Component 1', fontsize = 15)
    plt.ylabel('Random Component 2', fontsize = 15)
    plt.title('2 component Random Projection', fontsize = 20)

    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'g', 'b']
    for iris_class, color in zip(classes,colors):
        indicesToKeep = new_df['class'] == iris_class
        plt.scatter(new_df.loc[indicesToKeep, 'random component 1']
               , new_df.loc[indicesToKeep, 'random component 2']
               , c = color
               , s = 50)

    plt.legend(classes)
    plt.savefig('figures/RandomProjection_iris.png')
    plt.clf()

    analyze_kmeans(new_df, ['random component 1', 'random component 2'])
    analyze_em(new_df, ['random component 1', 'random component 2'])

def analyze_feature_agglomeration(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    transformer = FeatureAgglomeration(n_clusters=2)
    agglomeratedComponents = transformer.fit_transform(df_scaled)

    agglomeratedDf = pd.DataFrame(data = agglomeratedComponents
             , columns = ['agglomerated component 1', 'agglomerated component 2'])

    new_df = pd.concat([agglomeratedDf, df[['class']]], axis=1)

    plt.xlabel('Agglomerated Component 1', fontsize = 15)
    plt.ylabel('Agglomerated Component 2', fontsize = 15)
    plt.title('2 component Feature Agglomeration', fontsize = 20)

    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    colors = ['r', 'g', 'b']
    for iris_class, color in zip(classes,colors):
        indicesToKeep = new_df['class'] == iris_class
        plt.scatter(new_df.loc[indicesToKeep, 'agglomerated component 1']
               , new_df.loc[indicesToKeep, 'agglomerated component 2']
               , c = color
               , s = 50)

    plt.legend(classes)
    plt.savefig('figures/FeatureAgglomeration_iris.png')
    plt.clf()

    analyze_kmeans(new_df, ['agglomerated component 1', 'agglomerated component 2'])
    analyze_em(new_df, ['agglomerated component 1', 'agglomerated component 2'])

if (__name__=="__main__"):
    features_iris = ['sepal length','sepal width','petal length','petal width']

    df_iris = pd.read_csv('data/iris-data.csv')

    analyze_kmeans(df_iris, features_iris)
    analyze_em(df_iris, features_iris)
    analyze_PCA(df_iris, features_iris)
    analyze_ICA(df_iris, features_iris)
    analyze_random_projection(df_iris, features_iris)
    analyze_feature_agglomeration(df_iris, features_iris)