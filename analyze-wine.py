import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn import utils
from sklearn import neural_network
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import scipy
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore",category=UndefinedMetricWarning)

def analyze_kmeans(df, features, num_clusters=2):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    kmeans = KMeans(n_clusters=num_clusters, random_state=69)

    kmeans.fit(df_scaled[features])
    df['labels'] = kmeans.labels_

    for i in range(num_clusters):
        df_label = df[df['labels']==i]

        print(df_label['quality'].value_counts())

    df.drop(columns=['labels'], inplace=True)

def analyze_em(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    em = GaussianMixture(n_components=2, random_state=69)
    predictions = em.fit_predict(df_scaled[features])
    df['labels'] = predictions

    df_label0 = df[df['labels']==0]

    print(df_label0['quality'].value_counts())

    df_label1 = df[df['labels']==1]

    print(df_label1['quality'].value_counts())

    df.drop(columns=['labels'], inplace=True)

def analyze_PCA(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    pca = PCA(n_components=0.95, random_state=69)
    principalComponents = pca.fit_transform(df_scaled)

    print(pca.singular_values_)

    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 
             'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8'])

    new_df = pd.concat([principalDf, df[['quality']]], axis=1)

    analyze_kmeans(new_df, ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 
             'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8'])
    
    analyze_em(new_df, ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 
             'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8'])

def analyze_ICA(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    ica = FastICA(n_components=8, random_state=69)
    independentComponents = ica.fit_transform(df_scaled)

    print(scipy.stats.kurtosis(independentComponents))

    independentDf = pd.DataFrame(data = independentComponents
             , columns = ['independent component 1', 'independent component 2', 'independent component 3', 'independent component 4', 
             'independent component 5', 'independent component 6', 'independent component 7', 'independent component 8'])

    new_df = pd.concat([independentDf, df[['quality']]], axis=1)

    analyze_kmeans(new_df, ['independent component 1', 'independent component 2', 'independent component 3', 'independent component 4', 
             'independent component 5', 'independent component 6', 'independent component 7', 'independent component 8'])
    
    analyze_em(new_df, ['independent component 1', 'independent component 2', 'independent component 3', 'independent component 4', 
             'independent component 5', 'independent component 6', 'independent component 7', 'independent component 8'])

def analyze_random_projection(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    transformer = GaussianRandomProjection(n_components=8)
    randomComponents = transformer.fit_transform(df_scaled)

    randomDf = pd.DataFrame(data = randomComponents
             , columns = ['random component 1', 'random component 2', 'random component 3', 'random component 4', 
             'random component 5', 'random component 6', 'random component 7', 'random component 8'])

    new_df = pd.concat([randomDf, df[['quality']]], axis=1)

    analyze_kmeans(new_df, ['random component 1', 'random component 2', 'random component 3', 'random component 4', 
             'random component 5', 'random component 6', 'random component 7', 'random component 8'])
    analyze_em(new_df, ['random component 1', 'random component 2', 'random component 3', 'random component 4', 
             'random component 5', 'random component 6', 'random component 7', 'random component 8'])

def analyze_PCA_neural_network(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    pca = PCA(n_components=0.95, random_state=69)
    principalComponents = pca.fit_transform(df_scaled)

    features = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 
             'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8']

    principalDf = pd.DataFrame(data = principalComponents
             , columns = features)

    principalDf = pd.concat([principalDf, df[['good']]], axis=1)

    # Splitting dataset into train, test sets
    train_df, test_df = train_test_split(principalDf, test_size=0.2, random_state=69)
    
    # Balancing the training set
    train_good_df = train_df[train_df['good']==1]
    train_df = pd.concat([train_df[train_df['good']==0].head(len(train_good_df)), train_good_df])
    train_df = utils.shuffle(train_df, random_state=69)

    train_X = train_df[features]
    test_X = test_df[features]
    train_Y = train_df['good']
    test_Y = test_df['good']

    # Tuning hyper parameters

    num_hidden_units_range = []
    for i in range(3, 31, 3):
        num_hidden_units_range.append((i,i))

    alpha_range = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

    param_dict = {"hidden_layer_sizes": num_hidden_units_range, "alpha": alpha_range}

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', learning_rate_init=0.02, early_stopping=True)
    random_search = RandomizedSearchCV(clf, param_dict, random_state=69, scoring='accuracy', cv=5, n_iter=50, n_jobs=4)
    random_search = random_search.fit(train_X, train_Y)

    params = random_search.best_params_

    print(params)

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', early_stopping=True, hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=0.02, alpha=params['alpha'])

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Optimized Neural Network accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Optimized Neural Network accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    print(confusion_matrix(test_Y, predict_Y))

def analyze_ICA_neural_network(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    ica = FastICA(n_components=8, random_state=69)
    independentComponents = ica.fit_transform(df_scaled)

    features = ['independent component 1', 'independent component 2', 'independent component 3', 'independent component 4', 
             'independent component 5', 'independent component 6', 'independent component 7', 'independent component 8']

    independentDf = pd.DataFrame(data = independentComponents
             , columns = features)

    independentDf = pd.concat([independentDf, df[['good']]], axis=1)

    # Splitting dataset into train, test sets
    train_df, test_df = train_test_split(independentDf, test_size=0.2, random_state=69)
    
    # Balancing the training set
    train_good_df = train_df[train_df['good']==1]
    train_df = pd.concat([train_df[train_df['good']==0].head(len(train_good_df)), train_good_df])
    train_df = utils.shuffle(train_df, random_state=69)

    train_X = train_df[features]
    test_X = test_df[features]
    train_Y = train_df['good']
    test_Y = test_df['good']

    # Tuning hyper parameters

    num_hidden_units_range = []
    for i in range(3, 31, 3):
        num_hidden_units_range.append((i,i))

    alpha_range = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

    param_dict = {"hidden_layer_sizes": num_hidden_units_range, "alpha": alpha_range}

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', learning_rate_init=0.02, early_stopping=True)
    random_search = RandomizedSearchCV(clf, param_dict, random_state=69, scoring='accuracy', cv=5, n_iter=50, n_jobs=4)
    random_search = random_search.fit(train_X, train_Y)

    params = random_search.best_params_

    print(params)

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', early_stopping=True, hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=0.02, alpha=params['alpha'])

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Optimized Neural Network accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Optimized Neural Network accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    print(confusion_matrix(test_Y, predict_Y))

def analyze_RCA_neural_network(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    transformer = GaussianRandomProjection(n_components=8)
    randomComponents = transformer.fit_transform(df_scaled)

    features = ['random component 1', 'random component 2', 'random component 3', 'random component 4', 
             'random component 5', 'random component 6', 'random component 7', 'random component 8']

    randomDf = pd.DataFrame(data = randomComponents
             , columns = features)

    randomDf = pd.concat([randomDf, df[['good']]], axis=1)

    # Splitting dataset into train, test sets
    train_df, test_df = train_test_split(randomDf, test_size=0.2, random_state=69)
    
    # Balancing the training set
    train_good_df = train_df[train_df['good']==1]
    train_df = pd.concat([train_df[train_df['good']==0].head(len(train_good_df)), train_good_df])
    train_df = utils.shuffle(train_df, random_state=69)

    train_X = train_df[features]
    test_X = test_df[features]
    train_Y = train_df['good']
    test_Y = test_df['good']

    # Tuning hyper parameters

    num_hidden_units_range = []
    for i in range(3, 31, 3):
        num_hidden_units_range.append((i,i))

    alpha_range = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

    param_dict = {"hidden_layer_sizes": num_hidden_units_range, "alpha": alpha_range}

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', learning_rate_init=0.02, early_stopping=True)
    random_search = RandomizedSearchCV(clf, param_dict, random_state=69, scoring='accuracy', cv=5, n_iter=50, n_jobs=4)
    random_search = random_search.fit(train_X, train_Y)

    params = random_search.best_params_

    print(params)

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', early_stopping=True, hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=0.02, alpha=params['alpha'])

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Optimized Neural Network accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Optimized Neural Network accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    print(confusion_matrix(test_Y, predict_Y))

def analyze_PCA_kmeans_cluster_neural_network(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    pca = PCA(n_components=0.95, random_state=69)
    principalComponents = pca.fit_transform(df_scaled)

    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 
             'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8'])

    new_df = pd.concat([principalDf, df[['quality']]], axis=1)

    kmeans = KMeans(n_clusters=2, random_state=69)

    kmeans.fit(new_df[['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 
             'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8']])
    
    final_df = pd.concat([df_scaled, pd.get_dummies(kmeans.labels_), df['good']], axis=1)

    features.append(0)
    features.append(1)

    # Splitting dataset into train, test sets
    train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=69)
    
    # Balancing the training set
    train_good_df = train_df[train_df['good']==1]
    train_df = pd.concat([train_df[train_df['good']==0].head(len(train_good_df)), train_good_df])
    train_df = utils.shuffle(train_df, random_state=69)

    train_X = train_df[features]
    test_X = test_df[features]
    train_Y = train_df['good']
    test_Y = test_df['good']

    # Tuning hyper parameters

    num_hidden_units_range = []
    for i in range(3, 31, 3):
        num_hidden_units_range.append((i,i))

    alpha_range = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

    param_dict = {"hidden_layer_sizes": num_hidden_units_range, "alpha": alpha_range}

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', learning_rate_init=0.02, early_stopping=True)
    random_search = RandomizedSearchCV(clf, param_dict, random_state=69, scoring='accuracy', cv=5, n_iter=50, n_jobs=4)
    random_search = random_search.fit(train_X, train_Y)

    params = random_search.best_params_

    print(params)

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', early_stopping=True, hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=0.02, alpha=params['alpha'])

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Optimized Neural Network accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Optimized Neural Network accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    print(confusion_matrix(test_Y, predict_Y))

def analyze_ICA_kmeans_cluster_neural_network(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    ica = FastICA(n_components=8, random_state=69)
    independentComponents = ica.fit_transform(df_scaled)

    independentDf = pd.DataFrame(data = independentComponents
             , columns = ['independent component 1', 'independent component 2', 'independent component 3', 'independent component 4', 
             'independent component 5', 'independent component 6', 'independent component 7', 'independent component 8'])

    new_df = pd.concat([independentDf, df[['quality']]], axis=1)

    kmeans = KMeans(n_clusters=2, random_state=69)

    kmeans.fit(new_df[['independent component 1', 'independent component 2', 'independent component 3', 'independent component 4', 
             'independent component 5', 'independent component 6', 'independent component 7', 'independent component 8']])
    
    final_df = pd.concat([df_scaled, pd.get_dummies(kmeans.labels_), df['good']], axis=1)

    features.append(0)
    features.append(1)

    # Splitting dataset into train, test sets
    train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=69)
    
    # Balancing the training set
    train_good_df = train_df[train_df['good']==1]
    train_df = pd.concat([train_df[train_df['good']==0].head(len(train_good_df)), train_good_df])
    train_df = utils.shuffle(train_df, random_state=69)

    train_X = train_df[features]
    test_X = test_df[features]
    train_Y = train_df['good']
    test_Y = test_df['good']

    # Tuning hyper parameters

    num_hidden_units_range = []
    for i in range(3, 31, 3):
        num_hidden_units_range.append((i,i))

    alpha_range = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

    param_dict = {"hidden_layer_sizes": num_hidden_units_range, "alpha": alpha_range}

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', learning_rate_init=0.02, early_stopping=True)
    random_search = RandomizedSearchCV(clf, param_dict, random_state=69, scoring='accuracy', cv=5, n_iter=50, n_jobs=4)
    random_search = random_search.fit(train_X, train_Y)

    params = random_search.best_params_

    print(params)

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', early_stopping=True, hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=0.02, alpha=params['alpha'])

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Optimized Neural Network accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Optimized Neural Network accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    print(confusion_matrix(test_Y, predict_Y))

def analyze_RCA_kmeans_cluster_neural_network(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    transformer = GaussianRandomProjection(n_components=8)
    randomComponents = transformer.fit_transform(df_scaled)

    randomDf = pd.DataFrame(data = randomComponents
             , columns = ['random component 1', 'random component 2', 'random component 3', 'random component 4', 
             'random component 5', 'random component 6', 'random component 7', 'random component 8'])

    new_df = pd.concat([randomDf, df[['quality']]], axis=1)

    kmeans = KMeans(n_clusters=2, random_state=69)

    kmeans.fit(new_df[['random component 1', 'random component 2', 'random component 3', 'random component 4', 
             'random component 5', 'random component 6', 'random component 7', 'random component 8']])
    
    final_df = pd.concat([df_scaled, pd.get_dummies(kmeans.labels_), df['good']], axis=1)

    features.append(0)
    features.append(1)

    # Splitting dataset into train, test sets
    train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=69)
    
    # Balancing the training set
    train_good_df = train_df[train_df['good']==1]
    train_df = pd.concat([train_df[train_df['good']==0].head(len(train_good_df)), train_good_df])
    train_df = utils.shuffle(train_df, random_state=69)

    train_X = train_df[features]
    test_X = test_df[features]
    train_Y = train_df['good']
    test_Y = test_df['good']

    # Tuning hyper parameters

    num_hidden_units_range = []
    for i in range(3, 31, 3):
        num_hidden_units_range.append((i,i))

    alpha_range = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

    param_dict = {"hidden_layer_sizes": num_hidden_units_range, "alpha": alpha_range}

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', learning_rate_init=0.02, early_stopping=True)
    random_search = RandomizedSearchCV(clf, param_dict, random_state=69, scoring='accuracy', cv=5, n_iter=50, n_jobs=4)
    random_search = random_search.fit(train_X, train_Y)

    params = random_search.best_params_

    print(params)

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', early_stopping=True, hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=0.02, alpha=params['alpha'])

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Optimized Neural Network accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Optimized Neural Network accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    print(confusion_matrix(test_Y, predict_Y))  

def analyze_PCA_EM_cluster_neural_network(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    pca = PCA(n_components=0.95, random_state=69)
    principalComponents = pca.fit_transform(df_scaled)

    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 
             'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8'])

    new_df = pd.concat([principalDf, df[['quality']]], axis=1)

    em = GaussianMixture(n_components=2, random_state=69)
    predictions = em.fit_predict(new_df[['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 
             'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8']])
    
    final_df = pd.concat([df_scaled, pd.get_dummies(predictions), df['good']], axis=1)

    features.append(0)
    features.append(1)

    # Splitting dataset into train, test sets
    train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=69)
    
    # Balancing the training set
    train_good_df = train_df[train_df['good']==1]
    train_df = pd.concat([train_df[train_df['good']==0].head(len(train_good_df)), train_good_df])
    train_df = utils.shuffle(train_df, random_state=69)

    train_X = train_df[features]
    test_X = test_df[features]
    train_Y = train_df['good']
    test_Y = test_df['good']

    # Tuning hyper parameters

    num_hidden_units_range = []
    for i in range(3, 31, 3):
        num_hidden_units_range.append((i,i))

    alpha_range = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

    param_dict = {"hidden_layer_sizes": num_hidden_units_range, "alpha": alpha_range}

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', learning_rate_init=0.02, early_stopping=True)
    random_search = RandomizedSearchCV(clf, param_dict, random_state=69, scoring='accuracy', cv=5, n_iter=50, n_jobs=4)
    random_search = random_search.fit(train_X, train_Y)

    params = random_search.best_params_

    print(params)

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', early_stopping=True, hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=0.02, alpha=params['alpha'])

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Optimized Neural Network accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Optimized Neural Network accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    print(confusion_matrix(test_Y, predict_Y))

def analyze_ICA_EM_cluster_neural_network(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    ica = FastICA(n_components=8, random_state=69)
    independentComponents = ica.fit_transform(df_scaled)

    independentDf = pd.DataFrame(data = independentComponents
             , columns = ['independent component 1', 'independent component 2', 'independent component 3', 'independent component 4', 
             'independent component 5', 'independent component 6', 'independent component 7', 'independent component 8'])

    new_df = pd.concat([independentDf, df[['quality']]], axis=1)

    em = GaussianMixture(n_components=2, random_state=69)
    predictions = em.fit_predict(new_df[['independent component 1', 'independent component 2', 'independent component 3', 'independent component 4', 
             'independent component 5', 'independent component 6', 'independent component 7', 'independent component 8']])
    
    final_df = pd.concat([df_scaled, pd.get_dummies(predictions), df['good']], axis=1)

    features.append(0)
    features.append(1)

    # Splitting dataset into train, test sets
    train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=69)
    
    # Balancing the training set
    train_good_df = train_df[train_df['good']==1]
    train_df = pd.concat([train_df[train_df['good']==0].head(len(train_good_df)), train_good_df])
    train_df = utils.shuffle(train_df, random_state=69)

    train_X = train_df[features]
    test_X = test_df[features]
    train_Y = train_df['good']
    test_Y = test_df['good']

    # Tuning hyper parameters

    num_hidden_units_range = []
    for i in range(3, 31, 3):
        num_hidden_units_range.append((i,i))

    alpha_range = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

    param_dict = {"hidden_layer_sizes": num_hidden_units_range, "alpha": alpha_range}

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', learning_rate_init=0.02, early_stopping=True)
    random_search = RandomizedSearchCV(clf, param_dict, random_state=69, scoring='accuracy', cv=5, n_iter=50, n_jobs=4)
    random_search = random_search.fit(train_X, train_Y)

    params = random_search.best_params_

    print(params)

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', early_stopping=True, hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=0.02, alpha=params['alpha'])

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Optimized Neural Network accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Optimized Neural Network accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    print(confusion_matrix(test_Y, predict_Y))

def analyze_RCA_EM_cluster_neural_network(df, features):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=features)

    transformer = GaussianRandomProjection(n_components=8)
    randomComponents = transformer.fit_transform(df_scaled)

    randomDf = pd.DataFrame(data = randomComponents
             , columns = ['random component 1', 'random component 2', 'random component 3', 'random component 4', 
             'random component 5', 'random component 6', 'random component 7', 'random component 8'])

    new_df = pd.concat([randomDf, df[['quality']]], axis=1)

    em = GaussianMixture(n_components=2, random_state=69)
    predictions = em.fit_predict(new_df[['random component 1', 'random component 2', 'random component 3', 'random component 4', 
             'random component 5', 'random component 6', 'random component 7', 'random component 8']])
    
    final_df = pd.concat([df_scaled, pd.get_dummies(predictions), df['good']], axis=1)

    features.append(0)
    features.append(1)

    # Splitting dataset into train, test sets
    train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=69)
    
    # Balancing the training set
    train_good_df = train_df[train_df['good']==1]
    train_df = pd.concat([train_df[train_df['good']==0].head(len(train_good_df)), train_good_df])
    train_df = utils.shuffle(train_df, random_state=69)

    train_X = train_df[features]
    test_X = test_df[features]
    train_Y = train_df['good']
    test_Y = test_df['good']

    # Tuning hyper parameters

    num_hidden_units_range = []
    for i in range(3, 31, 3):
        num_hidden_units_range.append((i,i))

    alpha_range = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

    param_dict = {"hidden_layer_sizes": num_hidden_units_range, "alpha": alpha_range}

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', learning_rate_init=0.02, early_stopping=True)
    random_search = RandomizedSearchCV(clf, param_dict, random_state=69, scoring='accuracy', cv=5, n_iter=50, n_jobs=4)
    random_search = random_search.fit(train_X, train_Y)

    params = random_search.best_params_

    print(params)

    clf = neural_network.MLPClassifier(random_state=69, solver='sgd', max_iter=2000, learning_rate='adaptive', early_stopping=True, hidden_layer_sizes=params['hidden_layer_sizes'], learning_rate_init=0.02, alpha=params['alpha'])

    clf.fit(train_X, train_Y)

    predict_Y = clf.predict(train_X)

    print("Optimized Neural Network accuracy score on training set: " + str(accuracy_score(train_Y, predict_Y)))

    predict_Y = clf.predict(test_X)

    print("Optimized Neural Network accuracy score on test set: " + str(accuracy_score(test_Y, predict_Y)))

    print(confusion_matrix(test_Y, predict_Y))   
    

if (__name__=="__main__"):
    features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density' ,'pH' ,'sulphates', 'alcohol']

    df_wine = pd.read_csv('data/winequality-white.csv')
    df_wine['good'] = df_wine['quality'] > 6.5
    df_wine['good'] = df_wine['good'].astype(int)

    analyze_kmeans(df_wine, features)
    analyze_em(df_wine, features)
    analyze_PCA(df_wine, features)
    analyze_ICA(df_wine, features)
    analyze_random_projection(df_wine, features)
    analyze_PCA_neural_network(df_wine, features)
    analyze_ICA_neural_network(df_wine, features)
    analyze_RCA_neural_network(df_wine, features)
    analyze_PCA_kmeans_cluster_neural_network(df_wine, features)
    analyze_ICA_kmeans_cluster_neural_network(df_wine, features)
    analyze_RCA_kmeans_cluster_neural_network(df_wine, features)
    analyze_PCA_EM_cluster_neural_network(df_wine, features)
    analyze_ICA_EM_cluster_neural_network(df_wine, features)
    analyze_RCA_EM_cluster_neural_network(df_wine, features)