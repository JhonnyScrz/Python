import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# import data files

df = pd.read_excel('SalesProspects.xlsx')
df.columns =df.columns.str.strip()
df = df[df['company_size'] > 0]
df['company_size_log'] = [np.log(x) for x in df['company_size']]
df = df.drop('company_size', 1)
# removing spaces before and after column names

# df = df.dropna()

# df.columns = df.columns.str.strip()
# handling outliers in "company_size by removing size 0 since its an error, and logging the rest"

# filling missing dates in sale date in order to process non buyers
df.bfill(axis=0, inplace=True)

# check for null values
print(df.isnull().any())

# user id is not relevant so i dropped it
df = df.drop(['user_id'], 1)

# dealing with category objects using binary encoding
objectlist = df.select_dtypes(object)
for key in objectlist:
    if key != 'user_id':
        df2 = pd.get_dummies(df, columns=[key])
        df = df.drop([key], 1)
        df = pd.DataFrame.merge(df, df2)

# dealing with dates by getting time in days between events
date_list = df.select_dtypes(np.datetime64)
df3 = pd.DataFrame()
# df3['contact_upto_sale_days'] = pd.Series(delta.days for delta in (df['sale_date'] - df['date_first_contact']))
# df3['join_upto_sale_days'] = pd.Series(delta.days for delta in (df['sale_date'] - df['joining_date']))
df = df.drop(['sale_date'], 1)
df = df.drop(['date_first_contact'], 1)
df = df.drop(['joining_date'], 1)
df = pd.DataFrame(pd.concat([df, df3], axis=1))


def accuracy_text_RF(data):
    labels = np.asarray(df.did_buy)
    data_features = data.to_dict(orient='records')
    vec = DictVectorizer()
    features = vec.fit_transform(data_features).toarray()
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels,
        test_size=0.20, random_state=499)
    # initialize
    clf = RandomForestClassifier()
    # train the classifier using the training data
    clf.fit(features_train, labels_train)
    # compute accuracy using test data
    acc_test = clf.score(features_test, labels_test)
    print("Test Accuracy:", acc_test)


def perform_linear_regression_prediction(df):
    # linear regression
    X = df.drop("did_buy", axis=1)
    y = df.did_buy
    featureslist = X.keys()
    lm = LinearRegression()
    lm.fit(X, y)
    LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
    print("estimated intercept coeffs: ", lm.intercept_)
    print('number of coefs: ', len(lm.coef_))
    predictions = lm.predict(X)[0:-1]
    print("prediction mean value: ", predictions.mean())
    print(predictions)
    # makeplot(predictions=predictions)
    save_model(lm)


def evaluate_regression_model(X, y):
    lm = LinearRegression()
    X, y = make_regression(100, n_features=3, n_informative=3, n_targets=1, noise=50, coef=False, random_state=1)
    crossvaluesscore = cross_val_score(lm, X, y, scoring='neg_mean_squared_error')
    print(crossvaluesscore)
    crossvaluesscore = cross_val_score(lm, X, y, scoring='r2')
    print(crossvaluesscore)


def cross_validation(X, y):
    # cross evaluation of models
    standardizer = StandardScaler()
    logit = LogisticRegression()
    pipline = make_pipeline(standardizer, logit)
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    cv_results = cross_val_score(pipline, X, y, cv=kf, scoring="accuracy", n_jobs=-1)
    print("cv results: ", cv_results)
    print(cv_results)
    print("cv results mean: ", cv_results.mean())


def save_model(model):
    print("saving model")
    scikit_version = joblib.__version__
    joblib.dump(model, "model_{version}.pkl".format(version=scikit_version))


def makeplot(predictions):
    plt.style.use('ggplot')
    plt.ylabel("intent")
    plt.xlabel("prediction")
    plt.plot(predictions)
    plt.show()


perform_linear_regression_prediction(df)

from scipy.stats.stats import pearsonr

X = df.drop("did_buy", axis=1)
y = df.did_buy
# print(pearsonr(df['did_buy'],df['company_size_log']))
# print(np.corrcoef(X,y))


positivecorlist = list()
negativecorelist =list()
for value in X:

    if y.corr(df[value]) > 0:
        print(value)
        print(y.corr(df[value]))
        positivecorlist.append(""+value+" : "+str(y.corr(df[value])))

    print(positivecorlist)


