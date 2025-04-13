import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_prepare_data(filepath="student-mat.csv"):
    df = pd.read_csv(filepath, sep=';')

    df['pass_fail'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

    df = df.drop(columns=['G1', 'G2', 'G3'])

    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop(columns=['pass_fail'])
    y = df_encoded['pass_fail']

    return X, y

def show_g3_distribution():
    df = pd.read_csv("student-mat.csv", sep=';')
    sns.histplot(df['G3'], kde=True)
    plt.title("Distribution of Final Grades (G3)")
    plt.xlabel("G3")
    plt.ylabel("Count")
    plt.show()
