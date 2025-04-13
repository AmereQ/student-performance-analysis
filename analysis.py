from eda import load_and_prepare_data, show_g3_distribution
from model import train_model, evaluate_model, plot_feature_importance
from utils import split_data

show_g3_distribution()

X, y = load_and_prepare_data()

X_train, X_test, y_train, y_test = split_data(X, y)

model = train_model(X_train, y_train)

y_pred = evaluate_model(model, X_test, y_test)

plot_feature_importance(model, X)
