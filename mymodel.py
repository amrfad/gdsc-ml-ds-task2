from sklearn.ensemble import RandomForestRegressor

class MyModel(RandomForestRegressor):
  def predict(self, X):
    y = super().predict(X)
    for i in range(len(X['FunctioningDay_No'])):
      if X['FunctioningDay_No'].iloc[i] == 1:
        y[i] = 0
    return y