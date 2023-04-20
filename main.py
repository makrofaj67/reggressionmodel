import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error
#data = pandas.read_csv('house_price_Test.csv')
#data = data.drop('Unnamed: 0', axis=1)
#cwd = os.getcwd()
#new_file_path = os.path.join(cwd, 'cleaned_file.csv')
#data.to_csv(new_file_path, index=False)

#data = pandas.read_csv("cleaned_file.csv")
#data = data.dropna()
#data.to_csv("cleaned_file2.csv", index=False)

data = pandas.read_csv("cleaned_file3.csv")
#new_df = data.iloc[:, 2:]
#new_df.to_csv('cleaned_file3.csv', index=False)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

X_train = train_data.drop('Price', axis=1)
y_train = train_data['Price']
X_test = test_data.drop('Price', axis=1)
y_test = test_data['Price']

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Root Mean Squared Error:", rmse)


