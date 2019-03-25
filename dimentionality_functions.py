import ann
import pandas as pd

# performs all the possible simluations of the ann
#   X: the DataFrame with the input data
#   y: the expected results
def dimensionality_reduction(X, y):
    column_names = list(X.columns.values)
    _all_combinations = all_combinations(X)
    n_rows = len(_all_combinations)
    n_cols = len(_all_combinations.iloc[0])
    for i in range(n_rows):
        aux_X = X
        for j in range(n_cols):
            if(_all_combinations.iloc[i][column_names[j]] == 0):
                #drop colum
                aux_X = aux_X.drop(column_names[j], axis=1)
        f= open("dimensionality/dimensionality_results.txt","a")
        f.write(str(len(list(aux_X.columns.values)))+' :'+str(list(aux_X.columns.values))+' \n')
        epochs = 50
        nn = ann.NeuralNetwork(aux_X, y, 3)
        nn.run_nn_simulation(aux_X, y, epochs, f)

# calculates all the possible combinations of which columns to use for a model
# params:
#   X: the DataFrame with the input data
def all_combinations(X):
    column_names = list(X.columns.values)
    n_cols = X.shape[1]
    num_combinations = 2 ** n_cols #exponent
    combinations= [[0]*n_cols]*num_combinations
    df_combinations = pd.DataFrame(combinations, columns=column_names)
    middle = int(num_combinations/2)
    half_1_half_0(0, middle, df_combinations, column_names, 1, 0)
    half_1_half_0(middle, num_combinations, df_combinations, column_names, 0, 0)
    return df_combinations

# aux function for recursion of all_combinations, assigns the value to the array
# combinations from start to end
# params:
#   start: int of where the assignation should start
#   end: int of where the assignation should end
#   combinations: the array holding all the possible combinations
#   column_names: the array of the column names
#   value: binary, 1 or 0 the value that will be assigned
#   current_col: int of the id of the current column
def half_1_half_0(start, end, combinations, column_names, value, current_col):
    if(start<end and start>=0 and current_col<len(column_names)):
        for i in range (start,end):
            combinations.loc[i][column_names[current_col]]= value
        middle=start+int((end-start)/2)
        half_1_half_0(start, middle, combinations, column_names, 1, current_col+1)
        half_1_half_0(middle, end, combinations, column_names, 0, current_col+1)


def main():
    X = pd.read_csv('../titanic/train_x.csv', sep=',', skiprows=1, names=['Pclass', 'Sex', 'Age', 'Parch', 'Fare'])
    y = pd.read_csv('../titanic/train_y.csv', sep=',', skiprows=1, names=['Survived'])

    #one hot encoding
    X = ann.one_hot_encoding(X, "Pclass")

    #regularization
    for column in X:
        X[column] = ann.regularization(X[column])

    dimensionality_reduction(X, y)

if __name__ == "__main__":
    main()
