>>> import numpy as np

def get_dimensions(matrix):

    # Return the dimensions of any given matrix
    return [len(matrix), len(matrix[0])]

def find_determinant(matrix, excluded=1):
   
    # Return the value of the determinant of any given matrix
    dimensions = get_dimensions(matrix)
    if dimensions == [2, 2]:
        return excluded * ((matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]))
    else:
        new_matrices = []
        excluded = []
        exclude_row = 0
        for exclude_column in range(dimensions[1]):
            tmp = []
            excluded.append(matrix[exclude_row][exclude_column])
            for row in range(1, dimensions[0]):
                tmp_row = []
                for column in range(dimensions[1]):
                    if (row != exclude_row) and (column != exclude_column):
                        tmp_row.append(matrix[row][column])
                tmp.append(tmp_row)
            new_matrices.append(tmp)
        determinants = [find_determinant(new_matrices[j], excluded[j]) for j in range(len(new_matrices))]
        determinant = 0
        for i in range(len(determinants)):
            determinant += ((-1)**i)*determinants[i]
        return determinant

def list_multiply(list1, list2):
    
    # Return the multiplication of two lists by treating each list as a factor.
    result = [0 for _ in range(len(list1) + len(list2) - 1)]
    for i in range(len(list1)):
        for j in range(len(list2)):
            result[i+j] += list1[i] * list2[j]
    return result

def list_add(list1, list2, sub=1):
    
    # Return the element wise addition of two lists
    return [i + (sub*j) for i, j in zip(list1, list2)]

def determinant_equation(matrix, excluded=[1, 0]):
    
    # Return the equation describing the determinant in terms of some unknown
    # variable. The index of each element in the list represents the power of the
    # unknown variable. For example, [1, 2, 3] corresponds to the equation
    # 1 + 2x + 3x^2.
    dimensions = get_dimensions(matrix)
    if dimensions == [2, 2]:
        tmp = list_add(list_multiply(matrix[0][0], matrix[1][1]), list_multiply(matrix[0][1], matrix[1][0]), sub=-1)
        return list_multiply(tmp, excluded)
    else:
        new_matrices = []
        excluded = []
        exclude_row = 0
        for exclude_column in range(dimensions[1]):
            tmp = []
            excluded.append(matrix[exclude_row][exclude_column])
            for row in range(1, dimensions[0]):
                tmp_row = []
                for column in range(dimensions[1]):
                    if (row != exclude_row) and (column != exclude_column):
                        tmp_row.append(matrix[row][column])
                tmp.append(tmp_row)
            new_matrices.append(tmp)
        determinant_equations = [determinant_equation(new_matrices[j],
                            excluded[j]) for j in range(len(new_matrices))]
        dt_equation = [sum(i) for i in zip(*determinant_equations)]
        return dt_equation

def identity_matrix(dimensions):

    # Return an identity matrix of any given dimensions.
    matrix = [[0 for j in range(dimensions[1])] for i in range(dimensions[0])]
    for i in range(dimensions[0]):
        matrix[i][i] = 1
    return matrix

def characteristic_equation(matrix):
    
    # Return the characteristic equation of a matrix.
    dimensions = get_dimensions(matrix)
    return [[[a, -b] for a, b in zip(i, j)] for i, j in zip(matrix,
            identity_matrix(dimensions))]

def find_eigenvalues(matrix):
    
    # Return the eigenvalues of a matrix.
    dt_equation = determinant_equation(characteristic_equation(matrix))
    return np.roots(dt_equation[::-1])

if __name__ == "__main__":
    A = [[2, 0, 0],
         [0, 4, 0],
         [0, 0, 2]]
    eigenvalues = find_eigenvalues(A)
    print(eigenvalues)
