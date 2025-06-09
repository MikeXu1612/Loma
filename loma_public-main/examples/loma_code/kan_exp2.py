def kan_network(X: In[Array[float, 2]], W: In[Array[float, 12]], A: In[Array[float, 30]], Y: Out[Array[float, 2]]):
    a: float
    b: float
    a = kan(X, 2, 2, [3], 6, W, A)[0]
    b = kan(X, 2, 2, [3], 6, W, A)[1]
    Y[0] = a
    Y[1] = b
