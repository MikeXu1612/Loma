def kan_forward(x : In[float]) -> float:
    # This function will be replaced by KAN implementation
    return x

def d_kan_forward(x : In[_dfloat]) -> _dfloat:
    # Forward differentiation of kan_forward
    return ForwardDiff("kan_forward") 