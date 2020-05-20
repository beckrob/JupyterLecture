import math
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

def localLinearRegression(trainingCoordinates, trainingResponses, testCoordinates, neighborNumber, trainingWeights=None):
    
    sampleNumber = testCoordinates.shape[0]
    dim=testCoordinates.shape[1]

    if trainingWeights is None:
        trainingWeights=np.ones(trainingCoordinates.shape[0])

    testResponses = np.zeros(sampleNumber)

    knnreg = KNeighborsRegressor(n_neighbors=neighborNumber, weights='uniform', algorithm='kd_tree', leaf_size=40, p=2)
    knnreg.fit(trainingCoordinates, trainingResponses)

    for sample in range(0, sampleNumber):
        # find kNN

        [dist, ni] = knnreg.kneighbors(testCoordinates[sample,:].reshape(1,-1), neighborNumber)

        X=np.concatenate((np.ones(neighborNumber).reshape(-1,1),trainingCoordinates[ni.squeeze(),:]),axis=1)
        y=trainingResponses[ni.squeeze()]
        w=np.diag(trainingWeights[ni.squeeze()])


        rhs=np.zeros(dim+1)
        
        rhs=y.dot(w.dot(X))
        #for m in range(0, dim+1):
        #    for k in range(0,neighborNumber):
        #        rhs[m]+=X[k,m]*y[k]*w[k]

        lhs=np.zeros((dim+1,dim+1))
        
        lhs=(X.T).dot(w.dot(X))
        #for m in range(0,dim+1):
        #    for n in range(0,dim+1):
        #        for k in range(0,neighborNumber):
        #            lhs[m,n]+=X[k,m]*X[k,n]*w[k]

        try:
            coefficients = np.linalg.solve(lhs, rhs)
        except:
            break

        testResponses[sample]=(np.concatenate((np.ones(1),testCoordinates[sample,:]),axis=0).reshape(1,-1)).dot(coefficients.reshape(-1,1))

    return testResponses,coefficients,ni