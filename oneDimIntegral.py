import numpy
import scipy.stats

# checked
def sigmoidFunction(x):
    return 1.0 / (1.0 + numpy.exp(-x))


# checked
def getLinearApproxGradientAndOffset(g, a, b, eta):
    assert(numpy.abs(a) < float("inf") and numpy.abs(b) < float("inf"))
    assert(a <= b)
    allGradients = numpy.zeros(eta)
    allOffsets = numpy.zeros(eta)
    all_b = numpy.zeros(eta)
    stepSize = (float(b - a) / float(eta))
    assert(stepSize > 0.0)
    for t in range(eta):
        current_b = a + stepSize * (t + 1) 
        previous_b = a + stepSize * t
        allGradients[t] = (g(current_b) - g(previous_b)) / (current_b - previous_b) 
        allOffsets[t] = g(previous_b) - allGradients[t] * previous_b
        all_b[t] = previous_b
        
    return allGradients, allOffsets, all_b, stepSize


# checked
def getNormalCDFpart(mu, sigma, lowerBound, upperBound):
    part = scipy.stats.norm.cdf(x = upperBound, loc = mu, scale = sigma) - scipy.stats.norm.cdf(x = lowerBound, loc = mu, scale = sigma)
    if numpy.isnan(part) or part < 0.0:
        print("ERROR HERE")
        print("part = ", part)
        print("mu = ", mu)
        print("sigma = ", sigma)
        print("lowerBound = ", lowerBound)
        print("upperBound = ", upperBound)
    assert(part >= 0.0)
    return part

# checked
def getIntegralPartLinearApprox(mu, sigma, m, v, previous_b, stepSize):
    sigmaSquare = sigma ** 2
    l = previous_b
    o = previous_b + stepSize
    intFirstTerm = (sigma / numpy.sqrt(2.0 * numpy.pi)) * (numpy.exp((-1.0 / (2.0 * sigmaSquare)) * (l - mu) ** 2) - numpy.exp((-1.0 / (2.0 * sigmaSquare)) * (o - mu) ** 2))
    return m * intFirstTerm + (v + m * mu) * getNormalCDFpart(mu, sigma, l, o)

    
# checked
def getIntegralEstimate(mu, sigma, leftBound, rightBound):
    assert(leftBound < rightBound)
    
    ETA = 100
    BOUND_COORDINATE = 10.0
    
    if leftBound < -BOUND_COORDINATE and rightBound < -BOUND_COORDINATE:
        # for all values smaller than -10.0, we assume that the integrand is 0.0
        return 0.0
    elif leftBound > BOUND_COORDINATE and rightBound > BOUND_COORDINATE: 
        # for all values larger than 10.0, we assume that the sigmoid function is constant 1.0
        return getNormalCDFpart(mu, sigma, leftBound, rightBound)
    else:
        if leftBound < -BOUND_COORDINATE:
            # for all values smaller than -10.0, we assume that the integrand is 0.0
            leftBound = -BOUND_COORDINATE
        
        if rightBound > BOUND_COORDINATE:
            # for all values larger than 10.0, we assume that the sigmoid function is constant 1.0
            integralEstimate = getNormalCDFpart(mu, sigma, BOUND_COORDINATE, rightBound)
            rightBound = BOUND_COORDINATE 
        else:
            integralEstimate = 0.0
            
        assert(leftBound < rightBound)
        
        
        allGradients, allOffsets, all_b, stepSize = getLinearApproxGradientAndOffset(sigmoidFunction, leftBound, rightBound, ETA)
        assert((all_b[0] == leftBound) and numpy.abs(all_b[all_b.shape[0] - 1] + stepSize - rightBound) < 0.00000001)
        
        for t in range(allGradients.shape[0]):
            integralEstimate += getIntegralPartLinearApprox(mu, sigma, allGradients[t], allOffsets[t], all_b[t], stepSize)
            
        return integralEstimate


