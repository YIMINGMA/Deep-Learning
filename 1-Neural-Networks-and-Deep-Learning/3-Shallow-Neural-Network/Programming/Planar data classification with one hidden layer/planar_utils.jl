module planar_utils

export plot_decision_boundary
export sigmoid

using PyPlot
using ScikitLearn
@sk_import linear_model: LogisticRegression


function plot_decision_boundary(model, X, y)
    x_min, x_max = minimum(X[1, :]) - 1, maximum(X[1, :]) + 1
    y_min, y_max = minimum(X[2, :]) - 1, maximum(X[1, :]) + 1
    h = 0.01
    xx = x_min: h: x_max
    yy = y_min: h: y_max
    xx, yy = xx' .* ones(length(yy)), ones(length(xx))' .* yy
    Z = model(hcat(vec(xx), vec(yy)))
    Z = reshape(Z, size(xx))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.scatter(X[1, :], X[2, :], c=y, cmap=plt.cm.Spectral)
end


"""
Compute the sigmoid of x

Arguments:
x -- A scalar or array of any size.

Return:
s -- sigmoid(x)
"""
function sigmoid(x)
    s = 1 ./ (1 .+ exp.(-x))
    return s
end


end
