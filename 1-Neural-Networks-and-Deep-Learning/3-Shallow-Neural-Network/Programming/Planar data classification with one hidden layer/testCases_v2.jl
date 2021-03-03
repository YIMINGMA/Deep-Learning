module testCases_v2

export layer_sizes_test_case
export initialize_parameters_test_case
export forward_propagation_test_case
export compute_cost_test_case
export backward_propagation_test_case
export update_parameters_test_case
export nn_model_test_case
export predict_test_case
using Random


function layer_sizes_test_case()
    rng = MersenneTwister(1)
    X_assess = randn(rng, (5, 3))
    Y_assess = randn(rng, (2, 3))
    return X_assess, Y_assess
end  # module


function initialize_parameters_test_case()
    n_x, n_h, n_y = 2, 4, 1
    return n_x, n_h, n_y
end


function forward_propagation_test_case()
    rng = MersenneTwister(1)
    X_assess = randn(rng, (2, 3))
    b1 = randn(rng, (4, 1))
    b2 = reshape([-1.3], (1, 1))

    parameters = Dict(
        "W1" => [0.007396206598864331 -0.0067561560740237135;
        -0.007445071021408705 0.005566463333701307;
        -0.006085075626113596 -0.008615844731669183;
        -0.017234565107957983 0.005180851077313978],
        "W2" => [0.012482298080972934 0.011691129549524617 -0.0009301787986362445 -0.012924032572712177],
        "b1" => b1,
        "b2" => b2
    )

    return X_assess, parameters
end


function compute_cost_test_case()
    rng = MersenneTwister(1)
    Y_assess = randn(rng, (1, 3)) .> 0
    parameters = Dict(
        "W1" => [0.007396206598864331 -0.0067561560740237135;
        -0.007445071021408705 0.005566463333701307;
        -0.006085075626113596 -0.008615844731669183;
        -0.017234565107957983 0.005180851077313978],
        "W2" => reshape([0.012482298080972934 0.011691129549524617 -0.0009301787986362445 -0.012924032572712177], (1, 4)),
        "b1" => reshape([0. 0. 0. 0.], (4, 1)),
        "b2" => reshape([0.], (1, 1))
    )
    a2 = reshape([0.5002307, 0.49985831, 0.50023963], (1, 3))

    return a2, Y_assess, parameters
end


function backward_propagation_test_case()
    rng = MersenneTwister(1)
    X_assess = randn(rng, (2, 3))
    Y_assess = Float64.(randn(rng, (1, 3)) .> 0)
    parameters = Dict(
        "W1" => [
            -0.00416758 -0.00056267;
            -0.02136196 0.01640271;
            -0.01793436 -0.00841747;
            0.00502881 -0.01245288
        ],
        "W2" => reshape([-0.01057952, -0.00909008,  0.00551454,  0.02292208], (1, 4)),
        "b1" => reshape([0. 0. 0. 0.], (4, 1)),
        "b2" => reshape([0.], (1, 1))
    )

    cache = Dict(
        "A1" => [
            -0.00616578 0.0020626 0.00349619;
            -0.05225116 0.02725659 -0.02646251;
            -0.02009721 0.0036869 0.02883756;
            0.02152675 -0.01385234 0.02599885
        ],
        "A2" => reshape([0.5002307, 0.49985831, 0.50023963], (1, 3)),
        "Z1" => [
            -0.00616586 0.0020626 0.0034962;
            -0.05229879 0.02726335 -0.02646869;
            -0.02009991 0.00368692 0.02884556;
            0.02153007 -0.01385322 0.02600471
        ],
        "Z2" => reshape([0.00092281, -0.00056678, 0.00095853], (1, 3))
    )

    return parameters, cache, X_assess, Y_assess
end


function update_parameters_test_case()
    parameters = Dict(
        "W1" => [
            -0.00615039 0.0169021;
            -0.02311792 0.03137121;
            -0.0169217 -0.01752545;
            0.00935436 -0.05018221
        ],
        "W2" => reshape([-0.0104319, -0.04019007, 0.01607211, 0.04440255], (1, 4)),
        "b1" => reshape([-8.97523455e-07 8.15562092e-06 6.04810633e-07 -2.54560700e-06], (4, 1)),
        "b2" => reshape([9.14954378e-05], (1, 1))
    )
    grads = Dict(
        "dW1" => [
            0.00023322 -0.00205423;
            0.00082222 -0.00700776;
            -0.00031831 0.0028636;
            -0.00092857 0.00809933
        ],
        "dW2" => reshape([-1.75740039e-05 3.70231337e-03 -1.25683095e-03 -2.55715317e-03], (1, 4)),
        "db1" => reshape([1.05570087e-07 -3.81814487e-06 -1.90155145e-07 5.46467802e-07], (4, 1)),
        "db2" => reshape([-1.08923140e-05], (1, 1))
    )
    return parameters, grads
end


function nn_model_test_case()
    rng = MersenneTwister(1)
    X_assess = randn(rng, (2, 3))
    Y_assess = (randn(rng, (1, 3)) .> 0)
    return X_assess, Y_assess
end


function predict_test_case()
    rng = MersenneTwister(1)
    X_assess = randn(rng, (2, 3))
    parameters = Dict(
        "W1" => [
            -0.00615039 0.0169021;
            -0.02311792 0.03137121;
            -0.0169217 -0.01752545;
            0.00935436 -0.05018221
        ],
        "W2" => reshape([-0.0104319 -0.04019007 0.01607211 0.04440255], (1, 4)),
        "b1" => reshape([-8.97523455e-07 8.15562092e-06 6.04810633e-07 -2.54560700e-06], (4, 1)),
        "b2" => reshape([9.14954378e-05])
    )
    return parameters, X_assess
end


end
