module ACTPolLite

using Artifacts
using LoopVectorization
using NPZ

import Base.@kwdef

@kwdef  struct WindowFunction
    D_TT::Matrix
    D_TE::Matrix
    D_EE::Matrix
    W_TT::Matrix
    W_TE::Matrix
    W_EE::Matrix
end

function __init__()

    global win_func_d = npzread(joinpath(artifact"DR4_data", "win_func_d.npy"))
    global win_func_w = npzread(joinpath(artifact"DR4_data", "win_func_w.npy"))
    global cov_ACT = npzread(joinpath(artifact"DR4_data", "cov_ACT.npy"))
    global data = npzread(joinpath(artifact"DR4_data", "data.npy"))

end

end # module ACTPolLite
