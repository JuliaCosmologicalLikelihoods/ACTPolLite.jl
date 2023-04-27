module ACTPolLite

using Artifacts
using NPZ

function __init__()

    global win_func_d = npzread(joinpath(artifact"DR4_data", "win_func_d.npy"))
    global win_func_w = npzread(joinpath(artifact"DR4_data", "win_func_w.npy"))
    global cov_ACT = npzread(joinpath(artifact"DR4_data", "cov_ACT.npy"))
    global data = npzread(joinpath(artifact"DR4_data", "data.npy"))

end

end # module ACTPolLite
