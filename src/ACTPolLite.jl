module ACTPolLite

using Artifacts
using LoopVectorization
using NPZ

import Base.@kwdef

@kwdef struct WindowFunction
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

    bmax = 52 #hardcoded numbers
    lmax_win = 4999

    win_func_d_tt = win_func_d[2 * bmax + 1: 3  * bmax, 1:lmax_win]
    win_func_d_te = win_func_d[6 * bmax + 1: 7  * bmax, 1:lmax_win]
    win_func_d_ee = win_func_d[9 * bmax + 1: 10 * bmax, 1:lmax_win]
    win_func_w_tt = win_func_w[2 * bmax + 1: 3  * bmax, 1:lmax_win]
    win_func_w_te = win_func_w[6 * bmax + 1: 7  * bmax, 1:lmax_win]
    win_func_w_ee = win_func_w[9 * bmax + 1: 10 * bmax, 1:lmax_win]

    global WF =  WindowFunction(D_TT = win_func_d_tt,
                                D_TE = win_func_d_te,
                                D_EE = win_func_d_ee,
                                W_TT = win_func_w_tt,
                                W_TE = win_func_w_te,
                                W_EE = win_func_w_ee)

end

function compone_window_Cℓ(tt::AbstractArray{T}, te, ee, WF::WindowFunction) where {T}
    #bmax = 52
    #lmax_win = 4999

    cl_tt_d = Array{T}(zeros(52))
    cl_te_d = Array{T}(zeros(52))
    cl_ee_d = Array{T}(zeros(52))
    cl_tt_w = Array{T}(zeros(52))
    cl_te_w = Array{T}(zeros(52))
    cl_ee_w = Array{T}(zeros(52))

    mygemmavx!(cl_tt_d, WF.D_TT, tt)
    mygemmavx!(cl_te_d, WF.D_TE, te)
    mygemmavx!(cl_ee_d, WF.D_EE, ee)
    mygemmavx!(cl_tt_w, WF.W_TT, tt)
    mygemmavx!(cl_te_w, WF.W_TE, te)
    mygemmavx!(cl_ee_w, WF.W_EE, ee)

    b0 = 5
    nbintt = 40
    nbinte = 45
    #nbinee = 45

    X_model = vcat(cl_tt_d[b0+1 : b0 + nbintt], cl_te_d[1:nbinte], cl_ee_d[1:nbinte],
                   cl_tt_w[b0+1 : b0 + nbintt], cl_te_w[1:nbinte], cl_ee_w[1:nbinte])

    return X_model

end

function mygemmavx!(C, A, B)
    @turbo for m ∈ axes(A,1)
        Cm = zero(eltype(B))
        for k ∈ axes(A,2)
            Cm += A[m,k] * B[k]
        end
        C[m] = Cm
    end
end


end # module ACTPolLite
