export solve_vf_all!

function solve_vf_all!(
    EV::AbstractArray3, dEV::AbstractArray4, dEV_σ::AbstractArray4,                           # complete VF
    uin::AbstractArray4, uex::AbstractArray3,                                                 # flow payofs
    duin::AbstractArray5, duex::AbstractArray4, duexσ::AbstractArray4,                        # flow gradient
    ubVfull::AbstractArray3, dubVfull::AbstractArray4, dubV_σ::AbstractArray4,                # choice-specific VF
    q::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix,  # temp vars
    wp::well_problem, Πz::AbstractMatrix, βΠψ::AbstractMatrix, βdΠψ::AbstractMatrix, β::Real; # transitions, etc
    kwargs...
    )
    solve_vf_terminal!(EV, dEV, dEV_σ)
    solve_vf_infill!(  EV, dEV,        uin, duin,        ubVfull, dubVfull,            lse, tmp, IminusTEVp, wp, Πz,            β)
    solve_vf_explore!( EV, dEV, dEV_σ, uex, duex, duexσ, ubVfull, dubVfull, dubV_σ, q, lse, tmp,             wp, Πz, βΠψ, βdΠψ, β)
end


function solve_vf_all!(EV::AbstractArray3, uin::AbstractArray4, uex::AbstractArray3, ubVfull::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix, wp::well_problem, Πz::AbstractMatrix, βΠψ::AbstractMatrix,  β::Real; kwargs...)
    solve_vf_terminal!(EV)
    solve_vf_infill!(  EV, uin, ubVfull, lse, tmp, IminusTEVp, wp, Πz,      β)
    solve_vf_explore!( EV, uex, ubVfull, lse, tmp,             wp, Πz, βΠψ, β)
end
