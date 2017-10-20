export solve_vf_all!


function solve_vf_all!(e::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θ::AbstractVector, σ::Real, roy::AbstractFloat, dograd::Type{Val{false}}; kwargs...)
    fillflows!(t, p, θ, σ, roy)
    solve_vf_terminal!(e)
    solve_vf_infill!(e, t, p, dograd, kwargs...)
    learningUpdate!(t, e, p, σ, dograd)
    solve_vf_explore!(e, t, p, dograd)
end


function solve_vf_all!(e::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θ::AbstractVector, σ::Real, roy::AbstractFloat, dograd::Type{Val{true}}; kwargs...)
    fillflows_grad!(t, p, θ, σ, roy)
    solve_vf_terminal!(e)
    solve_vf_infill!(e, t, p, dograd, kwargs...)
    learningUpdate!(t, e, p, σ, dograd)
    solve_vf_explore!(e, t, p, dograd)
end


function solve_vf_all!(e::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θ::AbstractVector, σ::T, roy::AbstractFloat, v::Real=0.0, h::T=zero(T); kwargs...) where {T}
    dograd = Val{false}
    fillflows!(t, p, θ, σ, roy, v, h)
    solve_vf_terminal!(e)
    solve_vf_infill!(e, t, p, dograd, kwargs...)
    learningUpdate!(t, e, p, σ, v, h)
    solve_vf_explore!(e, t, p, dograd)
end

solve_vf_all!(e::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θ::AbstractVector,   σ::Real, roy::AbstractFloat, dograd::Bool ; kwargs...) = solve_vf_all!(e, t, p, θ, σ, roy, Val{dograd}; kwargs...)
solve_vf_all!(e::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θ::AbstractVector, roy::Real, geoid::Integer, dograd::Type     ; kwargs...) = solve_vf_all!(e, t, p, _θt(θ, geoid), _σv(θ), roy, dograd; kwargs...)
solve_vf_all!(e::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θ::AbstractVector, roy::Real, geoid::Integer, dograd::Bool=true; kwargs...) = solve_vf_all!(e, t, p, _θt(θ, geoid), _σv(θ), roy, Val{dograd}; kwargs...)


@GenGlobal g_dcdp_primitives g_dcdp_Emax g_dcdp_tmpvars

function solve_vf_all!(θ::AbstractVector, roy::Real, geoid::Integer, dograd::Type     ; kwargs...)
    global g_dcdp_Emax
    global g_dcdp_tmpvars
    global g_dcdp_primitives
    solve_vf_all!(g_dcdp_Emax, g_dcdp_tmpvars, g_dcdp_primitives, θ, roy, geoid, dograd; kwargs...)
end

# function solve_vf_all!(
#     EV::AbstractArray3, dEV::AbstractArray4, dEV_σ::AbstractArray4,                           # complete VF
#     uin::AbstractArray4, uex::AbstractArray3,                                                 # flow payofs
#     duin::AbstractArray5, duex::AbstractArray4, duexσ::AbstractArray4,                        # flow gradient
#     ubVfull::AbstractArray3, dubVfull::AbstractArray4, dubV_σ::AbstractArray4,                # choice-specific VF
#     q::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix,  # temp vars
#     wp::well_problem, Πz::AbstractMatrix, βΠψ::AbstractMatrix, βdΠψ::AbstractMatrix, β::Real; # transitions, etc
#     kwargs...
#     )
#
#
#     solve_vf_terminal!(EV, dEV, dEV_σ)
#     solve_vf_infill!(  EV, dEV,        uin, duin,        ubVfull, dubVfull,            lse, tmp, IminusTEVp, wp, Πz,            β)
#     solve_vf_explore!( EV, dEV, dEV_σ, uex, duex, duexσ, ubVfull, dubVfull, dubV_σ, q, lse, tmp,             wp, Πz, βΠψ, βdΠψ, β)
# end
#
#
# function solve_vf_all!(EV::AbstractArray3, uin::AbstractArray4, uex::AbstractArray3, ubVfull::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix, wp::well_problem, Πz::AbstractMatrix, βΠψ::AbstractMatrix,  β::Real; kwargs...)
#
#     fillflows!(tmpv, p, θt, σ, roy, v, -h)
#
#     solve_vf_terminal!(EV)
#     solve_vf_infill!(  EV, uin, ubVfull, lse, tmp, IminusTEVp, wp, Πz,      β)
#     solve_vf_explore!( EV, uex, ubVfull, lse, tmp,             wp, Πz, βΠψ, β)
# end
