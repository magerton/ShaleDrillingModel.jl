export solve_vf_all!


function solve_vf_all!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θ::AbstractVector, σ::Real, itype::Tuple, dograd::Type{Val{false}}; kwargs...)
    fillflows!(t, p, θ, σ, itype...)
    solve_vf_terminal!(evs,    p)
    solve_vf_infill!(  evs, t, p,    dograd; kwargs...)
    learningUpdate!(   evs, t, p, σ, dograd)
    solve_vf_explore!( evs, t, p,    dograd)
end


function solve_vf_all!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θ::AbstractVector, σ::Real, itype::Tuple, dograd::Type{Val{true}}; kwargs...)
    fillflows_grad!(t, p, θ, σ, itype...)
    solve_vf_terminal!(evs,    p)
    solve_vf_infill!(  evs, t, p,    dograd; kwargs...)
    learningUpdate!(   evs, t, p, σ, dograd)
    solve_vf_explore!( evs, t, p,    dograd)
end


solve_vf_all!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θ::AbstractVector,   σ::Real, itype::Tuple, dograd::Bool     ; kwargs...) = solve_vf_all!(evs, t, p,     θ,                 σ,    itype, Val{dograd}; kwargs...)
solve_vf_all!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θ::AbstractVector,            itype::Tuple, dograd::Type     ; kwargs...) = solve_vf_all!(evs, t, p, _θt(θ, p), _σv(θ), itype,      dograd; kwargs...)
solve_vf_all!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θ::AbstractVector,            itype::Tuple, dograd::Bool=true; kwargs...) = solve_vf_all!(evs, t, p, _θt(θ, p), _σv(θ), itype, Val{dograd}; kwargs...)
