export solve_vf_all!

function solve_vf_all!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θt::AbstractVector, σ::Real, itype::Tuple, dograd::Bool; kwargs...)

    solve_vf_terminal!(evs,    p)
    solve_vf_infill!(  evs, t, p, θt, σ, dograd, itype; kwargs...)
    learningUpdate!(   evs, t, p,     σ, dograd)
    solve_vf_explore!( evs, t, p, θt, σ, dograd, itype)
end

function solve_vf_all!(evs, t, p, θ, itype, dograd; kwargs...)
    solve_vf_all!(evs, t, p, _θt(θ, p), _σv(θ), itype, dograd; kwargs...)
end
