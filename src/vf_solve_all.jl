export solve_vf_all!


function solve_vf_all!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θ::AbstractVector, σ::Real, itype::Tuple, dograd::Bool; kwargs...)

    if dograd
        fillflows_grad!(t, p, θ, σ, itype...)
    else
        fillflows!(t, p, θ, σ, makepdct(p, Val{:u})..., itype...)
    end
    solve_vf_terminal!(evs,    p)
    solve_vf_infill!(  evs, t, p,    dograd; kwargs...)
    learningUpdate!(   evs, t, p, σ, dograd)
    solve_vf_explore!( evs, t, p,    dograd)
end

solve_vf_all!(evs, t, p, θ, itype, dograd; kwargs...) = solve_vf_all!(evs, t, p, _θt(θ, p), _σv(θ), itype, dograd; kwargs...)
