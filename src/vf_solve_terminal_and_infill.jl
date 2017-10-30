export solve_vf_terminal!, solve_vf_infill!, zero!


zero!(x::AbstractArray{T}) where {T<:Real} = fill!(x, zero(T))

function solve_vf_terminal!(EV::AbstractArray3)
    @views zero!(EV[:,:,end])
    @views zero!(EV[:,:,exploratory_terminal(wp)])
end


function solve_vf_terminal!(EV::AbstractArray3, dEV::AbstractArray4, dEVσ::AbstractArray3, wp::well_problem)
    @views zero!(EV[:,:,end])
    @views zero!(dEV[:,:,:,end])
    @views zero!(dEVσ[:,:,end])
    # @views zero!(dEV_ψ[:,:,end])
    exp_trm = exploratory_terminal(wp)
    @views zero!(EV[   :,:,  exp_trm])
    @views zero!(dEV[  :,:,:,exp_trm])
    @views zero!(dEVσ[:,:,  exp_trm])
    # @views zero!(dEV_ψ[:,:,  exp_trm])
end

solve_vf_terminal!(evs::dcdp_Emax, wp::well_problem) = solve_vf_terminal!(evs.EV, evs.dEV, evs.dEVσ, wp)
solve_vf_terminal!(evs::dcdp_Emax, prim::dcdp_primitives) = solve_vf_terminal!(evs.EV, evs.dEV, evs.dEVσ, prim.wp)

# ---------------------------------------------

function solve_vf_infill!(
    EV::AbstractArray3, dEV::AbstractArray4,                                  # value function
    uin::AbstractArray4, duin::AbstractArray5,                                # flow utilities
    ubVfull::AbstractArray3, dubVfull::AbstractArray4,                        # choice-specific VF
    lse::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix,     # temp vars
    wp::well_problem, Πz::AbstractMatrix, β::Real;                            # problem structure
    maxit0::Integer=12, maxit1::Integer=20, vftol::Real=1e-11                 # convergence options
    )

    nz,nψ,nS = size(EV)
    dmaxp1 = dmax(wp)+1

    dograd = (length(dEV) > 0)

    # ------------------------ size checks ----------------------------------

    (nz,nψ,dmaxp1,2) == size(uin)           || throw(DimensionMismatch())  # uin[z,ψ,d,d1]
    (nz,nz) == size(IminusTEVp) == size(Πz) || throw(DimensionMismatch())
    (nz,nψ) == size(tmp) == size(lse)       || throw(DimensionMismatch())
    (nz,nψ,dmaxp1) == size(ubVfull)         || throw(DimensionMismatch())

    if dograd
        nθ = size(dEV,3)
        (nz,nψ,nθ,nS)       == size(dEV)    || throw(DimensionMismatch())
        (nz,nψ,nθ,dmaxp1,2) == size(duin)   || throw(DimensionMismatch())
        (nz,nψ,nθ,dmaxp1) == size(dubVfull) || throw(DimensionMismatch())
    end

    # ------------------------ compute things ----------------------------------

    for i in ind_inf(wp)
        idxd, idxs, horzn, s = wp_info(wp, i)

        ubV  = @view(ubVfull[:,:,idxd])
        EV0  = @view(EV[:,:,i])
        @views ubV .= uin[:,:,idxd,1+s.d1] .+ β .* EV[:,:,idxs]

        if dograd
            dubV = @view(dubVfull[:,:,:,idxd])
            dEV0 = @view(dEV[:,:,:,i])
            dEV0 .= 0.0
            dubV .= @view(duin[:,:,:,idxd,1+s.d1]) .+ β .* @view(dEV[:,:,:,idxs])
        end

        if horzn == :Finite
            dograd || vfit!(EV0,       ubV,       lse, tmp, Πz)
            dograd && vfit!(EV0, dEV0, ubV, dubV, lse, tmp, Πz)

        elseif horzn == :Infinite
            solve_inf_vfit!(EV0, ubV, lse, tmp, Πz, β; maxit=maxit0, vftol=vftol)
            converged, iter, bnds = solve_inf_pfit!(EV0, ubV, lse, tmp, IminusTEVp, Πz, β; maxit=maxit1, vftol=vftol)
            # converged || warn("Did not converge at state $i after $iter pfit. McQueen-Porteus bnds: $bnds")
            if dograd
                # TODO: only allows 0-payoff if no action
                ubV[:,:,1] .= β .* EV0
                gradinf!(dEV0, ubV, dubV, lse, tmp, IminusTEVp, Πz, β)   # note: destroys ubV & dubV
            end
        else
            throw(error("horizon must be :Finite or :Infinite"))
        end
    end
end

# ---------------------------- wrappers ----------------------------------


function solve_vf_infill!(EV::AbstractArray3{T}, uin::AbstractArray4, ubVfull::AbstractArray3, lse::AbstractMatrix, tmp::AbstractMatrix, IminusTEVp::AbstractMatrix, wp::well_problem, Πz::AbstractMatrix, β::Real; kwargs...) where {T}
    zeros4 = Array{T}(0,0,0,0)
    zeros5 = Array{T}(0,0,0,0,0)
    solve_vf_infill!(EV, zeros4, uin, zeros5, ubVfull, zeros4, lse, tmp, IminusTEVp, wp, Πz, β; kwargs...)
end

solve_vf_infill!(e::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, ::Type{Val{true}} ; kwargs...) = solve_vf_infill!(e.EV, e.dEV, t.uin, t.duin, t.ubVfull, t.dubVfull, t.lse, t.tmp, t.IminusTEVp, p.wp, p.Πz, p.β; kwargs...)
solve_vf_infill!(e::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, ::Type{Val{false}}; kwargs...) = solve_vf_infill!(e.EV,        t.uin,         t.ubVfull,             t.lse, t.tmp, t.IminusTEVp, p.wp, p.Πz, p.β; kwargs...)
solve_vf_infill!(e::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, dograd::Bool=true; kwargs...) = solve_vf_infill!(e,t,p,Val{dograd};kwargs...)
