export solve_vf_terminal!, solve_vf_infill!, zero!


zero!(x::AbstractArray{T}) where {T<:Real} = fill!(x, zero(T))

function solve_vf_terminal!(EV::AbstractArray3)
    @views zero!(EV[:,:,end])
    @views zero!(EV[:,:,exploratory_terminal(wp)])
end


function solve_vf_terminal!(EV::AbstractArray3, dEV::AbstractArray4, dEVσ::AbstractArray3, wp::AbstractUnitProblem)
    @views zero!(EV[:,:,end])
    @views zero!(dEV[:,:,:,end])
    @views zero!(dEVσ[:,:,end])

    exp_trm = exploratory_terminal(wp)
    @views zero!(EV[  :,:,  exp_trm])
    @views zero!(dEV[ :,:,:,exp_trm])
    @views zero!(dEVσ[:,:,  exp_trm])
end

solve_vf_terminal!(evs::dcdp_Emax, wp::AbstractUnitProblem) = solve_vf_terminal!(evs.EV, evs.dEV, evs.dEVσ, wp)
solve_vf_terminal!(evs::dcdp_Emax, prim::dcdp_primitives) = solve_vf_terminal!(evs.EV, evs.dEV, evs.dEVσ, prim.wp)

# ---------------------------------------------

function solve_vf_infill!(evs::dcdp_Emax, t::dcdp_tmpvars, p::dcdp_primitives, θt::AbstractVector, σ::Real, dograd::Bool, itype::Tuple; maxit0::Integer=40, maxit1::Integer=20, vftol::Real=1e-9)

    EV         = evs.EV
    dEV        = evs.dEV
    wp         = p.wp
    Πz         = p.Πz
    β          = p.β

    nz,nψ,nS = size(EV)
    dmaxp1 = _dmax(wp)+1

    # ------------------------ size checks ----------------------------------

    (nz,nψ,dmaxp1) == size(t.ubVfull)         || throw(DimensionMismatch())
    (nz,nz) == size(t.IminusTEVp) == size(Πz) || throw(DimensionMismatch())
    (nz,nψ) == size(t.tmp) == size(t.lse)     || throw(DimensionMismatch())

    if dograd
        nθ = size(dEV,3)
        (nz,nψ,nθ,nS)     == size(dEV)        || throw(DimensionMismatch())
        (nz,nψ,nθ,dmaxp1) == size(t.dubVfull) || throw(DimensionMismatch())
    end

    # ------------------------ compute things ----------------------------------

    for i in ind_inf(wp)
        idxd, idxs, horzn = dp1space(wp,i), collect(sprimes(wp,i)), _horizon(wp,i)

        tvw = dcdp_tmpvars_view(t,idxd)
        ubV  = tvw.ubVfull
        dubV = tvw.dubVfull
        @views EV0 = EV[:,:,i]
        @views dEV0 = dEV[:,:,:,i]

        fillflows!(ubV, flow, p, θt, σ, i, itype...)
        @views ubV .+= β .* EV[:,:,idxs]

        if dograd
            fillflows_grad!(dubV, flowdθ, p, θt, σ, i, itype...)
            fill!(dEV0, 0.0)
            @views dubV .+= β .* dEV[:,:,:,idxs]
        end

        if horzn == :Finite
            if dograd
                vfit!(EV0, dEV0, tvw, p)
            else
                vfit!(EV0, tvw, p)
            end

        elseif horzn == :Infinite
            converged, iter, bnds =  solve_inf_vfit_pfit!(EV0, tvw, p; vftol=vftol, maxit0=maxit0, maxit1=maxit1)
            converged || @warn "Did not converge at state $i after $iter pfit. McQueen-Porteus bnds: $bnds. θt = $θt, σ = $σ"

            if dograd
                # TODO: only allows 0-payoff if no action
                ubV[:,:,1] .= β .* EV0
                gradinf!(dEV0, tvw, p)   # note: destroys ubV & dubV
            end
        else
            throw(error("i = $i, horzn = $horzn but must be :Finite or :Infinite"))
        end
    end
end
