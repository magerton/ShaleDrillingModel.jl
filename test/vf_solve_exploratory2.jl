T = Float64
EV1 = zeros(T, size(evs.EV))
EV2 = zeros(T, size(evs.EV))
t = tmpv
p = prim
nsexp1 = ShaleDrillingModel._nSexp(wp)
σ = σv
roy = 0.2
geoid = 2
itype = (geoid, roy,)

println("testing dEV/dσ")

solve_vf_terminal!(evs, prim)
solve_vf_infill!( evs, tmpv, prim, θt, σv, false, itype)
learningUpdate!(  evs, tmpv, prim,     σv, false)

EV       = evs.EV
dEV      = evs.dEV
dEVσ     = evs.dEVσ
ubVfull  = t.ubVfull
dubVfull = t.dubVfull
dubV_σ   = t.dubV_σ
q        = t.q
lse      = t.lse
tmp      = t.tmp
wp       = p.wp
Πz       = p.Πz
β        = p.β

nz,nψ,nS = size(EV)
nSexp, dmaxp1, nd = ShaleDrillingModel._nSexp(wp), ShaleDrillingModel._dmax(wp)+1, ShaleDrillingModel._dmax(wp)+1

(nz,nψ,nd) == size(ubVfull)       || throw(DimensionMismatch())
(nz,nz) == size(Πz)               || throw(DimensionMismatch())
(nz,nψ) == size(lse) == size(tmp) || throw(DimensionMismatch())


# Views of ubV so we can efficiently access them
@views ubV0    =  ubVfull[:,:,  1]
@views ubV1    =  ubVfull[:,:,  2:dmaxp1]
@views dubV0   = dubVfull[:,:,:,1]
@views dubV1   = dubVfull[:,:,:,2:dmaxp1]
@views dubV_σ0 = dubV_σ[  :,:,  1]
@views dubV_σ1 = dubV_σ[  :,:,  2:dmaxp1]

exp2lrn = ShaleDrillingModel.exploratory_learning(wp)
@views βEV1   =  EV[ :,:,  exp2lrn]
@views βdEV1  = dEV[ :,:,:,exp2lrn]
@views βdEVσ1 = dEVσ[:,:,  exp2lrn]

i = 1
ip = sprime(wp,i,0)

@views EV0 = EV[:,:,ip]

# compute u + βEV(d) ∀ d ∈ actionspace(wp,i)
fillflows!(ubVfull, flow, p, θt, σ, i, itype...)

# if dograd
@views dEV0 = dEV[:,:,:,ip]
@views dEVσ1 = dEVσ[ :,:,ip]
fillflows_grad!(dubVfull, flowdθ, p, θt, σ, i, itype...)
fillflows!(       dubV_σ, flowdσ, p, θt, σ, i, itype...)
dubV0   .+= β .* dEV0
dubV1   .+= βdEV1  # β already baked in
dubV_σ0 .+= β .* dEVσ1
dubV_σ1 .+= βdEVσ1 # β already baked in

# this does EV0 & ∇EV0
@views ShaleDrillingModel.vfit!(EV[:,:,i], dEV[:,:,:,i], ubVfull, dubVfull, q, lse, tmp, Πz)

# ∂EV/∂σ = I ⊗ Πz * ∑( Pr(d) * ∂ubV/∂σ[zspace, ψspace, d]  )
ShaleDrillingModel.sumprod!(tmp, dubV_σ, q)
using AxisAlgorithms
@views A_mul_B_md!(dEVσ[:,:,i], Πz, tmp, 1)

dEVσ[:,:,1] .- tmp
# else
#     @views vfit!(EV[:,:,i], ubVfull, lse, tmp, Πz)
# end
