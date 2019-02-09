using MarkovTransitionMatrices

# export makeSparseTransition

# given volatility parameters, regimes & regime-transitions, approximate a
# transition for a bivariate (independent) random walk using Farmer & Toda (2016)
# approach to matching moments of distribution

function makeSparseTransition(nP::Integer=31, nC::Integer=37, extendSD::Real = 1.5, minP::Real = 1e-6)
  n = nP * nC

  T = Float64

  σp = [0x1.269c263357e05p-4, 0x1.9bec08eb57abfp-5]
  σc = [0x1.2c9a3a5b8b5bap-5, 0x1.09d5d411d8a33p-7]

  Πk = [0x1.d7e13e2f38db1p-1 0x1.40f60e8638b53p-4;
        0x1.098244df0fddfp-4 0x1.decfb7641df78p-1]
  all(sum(Πk,2) .≈ 1.0) || throw(error("each row of π must sum to 1"))

  K = size(Πk,2)
  extrema_p = [0.8776572, 2.485073]
  extrema_c = [0.2437302, 1.529529]

  # state-spaces
  pspace = range(extrema_p[1]-log(extendSD), stop=extrema_p[2] + log(extendSD), length=nP)
  cspace = range(extrema_c[1]-log(extendSD), stop=extrema_c[2] + log(extendSD), length=nC)
  zspace = (pspace, cspace, 1:K,)

  # make deviations
  zrandwalk(x::Real, st::Real, σ::Real) = (x - st) / σ

  # allocate giant transition matrix
  P = zeros(T,n*K,n*K)

  # match moments for logp
  Pp1, JN, Λ, L_p1, approxErr = discreteNormalApprox(pspace, pspace, (x::Real,st::Real) -> zrandwalk(x,st,σp[1]), 15)
  Pp2, JN, Λ, L_p2, approxErr = discreteNormalApprox(pspace, pspace, (x::Real,st::Real) -> zrandwalk(x,st,σp[2]), 15)
  # plot(pspace, [L_p1,L_p2], yticks = 1:15, labels=["Hi vol", "Lo vol"], xlabel="Log p", ylabel="Moments matched")

  # match moments for logc
  Pc1, JN, Λ, L_c1, approxErr = discreteNormalApprox(cspace, cspace, (x::Real,st::Real) -> zrandwalk(x,st,σc[1]), 13)
  Pc2, JN, Λ, L_c2, approxErr = discreteNormalApprox(cspace, cspace, (x::Real,st::Real) -> zrandwalk(x,st,σc[2]), 5)
  # plot(cspace, [L_c1,L_c2], yticks = 1:15, labels=["Hi vol", "Lo vol"], xlabel="Log c", ylabel="Moments matched")

  # sparse versions
  sPp1 = MarkovTransitionMatrices.sparsify!(Pp1, minp)
  sPp2 = MarkovTransitionMatrices.sparsify!(Pp2, minp)
  sPc1 = MarkovTransitionMatrices.sparsify!(Pc1, minp)
  sPc2 = MarkovTransitionMatrices.sparsify!(Pc2, minp)

  # do the big matrix
  P12 = (kron(sPc1, sPp1), kron(sPc2, sPp2))
  for r2 in 1:K
    for r1 in 1:K
      P[(r1-1)*n+1:r1*n,  (r2-1)*n+1:r2*n] .= Πk[r1, r2] .* P12[r2]
    end
  end

  sP = sparse(P)

  return (zspace, sP)
end

















#
