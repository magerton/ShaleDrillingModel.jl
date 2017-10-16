export tauchen86_ρ!, tauchen86_c!

zθ( x2::Real, x1::Real , θ::Real         ) = (x2 - θ^2.*x1)/sqrt(1.-θ)
dzθ(x2::Real, x1::Real , θ::Real, z::Real) = -2.*θ*x1/sqrt(1.-θ) + 0.5*z/(1.-θ)
d2zθ(x2::Real, x1::Real, θ::Real, z::Real) = -2.*x1*(1.+θ/(1.-θ))/sqrt(1.-θ) + 0.75*z/(1.-θ)^2

dPdθ( x2::Real, x1::Real, θ::Real, z::Real) = normpdf(z) * dzθ(x2,x1,θ,z)
d2Pdθ(x2::Real, x1::Real, θ::Real, z::Real) = normpdf(z) * ( d2zθ(x2,x1,θ,z) - z * dzθ(x2,x1,θ,z)^2 )

ρ2_σ(σ::Real) = one(σ)/(one(σ)+σ^2)
ρ_σ(σ::Real) = one(σ)/sqrt(one(σ)+σ^2)
dρ_σ(σ::Real)  = -σ*(one(σ)+σ^2)^-1.5
d2ρ_σ(σ::Real) = -(one(σ)+σ^2.)^-1.5 + 3.*σ^2.*(one(σ)+σ^2.)^-2.5

function tauchen86_σ!(P::AbstractMatrix, dP::AbstractMatrix, d2P::AbstractMatrix, z::Vector, y::StepRangeLen, σ::Real)
    length(d2P) > 0  && length(dP) == 0  && throw(error("Must do d2P and dP!"))
    tauchen86_ρ!(P, dP, d2P, z, y, ρ_σ(σ))
    if length(d2P) > 0
        d2P .*= (dρ_σ(σ)^2)
        d2P .+= dP .* d2ρ_σ(σ)
    end
    if length(dP) > 0
        dρ = dρ_σ(σ)
        dP .*= dρ
    end
end

function tauchen86_ρ!(P::AbstractMatrix, dP::AbstractMatrix, d2P::AbstractMatrix, z::Vector, y::StepRangeLen, θ::Real)
    n = length(y)
    (n,n,)  == size(P) || throw(DimensionMismatch())
    n == length(z)   || throw(DimensionMismatch())
    -1.0 < ρ < 1.0     || throw(DomainError("need -1 < ρ < 1"))

    doP = length(P) > 0
    dodP = length(dP) > 0
    dod2P = length(d2P) > 0

    Δ = 0.5 * step(y) / sqrt(1.-θ)

    for (j,yj) in enumerate(y)
        z .= zθ.(yj, y, θ)
        if j == 1
            z .+= Δ
            doP   &&  ( P[ :,j] .= normcdf.(z))
            dodP  &&  (dP[ :,j] .= dPdθ.( yj, y, θ, z))
            dodP && dod2P &&  (d2P[:,j] .= d2Pdθ.(yj, y, θ, z))
        elseif j == n
            z .-= Δ
            doP   &&  ( P[ :,j] .= normccdf.(z))
            dodP  &&  (dP[ :,j] .= -dPdθ.( yj, y, θ, z))
            dodP && dod2P &&  (d2P[:,j] .= -d2Pdθ.(yj, y, θ, z))
        else
            doP   &&  ( P[ :,j] .= normcdf.(z.+Δ) .- normcdf.(z.-Δ))
            dodP  &&  (dP[ :,j] .= dPdθ.( yj, y, θ, z.+Δ) .- dPdθ.( yj, y, θ, z.-Δ))
            dodP && dod2P && (d2P[:,j] .= d2Pdθ.(yj, y, θ, z.+Δ) .- d2Pdθ.(yj, y, θ, z.-Δ))
        end
    end
end

for typ in (:σ, :ρ)
    fun = Symbol("tauchen86_$(typ)!")
    @eval ($fun)(P::AbstractMatrix, dP::AbstractMatrix,                      z::Vector, y::StepRangeLen{T}, θ::Real) where {T} = ($fun)(P, dP, Matrix{T}(0,0), z, y,θ)
    @eval ($fun)(P::AbstractMatrix,                                          z::Vector, y::StepRangeLen{T}, θ::Real) where {T} = ($fun)(P,     Matrix{T}(0,0), z, y,θ)
    @eval ($fun)(P::AbstractMatrix, dP::AbstractMatrix, d2P::AbstractMatrix,            y::StepRangeLen{T}, θ::Real) where {T} = ($fun)(P, dP, d2P, zeros(T,length(y)), y,θ)
    @eval ($fun)(P::AbstractMatrix, dP::AbstractMatrix,                                 y::StepRangeLen{T}, θ::Real) where {T} = ($fun)(P, dP,      zeros(T,length(y)), y,θ)
    @eval ($fun)(P::AbstractMatrix,                                                     y::StepRangeLen{T}, θ::Real) where {T} = ($fun)(P,          zeros(T,length(y)), y,θ)
end
