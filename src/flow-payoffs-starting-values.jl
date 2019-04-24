export starting_values

starting_values(x::AbstractPayoffFunction) = rand(length(x))
starting_values(x::StaticDrillingPayoff) = rand(length(x)+1)

starting_values(x::StaticDrillingPayoff{DrillingRevenue{Unconstrained,NoTrend}, DrillingCost_constant, ExtensionCost_Constant}) = [-0x1.5abc9b2ad486fp+1, STARTING_log_ogip, STARTING_α_ψ, -0x1.efb4349a3e38cp+2,                        -0x1.6ce81fc7e1acep-3, 1.0, ]
starting_values(x::StaticDrillingPayoff{DrillingRevenue{Unconstrained,NoTrend}, DrillingCost_dgt1,     ExtensionCost_Constant}) = [-0x1.30492070192dp+2,  STARTING_log_ogip, STARTING_α_ψ, -0x1.a8ca50a153ecbp+2, -0x1.45888b1d8fc67p+2, -0x1.c68737f1a1d98p-3, 1.0, ]

function starting_values(x::DrillingCost_TimeFE)
    if startstop(x) == (2008,2012,)
        timefe = [-13.1534, -9.18679, -7.90885, -7.16243, -6.77896,]
    else
        throw(error())
    end
    return vcat(timefe, 0x1.6a9e417fe7358p+0,)  # timeFE, multi-well discount
end

function starting_values(x::DrillingCost_TimeFE_rigrate)
    if startstop(x) == (2008,2012,)
        timefe = [-11.5378, -7.84324, -6.52969, -5.65049, -5.40226,]
    else
        throw(error())
    end
    return vcat(timefe, 1.47941, -0.811621,)  # timeFE, multi-well discount, rig-rate
end


function starting_values(x::StaticDrillingPayoff{<:DrillingRevenue{Unconstrained,NoTrend}, <:AbstractDrillingCost_TimeFE,  ExtensionCost_Constant})
    return vcat(-0x1.a1c8146018faap+1, STARTING_log_ogip, STARTING_α_ψ, starting_values(x.drillingcost), -0.9088156073909275, 1.037876396029411)
end

function starting_values(x::StaticDrillingPayoff{<:DrillingRevenue{Constrained,NoTrend}})
    thet = starting_values(UnconstrainedProblem(x))
    deleteat!(thet, constrained_parms(x))
    return thet
end


function starting_values(x::StaticDrillingPayoff{DrillingRevenue{Unconstrained,NoTrend,WithTaxes}, DrillingCost_TimeFE, ExtensionCost_Constant})

    revenue = [-0x1.26a3afb34b93dp+1, 0x1.401755c339009p-1, 0x1.7587cc6793516p-2, ]
    if startstop(x.drillingcost) == (2008,2012)
        drillcost = [-0x1.9183a0a1751ap+3, -0x1.16eb650cefff5p+3, -0x1.d6e2d10036d19p+2, -0x1.a78c5f0290dd9p+2, -0x1.919a8fa25c0e9p+2, 0x1.7acb0d22485f5p+0, ]
    else
        drillcost = starting_values(x.drillingcost)
    end
    extension, σ = -0x1.a56fc68c70ad7p-1, 0x1.0ee29553db387p+0
    return vcat(revenue, drillcost, extension, σ)
end


function starting_values(x::StaticDrillingPayoff{DrillingRevenue{Unconstrained,NoTrend,NoTaxes}, DrillingCost_TimeFE, ExtensionCost_Constant})

    revenue = [-0x1.52fec10df2f1ap+1, 0x1.2f754310c607ep-1, 0x1.666490fb4962p-2, ]
    if startstop(x.drillingcost) == (2008,2012)
        drillcost = [-0x1.a4e811fc110ccp+3, -0x1.29037a50f0256p+3, -0x1.fa94c6f064af4p+2, -0x1.cbf0ac440c092p+2, -0x1.b5029f65b30a1p+2, 0x1.74d357af0b952p+0,]
    else
        drillcost = starting_values(x.drillingcost)
    end
    extension, σ = -0x1.b8b6f1aad23abp-1, 0x1.fa9f78e45165fp-1

    return vcat(revenue, drillcost, extension, σ)
end
