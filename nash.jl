using JuMP
using GLPK
using XLSX
using Ipopt
using D3Trees
using BenchmarkTools
import Base.Iterators: product
import Random.shuffle

struct POMG
   γ # discount factor
   ℐ # agents
   𝒮 # state space
   𝒜 # joint action space
   𝒪 # joint observation space
   T # transition function
   O # joint observation function
   R # joint reward function
end

struct SimpleGame
    γ # discount factor
    ℐ # agents
    𝒜 # joint action space
    R # joint reward function
end

struct NashEquilibrium 
end

struct ConditionalPlan
    a # action to take at root
    subplans # dictionary mapping observations to subplans
end
ConditionalPlan(a) = ConditionalPlan(a, Dict())
(π::ConditionalPlan)() = π.a
(π::ConditionalPlan)(o) = π.subplans[o]

SATED = 1
HUNGRY = 2
FEED = 1
IGNORE = 2
SING = 3
CRYING = true
QUIET = false
r_hungry = -10.0
r_feed = -5.0
r_sing = -0.5
p_become_hungry = 0.1
p_cry_when_hungry = 0.8
p_cry_when_not_hungry = 0.1
p_cry_when_hungry_in_sing = 0.9
discount_factor = 0.9
n_agents = 2
ordered_states() = [SATED, HUNGRY]
ordered_actions(i::Int) = [FEED, IGNORE, SING]
ordered_observations(i::Int) = [CRYING, QUIET]

function transition(s, a, s1)
   # Regardless, feeding makes the baby sated.
   if a[1] == FEED || a[2] == FEED
       if s1 == SATED
           return 1.0
       else
           return 0.0
       end
   else
       # If neither caretaker fed, then one of two things happens.
       # First, a baby that is hungry remains hungry.
       if s == HUNGRY
           if s1 == HUNGRY
               return 1.0
           else
               return 0.0
           end
       # Otherwise, it becomes hungry with a fixed probability.
       else
           probBecomeHungry = 0.5 #pomg.babyPOMDP.p_become_hungry
           if s1 == SATED
               return 1.0 - probBecomeHungry
           else
               return probBecomeHungry
           end
       end
   end
end

function joint_observation(a, s1, o)
   # If at least one caregiver sings, then both observe the result.
   if a[1] == SING || a[2] == SING
       # If the baby is hungry, then the caregivers both observe crying/silent together.
       if s1 == HUNGRY
           if o[1] == CRYING && o[2] == CRYING
               return p_cry_when_hungry_in_sing
           elseif o[1] == QUIET && o[2] == QUIET
               return 1.0 - p_cry_when_hungry_in_sing
           else
               return 0.0
           end
       # Otherwise the baby is sated, and the baby is silent.
       else
           if o[1] == QUIET && o[2] == QUIET
               return 1.0
           else
               return 0.0
           end
       end
   # Otherwise, the caregivers fed and/or ignored the baby.
   else
       # If the baby is hungry, then there's a probability it cries.
       if s1 == HUNGRY
           if o[1] == CRYING && o[2] == CRYING
               return p_cry_when_hungry
           elseif o[1] == QUIET && o[2] == QUIET
               return 1.0 - p_cry_when_hungry
           else
               return 0.0
           end
       # Similarly when it is sated.
       else
           if o[1] == CRYING && o[2] == CRYING
               return p_cry_when_not_hungry
           elseif o[1] == QUIET && o[2] == QUIET
               return 1.0 - p_cry_when_not_hungry
           else
               return 0.0
           end
       end
   end
end

function joint_reward(s, a)
   r = [0.0, 0.0]

   # Both caregivers do not want the child to be hungry.
   if s == HUNGRY
       r += [r_hungry, r_hungry]
   end

   # One caregiver prefers to feed.
   if a[1] == FEED
       r[1] += r_feed / 2.0
   elseif a[1] == SING
       r[1] += r_sing
   end

   # One caregiver prefers to sing.
   if a[2] == FEED
       r[2] += r_feed
   elseif a[2] == SING
       r[2] += r_sing / 2.0
   end

   # Note that caregivers only experience a cost if they do something.

   return r
end

function POMG_Init()
   return POMG(
       discount_factor,
       vec(collect(1:n_agents)),
       ordered_states(),
       [ordered_actions(i) for i in 1:n_agents],
       [ordered_observations(i) for i in 1:n_agents],
       (s, a, s1) -> transition(s, a, s1),
       (a, s1, o) -> joint_observation(a, s1, o),
       (s, a) -> joint_reward(s, a)
   )
end

# Policy evaluation  / Evaluation plan 

function lookahead(𝒫::POMG, U, s, a)
   𝒮, 𝒪, T, O, R, γ = 𝒫.𝒮, joint(𝒫.𝒪), 𝒫.T, 𝒫.O, 𝒫.R, 𝒫.γ
   u′ = sum(T(s,a,s′)*sum(O(a,s′,o)*U(o,s′) for o in 𝒪) for s′ in 𝒮)
   return R(s,a) + γ*u′
end

# Pi is the vector of action

function evaluate_plan(𝒫::POMG, π, s)
   a = Tuple(πi() for πi in π)
   U(o,s′) = evaluate_plan(𝒫, [πi(oi) for (πi, oi) in zip(π,o)], s′)
   return isempty(first(π).subplans) ? 𝒫.R(s,a) : lookahead(𝒫, U, s, a)
end

function utility(𝒫::POMG, b, π)
   u = [evaluate_plan(𝒫, π, s) for s in 𝒫.𝒮]
   return sum(bs * us for (bs, us) in zip(b, u))
end

function tensorform(𝒫::SimpleGame)
    ℐ, 𝒜, R = 𝒫.ℐ, 𝒫.𝒜, 𝒫.R
    ℐ′ = eachindex(ℐ)
    𝒜′ = [eachindex(𝒜[i]) for i in ℐ]
    R′ = [R(a) for a in joint(𝒜)]
    return ℐ′, 𝒜′, R′
end

struct SimpleGamePolicy
    p # dictionary mapping actions to probabilities
    function SimpleGamePolicy(p::Base.Generator)
        return SimpleGamePolicy(Dict(p))
    end

    function SimpleGamePolicy(p::Dict)
        vs = collect(values(p))
        vs ./= sum(vs)
        return new(Dict(k => v for (k,v) in zip(keys(p), vs)))
    end
    SimpleGamePolicy(ai) = new(Dict(ai => 1.0))
end
(πi::SimpleGamePolicy)(ai) = get(πi.p, ai, 0.0)
function (πi::SimpleGamePolicy)()
    D = SetCategorical(collect(keys(πi.p)), collect(values(πi.p)))
    return rand(D)
end
joint(X) = vec(collect(product(X...)))
joint(π, πi, i) = [i == j ? πi : πj for (j, πj) in enumerate(π)]
function utility(𝒫::SimpleGame, π, i)
    𝒜, R = 𝒫.𝒜, 𝒫.R
    p(a) = prod(πj(aj) for (πj, aj) in zip(π, a))
    return sum(R(a)[i]*p(a) for a in joint(𝒜))
end

function solve(M::NashEquilibrium, 𝒫::SimpleGame)
    ℐ, 𝒜, R = tensorform(𝒫)
    model = Model(Ipopt.Optimizer)
    @variable(model, U[ℐ])
    @variable(model, π[i=ℐ, 𝒜[i]] ≥ 0)
    @NLobjective(model, Min,
        sum(U[i] - sum(prod(π[j,a[j]] for j in ℐ) * R[y][i]
        for (y,a) in enumerate(joint(𝒜))) for i in ℐ))
    @NLconstraint(model, [i=ℐ, ai=𝒜[i]],U[i] ≥ sum(
        prod(j==i ? (a[j]==ai ? 1.0 : 0.0) : π[j,a[j]] for j in ℐ)
        * R[y][i] for (y,a) in enumerate(joint(𝒜))))
    @constraint(model, [i=ℐ], sum(π[i,ai] for ai in 𝒜[i]) == 1)
    optimize!(model)
    πi′(i) = SimpleGamePolicy(𝒫.𝒜[i][ai] => value(π[i,ai]) for ai in 𝒜[i])
    return [πi′(i) for i in ℐ]
end

# Nash equilibrium 

struct POMGNashEquilibrium
   b # initial belief
   d # depth of conditional plans
end

depthOfPlan = 4
beliefInit = [0.5, 0.5]
function POMGNashEquilibrium_Init()
    return POMGNashEquilibrium(
        beliefInit,
        depthOfPlan
    )
end

function create_conditional_plans(𝒫, d)
   ℐ, 𝒜, 𝒪 = 𝒫.ℐ, 𝒫.𝒜, 𝒫.𝒪
   Π = [[ConditionalPlan(ai) for ai in 𝒜[i]] for i in ℐ]
   for t in 1:d
      Π = expand_conditional_plans(𝒫, Π)
   end
   return Π
end

function expand_conditional_plans(𝒫, Π)
   ℐ, 𝒜, 𝒪 = 𝒫.ℐ, 𝒫.𝒜, 𝒫.𝒪
   return [[ConditionalPlan(ai, Dict(oi => πi for oi in 𝒪[i])) for πi in Π[i] for ai in 𝒜[i]] for i in ℐ]
end

function solve(M::POMGNashEquilibrium, 𝒫::POMG)
   ℐ, γ, b, d = 𝒫.ℐ, 𝒫.γ, M.b, M.d
   Π = create_conditional_plans(𝒫, d)
   U = Dict(π => utility(𝒫, b, π) for π in joint(Π))
   𝒢 = SimpleGame(γ, ℐ, Π, π -> U[π])
   π = solve(NashEquilibrium(), 𝒢)
   return Tuple(argmax(πi.p) for πi in π)
end

myPOMG = POMG_Init()
myNash = POMGNashEquilibrium_Init()
@time solve(myNash, myPOMG)
# for (idx, i) in enumerate(result)
#     # println(i)
#     println("Agent ", idx)
#     println(i)
# end

n_loop = 400
function finalResultSave()
    finalRes = []
    finalCount = []
    for i in 1:n_loop
        myPOMG = POMG_Init()
        myNash = POMGNashEquilibrium_Init()
        result = solve(myNash, myPOMG)
        strAgent = [string(result[1]), string(result[2])]
        concatAgent = strAgent[1]*strAgent[2]
        if concatAgent in finalRes
            for (j, resi) in enumerate(finalCount)
                if resi[1][1]*resi[1][2]==concatAgent
                    finalCount[j][2]  = finalCount[j][2] + 1
                end
            end
        else
            push!(finalCount, [strAgent, 1])
            push!(finalRes, concatAgent)
        end
    end

    XLSX.openxlsx("Result3DepthNash.xlsx", mode="rw") do xf
        sheet = xf[1]
        for (idx, res) in  enumerate(finalCount)
            idxStr = string(idx + 1)
            sheet["A"*idxStr] = idx #row number = B2
            sheet["B"*idxStr] = string(res[1][1])
            sheet["C"*idxStr] = string(res[1][2])
            sheet["D"*idxStr] = res[2]
        end
    end
    #saveResult(finalCount, 'res.xlsx')
end
#finalResultSave()

# Tao 1 ham xuly kq vong for de println
# Print kq vao file de visualize

number = Vector([1])
children = Vector{Vector}()
text = Vector{String}()

function create_tree(π::ConditionalPlan, number::Array, children::Array, text::Array, observation::Bool)
    depth = number[1]
    o = observation == QUIET ? "QUIET" : "CRYING"
    push!(children, [])
    push!(text, "a"*string(number[1])*"\n"*o)
    if (length(π.subplans) == 0)
        return
    end
    
    number[1] += 1
    push!(children[depth], number[1])
    create_tree(π.subplans[QUIET], number, children, text, QUIET)
    number[1] += 1
    push!(children[depth], number[1])
    create_tree(π.subplans[CRYING], number, children, text, CRYING)
end

# create_tree(result[1], number, children, text, QUIET)
# t = D3Tree(children, text=text)