struct POMDP
   γ # discount factor
   𝒮 # state space
   𝒜 # action space
   𝒪 # observation space
   T # transition function
   R # reward function
   O # observation function
   TRO # sample transition, reward, and observation
end

# Algorithm 19.2. A method that
# updates a discrete belief based on
# equation (19.7), where b is a vector and 𝒫 is the POMDP model. If
# the given observation has a zero
# likelihood, a uniform distribution
# is returned.
function update(b::Vector{Float64}, 𝒫, a, o)
   𝒮, T, O = 𝒫.𝒮, 𝒫.T, 𝒫.O
   b′ = similar(b)
   for (i′, s′) in enumerate(𝒮)
      po = O(a, s′, o)
      b′[i′] = po * sum(T(s, a, s′) * b[i] for (i, s) in enumerate(𝒮))
   end
   if sum(b′) ≈ 0.0
      fill!(b′, 1)
   end
   return normalize!(b′, 1)
end

# Algorithm 19.3. The Kalman filter,
# which updates beliefs in the form
# of Gaussian distributions. The current belief is represented by μb and
# Σb, and 𝒫 contains the matrices that
# define linear Gaussian dynamics
# and observation model. This 𝒫 can
# be defined using a composite type
# or a named tuple.

struct KalmanFilter
   μb # mean vector
   Σb # covariance matrix
end

function update(b::KalmanFilter, 𝒫, a, o)
   μb, Σb = b.μb, b.Σb
   Ts, Ta, Os = 𝒫.Ts, 𝒫.Ta, 𝒫.Os
   Σs, Σo = 𝒫.Σs, 𝒫.Σo
   # predict
   μp = Ts*μb + Ta*a
   Σp = Ts*Σb*Ts' + Σs
   # update
   K = Σp*Os'/(Os*Σp*Os' + Σo)
   μb′ = μp + K*(o - Os*μp)
   Σb′ = (I - K*Os)*Σp
   return KalmanFilter(μb′, Σb′)
end

struct ExtendedKalmanFilter
   μb # mean vector
   Σb # covariance matrix
end

import ForwardDiff: jacobian
function update(b::ExtendedKalmanFilter, 𝒫, a, o)
   μb, Σb = b.μb, b.Σb
   fT, fO = 𝒫.fT, 𝒫.fO
   Σs, Σo = 𝒫.Σs, 𝒫.Σo
   # predict
   μp = fT(μb, a)
   Ts = jacobian(s->fT(s, a), μb)
   Os = jacobian(fO, μp)
   Σp = Ts*Σb*Ts' + Σs
   # update
   K = Σp*Os'/(Os*Σp*Os' + Σo)
   μb′ = μp + K*(o - fO(μp))
   Σb′ = (I - K*Os)*Σp
   return ExtendedKalmanFilter(μb′, Σb′)
end

# Algorithm 19.5. The unscented
# Kalman filter, an extension of the
# Kalman filter to problems with
# nonlinear Gaussian dynamics. The
# current belief is represented by
# mean μb and covariance Σb. The
# problem 𝒫 specifies the nonlinear
# dynamics using the mean transition dynamics function fT and
# mean observation dynamics function fO. The sigma points used in
# the unscented transforms are controlled by the scalars α, β, and κ.

struct UnscentedKalmanFilter
   μb # mean vector
   Σb # covariance matrix
   λ # spread parameter
end

function unscented_transform(μ, Σ, f, λ, ws)
   n = length(μ)
   Δ = sqrt((n + λ) * Σ)
   S = [μ]
   for i in 1:n
      push!(S, μ + Δ[:,i])
      push!(S, μ - Δ[:,i])
   end
   S′ = f.(S)
   μ′ = sum(w*s for (w,s) in zip(ws, S′))
   Σ′ = sum(w*(s - μ′)*(s - μ′)' for (w,s) in zip(ws, S′))
   return (μ′, Σ′, S, S′)
end

function update(b::UnscentedKalmanFilter, 𝒫, a, o)
   μb, Σb, λ = b.μb, b.Σb, b.λ
   fT, fO = 𝒫.fT, 𝒫.fO
   n = length(μb)
   ws = [λ / (n + λ); fill(1/(2(n + λ)), 2n)]
   # predict
   μp, Σp, Sp, Sp′ = unscented_transform(μb, Σb, s->fT(s,a), λ, ws)
   Σp += 𝒫.Σs
   # update
   μo, Σo, So, So′ = unscented_transform(μp, Σp, fO, λ, ws)
   Σo += 𝒫.Σo
   Σpo = sum(w*(s - μp)*(s′ - μo)' for (w,s,s′) in zip(ws, So, So′))
   K = Σpo / Σo
   μb′ = μp + K*(o - μo)
   Σb′ = Σp - K*Σo*K'
   return UnscentedKalmanFilter(μb′, Σb′, λ)
end

struct ParticleFilter
   states # vector of state samples
end

function update(b::ParticleFilter, 𝒫, a, o)
   T, O = 𝒫.T, 𝒫.O
   states = [rand(T(s, a)) for s in b.states]
   weights = [O(a, s′, o) for s′ in states]
   D = SetCategorical(states, weights)
   return ParticleFilter(rand(D, length(states)))
end

struct RejectionParticleFilter
   states # vector of state samples
end

function update(b::RejectionParticleFilter, 𝒫, a, o)
   T, O = 𝒫.T, 𝒫.O
   states = similar(b.states)
   i = 1
   while i ≤ length(states)
      s = rand(b.states)
      s′ = rand(T(s,a))
      if rand(O(a,s′)) == o
         states[i] = s′
         i += 1
      end
   end
   return RejectionParticleFilter(states)
end

struct InjectionParticleFilter
   states # vector of state samples
   m_inject # number of samples to inject
   D_inject # injection distribution
end

function update(b::InjectionParticleFilter, 𝒫, a, o)
   T, O, m_inject, D_inject = 𝒫.T, 𝒫.O, b.m_inject, b.D_inject
   states = [rand(T(s, a)) for s in b.states]
   weights = [O(a, s′, o) for s′ in states]
   D = SetCategorical(states, weights)
   m = length(states)
   states = vcat(rand(D, m - m_inject), rand(D_inject, m_inject))
   return InjectionParticleFilter(states, m_inject, D_inject)
end

mutable struct AdaptiveInjectionParticleFilter
   states # vector of state samples
   w_slow # slow moving average
   w_fast # fast moving average
   α_slow # slow moving average parameter
   α_fast # fast moving average parameter
   ν # injection parameter
   D_inject # injection distribution
end

function update(b::AdaptiveInjectionParticleFilter, 𝒫, a, o)
   T, O = 𝒫.T, 𝒫.O
   w_slow, w_fast, α_slow, α_fast, ν, D_inject =
   b.w_slow, b.w_fast, b.α_slow, b.α_fast, b.ν, b.D_inject
   states = [rand(T(s, a)) for s in b.states]
   weights = [O(a, s′, o) for s′ in states]
   w_mean = mean(weights)
   w_slow += α_slow*(w_mean - w_slow)
   w_fast += α_fast*(w_mean - w_fast)
   m = length(states)
   m_inject = round(Int, m * max(0, 1.0 - ν*w_fast / w_slow))
   D = SetCategorical(states, weights)
   states = vcat(rand(D, m - m_inject), rand(D_inject, m_inject))
   b.w_slow, b.w_fast = w_slow, w_fast
   return AdaptiveInjectionParticleFilter(states, w_slow, w_fast, α_slow, α_fast, ν, D_inject)
end