### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 24deb3ce-2789-4a07-9bbf-87ee9a9d25ac
begin
	using PlutoUI
	using Plots
	import PlotlyBase
	plotly()
	md"""
	
	#### Julia & Pluto setup
	"""
end

# ╔═╡ f941eb7c-d007-41c2-8cd2-be028759378d
using LinearAlgebra: Hermitian, Diagonal, tr, eigvals, eigvecs, isposdef

# ╔═╡ 1672b75c-2218-4da1-822a-69381abde65a
using QuadGK     # for numerical integration

# ╔═╡ 4f40abdd-e0ea-4ddf-b411-01b2c8172147
using Zygote     # for automatic differentiation (to test against)

# ╔═╡ da361613-f36b-4679-a7cd-d77b77588fdd
using Statistics # for percentiles and whatnot

# ╔═╡ b2f182ba-4f5e-477f-9a68-40f0a6e6b7fa
TableOfContents()

# ╔═╡ 44a73b2d-2aa0-4d92-9584-1644ed9c86f2
md"""
# *Nyquist vs Banchy-Crooks*
This Julia Pluto-notebook is part of the supplementary material for the paper 

> Dirk Oliver Theis, *“Proper” Shift Rules for Derivatives of Perturbed-Parametric
> Quantum Evolutions*, [arXiv:2207.01587](https://arxiv.org/abs/2207.01587)

#### Copyright and license information
Copyright lies with the University of Tartu, Estonia, and, to the extent mandated by law, with the author. The University of Tartu has applied for patent protection for some of the methods and processes encoded in this software. 

Permission is hereby granted to view, run, and experiment with this document. Rights to use either the software or the algorithms, methods, and processes encoded in it are *not* granted.

Address inquiries to:

	University of Tartu
	Centre for Entrepreneurship and Innovation
	Narva mnt 18
	51009 Tartu linn,
	Tartu linn, Tartumaa
	Estonia
	+372 737 4809
	eik@ut.ee
	https://eik.ut.ee
"""

# ╔═╡ 8bbce8bc-a587-464b-ba15-4601914447b0
md"""
# Introduction
This Pluto notebook is concerned with methods to estimate derivatives, with respect to θ of parameterized quantum expectation-values of the form
```math
\tag{$*$}
f\colon (t,\theta) \mapsto \mathrm{tr}( M e^{i t(\theta A+B)/\hbar} \varrho e^{-i t(\theta A+B)/\hbar})
```
The unitary $e^{it(\theta A+B)/\hbar}$ expresses a quantum evolution with Hamiltonian $H := -(\theta A+B)$ for a time $t$.

It implements, for the sake of comparison, the following methods:
* Banchi-Crooks' *Stochastic Approximate Parameter Shift Rule* [arXiv:2005.10299](https://arxiv.org/abs/2005.10299) and
* the *truncated Nyquist shift rule* [arXiv:2207.01587](https://arxiv.org/abs/2207.01587).

When implemented on quantum devices, these methods are estimators, i.e., the output is random and the expected output is off from the sought derivative by a (hopefully only) small bias (approximation error).

In this comparison, we are not interested in the stochastic properties (which, on paper, are essentially identical) but in the magnitude of the approximation error.

Both methods require large magnitudes of the $\theta$ parameter in order to approximate the derivative well. We will compare the effect of the magnitude of $\theta$ on the approximation error in both methods.
"""

# ╔═╡ 1a48fa48-b1ce-11ec-2a39-0da09392a7ca
begin
	import Base.:*

	const ⋅  = *
	const ¬  = !

	const ℝ  = Float64
	const ℂ  = Complex{ℝ}

	const ℜ  = real
	const ℑ  = imag

	const 𝒊  = ℂ(im)
	const π  = ℝ(pi)
	const π𝒊 = π⋅𝒊


	const ∫ = quadgk


	md"""
	We use syntactic sugar to make Julia look more like math: "$\cdot$" instead of "$*$" etc, "$\lnot$" instead of "!"...
	"""
end

# ╔═╡ 050279a5-26a2-45e2-850b-cb7c22c46042
md"""
# Expectation-value function and data
We make available a Julia function ``f(t,\theta ; \texttt{::Data})``, along with helpers and variants.

We take $h:=1$ for the Planck constant, i.e., $\hbar = 1/2\pi$. You don't like it, suck it.
"""

# ╔═╡ 766c707f-4c44-4fed-8115-660d13178bf3
begin
	const Data = @NamedTuple{M::Hermitian, ϱ::Hermitian, A::Hermitian, B::Hermitian}

	md"""
	`Data = @NamedTuple{M,ϱ,A,B}` is a helper data structure to hold the four Hermitian matrices.
	"""
end

# ╔═╡ c48a709d-181d-49aa-8c5a-6c6e594c9fa7
md"""
### Perturbed-parametric unitary function
```math
U\colon (t,\theta) \mapsto e^{2\pi i t( \theta A + B)}
```
"""

# ╔═╡ 3b4c3356-65fb-47b4-843e-928fbff4db20
begin

	mutable struct Eval_Stats_t
		n_calls   ::Int
		θₘₐₓ      ::ℝ
	end

	const eval_stats = Eval_Stats_t(0,0.0)

	"""
	Function `U(t::ℝ, θ::ℝ ; A::Hermitian{ℂ}, B::Hermitian{ℂ}) ::Matrix{ℂ}`
	"""
	function U(t::ℝ, θ::ℝ ; A::Hermitian{ℂ}, B::Hermitian{ℂ}) ::Matrix{ℂ}
		eval_stats.n_calls   += 1
		eval_stats.θₘₐₓ      =  max(eval_stats.θₘₐₓ ,  abs(θ) )
		exp( 2π𝒊⋅t⋅(θ⋅A+B) )
	end
end

# ╔═╡ 310a4e67-de57-459e-951c-bc6cbefd7536
md"""
### Expectation-value function
```math
f\colon (t,\theta) \mapsto \mathrm{tr}( M \, U(t,\theta) \, \varrho \, U(t,\theta)^\dagger)
```
"""

# ╔═╡ c1682e87-e441-4b7d-b6ab-697235d7d944
"""
Function
```
	f(t::ℝ, θ::ℝ
      ;     D::Data ) ::ℝ
```
"""
f(t::ℝ, θ::ℝ  ;  D::Data  ) ::ℝ   =
	begin
		M,ϱ,A,B = D
		Uₜ₀ = U(t,θ;A,B) ; tr( M⋅Uₜ₀⋅ϱ⋅Uₜ₀' ) |> ℜ
	end

# ╔═╡ f293797c-0ffd-40ea-85ea-0469c256c525
md"""
### Randomized input data

To compare the methods, in ($*$) above, we choose
* ``M`` a random Hermitian matrix with eigenvalues in ``\{\pm1\}``;
* ``\varrho`` a random positive definite trace-1 matrix
* ``A`` a random Hermitian matrix with eigenvalues in ``\{\pm1\}``;
* ``B`` a random Hermitian matrix with iid standard-normal complex entries.

This setting corresponds to the most common application in quantum computing: The observable is a Pauli operator, and the 1-qubit drive is a Pauli rotation. (Our setting is negligibly more general.)
"""

# ╔═╡ 5438fd29-d30a-4944-b779-f0e131fc1a69
begin
	function rand_Herm_given_evals(λ ::Vector{ℝ}) ::Hermitian{ℂ}
		d = length(λ)
		@assert d ≥ 2

		_A  = Hermitian( randn(ℂ,d,d) )
		U   = eigvecs(_A)
		return Hermitian( U⋅Diagonal(λ)⋅U' )
	end

	function non_const_sample(X::Vector{T},d::Int) ::Vector{T} where T
		sample = T[]
		while isempty(sample)
			sample = rand(X,d)
			if all( x -> x ≈ sample[1] , sample[2:end])
				sample = T[]
			end
		end
		return sample
	end

	"""
	Function `gimme_data(d ::Int) :: Data`
	"""
	function gimme_data(d ::Int) ::Data
		@assert d ≥ 2

		M   = rand_Herm_given_evals( non_const_sample([-1.0,+1.0],d) )

		C   = randn(ℂ,d,d)
		_ϱ  =  C'⋅C
		_ϱ ./= tr(_ϱ)
		ϱ   = Hermitian(_ϱ)

		A   = rand_Herm_given_evals( non_const_sample([-1.0,+1.0],d) )
		B   = Hermitian( randn(ℂ,d,d) )

		return (M=M,ϱ=ϱ,A=A,B=B)
	end
end

# ╔═╡ 2414502a-80aa-4fde-b2cd-40c302e6e97f
md"""
#### Let's make some data, just for fun...
"""

# ╔═╡ 6ccf3122-760c-4084-a70c-109354979fd8
@bind DATA_MAKING PlutoUI.combine() do child
	md"""
	The dimension is $(child( Scrubbable( 2:100 , default=4))).
	
	* Give me a new set of matrices, $(child( Button("hop, hop!") ))

	Eigenvalues:
	"""
end

# ╔═╡ f579fab7-42b6-4869-84f8-3d32d5613d84
begin
	DATA_MAKING[2] ;  # react to button press

	dat = gimme_data(DATA_MAKING[1])

	@assert isposdef(dat.ϱ)

	with_terminal() do
		println("M: ", eigvals(dat.M))
		println("ϱ: ", eigvals(dat.ϱ))
		println("A: ", eigvals(dat.A))
		println("B: ", eigvals(dat.B))
	end
end

# ╔═╡ 13161493-4a60-43f9-89f9-b1adb7ec4a80
md"""
##### ... and plot it...
... or not: Plotting takes several seconds.  Do you really want to plot? $( @bind PLOTEVF_YES Select([true=>"Yes",false=>"No"],default=false) )
"""

# ╔═╡ 4f1cc4cc-de3b-4fa8-aaee-d61ad36fdf03
if PLOTEVF_YES
	md"Plotting:"
else
	md"(Not plotting.)"
end

# ╔═╡ da1fe645-b884-4361-81aa-a7551a14110f
if PLOTEVF_YES
	contour( -25:0.01:+25, 0:0.01:1, (θ,t)->f(t,θ ; D=dat) , fill=true ;
	xaxis="θ",yaxis="t")
end

# ╔═╡ d4d0e767-a57f-4002-9c48-ec025f2c770c
md"""
### Analytic derivative by Julia

"""

# ╔═╡ 5505698e-998f-427e-bdb5-58003ddf01c8
begin
	"""
	Function `j∂(θ ; D::Data)`
	makes available the "true" derivative of the expectation value function ``θ ↦ f(1,θ)`` for the given data, using Julia's automatic differentiation based on code reflection (Zygote package).
	"""
	j∂(D::Data;t=1.0) = (θ -> f(t,θ ; D))'
end

# ╔═╡ 776be70c-471c-4033-b8d4-82de9ba1462d
md"""
# Implementation of Banchi-Crooks
"""

# ╔═╡ 91ddca2f-36ee-4538-be6b-3e538c52e361
md"""
### Expectation-value function for BC ASPSR

`fₐₛₚ2()` is the function under the integral in Banchi-Crooks ASPSR rule to approximate ``\partial f``.  For the derivative of ``f`` as in (``*``), in `fₐₛₚ2(`𝑠,θ,ε`)`, the unitary ``e^{2\pi i (\theta A+B)}`` in (``*``) is replaced by 
```math
e^{2\pi i s(\theta A+B)} e^{2\pi i \varepsilon (\pm\frac{1}{8\varepsilon}A + B)} e^{2\pi i (1 - s)(\theta A+B)}
```

With the "``\pm``" matching the superscript plus and minus in the following expression, the function `fₐₛₚ2()` is used in the BC's ASPSR as follows:
```math
\partial f(\theta) \approx 2\pi \int_0^1 \left( \; f_{\text{asp}}(s,\theta,\varepsilon)^+ - f_{\text{asp}}(s,\theta,\varepsilon)^- \; \right) \; ds
```
"""

# ╔═╡ 6bdc6667-9b89-4365-ab1a-548dc533e686
"""
Function
```
	fₐₛₚ2( s::ℝ, θ ::ℝ, ε ::ℝ
				;
                D ::Data   ) ::@NamedTuple{plus::ℝ,minus::ℝ}
```
returns the results of two evaluations of the "Approximate Stochastic Parameter" function as in Algorithm 3 of Banchi & Crooks; the two values are those for 𝑚=±1.
"""
fₐₛₚ2(s::ℝ, θ::ℝ, ε::ℝ;   D ::Data) ::@NamedTuple{plus::ℝ,minus::ℝ}   =
	let
		M,ϱ,A,B = D
		Uₛ   =      U(s,      θ ; A, B)
		U₁₋ₛ =      U(1-s,    θ ; A, B)

		U₊   = Uₛ ⋅ U(ε,  +1/8ε ; A, B) ⋅ U₁₋ₛ
		U₋   = Uₛ ⋅ U(ε,  -1/8ε ; A, B) ⋅ U₁₋ₛ

		return (   plus=ℜ( tr(M⋅U₊⋅ϱ⋅U₊') ) ,   minus=ℜ( tr(M⋅U₋⋅ϱ⋅U₋') )   )
	end

# ╔═╡ 7b4bcef0-808a-4087-b8c0-2856e3ea046e
md"""
### BC deterministic derivative
We implement Banchi-Crooks pseudo-shift rule deterministically, with the "shots" (Monte-Carlo integration) replaced by numerical integration (QuadGK package).

The BC method has a parameter, ``\varepsilon``, affecting the accuracy.
"""

# ╔═╡ 632b88b9-732a-4560-8915-03cf716324f5
begin
	"""
	Function `bcₐₚₓ(θ::ℝ ; ε     ::ℝ,
						   order ::Int,
						   D     ::Data ) ::@NamedTuple{∂::ℝ,err::ℝ}`
	
	Original BC pseudo-shift rule for 𝐻 with eigenvalues ±1.

	Performs the numerical, deterministic approximation of the derivative at `θ`.
	
	Numerical integration adds the parameter `order`: It goes into the QuadGK numerical integration package (the log₁₀ of the maximum number of function evaluations).

	*Return value:* Named tuple w/ 1st entry the derivative, `∂`, 2nd entry the numerical error of the integration, `err`.
	"""
	function bcₐₚₓ(θ::ℝ  ; 	ε     ::ℝ,
							order ::Int,
							D     ::Data) ::@NamedTuple{∂::ℝ,err::ℝ}
		int,err =
			∫( 0,1 ; order ) do s
						𝑓ₐₛₚ = fₐₛₚ2( s, θ, ε ; D)
						𝑓ₐₛₚ.plus - 𝑓ₐₛₚ.minus
				end
		return (∂=2π⋅int,err=err)
	end
end

# ╔═╡ a036848b-07e7-46ec-aba6-92e3c4a832a8
md"""
# Implementation of Nyquist shift rule
"""

# ╔═╡ a48b47dc-a4df-40ca-ae6f-bf1bee42673a
md"""
### Helpers
"""

# ╔═╡ 8e9368c1-37c8-4acf-bbdf-2379ad874b36
"""
Function
```
	f₁2(θ::ℝ, a::ℝ
                   ; T ::ℝ,
					 D ::Data ) ::@NamedTuple{plus::ℝ,minus::ℝ}
```
two `f`-values: 𝑓(1,θ−𝑎), and 𝑓(1,θ+𝑎); each one of them is replaced by 0 if the parameter value falls outside of the interval ``[-T, +T]``.
"""
f₁2(θ::ℝ, a::ℝ
	;
	T ::ℝ,
	D ::Data) ::@NamedTuple{plus::ℝ,minus::ℝ}  =
	( 	plus =  ( -T ≤ θ+a ≤ T ?   f(1.0, θ + a  ; D)  : 0.0 ),
		minus = ( -T ≤ θ-a ≤ T ?   f(1.0, θ - a  ; D)  : 0.0 )
	)

# ╔═╡ 6a501d7b-a7c1-4b0f-ae66-da8fcc1cef38
md"""
### Truncated Nyquist shift rule
We approximate the (analytical!) Nyquist derivative by truncating the sum at the evaluation points of the Banchi-Crooks derivative, ``1/\varepsilon`` (where ``\varepsilon`` is as in `bcₐₚₓ()`).
"""

# ╔═╡ c5f6b999-2473-4a6e-80ca-2836738cc921
"""
Function `nyₜᵣᵤₙ(θ ; ε , D::Data)::ℝ`

Approximates the Nyquist derivative deterministically by truncating the sum in such a way that the parameter values stay within ``[-T,+T]`` for ``T:= \\pi/4\\epsilon``.  In other way, the parameter values stay within the same window as in Banchi-Crooks.
"""
function nyₜᵣᵤₙ( θ ; ε::Float64, D::Data) ::ℝ
	T = 1/8ε     ; @assert T > 1

	sum(0.5: 1.0 :4T+1) do a
		𝑓2 = f₁2(θ, -a/4 ; T, D)
		4 ⋅ (-1)^(Int(a+1/2)) ⋅ ( 𝑓2.plus - 𝑓2.minus ) / ( π ⋅ a^2 )
	end
end

# ╔═╡ 6b7fc502-cf2a-4c2a-a21a-e487edf3b9d6
md"""
# Visualization
"""

# ╔═╡ 638bda3f-bb17-4831-8a83-2feb150cb430
@bind BC_PLOT PlutoUI.combine() do child
	md"""
	##### Define the quantities:
	
	* Plot window: ``[`` $(child( NumberField( -100.0:0.1:0.0 , default=-1.0))) ``,`` $(child( NumberField( 0.0:0.1:100.0 , default=+1.0))) ``]``
	* Number of plot points in the window: $(child( NumberField( 1 : 1000, default=10)))
	* Banchi-Crooks ``\varepsilon = e\cdot 10^{-\ell}`` where ``e=`` $(child( NumberField(1.0:9.9999999))) and ``\ell=`` $(child( NumberField( 1:1:30 , default=1)))
	* Numerical integration order parameter $(child( NumberField( 1:1:30 , default=10)))
	"""
end

# ╔═╡ 5f3603da-4edc-43f2-9c41-83df6981f038
md"""
Expectation value function and derivative (error-free automatic differentiation by Julia):
"""

# ╔═╡ 056c039e-b496-4928-9640-f44a2232116c
let
	a     = BC_PLOT[1]
	b     = BC_PLOT[2]
	n     = BC_PLOT[3]
	ε     = 10.0^(-BC_PLOT[4])
	order = BC_PLOT[5]

	σ         = (b-a)/(n+1)
	α         = a+σ
	β         = b-σ/2
	plotrange = α: σ/100 :β

	∂f = j∂(dat)
	theplot = plot( ∂f , plotrange
		; 	label="",
			xaxis="θ", xlimits=(a,b+σ/4),
			yaxis="black: f(1,θ), blue: ∂₂f(1,θ)",
			color=:blue)
	plot!(theplot, θ->f(1.0,θ;D=dat) , plotrange
		; 	label="",
			color=:black)
end

# ╔═╡ ec8fc8ab-f45c-4749-9dc3-047e9b529ce1
md"""
##### Plot BC error
"""

# ╔═╡ 7aee6d2e-6b41-40cc-96f8-74b725428c62
begin
visualize_plot =
let	
	a     = BC_PLOT[1]
	b     = BC_PLOT[2]
	n     = BC_PLOT[3]
	ε     = BC_PLOT[4]⋅10.0^(-BC_PLOT[5])
	order = BC_PLOT[6]

	σ         = (b-a)/(n+1)
	α         = a+σ
	β         = b-σ/2
	plotrange = α: σ :β

	∂f = j∂(dat;t=1.0)
	err(θ) = bcₐₚₓ(θ ; ε, order, D=dat).∂ - ∂f(θ)
	theplot = scatter( err, plotrange
		; 	label="",
			xaxis="θ", xlimits=(a,b+σ/4),
			yaxis="absolute error",
			markersize=0.5,
			markercolor=:red,
			markerstrokecolor=:red)
end
visualize_plot
end

# ╔═╡ 5898bd72-575b-454f-9841-bb2d832f4ff8
md"""
##### Plot truncated-Nyquist error (green)
Include BC-error in plot? $( @bind VISU_INCLUDE_BC CheckBox(true) )
"""

# ╔═╡ d468d44e-1df9-4dd6-a75f-585b0e697151
let
	if  VISU_INCLUDE_BC
		theplot = visualize_plot
	else
		theplot = plot()
	end

	a     = BC_PLOT[1]
	b     = BC_PLOT[2]
	n     = BC_PLOT[3]
	ε     = BC_PLOT[4]⋅10.0^(-BC_PLOT[5])
	order = BC_PLOT[6]

	σ         = (b-a)/(n+1)
	α         = a+σ
	β         = b-σ/2
	plotrange = α: σ :β

	∂f = j∂(dat)
	err(θ) = nyₜᵣᵤₙ(θ ; ε, D=dat) - ∂f(θ)
	scatter!(theplot, err , plotrange
		; 	label="",
			xaxis="θ", xlimits=(a,b+σ/4),
			yaxis="absolute error",
#			ylimits=[-0.1,+0.1],
			markersize=0.5,
			markercolor=:green,
			markerstrokecolor=:green)
end

# ╔═╡ 183092ba-9b70-4045-803a-4a445382fd5e
md"""
# Comparison
"""

# ╔═╡ 0d693856-d260-47ea-81c8-1c76c9f818b2
md"""
#### Data structures and functions for creating the data.
"""

# ╔═╡ 32ab7bf8-27f7-4f3f-8641-cbca29cac2da
begin

	"""
	Function `make_data(l; dim=2, RNDDATA_ITER=100, SAMPLE_PTS, order=20)`

	##### Parameters
	* `l` determines the range of ε's, it will be 𝑥⋅10ˡ for 𝑥∈[1,10[
	* `SAMPLE_PTS` is an iterator or iterable or so.

	##### Return value:
	* Named tuple `(εs,errors_bc,errors_ny,true_vals)`
	* `θs` is a vector of the θ's
	* `errors_𝑥𝑦` is a 2-dim array of data points
	* `true_vals` is a 2-dim array of the true derivatives
	In the arrays, 1st dim is repetition idx, 2nd dim is θ-idx
	"""
	function make_data(ε; dim, RNDDATA_ITER, SAMPLE_PTS, order=20)
		errors_bc = Array{ℝ}(undef,RNDDATA_ITER,length(SAMPLE_PTS))
		errors_ny = Array{ℝ}(undef,RNDDATA_ITER,length(SAMPLE_PTS))
		true∂     = Array{ℝ}(undef,RNDDATA_ITER,length(SAMPLE_PTS))

		errors_bc .= -Inf
		errors_ny .= -Inf
	
		for rdi = 1:RNDDATA_ITER
			D   = gimme_data(dim)
			for  (j,θ) in enumerate(SAMPLE_PTS)
				true∂[rdi,j] = j∂(D)(θ)
			end
			for  (j,θ) in enumerate(SAMPLE_PTS)
				bc  = bcₐₚₓ(θ;ε,order,D)
				err = abs(  bcₐₚₓ(θ;ε,order,D).∂    - true∂[rdi,j]  )
				errors_bc[rdi,j] = err
				if (bc.err > err/16)
					println("Problem: Numerical error $(bc.err) vs algorithmic error $(err)")
				end
			end
			for  (j,θ) in enumerate(SAMPLE_PTS)
				errors_ny[rdi,j] = abs(
					nyₜᵣᵤₙ(θ;ε,D)   - true∂[rdi,j]
				)
			end
		end

		return (θs=collect(SAMPLE_PTS),
		errors_bc=errors_bc,errors_ny=errors_ny,true_vals=true∂)
	end
end

# ╔═╡ 6cf39780-aab6-4b6b-9f5e-12b660786821
const Stats_t = @NamedTuple{mean::ℝ, median::ℝ, perc01::ℝ, perc10::ℝ, perc25::ℝ, perc90::ℝ, perc99::ℝ, min::ℝ, max::ℝ}

# ╔═╡ 62ad707f-4391-4b1b-b5d7-a660ce176629
begin
	"""
	Function `get_stats(θs,errors_bc,errors_ny,true_vals
	           ;  relative=false)`

	Input here is the output of `make_data()`

	Return value is a named tuple `(bc_err, ny_err, bc_err_minus_ny_err)` of vectors (indexed corresponding to `θs`) of `Stats_t`
	"""
	function get_stats(θs,_errors_bc,_errors_ny,true_vals
		               ;
					   relative::Bool=false)
		
		# Make stats

		errᵇᶜ  = Vector{Stats_t}(undef,length(θs))
		errⁿʸ  = Vector{Stats_t}(undef,length(θs))
		errdiff= Vector{Stats_t}(undef,length(θs))

		my_errors_bc = copy(_errors_bc)
		my_errors_ny = copy(_errors_ny)
		if relative
				my_errors_bc           ./= abs.(true_vals)
				my_errors_ny           ./= abs.(true_vals)
		end

		for (j,θ) in enumerate(θs)
			errᵇᶜ[j] = (
							mean    = mean(   my_errors_bc[:,j]),
							median  = median( my_errors_bc[:,j]),
							perc01  = quantile( vec(my_errors_bc[:,j]), 0.01 ),
							perc10  = quantile( vec(my_errors_bc[:,j]), 0.1 ),
							perc25  = quantile( vec(my_errors_bc[:,j]), 0.25 ),
							perc90  = quantile( vec(my_errors_bc[:,j]), 0.9 ),
							perc99  = quantile( vec(my_errors_bc[:,j]), 0.99 ),
							min     = minimum(my_errors_bc[:,j]),
							max     = maximum(my_errors_bc[:,j])
						)
		end
		for (j,θ) in enumerate(θs)
			errⁿʸ[j] = (
							mean    = mean(   my_errors_ny[:,j]),
							median  = median( my_errors_ny[:,j]),
							perc01  = quantile( vec(my_errors_ny[:,j]), 0.01 ),
							perc10  = quantile( vec(my_errors_ny[:,j]), 0.1 ),
							perc25  = quantile( vec(my_errors_ny[:,j]), 0.25 ),
							perc90  = quantile( vec(my_errors_ny[:,j]), 0.9 ),
							perc99  = quantile( vec(my_errors_ny[:,j]), 0.99 ),
							min     = minimum(my_errors_ny[:,j]),
							max     = maximum(my_errors_ny[:,j])
						)
		end
		for (j,θ) in enumerate(θs)
			diffⱼ = vec(  my_errors_bc[:,j] - my_errors_ny[:,j]  )
			errdiff[j] = (
							mean    = mean(     diffⱼ ),
							median  = median(   diffⱼ ),
							perc01  = quantile( diffⱼ , 0.01 ),
							perc10  = quantile( diffⱼ , 0.1 ),
							perc25  = quantile( diffⱼ , 0.25 ),
							perc90  = quantile( diffⱼ , 0.9 ),
							perc99  = quantile( diffⱼ , 0.99 ),
							min     = minimum(  diffⱼ ),
							max     = maximum(  diffⱼ )
						)
		end

		return (bc_err=errᵇᶜ,ny_err=errⁿʸ, bc_err_minus_ny_err=errdiff)
	end
end

# ╔═╡ f34ba029-d093-4fef-a60f-7d668054e1bc
md"""
#### Let's get cracking!
"""

# ╔═╡ 4e854293-38fb-45b2-ad9f-888387122f21
@bind MAKE_DATA_INPUT PlutoUI.combine() do child
	md"""
	How much data should be produced?
	* ``\varepsilon = 10^{-\ell}`` where ``\ell=`` $(
	                 child(NumberField(1:5,default=1)) )
	* Dimension: $( child(NumberField(2:64,default=2)) )
	* Number of random expectation-value functions (``M,\varrho,A,B`` as described above)
	   $( child(NumberField(1:1000,default=10)) )
	* Number of sample points:
	  $( child(NumberField(5:1:1000,default=5)) )

	Now making data.  This will take some time.  The terminal will show error messages (such as numerical issues).
	"""
end

# ╔═╡ fdcd719b-dbef-4cae-8a26-07f448ee0e57
begin
	data = nothing
	with_terminal(show_value=false) do
		ε             = 10.0^-MAKE_DATA_INPUT[1]
		dim           = MAKE_DATA_INPUT[2]
		SAMPLE_PTS    = range(-0.05/8ε,+1.05/8ε ; length=MAKE_DATA_INPUT[4])
		RNDDATA_ITER  = MAKE_DATA_INPUT[3]

		data = make_data( ε ;
				dim,
				RNDDATA_ITER,
				SAMPLE_PTS)
	end
end

# ╔═╡ 6059d76f-cfdc-48c4-9834-11da6918e323
md"""
Relative errors: $( @bind RELATIVE CheckBox(false) )
"""

# ╔═╡ bb798584-e4d9-49b8-8bbc-09da4ee96224
begin
	S = get_stats(data.θs, data.errors_bc, data.errors_ny, data.true_vals
	              ; relative=RELATIVE)
	ndata = length(data.θs)
	yaxistext = ( RELATIVE ? "Relative error" : "Absolute error")
end ;

# ╔═╡ 224cb8ff-3c2c-466d-bc0d-10bdee201035
@bind SHOW_SEPARATE PlutoUI.combine() do child
md"""
The next figure shows
* BC (red): 1st percentile, 10th percentile $(child(CheckBox(false))), median $(child(CheckBox(true))), 90th percentile $(child(CheckBox(false)))
* Ny (green): median, 90th percentile $(child(CheckBox(false))), 99th percentile $(child(CheckBox(false))), max $(child(CheckBox(false)))

Show it at all? $( child(CheckBox(true)) )
"""
end

# ╔═╡ 241387da-4dc5-4c80-a016-dcad799eb96c
if SHOW_SEPARATE[end]
	let theplot = plot()

	¬SHOW_SEPARATE[3] ||
	scatter!(theplot, data.θs, [ S.bc_err[k].perc90 for k in 1:ndata ]
		; 	label="",
			markersize=1,
			markercolor=:red,
			markerstrokecolor=:red)

	¬SHOW_SEPARATE[2] ||
	scatter!(theplot, data.θs, [ S.bc_err[k].median for k in 1:ndata ]
		; 	label="",
			markersize=1,
			markercolor=:red,
			markerstrokecolor=:red)
	
	¬SHOW_SEPARATE[1] ||
	scatter!(theplot, data.θs, [ S.bc_err[k].perc10 for k in 1:ndata ]
		; 	label="",
			markersize=1,
			markercolor=:red,
			markerstrokecolor=:red)

	scatter!(theplot, data.θs, [ S.bc_err[k].perc01 for k in 1:ndata ]
		; 	label="",
			markersize=1,
			markercolor=:red,
			markerstrokecolor=:red)

	¬SHOW_SEPARATE[6] ||
	scatter!(theplot, data.θs, [ S.ny_err[k].max for k in 1:ndata ]
		; 	label="",
			markersize=1,
			markercolor=:green,
			markerstrokecolor=:green)
	¬SHOW_SEPARATE[5] ||
	scatter!(theplot, data.θs, [ S.ny_err[k].perc99 for k in 1:ndata ]
		; 	label="",
			markersize=1,
			markercolor=:green,
			markerstrokecolor=:green)
	¬SHOW_SEPARATE[4] ||
	scatter!(theplot, data.θs, [ S.ny_err[k].perc90 for k in 1:ndata ]
		; 	label="",
			markersize=1,
			markercolor=:green,
			markerstrokecolor=:green)
	scatter!(theplot, data.θs, [ S.ny_err[k].median for k in 1:ndata ]
		; 	label="",
			xaxis="θ",
			yaxis=yaxistext*"s",
			markersize=1,
			markercolor=:green,
			markerstrokecolor=:green)
	theplot
	end
end

# ╔═╡ e313b6a1-2f35-41ce-ac47-546c90197ff3
@bind ERROR_DIFF_PLOT PlutoUI.combine() do child
md"""
The next figure shows statistics for the difference of absolute error for each individual data point, ``\text{err}_{\text{BC}} - \text{err}_{\text{Ny}}``.
The quantity is positive if Nyquist is better than BC, otherwise negative.
The statistics derived from that data set which are shown in the figures are, top to bottom:
* mean (black)  $(child( CheckBox(true) ))
* median (blue)  $(child( CheckBox(true) ))
* 25th-percentile (red)  $(child( CheckBox(true) ))
* 10th-percentile (green)  $(child( CheckBox(true) ))
* 1st-percentile (magenta)  $(child( CheckBox(true) ))
* minimum (cyan)   $(child( CheckBox(true) )).
"""
end

# ╔═╡ 89529e0d-989d-44fc-9862-1f09a73937c4
let ε = 10.0^-MAKE_DATA_INPUT[1]

	if RELATIVE
		theplot = plot(;
				xaxis="θ",
				xticks=[0.1j/8ε for j=0:10],
				yaxis=yaxistext*" differences",
				yticks=vcat([j/10 for j= -10:2:10],[j for j=-100.0:10.0:+100]))
	else
		theplot = plot(;
				xaxis="θ",
				xticks=[0.1j/8ε for j=0:10],
				yaxis=yaxistext*" differences")
	end

	¬ ERROR_DIFF_PLOT[1] ||
	plot!(theplot, data.θs, [ S.bc_err_minus_ny_err[k].mean for k in 1:ndata ]
		; 	st=:line,
			label="",
			markersize=1,
			color=:black)

	¬ ERROR_DIFF_PLOT[2] ||
	plot!(theplot, data.θs, [ S.bc_err_minus_ny_err[k].median for k in 1:ndata ]
		; 	st=:line,
			label="",
			color=:blue,
			markersize=1)

	¬ ERROR_DIFF_PLOT[3] ||
	plot!(theplot, data.θs, [ S.bc_err_minus_ny_err[k].perc25 for k in 1:ndata ]
		; 	st=:line,
			label="",
			color=:red,
			markersize=1)

	¬ ERROR_DIFF_PLOT[4] ||
	plot!(theplot, data.θs, [ S.bc_err_minus_ny_err[k].perc10 for k in 1:ndata ]
		; 	st=:line,
			label="",
			color=:green,
			markersize=1)

	¬ ERROR_DIFF_PLOT[5] ||
	plot!(theplot, data.θs, [ S.bc_err_minus_ny_err[k].perc01 for k in 1:ndata ]
		; 	st=:line,
			label="",
			color=:magenta,
			markersize=1)

	¬ ERROR_DIFF_PLOT[6] ||
	plot!(theplot, data.θs, [ S.bc_err_minus_ny_err[k].min for k in 1:ndata ]
		; 	st=:line,
			label="",
			color=:cyan,
			markersize=1)
	#
	theplot
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlotlyBase = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuadGK = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
PlotlyBase = "~0.8.18"
Plots = "~1.31.2"
PlutoUI = "~0.7.38"
QuadGK = "~2.4.2"
Zygote = "~0.6.37"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "dd9a607bdc36b68410bd8f9858d690cdc4589d69"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "61549d9b52c88df34d21bd306dba1d43bb039c87"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.51.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "be6ab11021cd29f0344d5c4357b163af05a48cba"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.21.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "96d823b94ba8d187a6d8f0826e731195a74b90e9"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "e17cc4dc2d0b0b568e80d937de8ed8341822de67"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.2.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "a3351bc577a6b49297248aadc23a4add1097c2ac"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.7.1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "cf0a9940f250dc3cb6cc6c6821b4bf8a4286cf9c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.66.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "bc9f7725571ddb4ab2c4bc74fa397c1c5ad08943"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.69.1+0"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "bb198ff907228523f3dee1070ceee63b9359b6ab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "659140c9375afa2f685e37c1a0b9c9a60ef56b40"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.7"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5e77dbf117412d4f164a464d610ee6050cc75272"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.6"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "eac00994ce3229a464c2847e956d77a2c64ad3a5"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.10"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "5007c1421563108110bbd57f63d8ad4565808818"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "5.2.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "1222116d7313cdefecf3d45a2bc1a89c4e7c9217"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.22+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "8c57307b5d9bb3be1ff2da469063628631d4d51e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.21"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    DiffEqBiologicalExt = "DiffEqBiological"
    ParameterizedFunctionsExt = "DiffEqBase"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e"
    DiffEqBiological = "eb300fae-53e8-50a0-950c-e21f52c2b7e0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "c3ce8e7420b3a6e071e0fe4745f5d4300e37b13f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.24"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1aa4b74f80b01c6bc2b89992b861b5f210e665b5"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.21+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "5a6ab2f64388fd1175effdf73fe5933ef1e0bac0"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "a19652399f43938413340b2068e11e55caa46b65"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.7"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "b478a748be27bd2f2c73a7690da219d0844db305"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "22c5201127d7b243b9ee1de3b43c408879dff60f"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.3.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "832afbae2a45b4ae7e831f86965469a24d1d8a83"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.26"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "521a0e828e98bb69042fec1809c1b5a680eb7389"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.15"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "5be3ddb88fc992a7d8ea96c3f10a49a7e98ebc7b"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.62"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "977aed5d006b840e2e40c0b48984f7463109046d"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.3"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─b2f182ba-4f5e-477f-9a68-40f0a6e6b7fa
# ╟─44a73b2d-2aa0-4d92-9584-1644ed9c86f2
# ╟─8bbce8bc-a587-464b-ba15-4601914447b0
# ╠═24deb3ce-2789-4a07-9bbf-87ee9a9d25ac
# ╠═f941eb7c-d007-41c2-8cd2-be028759378d
# ╠═1672b75c-2218-4da1-822a-69381abde65a
# ╠═4f40abdd-e0ea-4ddf-b411-01b2c8172147
# ╠═da361613-f36b-4679-a7cd-d77b77588fdd
# ╟─1a48fa48-b1ce-11ec-2a39-0da09392a7ca
# ╟─050279a5-26a2-45e2-850b-cb7c22c46042
# ╟─766c707f-4c44-4fed-8115-660d13178bf3
# ╟─c48a709d-181d-49aa-8c5a-6c6e594c9fa7
# ╟─3b4c3356-65fb-47b4-843e-928fbff4db20
# ╟─310a4e67-de57-459e-951c-bc6cbefd7536
# ╟─c1682e87-e441-4b7d-b6ab-697235d7d944
# ╟─f293797c-0ffd-40ea-85ea-0469c256c525
# ╟─5438fd29-d30a-4944-b779-f0e131fc1a69
# ╟─2414502a-80aa-4fde-b2cd-40c302e6e97f
# ╟─6ccf3122-760c-4084-a70c-109354979fd8
# ╟─f579fab7-42b6-4869-84f8-3d32d5613d84
# ╟─13161493-4a60-43f9-89f9-b1adb7ec4a80
# ╟─4f1cc4cc-de3b-4fa8-aaee-d61ad36fdf03
# ╟─da1fe645-b884-4361-81aa-a7551a14110f
# ╟─d4d0e767-a57f-4002-9c48-ec025f2c770c
# ╟─5505698e-998f-427e-bdb5-58003ddf01c8
# ╟─776be70c-471c-4033-b8d4-82de9ba1462d
# ╟─91ddca2f-36ee-4538-be6b-3e538c52e361
# ╟─6bdc6667-9b89-4365-ab1a-548dc533e686
# ╟─7b4bcef0-808a-4087-b8c0-2856e3ea046e
# ╟─632b88b9-732a-4560-8915-03cf716324f5
# ╟─a036848b-07e7-46ec-aba6-92e3c4a832a8
# ╟─a48b47dc-a4df-40ca-ae6f-bf1bee42673a
# ╟─8e9368c1-37c8-4acf-bbdf-2379ad874b36
# ╟─6a501d7b-a7c1-4b0f-ae66-da8fcc1cef38
# ╟─c5f6b999-2473-4a6e-80ca-2836738cc921
# ╟─6b7fc502-cf2a-4c2a-a21a-e487edf3b9d6
# ╟─638bda3f-bb17-4831-8a83-2feb150cb430
# ╟─5f3603da-4edc-43f2-9c41-83df6981f038
# ╟─056c039e-b496-4928-9640-f44a2232116c
# ╟─ec8fc8ab-f45c-4749-9dc3-047e9b529ce1
# ╟─7aee6d2e-6b41-40cc-96f8-74b725428c62
# ╟─5898bd72-575b-454f-9841-bb2d832f4ff8
# ╟─d468d44e-1df9-4dd6-a75f-585b0e697151
# ╟─183092ba-9b70-4045-803a-4a445382fd5e
# ╟─0d693856-d260-47ea-81c8-1c76c9f818b2
# ╟─32ab7bf8-27f7-4f3f-8641-cbca29cac2da
# ╟─6cf39780-aab6-4b6b-9f5e-12b660786821
# ╟─62ad707f-4391-4b1b-b5d7-a660ce176629
# ╟─f34ba029-d093-4fef-a60f-7d668054e1bc
# ╟─4e854293-38fb-45b2-ad9f-888387122f21
# ╟─fdcd719b-dbef-4cae-8a26-07f448ee0e57
# ╟─6059d76f-cfdc-48c4-9834-11da6918e323
# ╟─bb798584-e4d9-49b8-8bbc-09da4ee96224
# ╟─224cb8ff-3c2c-466d-bc0d-10bdee201035
# ╟─241387da-4dc5-4c80-a016-dcad799eb96c
# ╟─e313b6a1-2f35-41ce-ac47-546c90197ff3
# ╟─89529e0d-989d-44fc-9862-1f09a73937c4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
