### A Pluto.jl notebook ###
# v0.19.18

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

# ╔═╡ a13db58c-f340-11ea-0fa3-df05c859e0bd
begin
	using PlutoUI
	using Plots
	plotly() ;

md"""
### Julia in Quantumland
# Quantum Tunneling — A particle in a well with a barrier

* Copyright © University of Tartu, Estonia
* Author: Assoc. Prof. Dirk Oliver Theis, UTartu
* License: [CC-BY](https://creativecommons.org/licenses/by/4.0/): You are free to share (copy and redistribute the material in any medium or format) and adapt (remix, transform, and build upon the material for any purpose, even commercially), under the condition of [*Attribution:*](https://wiki.creativecommons.org/wiki/License_Versions#Detailed_attribution_comparison_chart) You must give appropriate credit, provide a link to the license, and indicate if changes were made; you may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
"""
end

# ╔═╡ b0d036dd-e9b9-4bb6-9a89-6ef70950e659
begin
	using Unitful
	using Unitful: ħ,
			nm, m,  kg, s, fs,  eV,
			Time, Length, Mass, Energy, Action,
			𝐋, 𝐌, 𝐓,
			ustrip
md"Let's get ourselves some units!"
end

# ╔═╡ e6c48838-30d0-11eb-05aa-29ad654daa2f
using QuadGK

# ╔═╡ e28f810e-bd0f-48b9-b6ce-699e89cc40d4
using IntervalSets

# ╔═╡ df9c4558-273b-11eb-2796-67d82da18f10
begin
	using LinearAlgebra: eigen, norm, Diagonal
	#using LinearAlgebra: dot as ⊙
end

# ╔═╡ 9a6007d0-30c4-11eb-20d8-e9e8f7ebf7fb
TableOfContents()

# ╔═╡ 66b9d55e-25a3-11eb-3096-19dd66316679
md"""
## 1. The story of the particle in the well continues...

We continue the story of our particle (an electron) in a well.  If you remember, the "well" comes from the potential energy.  The simplest particle in a well has this potential energy:

```math
x\mapsto V(x) :=
\begin{cases}
0,       &\text{if $0<x<L$;}\\
\infty,  &\text{otherwise.}
\end{cases}
```

The infinite potential energy outside of the "well" $\left]0,L\right[$ confines the particle wave function to be confined in that interval.  Inside of the well, though, the particle has the same potential energy: $0$.

This week, you will get to simulate the physics of a less trivial well, one which has a more complicated form.  For a small number $0<w<L$, and two numbers $E^+ \ge E_- \ge 0$, we set:

```math
x\mapsto V(x) :=
\begin{cases}
\infty,  &\text{if $x< 0$;}\\
0,       &\text{if $0 \le x < \frac{L-w}{2}$;}\\
E^+,     &\text{if $\frac{L-w}{2} \le x \le \frac{L+w}{2}$;}\\
E_-,     &\text{if $\frac{L+w}{2} < x \le L$;}\\
\infty,  &\text{if $L < x$.}
\end{cases}
```
"""

# ╔═╡ e06d3888-25a2-11eb-167a-47a2eeda7e39
begin
	const ⋅ = *
	const 𝐢 = im
	const ℝ = Float64
	const ℂ = Complex{ℝ}
	const ∑ = Base.sum
	const abs² = Base.abs2
	𝑒ⁱ(x) = exp(𝐢⋅x)

md"Let's have nice looking math..."
end

# ╔═╡ 3aaa19a7-69a3-465b-b921-3aa0caeee397
begin
	@unit eVs "eVs" ElectronVoltSecond 1eV⋅s false

md"Define unit for electron-volt seconds."
end

# ╔═╡ 9258c5ca-7638-4321-8c35-88d35e48681f
Unitful.register(@__MODULE__) ;

# ╔═╡ 78f4e4a5-4d9d-4766-9e3c-c20b0fa79e3b
begin
	const L  ::Length  = ℝ( π )nm
	const mₑ ::Mass    = 9.1093837015e-31kg

md"### 1.a. Let's get some definitions out of the way"
end

# ╔═╡ 093eeacf-28a9-4818-b27d-881729fc1f1a
begin
	@derived_dimension Per_Sqrt_Length 𝐋^(-1//2)
	@unit per_sqrt_m  "m⁻¹ᐟ²" Per_Sqrt_Meter 1/√m false
	@unit per_sqrt_nm "(nm)⁻¹ᐟ²" Per_Sqrt_Nanometer 1/√nm false

	@derived_dimension Energy_Per_Sqrt_Length 𝐋^(3//2)⋅𝐌⋅𝐓^(-2)
	@unit eV_per_sqrt_nm "eV/√nm" eVolt_Per_Sqrt_Nanometer 1eV/√nm false

md"Some more dimensions and units 🙂"
end

# ╔═╡ 8036d94a-25b8-11eb-3b93-350a6274ab6b
md"""
* Width of well ``L =`` $(L),
* Reduced Planck's constant ``\hbar =`` $ħ (``=`` kg m²/s)
* ... in eVs, that's ``\hbar = `` $(eVs(ħ))
* Mass of an electron ``m_e =`` $mₑ
"""

# ╔═╡ 0277c3cc-30c7-11eb-0c92-6325a0cdf19d
md"""
The energy levels of the simple particle in a well will be useful:
"""

# ╔═╡ 19e9be4a-30c7-11eb-3081-d3886a1e5066
E(n::Int)::Energy = ( @assert n≥1 ; eV( n^2 ⋅ ħ^2 / 2mₑ ⋅ (π/L)^2 )) ;

# ╔═╡ 1fb9e8b2-30c8-11eb-1eae-5f6a8ec2125d
md"""
### 1.b. Setup of operators and bases

As in the simple particle in a well, we need to:
1. make sure we understand the Hamiltonian
2. take a basis
3. create the matrices wrt the basis
4. let Julia find the eigenvalues & eigenvectors
5. transform the Julia output into something that we can plot
"""

# ╔═╡ 611eb7c4-30c8-11eb-1e65-8b0519d09001
md"""
#### The Hamiltonian

As usual, we have to add the kinetic energy operator to the potential energy operator.

Kinetic energy, as usual:
```math
H_\text{kin} = \psi \mapsto -\frac{\hbar^2}{2m}\psi''.
```

Potential energy:
```math
H_\text{pot} = \psi \mapsto [ x \mapsto V(x) \psi(x) ].
```

As in the simple particle in a well, the infinite potential energy outside of the "well" $\left]0,L\right[$ confines the wave function to a function $\psi\colon [0,L]\to\mathbb C$ with $\psi(0)=\psi(L)=0$ (which allows us to omit the $\infty$-parts).

The Hamiltonian of our particle in an uneven well is thus:
```math
H = \psi \mapsto \Bigl[ x \mapsto -\frac{\hbar^2}{2m} \psi''(x) + V(x)\psi(x) \Bigr].
```
"""

# ╔═╡ aec436c4-30c9-11eb-31e4-7185257f6029
md"""
#### The basis

Again, our pre-Hilbert space is infinite dimensional, so we need to "omit" parts of it to make it finite dimensional (so that we get finite-size matrices).

We could take the "position basis" $|x\rangle$, $x\in\left]0,L\right[$, again, and choose some subset of it — we did that for the particle in a well.  Been there done that, let's do something new.

We will take an orthonormal system consisting of eigenstates of a closely related Hamiltonian: That of the particle in a well.  The functions are: For $k\in\mathbb N$ (i.e., positive integers):
```math
\phi_k\colon x\mapsto \sqrt{\tfrac{2}{L}} \cdot \sin(k\pi/L\cdot x)
```

Let's have a function for it:
"""

# ╔═╡ c2c606fa-30cb-11eb-3cf8-d182760c9f2f
ϕ(x ::Length; k ::Int) ::Per_Sqrt_Length   = √(2/L) ⋅ sin(k⋅π⋅x/L) ;

# ╔═╡ a88141e6-30cc-11eb-2fc3-4bb8ccd5d5ac
md"""
Enter a frequency: 
𝑘 = $( @bind kₚₗₒₜ NumberField(1:1000 ; default=1) )
"""

# ╔═╡ f0ab0a48-30cb-11eb-3aef-81e84f1226f1
plot([x for x = L/2000 : L/1000 : L ],

	 x  ->  ϕ(x ; k=kₚₗₒₜ) |> ℝ ;

	 label="k=$(kₚₗₒₜ), E=$(E(kₚₗₒₜ))", xaxis="x", yaxis="ϕₖ(x)")

# ╔═╡ b0c2c612-30cd-11eb-1e1d-09ddf529a0d1
md"""
For our finite ONB, we will pick the first N energy levels, i.e., the functions $\phi_1,\phi_2,\dots,\phi_N$.
"""

# ╔═╡ aef58bbe-30cb-11eb-26b0-b7e50a3c694e
md"""
## 2. The matrices

We're now ready to set up the matrix, ``M``, that discretizes the Hamiltonian with respect to the chosen basis.  As you have by now learned 10 times:

* ``M`` is a ``N\times N`` matrix
* The entries are: ``\displaystyle M_{k,\ell} = (\phi_k \mid H \phi_\ell)``

Of course, we use the usual inner product for wave function Hilbert spaces — but we take into account that our wave functions are $0$ outside of $\left]0,L\right[$:

```math
(\phi\mid\psi) := \int\limits_0^L \phi(x)^* \psi(x)\,dx
```

This gives us:

```math
M_{k,\ell} = \int\limits_0^L  \phi_k(x)^* \Bigl( -\tfrac{\hbar^2}{2m} \phi_\ell''(x) + V(x)\phi_\ell(x)\Bigr) \, dx
```

Obviously, we are too lazy to compute the integral by hand: We let Julia do it for us.
"""

# ╔═╡ febedade-30cf-11eb-2943-f72544183cbd
md"""
### 2.a. The inner product function

Now let's have the the function for the $(\phi,\psi)\to (\phi\mid \psi)$.

We use the Julia package [QuadGK](https://juliamath.github.io/QuadGK.jl/latest/).  The basic syntax is:
```Julia
integral, error = quadgk(x -> exp(-x^2)/sqrt(x), 0, 1, rtol=...)
```
"""

# ╔═╡ d1a17ab2-4883-4089-ae8f-525561a522ee
md"""
The following defines the function `∫`, which takes a function $f\colon\mathbb R\to\mathbb R$ as input, and returns the real number $\displaystyle \int_0^L f(x)\,dx$.
"""

# ╔═╡ b2ecba78-30ec-11eb-2680-d906db765a1f
∫(f::Function ; I ::Interval=0.0nm..L)   = quadgk(f , I.left,I.right)[1]   ;

# ╔═╡ 9d498fc2-30d1-11eb-1df6-2d48282ff416
md"""
Here comes the inner product.

Remember that the Julia syntax for the complex conjugate (or adjoint) of `z` is `z'`.
"""

# ╔═╡ 8f585196-30d1-11eb-2c2f-7399ec1c5ece
(ϕ::Function | ψ::Function ) = ∫( x -> ϕ(x)'⋅ψ(x) ) ;

# ╔═╡ 0a001c28-30d5-11eb-14a8-e18d1200c783
md"""
Little tests:
"""

# ╔═╡ e51395f0-30d4-11eb-08b1-b9495f770b11
md"""
This should be zero:     $(
	let ϕ₁₀₀ = x -> ϕ(x;k=100),
		ϕ₁₀₁ = x -> ϕ(x;k=101)
		;
		( ϕ₁₀₀ | ϕ₁₀₁ )
	end
)
"""

# ╔═╡ 8b44eaf0-30d7-11eb-3156-9f9bd82b97de
md"""
This should be 1:       $(
	let ϕ₁₉₉     = x -> ϕ(x;k=199),
		ϕ₁₉₉_too = x -> ϕ(x;k=199)
		;
		( ϕ₁₉₉ | ϕ₁₉₉_too )
	end
)
"""

# ╔═╡ 92d86576-30d2-11eb-32ee-a9811228efce
md"""
### 2.b. Computing the Hamiltonian

Let's have a function $H$ which computes $H$: It takes a function $\psi$ as input, and returns the function $H\psi$.  At this point, we fixe $m := m_e$, the mass of an electron.

##### Kinetic energy term
We need the second derivative of the input function $\psi$.  For simplicity, let's only compute the input functions that we need: The $\phi_k$.  So we need the second derivatives of them.
"""

# ╔═╡ 550d13ac-30ed-11eb-3b34-fb0a048b1827
∂²ϕ(x ::Length; k ::Int) = -√(2/L) ⋅ sin(k⋅π⋅x/L) ⋅ k^2⋅π^2/L^2 ;

# ╔═╡ e877eef9-baf2-45da-a65a-1e4bdb477a2e
md"""
##### Potential energy term
"""

# ╔═╡ c8f5ad2a-30c4-11eb-05e3-9faee17cd60c
"""
Function `V(x::Length ; E⁺::Energy, E₋::Energy, w::Length) :: Energy`

The potential as above.
"""
V(x::Length
  ; 
  E⁺::Energy, E₋::Energy, w::Length) :: Energy   =
begin	
		@assert 0eV ≤ E₋ < 1e10eV
		@assert 0eV ≤ E⁺ < 1e10eV
		@assert 0nm < w < L

		if           x < 0nm          return  Inf      end
		if 0nm     ≤ x < (L-w)/2      return  0.0eV    end
		if (L-w)/2 ≤ x ≤ (L+w)/2      return  E⁺       end
		if (L+w)/2 < x ≤ L            return  E₋       end
		if L       < x                return  Inf      end
end

# ╔═╡ 62b547ee-30c6-11eb-008c-5f1410db2544
plot([x for x = 0nm : L/1001 : L ],

	 x  ->  V(x ; w=L/10, E⁺=E(7), E₋=E(3))  ;

	 yaxis="V ", xaxis="x"   , label="")

# ╔═╡ 80c28c1c-30dd-11eb-06d8-6f7fad08b7c1
md"""
The Hamiltonian, applied to $\phi_k$.
"""

# ╔═╡ a0a88398-30d2-11eb-3ed9-f5483557b145
@doc raw"
Function `H( k ; E⁺::Energy, E₋::Energy, w::Length) ::Function`

Returns function that results if the Hamiltonian with the given parameters is applied to the particle-in-a-well basis function ``\phi_k``.
* `w` — width of the potential energy barrier;
* `E⁺` — potential energy of the barrier;
* `E₋` — potential energy to the right of the barrier.
"
H( k ::Int
   ;
   E⁺::Energy, E₋::Energy, w::Length) ::Function =
begin
	@assert k ≥ 1
	x::Length -> eV_per_sqrt_nm(
					- ħ^2/2mₑ ⋅ ∂²ϕ(x;k)  +  V(x;w,E⁺,E₋) ⋅ ϕ(x;k)
				)
end

# ╔═╡ 5051be2c-30d8-11eb-0790-77ff1f07c5cc
let λ₁₀₁ = ( (x->ϕ(x;k=101)) | H(101 ;  w=L/10, E⁺=0.0eV, E₋=0.0eV) )
	md"""
	This should be equal to $( E(101) ):    $(λ₁₀₁)...  $(
	if λ₁₀₁≈E(101)
		"Yep! 👍👍👍"
	else
		"Hmmm, something's wrong...🤔"
	end
	)"""
end

# ╔═╡ b23a2ef2-30ed-11eb-00b3-dbb88552f074
let ip = ( (x->ϕ(x;k=101)) | H(102 ;  w=L/10, E⁺=0.0eV, E₋=0.0eV) )
	md"""
	This should be equal to zero:   $(ip)... $(
	if 1eV + ip ≈ 1eV
		"Yep! 👍👍👍"
	else
		"Hmmm, something's wrong...🤔"
	end
	)
	"""
end

# ╔═╡ f9b86b0c-30d1-11eb-2bd7-5501d4dc02c5
md"""
### 2.c. Filling the matrix
"""

# ╔═╡ efc01aad-89b7-4a2d-9727-a24b60046c57
md"""
The matrix would have entries of dimension *Energy*, but for computational speed, we will just take unit-less complex numbers. So we need to `ustrip`, i.e., strip the unit:
"""

# ╔═╡ 086523ac-30d2-11eb-239e-d7c121b6057e
function make_matrix(N ::Int ; E⁺::Energy, E₋::Energy, w::Length) :: Matrix{ℂ}
	@assert N ≥ 2

	M = Matrix{ℂ}(undef,N,N)

	for k = 1 : N
		for ℓ = 1 : N
			M[k,ℓ] = ustrip(ℂ, eV, ( (x->ϕ(x;k=k)) | H(ℓ ; w,E⁺,E₋) ))
		end
	end

	return M
end ;

# ╔═╡ 7b65f758-30d4-11eb-36b9-19da0fb76384
let M = make_matrix(2 ; w=L/10, E⁺=E(7), E₋=E(3))
	M
end

# ╔═╡ dfaf0a1c-273b-11eb-2efc-397e49b58331
md"""
## 3. The numerical work

We delegate finding the eigenvectors and eigenvalues of our matrix to Julia.
"""

# ╔═╡ 38223522-2749-11eb-3bd6-87f48071f8ae
md"""
Enter the number of basis elements:
𝑁 = $( @bind N NumberField(4:1000 ; default=10) )
"""

# ╔═╡ e8af2840-30f0-11eb-39b2-77447cd7d307
md"""
Choose the width of the potential energy barrier: $( @bind _w NumberField( 1 : 999 ; default=100) )‰ of width of well
"""

# ╔═╡ 6dccd70c-30f1-11eb-3a61-011a57f16c4f
begin
	w = _w/1000⋅L
md"w = $( _w/1000 ⋅ L )"
end

# ╔═╡ 82af2b2a-30f1-11eb-2438-1d9484a2f8c7
md"""
Choose the height of the potential energy barrier: $(
@bind n_E⁺ NumberField( 0 : 100 ; default=5) )
"""

# ╔═╡ bbec4332-30f1-11eb-1da3-b509990b6260
begin
	E⁺ = n_E⁺==0 ? 0.0 : E(n_E⁺)
	md"""
	𝐸⁺ = Energy of the $( n_E⁺ )th energy level of the particle in the simple well: $( E⁺ )
	"""
end

# ╔═╡ 11cf7f12-30f2-11eb-0cd7-bde453f549fb
md"""
Choose the level of the local minimum: $(
@bind n_E₋ NumberField( 1 : 100 ; default=1) )
"""

# ╔═╡ 287285fc-30f2-11eb-3426-ad94ec5786c3
begin
	E₋ = n_E₋==0 ? 0.0 : E(n_E₋)
	md"""
	𝐸₋ = 
	Energy of the $(  n_E₋ )th energy level of the particle in the simple well: $(E₋)
	"""
end

# ╔═╡ 48fdea46-30f2-11eb-1fcb-4532711a0de3
M = make_matrix(N ; w, E⁺, E₋) ;

# ╔═╡ df839cd6-273b-11eb-0f00-0bd3bd1c565e
md"""
The function `eigen()` computes the eigenvalues and eigenvectors of
a square matrix. See the [Julia docs](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Eigen).
"""


# ╔═╡ 449b68d4-273f-11eb-283e-d506d893c613
e = eigen(M) ;

# ╔═╡ 2ae6223c-056c-4a0d-9e3a-33abc2f0cc54
λ = [ real( e.values[ℓ] )⋅eV for ℓ=1:N ]

# ╔═╡ 2f16b9be-273d-11eb-322e-d16319a95cfe
md"""
Just a little sanity check: Spectral decomposition: $\| M - U \, \texttt{Diag}(\vec\lambda) \, U^\dagger\|=$
$( norm( M - e.vectors⋅Diagonal(e.values)⋅e.vectors' ) )
"""

# ╔═╡ f1a23cf0-2743-11eb-04bc-93a51e917a5c
md"""
### 3.a. Energy levels

Helper function: Evaluate the function defined by the coefficients
$\gamma$ at point $x$.
"""

# ╔═╡ 7af2b274-30f8-11eb-37ca-130467aec49a
ψ(x ; γ::Vector{ℂ}) = ∑( γ[k]⋅ϕ(x;k)  for k=1:N )   ;

# ╔═╡ 0cb64324-30f9-11eb-01f9-b5d257d769d0
md"""
Choose an energy level: ``\ell =`` $( @bind lvl NumberField(1:N ; default=1) )
"""

# ╔═╡ b6803f27-0f9d-4a31-b314-17da3dab429c
md"""
* ``\ell=`` $(lvl)
* ``E= `` $( λ[lvl] )
"""

# ╔═╡ dab35bd2-30f8-11eb-3c5e-ebe1b309362d
let plt = plot([x for x = L/2000 : L/1000 : L ],

	 x  ->  real( ψ(x ; γ=e.vectors[:,lvl]) );

	 label="Re", color="blue")
	
	plot!(plt, [x for x = L/2000 : L/1000 : L ],

	 x  ->  imag( ψ(x ; γ=e.vectors[:,lvl]) );

	 label="Im", color="green")	
end

# ╔═╡ 8b6078e0-325e-487c-9526-1d4f2a28dd70
md"Probability of being left (blue), righ (green) or on top of (orange) the barrier:"

# ╔═╡ 2aec02ca-260e-4cc4-8c2d-65b1b80d93a7
let
	p(x)   = abs2( ψ(x;γ=e.vectors[:,lvl]) )
	left   = quadgk(p, 0nm,     (L-w)/2)[1]
	right  = quadgk(p, (L+w)/2, L      )[1]
	middle = quadgk(p, (L-w)/2, (L+w)/2)[1]

	pie( [ left, middle, right ] )
end

# ╔═╡ fadaf664-30f2-11eb-20f9-eb2bffc48b6a
plot([x for x = 0nm : L/1001 : L ],

	 x  ->  V(x ; w, E⁺, E₋)  ;

	 label="", xaxis="x", yaxis="Potential energy ", color="red")

# ╔═╡ 165b3351-4264-4ff8-a44e-cd11d7203c87
md"""
### 3.b. Preparations for time evolution
"""

# ╔═╡ e70648a1-b7ea-45ac-99b2-9771886d2824
md"""
Let's put all probability density on a point in the higher-energy low plateau to the right of the barrier. We approximate a Dirac-function with the $(N) basis functions that we have.  Here is the vector of coefficients:
"""

# ╔═╡ 9067039c-77cd-4f9c-a6e1-ff6af1394368
begin
	δ_interval = 90L/100..92L/100
	δ_height   = √(1/(δ_interval.right - δ_interval.left))
	δ          = ℂ[  ∫( x->δ_height⋅ϕ(x;k) ; I=δ_interval )[1]   for k=1:N ]
end

# ╔═╡ 6926923a-c08b-4ab3-a39e-383d794e6937
let plt = plot([x for x = L/2000 : L/1000 : L ],

	 x  ->  real( ψ(x ; γ=δ) );

	 label="", color="blue")
	
	plot!(plt, [x for x = L/2000 : L/1000 : L ],

	 x  ->  imag( ψ(x ; γ=δ) );

	 label="", color="green")	
end

# ╔═╡ 76557c1d-c491-4f8a-a300-6c30e9bbd570
md"""
Let's see how well we're approximating our ``\delta`` with our ``N=`` $(N) basis elements:
* The norm of the projection is ``\lVert\mathrm{Prj}\; δ \rVert =`` $(norm(δ)) ``< 1 = \lVert\delta\rVert``.
* In other words, the approximation error is $( round( (1-norm(δ))*1000 )/10 )%.
"""

# ╔═╡ 846ad4fb-bd77-4e06-9681-e7ba282283a3
md"""
To realize time evolution, as we have already diagonalized the Hamiltonian, we'll just write our $\delta$-function (approximation) as a linear combination of eigenstates of the Hamiltonian.  Here are the coefficients:
"""

# ╔═╡ c6ba5644-e449-4da7-a8f0-7adba484854b
δᴴ = ℂ[ e.vectors[:,ℓ]'⋅δ  for ℓ=1:N ]

# ╔═╡ 9582d969-7e12-4178-bf60-983f75d7c402
md"""
For going back from ``H``-eigenstate coefficients to position basis, we make another function — like ``\psi`` above, just for the ``H``-basis:
"""

# ╔═╡ c59c8bc7-50f3-42d4-b77d-41bcb20925e7
ξ( x ; β::Vector{ℂ}) = ∑( ∑( β[ℓ]⋅e.vectors[k,ℓ]⋅ϕ(x;k) for ℓ=1:N ) for k=1:N )   ;

# ╔═╡ 31414485-eea3-478d-95e3-60e02a750f60
md"""
The moment of truth:
"""

# ╔═╡ 6f0e9a87-2b85-4d21-9025-95f6d9b3850d
md"""
The energy of our approximated ``\delta``-function is: $(
	∑( abs²(δᴴ[ℓ])⋅λ[ℓ]  for ℓ=1:N )
)
"""

# ╔═╡ fa54d54f-457f-4939-b28d-bf47072c3386
md"""
### 3.c. Let's tunnel!!
"""

# ╔═╡ 6ef8ebbd-79ad-49e8-a829-ff26b60f4d83
md"""
Duration: $(@bind _𝑡 Slider(0:0.1:100) )
"""

# ╔═╡ d148aef9-a4ab-40db-9e1d-f504f325d9f5
md"𝑡 = $(𝑡=_𝑡⋅fs)"

# ╔═╡ 85ee052e-b29b-42eb-8dd7-f6da93d6163c
ξ( x::Length, 𝑡::Time
	; β::Vector{ℂ} ) =
	∑( β[ℓ]⋅exp(-𝐢⋅λ[ℓ]⋅𝑡/ħ)⋅e.vectors[k,ℓ]⋅ϕ(x;k) for ℓ=1:N, k=1:N )   ;

# ╔═╡ 066481a9-f02a-4e63-8929-d5022da59df6
let plt = plot([x for x = L/2000 : L/1000 : L ],

			x  ->  real( ξ(x ; β=δᴴ) )
			;
			label="", color="blue")
	
	plot!(plt, [x for x = L/2000 : L/1000 : L ],

	 		x  ->  imag( ξ(x ; β=δᴴ) )
			;
			label="", color="green")
	
	plot!(plt, [x for x = L/2000 : L/1000 : L ],
		
			x -> ( x ∈ δ_interval ? δ_height : 0.0/√nm )
			;
			label="", color="black")
end

# ╔═╡ 8201594d-7ac6-4b84-a473-af8c71211be8
plot([x for x = L/2000 : L/1000 : L ],

	 x  ->  abs²( ξ(x,𝑡 ; β=δᴴ) );

	 label="Prb", color="black")

# ╔═╡ a7465ed5-bcd1-4ccc-8204-a3ef778a1f32
md"Probability of being left (blue), righ (green) or on top of (orange) the barrier:"

# ╔═╡ d6db5c1e-a53e-4bf9-a43c-bc8f8b66070a
let
	p(x)   = abs²( ξ(x,𝑡;β=δᴴ) )
	left   = quadgk(p, 0nm,     (L-w)/2)[1]
	right  = quadgk(p, (L+w)/2, L      )[1]
	middle = quadgk(p, (L-w)/2, (L+w)/2)[1]

	pie( [ left, middle, right ] )
end

# ╔═╡ a3c896e9-1d33-4e4d-9507-08473597d3d6
plot([x for x = 0nm : L/1001 : L ],

	 x  ->  V(x ; w, E⁺, E₋)  ;

	 label="", xaxis="x", yaxis="Potential energy ", color="red")

# ╔═╡ d2a4fd0c-f371-11ea-0119-ad1e498314cf
md"# Done!"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuadGK = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[compat]
IntervalSets = "~0.7.4"
Plots = "~1.35.5"
PlutoUI = "~0.7.49"
QuadGK = "~2.6.0"
Unitful = "~1.12.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.3"
manifest_format = "2.0"
project_hash = "15cf5f3489364eb5862f166716f437b761a55d15"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "84259bb6172806304b9101094a7cc4bc6f56dbc6"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "3ca828fe1b75fa84b021a7860bd039eaea84d2f2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.3.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "46d2680e618f8abd007bce0c3026cb0c4a8f2032"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.12.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "c36550cb29cbe373e95b3f40486b9a4148f89ffd"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.2"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

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

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "00a9d4abadc05b9476e937a5557fcce476b9e547"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.69.5"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "bc9f7725571ddb4ab2c4bc74fa397c1c5ad08943"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.69.1+0"

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
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "a97d47758e933cd5fe5ea181d178936a9fc60427"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.5.1"

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
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "16c0cc91853084cb5f58a78bd209513900206ce6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.4"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

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
git-tree-sha1 = "ab9aa169d2160129beb241cb2750ca499b4e90e9"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.17"

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
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

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
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

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
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

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
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "3c3c4a401d267b04942545b1e964a20279587fd7"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

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
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "6c01a9b494f6d2a9fc180a08b182fcb06f0958a0"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.2"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "21303256d239f6b484977314674aef4bb1fe4420"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "0a56829d264eb1bc910cf7c39ac008b5bcb5a0d9"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.35.5"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eadad7b14cf046de6eb41f13c9275e5aa2711ab6"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.49"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

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
git-tree-sha1 = "97aa253e65b784fd13e83774cadc95b38011d734"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.6.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "d12e612bba40d189cead6ff857ddb67bd2e6a387"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "9b1c0c8e9188950e66fc28f40bfe0f8aac311fe0"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.7"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

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
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

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

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

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
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d57a4ed70b6f9ff1da6719f5f2713706d57e0d66"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.12.0"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

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
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

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
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

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
# ╟─a13db58c-f340-11ea-0fa3-df05c859e0bd
# ╟─9a6007d0-30c4-11eb-20d8-e9e8f7ebf7fb
# ╟─66b9d55e-25a3-11eb-3096-19dd66316679
# ╟─e06d3888-25a2-11eb-167a-47a2eeda7e39
# ╠═b0d036dd-e9b9-4bb6-9a89-6ef70950e659
# ╠═3aaa19a7-69a3-465b-b921-3aa0caeee397
# ╠═093eeacf-28a9-4818-b27d-881729fc1f1a
# ╟─9258c5ca-7638-4321-8c35-88d35e48681f
# ╟─78f4e4a5-4d9d-4766-9e3c-c20b0fa79e3b
# ╟─8036d94a-25b8-11eb-3b93-350a6274ab6b
# ╟─0277c3cc-30c7-11eb-0c92-6325a0cdf19d
# ╠═19e9be4a-30c7-11eb-3081-d3886a1e5066
# ╟─1fb9e8b2-30c8-11eb-1eae-5f6a8ec2125d
# ╟─611eb7c4-30c8-11eb-1e65-8b0519d09001
# ╟─aec436c4-30c9-11eb-31e4-7185257f6029
# ╠═c2c606fa-30cb-11eb-3cf8-d182760c9f2f
# ╟─a88141e6-30cc-11eb-2fc3-4bb8ccd5d5ac
# ╟─f0ab0a48-30cb-11eb-3aef-81e84f1226f1
# ╟─b0c2c612-30cd-11eb-1e1d-09ddf529a0d1
# ╟─aef58bbe-30cb-11eb-26b0-b7e50a3c694e
# ╟─febedade-30cf-11eb-2943-f72544183cbd
# ╠═e6c48838-30d0-11eb-05aa-29ad654daa2f
# ╠═e28f810e-bd0f-48b9-b6ce-699e89cc40d4
# ╟─d1a17ab2-4883-4089-ae8f-525561a522ee
# ╠═b2ecba78-30ec-11eb-2680-d906db765a1f
# ╟─9d498fc2-30d1-11eb-1df6-2d48282ff416
# ╠═8f585196-30d1-11eb-2c2f-7399ec1c5ece
# ╟─0a001c28-30d5-11eb-14a8-e18d1200c783
# ╟─e51395f0-30d4-11eb-08b1-b9495f770b11
# ╟─8b44eaf0-30d7-11eb-3156-9f9bd82b97de
# ╟─92d86576-30d2-11eb-32ee-a9811228efce
# ╠═550d13ac-30ed-11eb-3b34-fb0a048b1827
# ╟─e877eef9-baf2-45da-a65a-1e4bdb477a2e
# ╠═c8f5ad2a-30c4-11eb-05e3-9faee17cd60c
# ╟─62b547ee-30c6-11eb-008c-5f1410db2544
# ╟─80c28c1c-30dd-11eb-06d8-6f7fad08b7c1
# ╠═a0a88398-30d2-11eb-3ed9-f5483557b145
# ╟─5051be2c-30d8-11eb-0790-77ff1f07c5cc
# ╟─b23a2ef2-30ed-11eb-00b3-dbb88552f074
# ╟─f9b86b0c-30d1-11eb-2bd7-5501d4dc02c5
# ╟─efc01aad-89b7-4a2d-9727-a24b60046c57
# ╠═086523ac-30d2-11eb-239e-d7c121b6057e
# ╠═7b65f758-30d4-11eb-36b9-19da0fb76384
# ╟─dfaf0a1c-273b-11eb-2efc-397e49b58331
# ╟─df9c4558-273b-11eb-2796-67d82da18f10
# ╟─38223522-2749-11eb-3bd6-87f48071f8ae
# ╟─e8af2840-30f0-11eb-39b2-77447cd7d307
# ╟─6dccd70c-30f1-11eb-3a61-011a57f16c4f
# ╟─82af2b2a-30f1-11eb-2438-1d9484a2f8c7
# ╟─bbec4332-30f1-11eb-1da3-b509990b6260
# ╟─11cf7f12-30f2-11eb-0cd7-bde453f549fb
# ╟─287285fc-30f2-11eb-3426-ad94ec5786c3
# ╠═48fdea46-30f2-11eb-1fcb-4532711a0de3
# ╟─df839cd6-273b-11eb-0f00-0bd3bd1c565e
# ╠═449b68d4-273f-11eb-283e-d506d893c613
# ╠═2ae6223c-056c-4a0d-9e3a-33abc2f0cc54
# ╟─2f16b9be-273d-11eb-322e-d16319a95cfe
# ╟─f1a23cf0-2743-11eb-04bc-93a51e917a5c
# ╠═7af2b274-30f8-11eb-37ca-130467aec49a
# ╟─0cb64324-30f9-11eb-01f9-b5d257d769d0
# ╟─b6803f27-0f9d-4a31-b314-17da3dab429c
# ╟─dab35bd2-30f8-11eb-3c5e-ebe1b309362d
# ╟─8b6078e0-325e-487c-9526-1d4f2a28dd70
# ╟─2aec02ca-260e-4cc4-8c2d-65b1b80d93a7
# ╟─fadaf664-30f2-11eb-20f9-eb2bffc48b6a
# ╟─165b3351-4264-4ff8-a44e-cd11d7203c87
# ╟─e70648a1-b7ea-45ac-99b2-9771886d2824
# ╠═9067039c-77cd-4f9c-a6e1-ff6af1394368
# ╟─6926923a-c08b-4ab3-a39e-383d794e6937
# ╟─76557c1d-c491-4f8a-a300-6c30e9bbd570
# ╟─846ad4fb-bd77-4e06-9681-e7ba282283a3
# ╠═c6ba5644-e449-4da7-a8f0-7adba484854b
# ╟─9582d969-7e12-4178-bf60-983f75d7c402
# ╠═c59c8bc7-50f3-42d4-b77d-41bcb20925e7
# ╟─31414485-eea3-478d-95e3-60e02a750f60
# ╟─066481a9-f02a-4e63-8929-d5022da59df6
# ╟─6f0e9a87-2b85-4d21-9025-95f6d9b3850d
# ╟─fa54d54f-457f-4939-b28d-bf47072c3386
# ╟─6ef8ebbd-79ad-49e8-a829-ff26b60f4d83
# ╟─d148aef9-a4ab-40db-9e1d-f504f325d9f5
# ╠═85ee052e-b29b-42eb-8dd7-f6da93d6163c
# ╟─8201594d-7ac6-4b84-a473-af8c71211be8
# ╟─a7465ed5-bcd1-4ccc-8204-a3ef778a1f32
# ╟─d6db5c1e-a53e-4bf9-a43c-bc8f8b66070a
# ╟─a3c896e9-1d33-4e4d-9507-08473597d3d6
# ╟─d2a4fd0c-f371-11ea-0119-ad1e498314cf
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
