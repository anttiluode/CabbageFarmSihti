# CabbageFarmSihti

**Finite example → learned stochastic transfer law → arbitrarily large new fields.**

This repo revisits the old [CabbageFarm](https://github.com/anttiluode/CabbageFarm) goal — continuous, effectively unbounded worlds from finite 2-D examples — from the opposite direction.

CabbageFarm tried to encode **one particular image/world** in a continuous coordinate network.

CabbageFarmSihti asks instead:

> **Can a finite texture reveal the multiscale stochastic law that makes fields of that kind, and can we apply that law to fresh coordinate-addressable noise?**

The immediate ancestry is:

\`\`\`text
CabbageFarm
    continuous coordinate field / infinite-query ambition
          \
           \
            CabbageFarmSihti
           /
          /
Sihti -> Sihti2
purifier / residue / transfer-law view
\`\`\`

The first version is deliberately conservative. It does **not** claim that Sihti residues magically solve texture synthesis. It starts with the strongest conventional attacker: a stationary Gaussian random field whose 2-D power spectrum is learned from the reference. Only after that baseline is measured do cross-octave / nonlinear Sihti relations earn a role.

## v0: learn the spectral law, not the pixels

For a stationary linear field

\`\`\`math
Y = H * \xi,
\`\`\`

with white noise \(\xi\),

\`\`\`math
S_Y(k)=|H(k)|^2 S_\xi(k)=|H(k)|^2.
\`\`\`

So the reference power spectrum gives a transfer law. CabbageFarmSihti fits that law in a PCA colour basis and interprets the normalized spectral power as a distribution over continuous spatial frequencies.

A new field is then generated as random Fourier features:

\`\`\`math
z_c(x)
=
\sqrt{\frac{2v_c}{M}}
\sum_{m=1}^{M}
\cos\!\left(2\pi k_{cm}\cdot x+\phi_{cm}\right),
\`\`\`

where

- \(v_c\) is the reference variance of PCA colour component \(c\),
- \(k_{cm}\) is sampled from that component's learned 2-D spectral density,
- \(\phi_{cm}\) is fresh random phase.

The frequencies are jittered continuously inside the finite reference's Fourier cells. The generated field is therefore defined directly at coordinates instead of as a repeated reference-sized tile.

That gives the first useful distinction from the old CabbageFarm:

\`\`\`text
CabbageFarm:
finite image -> function approximating that particular image

CabbageFarmSihti:
finite image -> stochastic law -> new fields drawn from that law
\`\`\`

## Coordinate-addressable means overlap must agree

A procedural field is not "infinite" merely because a program can make a huge bitmap.

For fixed model, seed and mode count,

\`\`\`python
crop(x=100, y=50, w=256, h=256)
\`\`\`

and

\`\`\`python
crop(x=228, y=50, w=256, h=256)
\`\`\`

must return identical values on their overlapping coordinates.

The test suite checks this directly. There is no hidden tile index in the generator.

## Why Sihti is still in the name

For a frozen purifier \(T\),

\`\`\`math
R_d=T^d(I-T^d)x.
\`\`\`

In one eigenmode \(T\phi=g\phi\),

\`\`\`math
R_d^{(\phi)}
=
c\,g^d(1-g^d)\phi.
\`\`\`

So bleed-out noise is not just "detail." It is a band-pass measurement over **relaxation lifetime**.

Adjacent octave residues are deterministically related:

\`\`\`math
R_{2d}=T^d(I+T^d)R_d.
\`\`\`

That suggests a richer generator than plain spectral matching:

\`\`\`math
P(R_{2d}\mid R_d),
\`\`\`

or a learned scale-transfer operator

\`\`\`math
R_{2d}\approx A_dR_d.
\`\`\`

But there is an important kill condition:

> For a stationary Gaussian linear process, octave-energy and linear cross-octave statistics are already consequences of the power spectrum.

So v0 records a Gaussian scale-space residue signature but does **not** count matching it as evidence that Sihti added information beyond spectral matching.

The next mechanism is only earned if real textures contain cross-scale relations that a phase-randomized, power-spectrum-matched field fails to reproduce.

## What "shape of the structure" means here

The progression is deliberately staged.

### Level 0 — ordinary spectrum

Learn

\`\`\`math
|H(k_x,k_y)|.
\`\`\`

This can already encode:

- characteristic feature sizes,
- anisotropy,
- preferred directions,
- broad \(1/f^\beta\)-like scaling.

It cannot generally encode the phase organization that distinguishes veins, sparse edges, branching, repeated motifs or coherent local geometry.

### Level 1 — Sihti/scale-space lifetime fingerprint

Measure

\`\`\`math
R_1,R_2,R_4,R_8,\ldots
\`\`\`

and ask how much energy dies at each scale.

### Level 2 — cross-octave relations

Ask whether structure at one lifetime predicts structure at the next:

\`\`\`math
P(R_{2d}\mid R_d).
\`\`\`

Now the object is not one band but **how bands develop into one another**.

### Level 3 — local/state-dependent transfer

Replace one global law with something like

\`\`\`math
H(x,y,k_x,k_y),
\`\`\`

or an operator written by a slower field.

That is where curved wood grain, mineral domains, branching terrain or other nonstationary material structure could live.

## Run

Install:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Fit a reference:

\`\`\`bash
python cabbage_sihti.py fit reference.jpg model.json --max-size 256 --max-bins 4096
\`\`\`

Generate a fresh field:

\`\`\`bash
python cabbage_sihti.py synth model.json generated.png \
  --width 768 --height 512 --seed 7 --modes 768
\`\`\`

Generate another crop from the **same global field**:

\`\`\`bash
python cabbage_sihti.py synth model.json crop.png \
  --width 512 --height 512 \
  --origin-x 10000 --origin-y -3000 \
  --seed 7 --modes 768
\`\`\`

Compare a generated sample with the reference:

\`\`\`bash
python cabbage_sihti.py compare reference.jpg generated.png
\`\`\`

The comparison reports:

- log radial power-spectrum error,
- gradient anisotropy,
- colour covariance,
- Gaussian scale-space residue energy by octave,
- adjacent residue correlation.

## The first real experiment

Use references from qualitatively different material families:

\`\`\`text
granite
wood
cloud
terrain
\`\`\`

Fit each law and apply all four to fresh noise.

The first question is not whether the output "looks cool." It is:

> **Which measurable structure survives a power-spectrum-only generator, and which reference statistics refuse to transfer without cross-octave relations?**

A useful matrix is:

\`\`\`text
                        PSD      octave energy    cross-octave spatial relation
Gaussian texture        pass         pass                 pass/irrelevant
granite                 ?            ?                    ?
wood                    ?            ?                    ?
cloud                   ?            ?                    ?
terrain                 ?            ?                    ?
phase-scrambled ref     pass         pass                 should expose the limit
\`\`\`

If spectrum alone works, that is a successful simple generator and Sihti adds no mechanism yet.

If spectrum matches while recognizable structure and cross-octave spatial statistics fail, we have a clean reason to add the next layer.

## Why this is not "statistically indistinguishable" yet

Matching a power spectrum fixes second-order statistics of a stationary Gaussian field. Natural textures can depend on much more:

- phase relationships,
- non-Gaussian marginal statistics,
- sparse boundaries,
- orientation fields,
- cross-scale dependencies,
- spatially varying statistics.

Two images can share Fourier magnitude and look dramatically different.

That is not a nuisance; it is the experimental ladder for this repo.

## Files

\`\`\`text
cabbage_sihti.py              fit / synth / compare CLI
tests/test_cabbage_sihti.py   coordinate and transfer-law invariants
index.html                    browser explanation + procedural field viewer
requirements.txt
\`\`\`

## North star

The old Cabbage question was:

> can a small representation pretend to contain an infinite world?

This branch asks a more physical question:

\`\`\`text
finite example
      |
      v
what stochastic law could have produced fields like this?
      |
      v
fresh coordinate-addressable noise
      |
      v
new arbitrarily large field with the learned multiscale character
\`\`\`

The reference contributes the law.

Noise contributes the novelty.

The residue tells us what the simple law still failed to explain.
