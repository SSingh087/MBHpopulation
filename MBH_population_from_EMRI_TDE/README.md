```mermaid
flowchart TD
  classDef input fill:#1f2937,color:#fff,stroke:#111;
  classDef process fill:#3b82f6,color:#fff,stroke:#1e40af;
  classDef decision fill:#f59e0b,color:#000,stroke:#b45309;

  A0["Inputs: z grid, GSMF tables, config params"]:::input

  subgraph POP[Galaxy population model]
    A1["GSMF\nlog10 φ(M*, z)"]:::process
    A2["Sample galaxies\n(z_gal, log10 M*)"]:::process
    A3["Check nucleation\np_NSC = max(models)"]:::decision
    A4["Galaxy object\nR_eff → σ → M_BH"]:::process
  end

  A0 --> A1 --> A2 --> A3 --> A4
```


```mermaid
flowchart TD
  classDef process fill:#10b981,color:#000,stroke:#065f46;
  classDef decision fill:#f59e0b,color:#000,stroke:#b45309;
  classDef highlight fill:#6366f1,color:#fff,stroke:#312e81;

  B1["NSC(galaxy)\nM_BH fixed"]:::process
  B2["r_infl = G M_BH / σ^2"]:::process
  B3["r_a = 4 r_infl"]:::process
  B4{"Channel?"}:::decision

  B5["EMRI cutoff\nr_k = 8GM/c^2"]:::highlight
  B6["TDE cutoff\nr_k ≈ tidal radius"]:::highlight

  C1["Dehnen profile n(r)"]:::process
  C2["ρ(r_infl)"]:::process

  B1 --> B2 --> B3 --> B4
  B4 -->|EMRI| B5 --> C1
  B4 -->|TDE|  B6 --> C1
  C1 --> C2
```


```mermaid
flowchart TD
  classDef process fill:#ef4444,color:#fff,stroke:#7f1d1d;
  classDef time fill:#14b8a6,color:#000,stroke:#0f766e;

  D1["Relaxation time\n t_rlx ∝ σ^3/(G^2 m̄ ρ lnΛ)"]:::process
  D2["Cosmology\nage(z), dt/dz"]:::time
  D3["Last Major Merger\nsample z_LMM"]:::process

  D4["t_LMM, t_obs"]:::time
  D5["t_on = t_LMM + κ t_rlx"]:::process
  D6["Cusp age\nT_c = max(0, t_obs - t_on)"]:::process

  D1 --> D5
  D2 --> D3 --> D4 --> D5 --> D6
```

```mermaid
flowchart TD
  classDef process fill:#8b5cf6,color:#fff,stroke:#4c1d95;
  classDef math fill:#eab308,color:#000,stroke:#854d0e;

  E1["RateModel\nΓ̂(M_BH,σ), t_peak"]:::process
  E2["τ = T_c / t_peak"]:::math
  E3["S_x(τ)\nEMRI: beta\nTDE: placeholder"]:::process
  E4["C(τ) = ∫ S_x"]:::math
  E5["N_expected = Γ̂ · t_peak · C(τ)"]:::process

  E1 --> E2 --> E3 --> E4 --> E5
```

```mermaid
flowchart LR
  classDef block fill:#374151,color:#fff,stroke:#111;

  A["Inputs"]:::block
  B["Galaxy Population"]:::block
  C["NSC Structure"]:::block
  D["Cusp Evolution"]:::block
  E["Rates"]:::block

  A --> B --> C --> D --> E
```