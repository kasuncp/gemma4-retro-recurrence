# Paper Banana prompt + source mermaid

**Source:** `papers/four-paths-one-phone-visual-guide.md`
**Target tool:** Paper Banana (illustrated-diagram generator)

---

## Prompt for Paper Banana

> Generate a set of 9 illustrated technical diagrams in a single consistent house style for a blog post titled **"A Visual Guide to Test-Time Compute on a Phone."** The post is a map of four candidate mechanisms for making a 2B-class on-device language model (Gemma 4 E2B) "think harder" under a fixed phone-class compute budget.
>
> **Style requirements (apply to every diagram):**
> - Clean, editorial, hand-illustrated feel — closer to an explainer in *Distill* or *The Pudding* than a PowerPoint flowchart.
> - Warm pastel palette: soft blue for datacenter/GPU/frontier, soft peach/orange for phone/CPU/on-device, muted green for "good / baseline / survivable," muted red for "broken / collapse," pale yellow for boundaries and ambiguity.
> - Label every box in legible sans-serif; keep arrow directionality obvious; avoid clutter.
> - Use simple, recognizable icons (a phone silhouette, a server rack, a brain, a chain loop, a routing switch) instead of generic rectangles where it helps the metaphor.
> - Horizontal layout where the mermaid source is LR; vertical layout where it is TB.
> - Each diagram is standalone and reads left-to-right or top-to-bottom without a legend.
>
> Render the nine diagrams below, preserving labels and arrow semantics exactly. Titles are for your reference; do not draw them inside the diagram unless specified.

---

## Diagram 1 — The gap between frontier and phone (LR)

```mermaid
flowchart LR
    F["🧠 Frontier model<br/>400B+ parameters<br/>Datacenter GPUs<br/>Excellent reasoning<br/>Reachable only via API"]:::frontier
    G["❓ The gap<br/>What goes here?"]:::gap
    P["📱 Your phone<br/>~6 GB usable memory<br/>4–8 W sustained<br/>2B-class model<br/>Limited reasoning"]:::phone
    F --- G
    G --- P
    classDef frontier fill:#c9ddff,stroke:#1f4aa0,color:#0b2040
    classDef phone fill:#ffd9b5,stroke:#a35a12,color:#3f1f00
    classDef gap fill:#fff3b0,stroke:#9c7a00,color:#2e2400
```

## Diagram 2 — Two-stage evaluation pipeline (LR)

```mermaid
flowchart LR
    S1["🖥️ Stage 1<br/>Datacenter GPU<br/>sweep configurations,<br/>measure cleanly"]:::gpu
    S1 --> C{"Worth<br/>validating<br/>on silicon?"}
    C -->|"yes"| S2["📱 Stage 2<br/>On-device measurement<br/>(real phone)"]:::phone
    C -->|"no"| X["📊 Report GPU-only<br/>numbers, move on"]:::gray
    classDef gpu fill:#d8e4ff,stroke:#1f4aa0,color:#0b2040
    classDef phone fill:#ffd9b5,stroke:#a35a12,color:#3f1f00
    classDef gray fill:#ececec,stroke:#666,color:#333
```

## Diagram 3 — Gemma 4 E2B with Per-Layer Embeddings (TB)

```mermaid
flowchart TB
    subgraph ACC["📱 Accelerator (GPU / NPU) — ~2 GB"]
        W["Core transformer weights<br/>(~2 B active parameters)"]
    end
    subgraph CPU["🧮 CPU RAM"]
        E["Per-Layer Embeddings (PLE)<br/>(the rest of ~2.3 B)"]
    end
    E -.->|"streamed per layer"| W
```

## Diagram 4 — Multimodal inputs into Gemma 4 E2B (LR)

```mermaid
flowchart LR
    T["📝 Text"] --> G
    I["🖼️ Image"] --> V["MobileNet-V5-300M<br/>vision encoder"]
    V2["🎥 Video"] --> V
    V --> G
    A["🔊 Audio"] --> U["USM audio encoder<br/>~6 tokens/sec"]
    U --> G
    G["Gemma 4 E2B"] --> OUT["📄 Text output"]
```

## Diagram 5 — The four paths and their taxes (LR)

```mermaid
flowchart LR
    B["🟢 Gemma 4 E2B<br/>baseline"]:::base
    B --> L1["Path 1<br/>📝 More CoT tokens<br/>longer generation"]:::path
    B --> L2["Path 2<br/>🔁 Depth-recurrence retrofit<br/>loop middle layers r times"]:::path
    B --> L3["Path 3<br/>📦 Quantized sibling<br/>Gemma 4 E4B at INT4"]:::path
    B --> L4["Path 4<br/>🎚️ Mixture-of-Depths<br/>per-token routing"]:::path
    L1 --> T1["tax: latency + energy"]:::tax
    L2 --> T2["tax: training + surgery"]:::tax
    L3 --> T3["tax: quantization loss"]:::tax
    L4 --> T4["tax: router training"]:::tax
    classDef base fill:#b7e4c7,stroke:#2d6a4f,color:#1a4a2a
    classDef path fill:#cfe2f3,stroke:#1f4aa0,color:#0b2040
    classDef tax fill:#fde2e2,stroke:#9c1b1b,color:#3b0808
```

## Diagram 6 — Path 2 depth-recurrence architecture (LR)

```mermaid
flowchart LR
    X["input tokens x"] --> P["Prelude P<br/>(early layers)"]
    P --> E["embeddings e"]
    E --> A["🔀 adapter<br/>ℝ²ʰ → ℝʰ"]
    S0["s₀ ~ 𝒩(0,σ²)<br/>(random init state)"] --> A
    A --> R["🔁 Recurrent block R<br/>(middle layers)<br/>run r times"]
    R -->|"sᵢ (state out)"| A
    R --> C["Coda C<br/>(late layers)"]
    C --> Y["output distribution"]
```

## Diagram 7 — Path 4 Mixture-of-Depths router (LR)

```mermaid
flowchart LR
    T["🔤 tokens at layer ℓ"] --> R["🎚️ router<br/>(score per token)"]
    R --> TK["top-k tokens"]:::hot
    R --> BP["bottom tokens"]:::cold
    TK --> AM["attention + MLP<br/>(full compute)"]
    BP --> SK["residual bypass<br/>(skip)"]
    AM --> M["merge"]
    SK --> M
    M --> NX["→ layer ℓ+1"]
    classDef hot fill:#ffcdb2,stroke:#b7451f,color:#5c2a0b
    classDef cold fill:#cfe2f3,stroke:#1f4aa0,color:#0b2040
```

## Diagram 8 — Shared measurement rig (LR)

```mermaid
flowchart LR
    M["📋 leg-specific<br/>model configuration"] --> H["🔧 shared eval harness"]
    H --> A1["ARC-Easy<br/>n = 500<br/>0-shot + 5-shot"]:::acc
    H --> A2["GSM8K<br/>n = 500<br/>8-shot CoT"]:::acc
    H --> C1["wall-clock<br/>tokens/sec"]:::compute
    H --> C2["analytical<br/>FLOPs/token"]:::compute
    H --> C3["peak VRAM<br/>(cuda.max_memory_allocated)"]:::compute
    A1 --> P["📈 Pareto plot:<br/>accuracy vs. compute budget"]:::result
    A2 --> P
    C1 --> P
    C2 --> P
    C3 --> P
    classDef acc fill:#d4edda,stroke:#155724,color:#0a3d1a
    classDef compute fill:#cce5ff,stroke:#004085,color:#0b2040
    classDef result fill:#fff3cd,stroke:#856404,color:#2e2400
```

## Diagram 9 — The loopable valley vs. the KV wall (TB)

```mermaid
flowchart TB
    A["Early layers — Prelude zone<br/>🚫 not a recurrence target"]:::zone
    B["Middle layers, above KV boundary<br/>💥 loop here → collapse"]:::bad
    C["⚡ KV producer / consumer boundary"]:::wall
    D["Middle layers, below KV boundary<br/>✅ the loopable valley"]:::good
    E["Late layers — Coda zone<br/>🚫 not a recurrence target"]:::zone
    A --> B
    B --> C
    C --> D
    D --> E
    classDef zone fill:#eeeeee,stroke:#888,color:#555
    classDef bad fill:#fde2e2,stroke:#9c1b1b,color:#3b0808
    classDef wall fill:#fff3b0,stroke:#9c7a00,color:#2e2400
    classDef good fill:#d4edda,stroke:#155724,color:#0a3d1a
```

---

## Per-diagram illustration hints (optional, for Paper Banana)

1. **The gap** — Three-panel strip: server rack ➜ fogged question-mark chasm ➜ phone. Emphasize visual distance between the two ends.
2. **Two-stage pipeline** — A wide GPU block feeding a diamond decision node; the "yes" branch exits to a small phone, the "no" branch exits to a clipboard icon.
3. **PLE architecture** — Stacked card: top card = accelerator (blue) holding transformer layers; bottom card = CPU RAM (peach) holding PLE blocks; dashed arrows showing per-layer streaming between them.
4. **Multimodal inputs** — Four input lanes (text, image, video, audio) converging through their encoders into a central Gemma brain icon, single text output on the right.
5. **Four paths** — Central green baseline node branching into four colored routes, each terminating in a small red "tax" tag. Keep the four routes visually parallel.
6. **Depth-recurrence** — Linear pipeline Prelude → Recurrent block → Coda, with a prominent self-loop around the recurrent block labeled "× r". Small adapter triangle at the block entrance fusing state + embedding.
7. **Mixture-of-Depths** — Tokens entering a router; two streams emerge (hot top-k through full compute, cold through a bypass arc); they reconverge before the next layer.
8. **Measurement rig** — Single funnel (harness) fed by a config card; fans out into two accuracy benchmarks and three compute metrics; all five flow into a Pareto plot thumbnail on the right.
9. **Loopable valley** — Vertical cross-section of the model: greyed prelude and coda zones at the ends, a red "collapse" band above a glowing yellow KV boundary, and a green "valley" band below it.
