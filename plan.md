Conditional microstructure generation from simulation metrics (RS, …) using geometric graph diffusion + sparse edge
  synthesis

  ## Summary

  Build a conditional generative pipeline that outputs (nodes, edges) for a tessellation-style geometric graph,
  conditioned on target simulation metrics (RS + optional other columns) and user-specified RD. Use:

  1. a strong forward surrogate Sφ(G, RD) → y (y includes RS and any other metrics you want to condition on),
  2. a node-coordinate generator (diffusion on 2D point sets with variable N),
  3. a sparse edge generator (edge sampling over local candidate pairs with hard validity constraints),
  4. sample-and-select (generate K candidates, score with the surrogate, return the top M).

  This avoids the scaling pain of fully autoregressive “edge-by-edge until stop token” generation for 500–1800 node
  graphs, while still supporting diversity and controllable constraints.

  ## Data and representations (fixed decisions)

  ### Graph format (from your files)

  - Node_n.txt: lines node_id, x, y where node_id is 1..N, (x,y) in ~[0.15, 0.85].
  - Connection_n.txt: lines edge_id, u, v where u,v reference node ids (undirected, no duplicates/self-loops).

  ### Training tuples

  For each tessellation index n and each RD level j:

  - Input: G_n = (V_n, E_n, coords_n), RD_j, target metrics y_{n,j} (at minimum RS; recommended: include additional
    columns you will have at inference).
  - Note: G_n is shared across the 5 RD rows; only RD_j and y_{n,j} change.

  ### Conditions you feed the generator

  - c := concat( RD, y_target )
      - y_target is the vector of metrics you want to match (at minimum RS; optionally a subset of the other columns
        you said you can condition on).
      - Use log(RS) (and log for any heavy-tailed positive metrics) for numerical stability.

  ## Model components (fixed decisions)

  ### 1) Forward surrogate (mandatory): Sφ(G, RD) → y

  Purpose: (a) evaluate generated graphs, (b) enable guided selection, (c) provide an acceptance test without running
  the expensive physics sim.

  - Backbone: SE(2)-equivariant message passing (EGNN-style) on the given edges, with node features [x,y], edge
    features [distance, Δx, Δy].
  - Global pooling: attention pooling or mean+max; concatenate RD embedding; MLP head to predict y.
  - Loss: Huber or MSE on transformed targets (e.g., log(RS)), plus per-metric weighting.

  ### 2) Node generator: conditional diffusion on point sets Pθ(coords | c, N)

  Goal: generate N 2D points (unordered set) in the domain.

  - First, predict node count:
      - N ~ pψ(N | c) as a categorical distribution over observed N (bucket if too many unique Ns).
  - Then generate coordinates with a set/point-cloud diffusion:
      - Forward: Gaussian noise on coordinates.
      - Denoiser network: kNN graph (k=12) built on current noisy coords each step; run an SE(2)-equivariant GNN;
        inject condition c via FiLM or cross-attention to a condition token.
  - Output: denoised coords in normalized domain; clamp/penalize outside bounds.

  Why diffusion here:

  - Parallel generation scales well to N≈1800; avoids O(N²) autoregressive edge-decoding.
  - Naturally supports multiple samples per condition.

  (Reference direction: discrete graph diffusion exists and is strong for adjacency generation—e.g., DiGress.
  citeturn0search1 But for your case with large geometric graphs, splitting coords vs edges is simpler and more
  scalable.)

  ### 3) Edge generator: sparse conditional edge sampling Pω(E | coords, c)

  Goal: output an undirected sparse edge list with degree constraints.

  - Candidate edge set construction (deterministic):
      - Build kNN candidates on generated coords with k=12 (or radius graph).
      - Consider only (u,v) in this candidate set (keeps O(Nk) complexity).
  - Edge model:
      - Run an EGNN over the candidate neighborhood graph to produce edge logits ℓ_uv.
      - Train with BCE on real edges vs sampled negatives from the same candidate set.
  - Hard validity constraints at sampling time:
      - No self-loops; undirected unique edges.
      - Enforce max degree (empirically your data is ≤6): for each node keep top-6 incident edges by probability, then
        symmetrize (mutual-or rule).
      - Enforce connectivity: if disconnected, add highest-prob edges between components until connected (or reject
        sample).

  This gives you edges that “look like” your dataset: sparse, local, bounded-degree, and geometry-consistent.

  ### 4) Sampling + selection loop (what you return)

  Given (RD_fixed, y_target):

  1. Sample K candidates:
      - sample N from pψ(N|c)
      - sample coords via diffusion
      - sample edges via edge model + constraints
  2. Score each candidate with the surrogate:
      - score = || Sφ(G, RD_fixed) - y_target || (in transformed space)
  This matches your “multiple samples” preference and avoids needing differentiable guidance through discrete edges on

  - Condition on RS + other metrics you expect to actually have at inference time.
  - If later you only truly have RS, the same framework works, but you should expect much higher diversity/
    uncertainty.

  ## Baselines to implement first (for de-risking)

  1. Train only the forward surrogate Sφ and verify it predicts RS accurately from real graphs + RD.
  2. Train the edge model on real coords (no node diffusion yet) to verify it can reproduce adjacency from kNN
     candidates.
  3. Train node diffusion to match coordinate distributions (even unconditionally first).
  4. Combine (node diffusion → edge sampling → surrogate selection).

  ## Tests and acceptance criteria (concrete)

  ### Data loader tests

  - For random n: max(node_id)=num_nodes, edge endpoints within [1..N], no duplicates/self-loops.
  - Degree histogram matches observed (max degree ≤6 on real data).

  ### Model sanity tests

  - Surrogate overfit test on ~10 tessellations × 5 RD: near-zero training error (ensures pipeline is correct).

  ## Public interfaces / outputs (explicit)

  - Input: RD (float), y_target (vector; includes RS at minimum), K (#samples).
  - Output:
      - nodes: list of (id, x, y) with ids 1..N
      - edges: list of (id, u, v) with ids 1..M
      - metrics_hat: surrogate-predicted metrics for each returned candidate
      - error: surrogate distance to target for each candidate

  ## Key assumptions (explicit)

  - Node ordering is arbitrary; only geometry + adjacency matter.
  - The graph family is local, sparse, approximately planar with bounded degree (consistent with your files).
  - You will condition on more than RS when possible; otherwise the inverse is highly non-identifiable and must be
    treated as sampling from a broad posterior.

  ## Alternatives (when to choose them)

  - Fully discrete graph diffusion (e.g., DiGress-style) if you later decide to ignore geometry and only model
    adjacency. citeturn0search1
  - Score-based SDE diffusion on graphs if you want a more unified “diffusion on structure” approach, but it’s more
    complex with discrete edges at your graph sizes. citeturn0search3
  - Autoregressive (GraphRNN/Transformer + stop tokens) only if you downsample to small graphs; for N≈1000+ it’s
    typically too slow and too sensitive to node ordering.

