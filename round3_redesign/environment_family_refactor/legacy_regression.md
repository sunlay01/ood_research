# Legacy regression

The compatibility snapshot records the concatenated `O_S` and `A` matrix hash
`ba982e44ddfd867ba75d4a9f0c10c91e842b1d93255223aa2cda47d1a0b95828`.

The expected legacy geometry is `O_S.shape=(275,8)`, `A.shape=(9,8)`,
`rank(O_S)=7`, and hidden-U information floor
`0.05118145108608892`.  Exposing U at the source gives floor zero.  The runner
also compares the existing 3C-B and 3E-B scalar CSVs when present, checks the
large-lambda V-REx invalid path, and checks the existing CORAL
fixed-representation/gauge-degenerate classification.

The matrix hash, not only rounded scalar summaries, is used to catch accidental
changes in the legacy local chart.
