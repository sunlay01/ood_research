# Mechanism families

本轮同时记录 moment coordinates 与 structural coordinates。前者适合 exact risk algebra，后者用于赋予干预生成含义。统计矩变化不自动等于 structural mechanism change；同一 observed moment 可能有多个 structural explanations。

当前模块包括 core mean/variance、nuisance intercept/innovation mean/variance 和 core-nuisance relation。每个模块都有 structural tangent direction；模块 tangent 发生 overlap 时只记录为 sum，不强行写成 direct sum。
