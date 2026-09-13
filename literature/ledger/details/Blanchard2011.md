# Blanchard2011 / Generalizing from Several Related Classification Tasks to a New Unlabeled Sample

## Problem
Use several related source tasks to classify a new, unlabeled task.

## Target
Unseen-domain risk under a meta-distribution over tasks.

## Representation
Domains are random draws from a domain-of-domains law; a task-level statistic is transferred to a new draw.

## Proof bottleneck
No target labels are available, so source examples alone are insufficient without a law relating domains.

## Proof-enabling property
Meta-distribution concentration separates within-domain estimation from between-domain sampling error.

## Main theorem
The main result gives a new-task generalization bound with source sample, task sample, and domain-complexity terms.

## Price of tractability
Related-task/meta-distribution and coverage assumptions; arbitrary unseen domains are excluded.

## Algorithm relationship / source-only status
POPULATION-ABSTRACTION; `YES` conditional on the domain-generating law.

## Evidence
NeurIPS 2011 main theorem; proceedings PDF.
