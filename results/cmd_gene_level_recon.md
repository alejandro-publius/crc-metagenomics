# cMD gene-level data reconnaissance

Generated: 2026-05-18 22:50:12.846026 
cMD version: 3.20.0 

## Test pull: GuptaA_2019 gene_families


- Object class: SummarizedExperiment 
- Dimensions (features x samples): 3058876 x 60 
- Memory size: ~849.8MB
- First 5 feature names (HUMAnN format):
    - UNMAPPED 
    - UniRef90_V0SIR0 
    - UniRef90_V0SIR0|g__Escherichia.s__Escherichia_coli 
    - UniRef90_A0A192CAU6 
    - UniRef90_A0A192CAU6|g__Escherichia.s__Escherichia_coli 
- Stratified features (contain '|'): 1805837 
- Unstratified features: 1253039 

- Median non-zero features per sample: 258883.5 
- Mean non-zero features per sample: 280175 

## Test pull: GuptaA_2019 pathway_abundance (looking for stratified)

- Dimensions: 12535 x 60 
- Stratified pathway features: 12092 (species-resolved)
- Unstratified pathway features: 443 (community-level)

## Headline finding

- gene_families ships at the HUMAnN gene-level granularity Tal asked for
- pathway_abundance includes STRATIFIED (species-resolved) pathways alongside unstratified
- This is the granularity step the project hasn't yet exploited
