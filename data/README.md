

# Transcription Factor Datasets

## E. Coli TF Dataset (*K-12 Strain*)

The dataset for E. Coli is found via a query to RegulonDB: 

`` ``

The ID: RDBECOLITF0001 specifies: 

- RDE = database
- ECOLI = organism 
- TF = transcription factor
- 0001 = ? code

From the TF dataset in Regulon we extract two important fields:
1. geneCodingForTF 
2. geneBnumberCodingForTF
3. id 
4. name




The B number obtained by (2) is then used to query RegulonDB again to get the GeneproductIdenitifers to get the otherDbsProductIDs

This otherDbsProductIDs contains the identifier we use to query the API of Uniprot for the amino acid sequences of a given transcription factor. 

We then query Uniprot API in batch (since their max is 100 calls) and obtain the following: 
1. Amino acid sequence 
2. sequence length 
3. primaryAccession 
4. uniprotbID
5. organism 
6. gene name

Then the program post processes the data to get: 




## Psuedomona Aerguinosa (*POA1*)

The dataset for Psuedomona Aerguinosa is initally pulled from (MiST4.0)[https://mistdb.com/mist/genomes/GCF_000006765.1]. We pull the (output domains)[https://mistdb.com/mist/genomes/GCF_000006765.1], which contain 477 domains. 


From our API call to Psuedomona Aerguinosa we extract:
1. organism_id
2. locus tag
3. GO ID

Then utilizing those we query Uniprot. All of the domains that don't have a matching Uniprot entry are dropped from the set, thus leading to a total of 377 transription factors. 



## Halobacterium Salinarum ()



## Haloferax volcanii ()

