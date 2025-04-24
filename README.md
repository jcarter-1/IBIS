IBIS
-----
The code and analysis presented here is presented in IBIS: an Integrated Bayesian approach for unique Initial thorium corrections and age-depth models in U-Th dating of Speleothems - Kinsley et al. (2025) 

This work is the result of a collaboration at BGC between Christopher Kinsley, Warren Sharp, and Jack Carter. This work was originally motivated by some complex data from Curaçao and then subsequently led to the quesiton - what else can we do with our algorithm? 

Why it matters?
---------------
U-Th dating is a pivotal chronometer for the estimation of accurate ages anchoring paleoclimate recored from ~present day to ~650 ka. For accurate U-Th dating, a sample must contain: 
  (1) Measurable amounts of U-238 - and the decay products U-234 and Th-230. 
  (2) It must have reamined a closed system with respect to U-Th isotopes since its formation
  (3) Any initial daughter throium-230 must be known or quantifable and then subtract. 
  

However, (3) - the challenge of correcting for initial throium-230 persists to varying degress in all stalagmites. This comprimises the accuracy of U-Th ages. 

Abstract 
--------
Proxies recorded in speleothems offer insight into past patterns of rainfall, wildfires, vegetation, and other components of the terrestrial climate system. For these records to be effective they must be accurately and precisely dated. Uranium-thorium (U-Th) dating is commonly applied to create chronological models for speleothems, including stalagmites, from present-day to ~650 ka. Making accurate and precise corrections for initial thorium (i.e., 230Th that is not a result of in-situ decay of uranium within the sample) can be the limiting factor in building robust speleothem age models. Two methods are broadly used to correct for initial thorium: (1) direct estimation using isochron techniques; or (2) assuming a constant, model value of the initial 230Th/232Th ratio. However, isochron approaches are resource-intensive, while corrections made using a single, assumed initial thorium value with arbitrary uncertainty may fail to characterize significant variations in initial thorium, leading to inflated age uncertainties and potentially inaccurate ages, thereby limiting speleothems that can be utilized for climate reconstruction. IBIS is a Bayesian model that uses a probabilistic approach to constrain the initial thorium isotope composition of each sample, thereby facilitating the derivation of accurate U-Th ages and errors and robust age-depth models for speleothems. The model framework formulates priors for all model parameters, including systematic priors (e.g., 230Th and 234U decay constants) and sample-specific priors (e.g., U-Th measurements and a likely range of initial thorium compositions expressed as 230Th/232Th activity ratios), and uses a likelihood that penalizes violations of stratigraphic order. To test the validity of the approach, we show the efficacy of IBIS when applied to case studies from the published literature that exhibit rapidly changing growth rates, uranium loss, growth hiatuses, and very young (<1 ka) samples with heterogeneous initial thorium. The ability of the IBIS modeling framework to make accurate, precise, and unique corrections of the initial thorium for each sample within a speleothem has the potential to extend the application of U-Th dating to more complex speleothem systems (e.g., variable initial thorium) and provide more accurate and precise age information for samples where initial Th corrections are significant and variable.


Thoth
-----
We create an algorithm for the estimation of prior for the initial Th composition. We design a Monte Carlo method that utilizes batch running 


Intial Thorium MCMC
-------------------

Age-Depth MCMC
--------------

Example List
------------




