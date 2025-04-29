IBIS
-----
The code and analysis presented here is presented in IBIS: an Integrated Bayesian approach for unique Initial thorium corrections and age-depth models in U-Th dating of Speleothems - Kinsley et al. (2025) 

This work is the result of a collaboration at Berkeley Geochronology Center (BGC) between Christopher Kinsley, Warren Sharp, and Jack Carter. This work was originally motivated by some complex data from Curaçao and then subsequently led to the quesiton - what else can we do with our algorithm? 

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
For the complete mathematical closure of the Bayesian model we are required to form a mathematical description - in the form of a distribution - for the 230Th/232Th initial activity ratio (230Th/232ThA0)for the speleothem sample. However, this requires careful thought as 230Th/232ThA0 can vary in space and time. Potentially showing variability throughout a single speleothem sample. Moseley et al. (2015) performed U-Th analysis on a speleothem from the Yucatán Peninsula and documented 4 distinct growth intervals. Isochron analysis on these four intervals inferred unique (230Th/232ThA0) ranging from ~3 to >20. In the Thoth 


We provide an illustration of how thoth works in the Thoth folder. We use the data of Faraji et al. (2021) - the twinned U-Th dating and lamina counting age model provides a unique scenario to valid the Thoth appraoch. 


Faraji, M., Borsato, A., Frisia, S., Hellstrom, J.C., Lorrey, A., Hartland, A., Greig, A. and Mattey, D.P., 2021. Accurate dating of stalagmites from low seasonal contrast tropical Pacific climate using Sr 2D maps, fabrics and annual hydrological cycles. Scientific Reports, 11(1), p.2178.

Moseley, G.E., Richards, D.A., Smart, P.L., Standish, C.D., Hoffmann, D.L., ten Hove, H. and Vinn, O., 2015. Early–middle Holocene relative sea-level oscillation events recorded in a submerged speleothem from the Yucatán Peninsula, Mexico. The Holocene, 25(9), pp.1511-1521.


Intial Thorium MCMC
-------------------
IBIS is a two stage model. The first makes inferences on the U-Th ages with unique initial thorium composition. This model makes use of a stratigraphic log-likelihood function to find the coupled three-vector of activity ratios (230Th/238U, 232Th/238U, and 234U/238U) and ($^{230}$Th/$^{232}$Th)$_{A0}$ which aligns the ages in stratigraphic order. 

The Initial Thorium move
========================
We use a cumulative density function (CDF)-space Metropolis move. We take the current parameter value, say x, and map it through its prior CDF. Our prior is non-parametric so we initially numerical calculate this through interpolation. We propose a move in CDF-space. Say u' is our proposed value, we propose u' = (u + $\delta$) mod 1, $\delta$ $\sim$ N(0, $\sigma$). Because the normal, N, is symmetric and we use a wrapped interval, the Hastings ratio from the proposal cancels. We then convert the "real" value by inverting the CDF. We then perform the traditional Metropolis acceptance criterion and see if the proposed value should be accepted or rejected (Gilks, Richardson, & Spiegelhalter 1996). 


Age-Depth MCMC
--------------
The second part of the IBIS model is a Bayesian Age-Depth model. This takes the output ages from the first part of the model and perform a second MCMC algorithm to make an inference on the age-depth relationship. This is a modular approach such that the U-Th ages from the initial part can be extract and a user can choose which age-model they prefer to use (e.g., Comas-Bru et al. 2020). 

Comas-Bru, L., Rehfeld, K., Roesch, C., Amirnezhad-Mozhdehi, S., Harrison, S.P., Atsawawaranunt, K., Ahmad, S.M., Ait Brahim, Y., Baker, A., Bosomworth, M. and Breitenbach, S.F., 2020. SISALv2: A comprehensive speleothem isotope database with multiple age-depth models. Earth System Science Data Discussions, 2020, pp.1-47.

Example List
------------
We provide a suite of examples to display the efficacy of the IBIS model framework. 

These are included in the manuscript which can be found here: 


References
----------

Faraji, M., Borsato, A., Frisia, S., Hellstrom, J.C., Lorrey, A., Hartland, A., Greig, A. and Mattey, D.P., 2021. Accurate dating of stalagmites from low seasonal contrast tropical Pacific climate using Sr 2D maps, fabrics and annual hydrological cycles. Scientific Reports, 11(1), p.2178.

Gilks, W.R., Richardson, S. and Spiegelhalter, D.J., 1996. (1996), Markov Chain Monte Carlo in Practice.

Hoffmann, D.L., Spötl, C. and Mangini, A., 2009. Micromill and in situ laser ablation sampling techniques for high spatial resolution MC-ICPMS U-Th dating of carbonates. Chemical Geology, 259(3-4), pp.253-261.

Moseley, G.E., Edwards, R.L., Wendt, K.A., Cheng, H., Dublyansky, Y., Lu, Y., Boch, R. and Spötl, C., 2016. Reconciliation of the Devils Hole climate record with orbital forcing. Science, 351(6269), pp.165-168.

Weber, M., Scholz, D., Schröder-Ritzrau, A., Deininger, M., Spötl, C., Lugli, F., Mertz-Kraus, R., Jochum, K.P., Fohlmeister, J., Stumpf, C.F. and Riechelmann, D.F., 2018. Evidence of warm and humid interstadials in central Europe during early MIS 3 revealed by a multi-proxy speleothem record. Quaternary Science Reviews, 200, pp.276-286.






