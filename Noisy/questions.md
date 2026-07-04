
1. Look at 3 cases. Try to understand which works and why the other don't
- physical one (2^-N T)
- error independent of how close the intervals are
- maybe it makes more sense to compare the mean derivative with mean squared error on the samples I know w.r.t. the measured function

maybe lambda 1,2 works better, or if it matters at all

compare derivative squared with mean squared error - they should be comparable!!

two possible effects as to why dividing by 1/2 does not work:
- 

while the aim of the least squares optimization is to bring second term to 0, the first term will never be zero because the real function does not have first term = 0.


2. Find a case where the first term is useful 
- take a TCI with less function evaluations

stop TCI earlier and then rely on ls to improve the tci