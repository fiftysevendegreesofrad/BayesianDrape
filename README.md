# BayesianDrape
Drape GPS traces and path networks over terrain using Bayesian maximum likelihood estimation

A bit like Kriging, but with a network, and the underlying data is the elevation model which we assume to be correct just lacking sufficient detail.
https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.12979

Also a bit like Kalman filtering, but making use of extra information specific to the task of draping paths over terrain.

As a physical analog you might like to think about a network made of rubber bands being stretched to fit over the terrain. This is not very accurate as (1) rubber bands only stretch over convex surfaces while this algorithm stretches over concave surfaces also; (2) the rubber band metaphor implies the ends of the rubber band are at a fixed position, whereas here, nothing is fixed. A better, though still approximate physical analog might be to imagine the rubber bands being made of statically charged clingfilm, with a tendency to stick to the surface regardless of whether concave or convex, balanced by an opposing stiffness such that they resist excessive bending. But (unless we allow vertical errors) they will never leave the surface of the terrain, they just have a little local flexibility in regard to exactly which part of it they stick to. This is as far as the analogy can go, because we do not really simulate the elastic and electrostatic forces governing our imaginary material. What we compute is a local maximum a posteriori likelihood for the error model, but given that caveat, visual thinkers may still prefer to imagine stiff clingfilm rubber bands.

this analogy helps not only for understanding, but for seeing why we hope the optimization surface might be relatively smooth. Also, terrain models by definition don't have cliffs at sub-cell level.

Splitting to 10m segments draped on 50m raster gives distribution of (hg+hl)/len for prior (although this includes outliers it should still be representative enough)
Maximum:	0.509859
Sum:	15044.436881
Mean:	0.042823
Standard Deviation:	0.039105

Actually we should check normality and remove outliers if needed. exported to ec_per_len.txt.

also do it by angle
Minimum:	0
Maximum:	27.0152
Sum:	859474.466267
Mean:	2.446443
Standard Deviation:	2.221301

Will need to check for precision issues in sum likelihoods.

If it comes to analytical differentiation, do I do it per point (assuming others don't move) or for the ensemble (e.g. automatically with autograd pyautodiff auto-diff pytorch)?
intro to autodiff suitable for jax: https://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/readings/L06%20Automatic%20Differentiation.pdf
https://github.com/google/jax
looks great! basically says supports numpy so let's get started with numpy sparse.
comparison of some autograds: https://arxiv.org/ftp/arxiv/papers/1606/1606.06311.pdf
http://gcucurull.github.io/deep-learning/2020/06/03/jax-sparse-matrix-multiplication/


profile visualisation
https://jiffyclub.github.io/snakeviz/
python -m cProfile -s cumtime BayesianDrape.py  --JUST-LLTEST >tmp && head -n 60 tmp


is my func broken? how to test when i've changed behaviour? add zeros back in, LLtest, rerun?

ok there is a precision issue with func

weirdly its fixed by the zero padding

putting bounds back in causes opt fail (differt opt alg)

right when did it work? not sure now :(

if it helps with approximator i could put in a gradient outside the widest limit


also i could revert to dense matrix + let optimizer run longer

NOW QUITE NICE THOUGH COMPLAINS PRECISION LOSS

try sparse again - very similar output

COMPARE LLTEST SPARSE ZEROPAD/NON - very similar so sparse isn't the problem

THEN TRY WITHOUT THE ZEROS AGAIN - LLTEST is vaguely similar not quite so much (numbers off by 1%), precision issue then?
but everything i checked is float64 - including interpolator outputs

use log neighbour grades?
could it be that we get too close to neighbours?
does this work with exact logpdf so i can rule that out for now? yes. keep using exact, but that's not the issue.

noticable how it really likes making flat roads right now

should i put lower limit on grade prior in case there are any super short distances messing us up? ok now done via return to slope as this also should fix any inverse distance/grade precision issues 

it's stopped complaining about precision? though my results are middling. priors are kinda wrong?

* if i go back to angles i should make a table again
* still not using bounds
* still not approximating gaussian prior
* fix for sparse

* if precision gets fixed maybe i can bring back inverse_distances


now with the exponential approx grade prior translated to slope, it complains of precision again

WTF the working one used slope = np.arctan(height,dist) !! not even a function! checkout and rerun testing types to be sure!
ok so my convergence was based on a bug - fix this bug and precision issues return even with gaussian slope prior.

***BUT WE KNOW THEY HAVE TO DO WITH SLOPE COMPUTATION OR PRIORS***

trying to invent a precise arctan i didn't do any better than numpy (funny that)

maybe make 2d slope prior - for short segments it shouldn't go nuts?

but distances minimum in the test case is 0.5 so we don't have insane small values there. 

np.sum always improves precision by pairwise add if no axis given. else python math.fsum may be worth a look but slow.

msum and fsum. not sure which is faster but fsum is exact.
or can we get a float128? not easily, if we can it may only be float64 underneath.


testing sum - by replacing with fsum - is this the issue? no, still precision loss.

VISUALISE PRIORS TO SANITY CHECK. MAYBE PREVIOUS BUGS (LIKE THE ZERO SUMMING) EFFECTIVELY CAUSED WEIRD PRIORS THAT MADE IT WORK OK. TWEAK PRIORS TO MAKE WORK?

- THE BUGGY ARCTAN DISCARDED DISTANCE - COULD DISTANCE BE BETTER RECOMPUTED EACH TIME
- THE BUGGY ZEROADD DISCARDED WHAT? WILL HAVE REDUCED INFLENCE OF ALL OTHER PRIORS. MAYBE ITS TRYING TO OPTIM TOO FAST? don't forget moving to differentiation is an option now.

or is it not really a precision issue but a convergence one?

WHEN DID WE GET THE REALLY NICE OUTPUT, WHAT WERE WE DOING?

(bayesiandrape) D:\BayesianDrape>python -m cProfile -o profile.prof -s cumtime BayesianDrape.py --TERRAIN-INPUT=data/all_os50_terrain.tif --POLYLINE-INPUT=data/test_awkward_link.shp --OUTPUT=data/test_output.shp --SLOPE-PRIOR-STD=2.2 --SPATIAL-MISMATCH-PRIOR-STD=25

oddnesses: not using bounds, not approximating gaussian prior

Slope prior should be linear composable?  and length weighted. Discuss meaning of deviation from exp distribution. But what is right segment length? Autocorrelation? Adjacent segments in single point case?

Auto correlation of heights isn't much concern because each parameter only influences 1 height mainly. Is this a general finding?

here's the argument: length weighting is a more useful model because it automatically gives smooth changes, you don't have to length weight but if you did you'd have to have a prior for gradient autocorr too. Length perfectly captures gradient autocorrelation in prediction of heights - but causes an autocorrelation problem versus the offset prior unless we normalize for average link length. :)

first attempt at lenwt prior: definitely flattens it though still convergence issue and short steep slopes issue. maybe because i shouldn't approximate the slope prior - now testing. no difference.

ok - maybe i should
1. test case the height gain prior situation
2. tweak height prior to prefer gradual gain. yes - this works - still complains about precision but if anything output is too smooth. suspect i should formally define and calibrate the prior.

model as normal times exponential.

later: options to ordinary drape / 3d fix / decouple bridge links with interior points being weighted average of exterior

autocorr stats:
scale of exp distribution for grades = 21.7
scale of difference between neighbouring grades = 56.7