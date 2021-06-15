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


profile visualisation
https://jiffyclub.github.io/snakeviz/
python -m cProfile -s cumtime BayesianDrape.py  --JUST-LLTEST >tmp && head -n 60 tmp

summing accuracy - i tried math.fsum which is exact, no apparent difference in results or optimizer warnings, and no need to use now. np.sum always improves precision by pairwise add if no axis given. everything in program is float64

(bayesiandrape) D:\BayesianDrape>python -m cProfile -o profile.prof -s cumtime BayesianDrape.py --TERRAIN-INPUT=data/all_os50_terrain.tif --POLYLINE-INPUT=data/test_awkward_link.shp --OUTPUT=data/test_output.shp --SLOPE-PRIOR-STD=2.2 --SPATIAL-MISMATCH-PRIOR-STD=25

python BayesianDrape.py --TERRAIN-INPUT=data/all_os50_terrain.tif --POLYLINE-INPUT=data/test_awkward_link.shp --OUTPUT=data/test_output_nocont.shp --SLOPE-PRIOR-STD=2.2 --SPATIAL-MISMATCH-PRIOR-STD=25 --SLOPE-CONTINUITY-PRIOR-SCALE=inf 


python BayesianDrape.py --TERRAIN-INPUT=data/all_os50_terrain.tif --POLYLINE-INPUT=data/biggertest.shp --OUTPUT=data/test_output.shp --SLOPE-PRIOR-STD=2.2 --SPATIAL-MISMATCH-PRIOR-STD=25

python BayesianDrape.py --TERRAIN-INPUT=data/all_os50_terrain.tif --POLYLINE-INPUT=data/bridge_fix_test1.shp --OUTPUT=data/test_output.shp --SIMPLE-DRAPE-FIELD=fix --DECOUPLE-FIELD=bridge

python BayesianDrape.py --TERRAIN-INPUT=data/all_os50_terrain.tif --POLYLINE-INPUT=data/bridge_fix_test1.shp --OUTPUT=data/fix_test_out2.shp --SIMPLE-DRAPE-FIELD=fix --DECOUPLE-FIELD=bridge



Slope prior should be linear composable?  and length weighted. Discuss meaning of deviation from exp distribution. But what is right segment length? Autocorrelation? Adjacent segments in single point case?

Auto correlation of heights isn't much concern because each parameter only influences 1 height mainly. Is this a general finding?

here's the argument: length weighting is a more useful model because it automatically gives smooth changes, you don't have to length weight but if you did you'd have to have a prior for gradient autocorr too. Length perfectly captures gradient autocorrelation in prediction of heights - but causes an autocorrelation problem versus the offset prior unless we normalize for average link length. :)


later: options to ordinary drape / 3d fix / decouple bridge links with interior points being weighted average of exterior

autocorr stats:
scale of exp distribution for grades = 21.7
scale of difference between neighbouring grades = 56.7
but i'm going to abandon that for now on computational grounds.


profiling now
25% terrain interpolator
20% fsum
20% sparse find
20% gaussian pdf
...ditching fsum, exact gauss (tested it still works on awkward link). 

todo: investigate opt termination. current settings are nice on biggertest but could it be quicker?


could make default max offset equal tile diagonal

conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch -c conda-forge
https://pytorch.org/docs/0.3.1/torch.html#torch.index_select
Torch has its own optimizer should we use it? potentially yes to keep all the optim tensors on gpu only
https://github.com/sbarratt/torch_interpolations

before torch: biggertest gradient takes 17.2. 

after torch even biggertest takes 0.008 (2000x speed) for gradient and it's not on gpu yet! increasing ncalls so its now 0.88

later: can i include torch_interpolations in yaml? or just add to my own repo

sounds like i can get working on laptop if i recompile from source:
https://github.com/pytorch/pytorch/issues/31285
https://discuss.pytorch.org/t/solved-pytorch-no-longer-supports-this-gpu-because-it-is-too-old/15444/19


publishing to conda-forge
https://www.colorado.edu/earthlab/2019/01/03/publishing-your-python-code-pip-and-conda-tips-and-best-practices

could have different priors for x and y mismatch? half cell size in each case

python -m pytest

RESIDS seem to get worse with scatter plots - this is very short links being flattened by bayes especially when they are dead ends.

time code in test_all.py. before refactor 0.0082
after 0.030! not so good. could it be reuse of n1_zs n2_zs requiring torch backtrace to preserve the graph? yes. fixed -> time=0.008. 

TODO split lines at appropriate points

how to test for myself whether precision affects optim? get rid of constant in prior; before summing get smallest value, after summing add it then subtract prev sum and assert smaller than smallval/100 or something

note - the bulwark link is not totally fixed with offset prior of 5, still a little spurious change remains (ground truth).
D:\sustrans\cyclemonData\intermediate-data\osm\connected_chepstow.shp


error model for the bayesiandrape output vs OS:
length weighted logpdf(grade): hugely influenced by length; long links good short links bad
logpdf(grade) very heteroscedastic, worst links all short
if we don't length weight then it's the long ones that are bad - but situation far better than the length weighted one
but what do i care about more?
i care about elev change per unit length.
it's heteroscedastic. 
that can be modelled.
but it's not about what i care about, it's what is.
the thing to analyse proability wise is links.


first run
sigma is 3.9
slope maybe not burned in - change starting conditions and run length
interactions are as you'd expect 

interactions on longer test are hardly there

do we care about log likelihood? it's the same thing as optimizing anyway I think and we do care about underlying process

see if mcmc results differ if we just filter for +ve outlier likelihood

what we'll do after this is
1. plot of mean error over each param (based on max ll)
2. plot of mean error of (simple drape high resids, low resids, bayes drape high resids, low resids) over each param
3. plot of regression param over the same


interactions between slope and contin are a problem.
ideas
1. try to engineer them out by fitting the prior to two impact points ******
2. see if it settles down somewhere if i allow slope up to 200 (trying now) (doh should be 90 but doesn't matter) - yes settles to 90 degree slope!!
3. remove exp prior just use normal?? maybe without sub link data this makes most sense. 
4. is the real issue that we don't have sub link data to properly calibrate sc

nb real reason for length weight is to solve MAUP (linear composable parts)

3 ABOVE: if slope fixed, contin doesn't converge
trying again without weird impact estimation

mismatch,slope,contin(A=grade scale otherwise impact)
42 90 0.75A -2225.138515253074 0.9902554944738716 0.006059345189255998
38 90 0.68A -2225.1531872282117 0.9902435474581606 0.0060552036192509096
5.68 85.4 0 -2229.198081130041 0.9905906276707698 0.006209601733760056 
6 39.2 0.02 -2250.590703284687 0.9895723391504408 0.006043401431946213
6 2 0.5 -2783.305895523943 0.9844628109417487 0.012248940699023658
6 2 1 -2782.9485554265366 0.9844588426667591 0.01225303269807083
  
is sigma really the same? comes out at 2.49 on a param set bayes had optimized to 2.76 (ignoring interactions i.e. feeding in median params). seems close. only affects LL calculation anyway not r2/mse.

does seeem my slope priors are wrong

is it just that my mismatch is way underestimated if i widen the prior it will peak and stick? try again newsigma-noexp-widemismatch

priors: compare exp with scale of 2 degrees to norm with scale=grade 0.68. They cross over at about 10 degrees (18% grade), which is vaguely sensible. 
what's weird is the direction of interaction (higher mismatch and more permissive continpar) unless there's no interaction and both were just underspecified to start with. which we'll find out now.
ah, they are: widening priors means they don't run away forever. but they still interact, and that makes sense - probabilities adjust in proportion to one another. 

try again with contin as  ratio of mismatch - yes it converges (though it's pretty agnostic on mismatch)
post_log_likelihood(97,0.0178,printstats=True)
lr.rvalue=0.9902729405931308
abs(resids).sum()/draped_net.SHAPE_Leng.sum()=0.006065016923154786
-2225.0916912063867
is that a weird convergence though? as one parameter burned in only after the other converged? 
post_log_likelihood(119.5,0.0178,printstats=True)
lr.rvalue=0.9902731473753407
abs(resids).sum()/draped_net.SHAPE_Leng.sum()=0.00606493722090627
-2225.0769589767024
this looks good as a model in that it doesn't much care what mismatch is.

in fact the prior could go even wider 
post_log_likelihood(200,0.0176,printstats=True)
lr.rvalue=0.9903299156766088
abs(resids).sum()/draped_net.SHAPE_Leng.sum()=0.006085272336636772
-2225.113266878689
we'll have to quantify this with evidence integral though

EXP ALONE, REMOVE INTERACT, WIDE PRIOR
post_log_likelihood(109,0.0067,printstats=True)
lr.rvalue=0.9892148522632574
abs(resids).sum()/draped_net.SHAPE_Leng.sum()=0.006625902214336829
-2308.222028393377



CURRENTLY I AM TESTING CURVATURE PRIOR AND IT IS NOT WORKING we get instant convergence hence no optimization
why?
have changed from curvature to just grade change in the code - same result.
is it a prior issue or a bug in measuring curvature or a subtlety of optimization? MAKE A VERY SMALL TEST NETWORK FOR CURVATURE. NO THAT SEEMS FINE.
trying one other point really screws up curvatures (and neighbour likelihoods) so optimizer gives up

HERE'S A THOUGHT. DO WE NEED TO DERIVE THE EQUIV OF THE EXPONENTIAL GRADE PRIOR, BUT FOR DIRECTION CHANGE I.E. ONE THAT IS INDIFFERENT TO SHAPE OF CHANGE BUT RESPONDS TO ITS SUM? it would have to be based on angle, not grade. and then f(2*theta)=f(theta)**2 implies log.

also the grade change code can return nan due to rounding errors. need to restrict arctrig input to -1,1.
then maybe all prior options will work....

expon angle prior works. (scale=1.28 degrees)

would expon distance weighted curvature work better? no, it's not needed as an expon angle prior ALREADY PREVENTS SHORT SEGMENTS FROM CHEATING.
(if i did want to:
    but there are such outliers. try truncating distance from 0.1 to 2 then fitting expon scale=0.65 for pitch/trunc_dist and NOT WEIGHTING IT
    implement both and MCMC test
    perhaps ultimately we argue that we don't want to do curvature tests as they involve looking at lots of neighbouring points so we can always test curvature over a set distance thereby removing outliers?
    this is alright though expon angles is better: (using pareto distributed distance weighted curvature and note the rather lax curvature scale)
(bayesiandrape) D:\BayesianDrape>python BayesianDrape.py --TERRAIN-INPUT=data/all_os50_terrain.tif --POLYLINE-INPUT=data/test_awkward_link.shp --OUTPUT=data/test_output_curvaturecont_PARETO.shp --SLOPE-PRIOR-SCALE=4 --SPATIAL-MISMATCH-PRIOR-STD=25 --SLOPE-CONTINUITY-PRIOR-SCALE=inf --CURVATURE-PRIOR-SCALE=0.91 --USE-CURVATURE-PRIOR --CURVATURE-PRIOR-SHAPE=1.78
)

ok so we need a version that supports any combination of expon grade, normal grade, expon angles.
set mismatch max to tile size

callback - maybe don't compute ll every time, it takes ages!

now mcmc test 
nb no bridges in these tests - fix
exp grade alone
norm grade alone known to be better
exp grade plus angle

WHEN I GET THIS ON GPU - INCLUDE BRIDGES IN TEST - SET HIGH MAXITER LIMIT

mismatch and sloperatrat seem to have interaction. return to sloperat only.
if it doesn't work, query max offset and try returning to direct slope param.
