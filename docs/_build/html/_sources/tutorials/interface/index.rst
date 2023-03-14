Using the SATLAS interface
==========================

As a stepping stone between SATLAS and SATLAS2, an interface has been
provided which can mostly be used as a drop-in replacement for code that
uses the SATLAS syntax. Note that not all functionalities have been
implemented in this fashion. For users that require these
functionalities, we recommend migrating to SATLAS2.

.. code:: ipython3

    import sys
    import time
    
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    
    sys.path.insert(0, '..\src')
    
    import satlas2
    import satlas as sat
    
    
    def modifiedSqrt(input):
        output = np.sqrt(input)
        output[input <= 0] = 1e-3
        return output

Fitting a single hyperfine spectrum
-----------------------------------

The most common task, and the one this interface is meant for, is
fitting a single hyperfine spectrum. A special class in SATLAS2 called
*HFSModel* has been created as a replacement for the equivalent SATLAS
*HFSModel*. Note that the normal hyperfine spectrum model in SATLAS2 is
called *HFS*.

.. code:: ipython3

    spin = 3.5
    J = [0.5, 1.5]
    A = [9600, 175]
    B = [0, 315]
    C = [0, 0]
    FWHMG = 135
    FWHML = 101
    centroid = 480
    bkg = [100]
    scale = 90
    
    x = np.arange(-17500, -14500, 40)
    x = np.hstack([x, np.arange(20000, 23000, 40)])
    
    rng = np.random.default_rng(0)
    hfs = satlas2.HFSModel(I=spin,
                           J=J,
                           ABC=[A[0], A[1], B[0], B[1], C[0], C[1]],
                           centroid=centroid,
                           fwhm=[FWHMG, FWHML],
                           scale=scale,
                           background_params=bkg,
                           use_racah=True)
    hfs.set_variation({'Cu': False})
    

The object called *hfs* can be used with the syntax of SATLAS.
Generating Poisson-distributed data is done by simply calling the
function with frequency values as an argument, and using the result for
the NumPy Poisson random number generator.

.. code:: ipython3

    y = satlas2.generateSpectrum(hfs, x, rng.poisson)

In order to demonstrate the difference in performance, the centroid is
offset by 100 from the actual value and the fitting is done by both the
interface and SATLAS.

.. code:: ipython3

    hfs.params['centroid'].value = centroid - 100
    # Normal SATLAS implementation
    hfs1 = sat.HFSModel(spin,
                        J, [A[0], A[1], B[0], B[1], C[0], C[1]],
                        centroid - 100, [FWHMG, FWHML],
                        scale=scale,
                        background_params=bkg,
                        use_racah=True)
    hfs1.set_variation({'Cu': False})
    
    # Interface fitting
    print('Fitting 1 dataset with chisquare (Pearson, satlas2)...')
    start = time.time()
    satlas2.chisquare_fit(hfs, x, y, modifiedSqrt(y))
    stop = time.time()
    print(hfs.display_chisquare_fit(show_correl=False))
    dt1 = stop - start
    
    # SATLAS fitting
    print('Fitting 1 dataset with chisquare (Pearson, satlas)...')
    start = time.time()
    sat.chisquare_fit(hfs1, x, y, modifiedSqrt(y))
    stop = time.time()
    hfs1.display_chisquare_fit(show_correl=False, scaled=True)
    dt2 = stop - start
    print('SATLAS2: {:.3} s'.format(dt1))
    print('SATLAS1: {:.3} s'.format(dt2))


.. parsed-literal::

    Fitting 1 dataset with chisquare (Pearson, satlas2)...
    define whether you want to see the correlations in display_chisquare_fit(...)
    [[Fit Statistics]]
        # fitting method   = leastsq
        # function evals   = 137
        # data points      = 150
        # variables        = 8
        chi-square         = 151.188938
        reduced chi-square = 1.06471083
        Akaike info crit   = 17.1842512
        Bayesian info crit = 41.2693335
    [[Variables]]
        Fit___HFModel__3_5___centroid:  482.548151 +/- 7.56664202 (1.57%) (init = 380)
        Fit___HFModel__3_5___Al:        9604.53249 +/- 6.41301505 (0.07%) (init = 9600)
        Fit___HFModel__3_5___Au:        176.460909 +/- 2.73509340 (1.55%) (init = 175)
        Fit___HFModel__3_5___Bl:        0 (fixed)
        Fit___HFModel__3_5___Bu:        348.564588 +/- 19.6945285 (5.65%) (init = 315)
        Fit___HFModel__3_5___Cl:        0 (fixed)
        Fit___HFModel__3_5___Cu:        0 (fixed)
        Fit___HFModel__3_5___FWHMG:     142.382607 +/- 57.6647366 (40.50%) (init = 135)
        Fit___HFModel__3_5___FWHML:     100.522879 +/- 63.5247619 (63.19%) (init = 101)
        Fit___HFModel__3_5___scale:     89.2398271 +/- 7.15348105 (8.02%) (init = 90)
        Fit___HFModel__3_5___Amp3to2:   0.4545455 (fixed)
        Fit___HFModel__3_5___Amp3to3:   0.4772727 (fixed)
        Fit___HFModel__3_5___Amp3to4:   0.3409091 (fixed)
        Fit___HFModel__3_5___Amp4to3:   0.1590909 (fixed)
        Fit___HFModel__3_5___Amp4to4:   0.4772727 (fixed)
        Fit___HFModel__3_5___Amp4to5:   1 (fixed)
        Fit___bkg___p0:                 100.670729 +/- 1.59295191 (1.58%) (init = 100)

    Fitting 1 dataset with chisquare (Pearson, satlas)...
    Chisquare fitting in progress (151.18893761580117): 172it [00:00, 182.60it/s]

    NDoF: 142, Chisquare: 151.18894, Reduced Chisquare: 1.0647108
    Akaike Information Criterium: 17.18425, Bayesian Information Criterium: 41.269333
    Errors scaled with reduced chisquare.
    [[Variables]]
        FWHMG:        142.398642 +/- 57.6603105 (40.49%) (init = 142.3868)
        FWHML:        100.507633 +/- 63.5294155 (63.21%) (init = 100.5189)
        TotalFWHM:    203.616069 +/- 21.3016922 (10.46%) == '0.5346*FWHML+(0.2166*FWHML**2+FWHMG**2)**0.5'
        Scale:        89.2388856 +/- 7.15309388 (8.02%) (init = 89.23958)
        Saturation:   0 (fixed)
        Amp3__2:      0.4546399 (fixed)
        Amp3__3:      0.4773649 (fixed)
        Amp3__4:      0.3410048 (fixed)
        Amp4__3:      0.1591578 (fixed)
        Amp4__4:      0.4773975 (fixed)
        Amp4__5:      1 (fixed)
        Al:           9604.53225 +/- 6.41310262 (0.07%) (init = 9604.532)
        Au:           176.461706 +/- 2.73513443 (1.55%) (init = 176.4611)
        Bl:           0 (fixed)
        Bu:           348.556409 +/- 19.6948333 (5.65%) (init = 348.5625)
        Cl:           0 (fixed)
        Cu:           0 (fixed)
        Centroid:     482.545220 +/- 7.56678464 (1.57%) (init = 482.5474)
        Background0:  100.670920 +/- 1.59296489 (1.58%) (init = 100.6708)
        N:            0 (fixed)
    SATLAS2: 0.043 s
    SATLAS1: 0.967 s
    

Note that the results are functionally identical: the slight difference
is due to a more modern implementation of the least squares fitting
routine that is used under the hood by SATLAS2. The speedup by using
SATLAS 2 is about a factor 20 for a single spectrum.

.. code:: ipython3

    left_x = x[x<0]
    right_x = x[x>0]
    
    left_y = y[x<0]
    right_y = y[x>0]
    
    fig = plt.figure(constrained_layout=True, figsize=(14, 9))
    gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)
    ax11 = fig.add_subplot(gs[0, 0])
    ax11.label_outer()
    ax12 = fig.add_subplot(gs[0, 1], sharey=ax11)
    ax12.label_outer()
    ax21 = fig.add_subplot(gs[1, 0], sharex=ax11)
    ax21.label_outer()
    ax22 = fig.add_subplot(gs[1, 1], sharex=ax12, sharey=ax21)
    ax22.label_outer()
    
    ax11.errorbar(left_x, left_y, modifiedSqrt(left_y), fmt='.', label='Artificial data')
    ax11.plot(left_x, hfs(left_x), '-', label='Fit')
    ax12.errorbar(right_x, right_y, modifiedSqrt(right_y), fmt='.', label='Artificial data')
    ax12.plot(right_x, hfs(right_x), '-', label='Fit')
    
    ax21.errorbar(left_x, left_y, modifiedSqrt(left_y), fmt='.', label='Artificial data')
    ax21.plot(left_x, hfs1(left_x), '-', label='SATLAS fit')
    ax22.errorbar(right_x, right_y, modifiedSqrt(right_y), fmt='.', label='Artificial data')
    ax22.plot(right_x, hfs1(right_x), '-', label='SATLAS fit')
    
    ax11.legend()
    ax21.legend()
    
    ax11.set_ylabel('SATLAS2')
    ax21.set_ylabel('SATLAS')
    
    plt.show()



.. image:: output_9_0.png


Overlapping hyperfine spectra
-----------------------------

The other most common usecase for SATLAS was analysis of spectra with an
isomer present, resulting in overlapping spectra. In the SATLAS
terminology, this would result in a *SumModel* being used. In SATLAS2, a
second *HFS* is simply added to the Source. However, the interface does
provide the folllowing functionality:

.. code:: ipython3

    J = [0.5, 1.5]
    FWHMG = 135
    FWHML = 101
    
    spin1 = 4
    A1 = [5300, 100]
    B1 = [0, 230]
    C1 = [0, 0]
    centroid1 = 400
    bkg1 = 60
    scale1 = 90
    
    spin2 = 7
    A2 = [3300, 60]
    B2 = [0, 270]
    C2 = [0, 0]
    centroid2 = -100
    bkg2 = 60
    scale2 = 160
    
    x = np.arange(-13000, -9000, 40)
    x = np.hstack([x, np.arange(11000, 14000, 40)])
    rng = np.random.default_rng(0)
    
    # Interface models
    hfs1 = satlas2.HFSModel(I=spin1,
                            J=J,
                            ABC=[A1[0], A1[1], B1[0], B1[1], C1[0], C1[1]],
                            centroid=centroid1,
                            fwhm=[FWHMG, FWHML],
                            scale=scale1,
                            background_params=[bkg1],
                            use_racah=True)
    hfs1.set_variation({'Cu': False})
    hfs2 = satlas2.HFSModel(I=spin2,
                            J=J,
                            ABC=[A2[0], A2[1], B2[0], B2[1], C2[0], C2[1]],
                            centroid=centroid2,
                            fwhm=[FWHMG, FWHML],
                            scale=scale2,
                            background_params=[bkg2],
                            use_racah=True)
    hfs2.set_variation({'Cu': False})
    y = satlas2.generateSpectrum([hfs1, hfs2, satlas2.Polynomial([bkg1])], x, rng.poisson)
    
    hfs1.params['centroid'].value = centroid1 - 100
    hfs2.params['centroid'].value = centroid2 - 100
    summodel = satlas2.SumModel([hfs1, hfs2], {
        'values': [bkg1, bkg2],
        'bounds': [0]
    })
    
    # SATLAS implementation
    hfs3 = sat.HFSModel(spin1,
                        J, [A1[0], A1[1], B1[0], B1[1], C1[0], C1[1]],
                        centroid1-100, [FWHMG, FWHML],
                        scale=scale1,
                        background_params=bkg,
                        use_racah=True)
    hfs4 = sat.HFSModel(spin2,
                        J, [A2[0], A2[1], B2[0], B2[1], C2[0], C2[1]],
                        centroid2-100, [FWHMG, FWHML],
                        scale=scale2,
                        background_params=[0],
                        use_racah=True)
    hfs3.set_variation({'Cu': False})
    hfs4.set_variation({'Background0': False, 'Cu': False})
    summodel2 = hfs3 + hfs4
    
    print('Fitting 1 dataset with chisquare (Pearson, satlas2)...')
    start = time.time()
    f = satlas2.chisquare_fit(summodel, x, y, modifiedSqrt(y))
    stop = time.time()
    print(summodel.display_chisquare_fit(show_correl=False))
    dt1 = stop - start
    start = time.time()
    sat.chisquare_fit(summodel2, x, y, modifiedSqrt(y))
    stop = time.time()
    summodel2.display_chisquare_fit(show_correl=False, scaled=True)
    dt2 = stop - start
    print('SATLAS2: {:.3} s'.format(dt1))
    print('SATLAS1: {:.3} s'.format(dt2))
    
    


.. parsed-literal::

    Fitting 1 dataset with chisquare (Pearson, satlas2)...
    [[Fit Statistics]]
        # fitting method   = leastsq
        # function evals   = 423
        # data points      = 175
        # variables        = 16
        chi-square         = 177.052463
        reduced chi-square = 1.11353750
        Akaike info crit   = 34.0405200
        Bayesian info crit = 84.6770956
    [[Variables]]
        Fit___HFModel__4___centroid:       392.980617 +/- 13.2182180 (3.36%) (init = 300)
        Fit___HFModel__4___Al:             5306.16636 +/- 9.74519323 (0.18%) (init = 5300)
        Fit___HFModel__4___Au:             103.560669 +/- 4.03858459 (3.90%) (init = 100)
        Fit___HFModel__4___Bl:             0 (fixed)
        Fit___HFModel__4___Bu:             195.784015 +/- 32.9150928 (16.81%) (init = 230)
        Fit___HFModel__4___Cl:             0 (fixed)
        Fit___HFModel__4___Cu:             0 (fixed)
        Fit___HFModel__4___FWHMG:          251.277769 +/- 25.0965330 (9.99%) (init = 135)
        Fit___HFModel__4___FWHML:          0.01000055 +/- 4.50439705 (45041.49%) (init = 101)
        Fit___HFModel__4___scale:          79.7727405 +/- 7.53870955 (9.45%) (init = 90)
        Fit___HFModel__4___Amp7_2to5_2:    0.5 (fixed)
        Fit___HFModel__4___Amp7_2to7_2:    0.4938272 (fixed)
        Fit___HFModel__4___Amp7_2to9_2:    0.3395062 (fixed)
        Fit___HFModel__4___Amp9_2to7_2:    0.1728395 (fixed)
        Fit___HFModel__4___Amp9_2to9_2:    0.4938272 (fixed)
        Fit___HFModel__4___Amp9_2to11_2:   1 (fixed)
        Fit___HFModel__7___centroid:      -104.843040 +/- 5.61216015 (5.35%) (init = -200)
        Fit___HFModel__7___Al:             3299.38314 +/- 2.54164939 (0.08%) (init = 3300)
        Fit___HFModel__7___Au:             60.0125639 +/- 0.99398820 (1.66%) (init = 60)
        Fit___HFModel__7___Bl:             0 (fixed)
        Fit___HFModel__7___Bu:             273.049192 +/- 15.5843734 (5.71%) (init = 270)
        Fit___HFModel__7___Cl:             0 (fixed)
        Fit___HFModel__7___Cu:             0 (fixed)
        Fit___HFModel__7___FWHMG:          121.107402 +/- 39.0810172 (32.27%) (init = 135)
        Fit___HFModel__7___FWHML:          112.746219 +/- 36.9166340 (32.74%) (init = 101)
        Fit___HFModel__7___scale:          163.484079 +/- 9.34512379 (5.72%) (init = 160)
        Fit___HFModel__7___Amp13_2to11_2:  0.6666667 (fixed)
        Fit___HFModel__7___Amp13_2to13_2:  0.5530864 (fixed)
        Fit___HFModel__7___Amp13_2to15_2:  0.3358025 (fixed)
        Fit___HFModel__7___Amp15_2to13_2:  0.2246914 (fixed)
        Fit___HFModel__7___Amp15_2to15_2:  0.5530864 (fixed)
        Fit___HFModel__7___Amp15_2to17_2:  1 (fixed)
        Fit___bkg___value1:                60.4476367 +/- 2.36128234 (3.91%) (init = 60)
        Fit___bkg___value0:                61.4896354 +/- 2.12969392 (3.46%) (init = 60)
    
    Chisquare fitting done: 421it [00:12, 32.65it/s]                            

    NDoF: 160, Chisquare: 177.29488, Reduced Chisquare: 1.108093
    Akaike Information Criterium: 32.27996, Bayesian Information Criterium: 79.751749
    Errors scaled with reduced chisquare.
    [[Variables]]
        s0_FWHMG:          250.753540 +/- 26.0746636 (10.40%) (init = 250.7535)
        s0_FWHML:          1.00000275 +/- 11.8677590 (1186.77%) (init = 1.000003)
        s0_TotalFWHM:      251.288574 +/- 24.9165138 (9.92%) == '0.5346*s0_FWHML+(0.2166*s0_FWHML**2+s0_FWHMG**2)**0.5'
        s0_Scale:          79.7123062 +/- 7.13677345 (8.95%) (init = 79.71231)
        s0_Saturation:     0 (fixed)
        s0_Amp7_2__5_2:    0.5000937 (fixed)
        s0_Amp7_2__7_2:    0.4939217 (fixed)
        s0_Amp7_2__9_2:    0.3396039 (fixed)
        s0_Amp9_2__7_2:    0.172911 (fixed)
        s0_Amp9_2__9_2:    0.4939521 (fixed)
        s0_Amp9_2__11_2:   1 (fixed)
        s0_Al:             5306.11719 +/- 9.76080435 (0.18%) (init = 5306.117)
        s0_Au:             103.549437 +/- 4.12089719 (3.98%) (init = 103.5494)
        s0_Bl:             0 (fixed)
        s0_Bu:             196.011593 +/- 32.8509112 (16.76%) (init = 196.0116)
        s0_Cl:             0 (fixed)
        s0_Cu:             0 (fixed)
        s0_Centroid:       392.909905 +/- 13.1577474 (3.35%) (init = 392.9099)
        s0_Background0:    181.069305 +/- 1.91537125 (1.06%) (init = 181.0693)
        s0_N:              0 (fixed)
        s1_FWHMG:          121.817424 +/- 39.1318124 (32.12%) (init = 121.8174)
        s1_FWHML:          112.056361 +/- 37.1055724 (33.11%) (init = 112.0564)
        s1_TotalFWHM:      192.416653 +/- 15.7236790 (8.17%) == '0.5346*s1_FWHML+(0.2166*s1_FWHML**2+s1_FWHMG**2)**0.5'
        s1_Scale:          163.317972 +/- 9.22593437 (5.65%) (init = 163.318)
        s1_Saturation:     0 (fixed)
        s1_Amp13_2__11_2:  0.666746 (fixed)
        s1_Amp13_2__13_2:  0.5531882 (fixed)
        s1_Amp13_2__15_2:  0.3359059 (fixed)
        s1_Amp15_2__13_2:  0.2247785 (fixed)
        s1_Amp15_2__15_2:  0.55321 (fixed)
        s1_Amp15_2__17_2:  1 (fixed)
        s1_Al:             3299.37436 +/- 2.48138492 (0.08%) (init = 3299.374)
        s1_Au:             60.0050608 +/- 0.98060493 (1.63%) (init = 60.00506)
        s1_Bl:             0 (fixed)
        s1_Bu:             273.161795 +/- 15.4999419 (5.67%) (init = 273.1618)
        s1_Cl:             0 (fixed)
        s1_Cu:             0 (fixed)
        s1_Centroid:      -104.833860 +/- 5.57438226 (5.32%) (init = -104.8339)
        s1_Background0:    0 (fixed)
        s1_N:              0 (fixed)
    SATLAS2: 0.226 s
    SATLAS1: 12.9 s
    

The difference in coding implementation is a result of the interface
automatically implementing a PiecewiseConstant background, where the background is a
constant for different regions in *x*-space. Notice here that the
speedup due using the SATLAS2 implementation has risen from a factor 20
for a single spectrum to almost a factor 60.

.. code:: ipython3

    left_x = x[x<0]
    right_x = x[x>0]
    
    left_y = y[x<0]
    right_y = y[x>0]
    
    fig = plt.figure(constrained_layout=True, figsize=(14, 9))
    gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)
    ax11 = fig.add_subplot(gs[0, 0])
    ax11.label_outer()
    ax12 = fig.add_subplot(gs[0, 1], sharey=ax11)
    ax12.label_outer()
    ax21 = fig.add_subplot(gs[1, 0], sharex=ax11)
    ax21.label_outer()
    ax22 = fig.add_subplot(gs[1, 1], sharex=ax12, sharey=ax21)
    ax22.label_outer()
    
    ax11.errorbar(left_x, left_y, modifiedSqrt(left_y), fmt='.', label='Artificial data')
    ax11.plot(left_x, hfs1(left_x), '-', label='SATLAS2 fit model 1')
    ax11.plot(left_x, hfs2(left_x), '-', label='SATLAS2 fit model 2')
    ax11.plot(left_x, summodel(left_x), '-', label='Sum of models')
    
    ax12.errorbar(right_x, right_y, modifiedSqrt(right_y), fmt='.', label='Artificial data')
    ax12.plot(right_x, hfs1(right_x), '-', label='SATLAS2 fit model 1')
    ax12.plot(right_x, hfs2(right_x), '-', label='SATLAS2 fit model 2')
    ax12.plot(right_x, summodel(right_x), '-', label='Sum of models')
    ax11.legend()
    
    ax21.errorbar(left_x, left_y, modifiedSqrt(left_y), fmt='.', label='Artificial data')
    ax21.plot(left_x, hfs3(left_x), '-', label='SATLAS fit model 1')
    ax21.plot(left_x, hfs4(left_x), '-', label='SATLAS fit model 2')
    ax21.plot(left_x, summodel2(left_x), '-', label='Sum of models')
    
    ax22.errorbar(right_x, right_y, modifiedSqrt(right_y), fmt='.', label='Artificial data')
    ax22.plot(right_x, hfs3(right_x), '-', label='SATLAS fit model 1')
    ax22.plot(right_x, hfs4(right_x), '-', label='SATLAS fit model 2')
    ax22.plot(right_x, summodel2(right_x), '-', label='Sum of models')
    ax21.legend()
    
    ax11.set_ylabel('SATLAS2')
    ax21.set_ylabel('SATLAS')
    plt.show()



.. image:: output_13_0.png


Different background for multiplets
-----------------------------------

To demonstrate the convenience of the PiecewiseConstant background, the same results
are coded with SATLAS, where the use of LinkedModel is required. Note
that here, the interface is *not* used.

.. code:: ipython3

    J = [0.5, 1.5]
    FWHMG = 135
    FWHML = 101
    
    spin1 = 4
    A1 = [5300, 100]
    B1 = [0, 230]
    C1 = [0, 0]
    centroid1 = 400
    bkg1 = 90
    scale1 = 90
    
    x = np.arange(-13000, -9000, 40)
    x = np.hstack([x, np.arange(11000, 14000, 40)])
    
    hfs = satlas2.HFS(spin1,
                       J=J,
                       A=[A1[0], A1[1]],
                       B=[B1[0], B1[1]],
                       C=[C1[0], C1[1]],
                       df=centroid1,
                       fwhmg=FWHMG,
                       fwhml=FWHML,
                       scale=scale1,
                       racah=True
                      )
    hfs.params['Cu'].vary = False
    bkg = satlas2.PiecewiseConstant([bkg1, bkg2], [0])
    
    y = satlas2.generateSpectrum([hfs1, bkg], x, rng.poisson)
    
    s = satlas2.Source(x, y, yerr=modifiedSqrt, name='Artificial')
    s.addModel(hfs)
    s.addModel(bkg)
    f = satlas2.Fitter()
    f.addSource(s)
    
    hfs2 = sat.HFSModel(spin1,
                        J, [A1[0], A1[1], B1[0], B1[1], C1[0], C1[1]],
                        centroid - 100, [FWHMG, FWHML],
                        scale=scale1,
                        background_params=[bkg1],
                        use_racah=True)
    hfs3 = sat.HFSModel(spin1,
                        J, [A1[0], A1[1], B1[0], B1[1], C1[0], C1[1]],
                        centroid - 100, [FWHMG, FWHML],
                        scale=scale1,
                        background_params=[bkg1],
                        use_racah=True)
    hfs2.set_variation({'Cu': False})
    hfs3.set_variation({'Cu': False})
    linkedmodel = sat.LinkedModel([hfs2, hfs3])
    linkedmodel.shared = ['Al', 'Au', 'Bl', 'Bu', 'Centroid']
    linked_x = [x[x<0], x[x>0]]
    linked_y = [y[x<0], y[x>0]]
    
    print('Fitting 1 dataset with chisquare (Pearson, satlas2)...')
    start = time.time()
    f.fit()
    stop = time.time()
    print(f.reportFit())
    dt1 = stop - start
    start = time.time()
    sat.chisquare_spectroscopic_fit(linkedmodel, linked_x, linked_y, func=modifiedSqrt)
    stop = time.time()
    linkedmodel.display_chisquare_fit(show_correl=False, scaled=True)
    dt2 = stop - start
    print('SATLAS2: {:.3} s'.format(dt1))
    print('SATLAS1: {:.3} s'.format(dt2))


.. parsed-literal::

    Fitting 1 dataset with chisquare (Pearson, satlas2)...
    [[Fit Statistics]]
        # fitting method   = leastsq
        # function evals   = 202
        # data points      = 175
        # variables        = 9
        chi-square         = 162.334878
        reduced chi-square = 0.97792095
        Akaike info crit   = 4.85319079
        Bayesian info crit = 33.3362646
    [[Variables]]
        Artificial___HFS___centroid:      379.439738 +/- 11.8479412 (3.12%) (init = 400)
        Artificial___HFS___Al:            5300.53685 +/- 8.60042067 (0.16%) (init = 5300)
        Artificial___HFS___Au:            100.910641 +/- 3.43441833 (3.40%) (init = 100)
        Artificial___HFS___Bl:            0 (fixed)
        Artificial___HFS___Bu:            167.829114 +/- 27.5840684 (16.44%) (init = 230)
        Artificial___HFS___Cl:            0 (fixed)
        Artificial___HFS___Cu:            0 (fixed)
        Artificial___HFS___FWHMG:         257.963959 +/- 23.7214758 (9.20%) (init = 135)
        Artificial___HFS___FWHML:         0.01005831 +/- 46.0743167 (458072.02%) (init = 101)
        Artificial___HFS___scale:         73.5969741 +/- 6.04333358 (8.21%) (init = 90)
        Artificial___HFS___Amp7_2to5_2:   0.5 (fixed)
        Artificial___HFS___Amp7_2to7_2:   0.4938272 (fixed)
        Artificial___HFS___Amp7_2to9_2:   0.3395062 (fixed)
        Artificial___HFS___Amp9_2to7_2:   0.1728395 (fixed)
        Artificial___HFS___Amp9_2to9_2:   0.4938272 (fixed)
        Artificial___HFS___Amp9_2to11_2:  1 (fixed)
        Artificial___PiecewiseConstant___value1:       122.518511 +/- 1.44251185 (1.18%) (init = 60)
        Artificial___PiecewiseConstant___value0:       151.305847 +/- 1.37967336 (0.91%) (init = 90)
    
    Chisquare fitting done: 619it [00:19, 31.30it/s]                            

    NDoF: 163, Chisquare: 158.72971, Reduced Chisquare: 0.97380192
    Akaike Information Criterium: 6.9229505, Bayesian Information Criterium: 44.900382
    Errors scaled with reduced chisquare.
    [[Variables]]
        s0_FWHMG:         287.317538 (init = 287.3175)
        s0_FWHML:         1.00000004 (init = 1)
        s0_TotalFWHM:     287.852515 == '0.5346*s0_FWHML+(0.2166*s0_FWHML**2+s0_FWHMG**2)**0.5'
        s0_Scale:         72.0818067 (init = 72.08181)
        s0_Saturation:    0 (fixed)
        s0_Amp7_2__5_2:   0.5000937 (fixed)
        s0_Amp7_2__7_2:   0.4939217 (fixed)
        s0_Amp7_2__9_2:   0.3396039 (fixed)
        s0_Amp9_2__7_2:   0.172911 (fixed)
        s0_Amp9_2__9_2:   0.4939521 (fixed)
        s0_Amp9_2__11_2:  1 (fixed)
        s0_Al:            5300.79815 (init = 5300.798)
        s0_Au:            101.129022 (init = 101.129)
        s0_Bl:            0 (fixed)
        s0_Bu:            171.971287 (init = 171.9713)
        s0_Cl:            0 (fixed)
        s0_Cu:            0 (fixed)
        s0_Centroid:      377.508491 (init = 377.5085)
        s0_Background0:   150.539789 (init = 150.5398)
        s0_N:             0 (fixed)
        s1_FWHMG:         208.133894 (init = 208.1339)
        s1_FWHML:         1.00001971 (init = 1.00002)
        s1_TotalFWHM:     208.669025 == '0.5346*s1_FWHML+(0.2166*s1_FWHML**2+s1_FWHMG**2)**0.5'
        s1_Scale:         82.8509918 (init = 82.85099)
        s1_Saturation:    0 (fixed)
        s1_Amp7_2__5_2:   0.5000937 (fixed)
        s1_Amp7_2__7_2:   0.4939217 (fixed)
        s1_Amp7_2__9_2:   0.3396039 (fixed)
        s1_Amp9_2__7_2:   0.172911 (fixed)
        s1_Amp9_2__9_2:   0.4939521 (fixed)
        s1_Amp9_2__11_2:  1 (fixed)
        s1_Al:            5300.79815 == 's0_Al'
        s1_Au:            101.129022 == 's0_Au'
        s1_Bl:            0.00000000 == 's0_Bl'
        s1_Bu:            171.971287 == 's0_Bu'
        s1_Cl:            0 (fixed)
        s1_Cu:            0 (fixed)
        s1_Centroid:      377.508491 == 's0_Centroid'
        s1_Background0:   123.248661 (init = 123.2487)
        s1_N:             0 (fixed)
    SATLAS2: 0.107 s
    SATLAS1: 19.8 s
    

.. code:: ipython3

    fig = plt.figure(constrained_layout=True, figsize=(14, 9))
    gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)
    ax11 = fig.add_subplot(gs[0, 0])
    ax11.label_outer()
    ax12 = fig.add_subplot(gs[0, 1], sharey=ax11)
    ax12.label_outer()
    ax21 = fig.add_subplot(gs[1, 0], sharex=ax11)
    ax21.label_outer()
    ax22 = fig.add_subplot(gs[1, 1], sharex=ax12, sharey=ax21)
    ax22.label_outer()
    
    ax11.errorbar(linked_x[0], linked_y[0], modifiedSqrt(linked_y[0]), fmt='.', label='Artificial data')
    ax11.plot(linked_x[0], s.evaluate(linked_x[0]), '-', label='Fit')
    ax12.errorbar(linked_x[1], linked_y[1], modifiedSqrt(linked_y[1]), fmt='.', label='Artificial data')
    ax12.plot(linked_x[1], s.evaluate(linked_x[1]), '-', label='SATLAS2 fit model 1')
    ax11.legend()
    
    ax21.errorbar(linked_x[0], linked_y[0], modifiedSqrt(linked_y[0]), fmt='.', label='Artificial data')
    ax21.plot(linked_x[0], linkedmodel.models[0](linked_x[0]), '-', label='Fit')
    ax22.errorbar(linked_x[1], linked_y[1], modifiedSqrt(linked_y[1]), fmt='.', label='Artificial data')
    ax22.plot(linked_x[1], linkedmodel.models[1](linked_x[1]), '-', label='Fit')
    ax21.legend()
    
    ax11.set_ylabel('SATLAS2')
    ax21.set_ylabel('SATLAS')
    plt.show()



.. image:: output_16_0.png


