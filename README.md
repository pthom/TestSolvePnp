# TestSolvePnp

A simple playground in order to test various pnp (3D pose estimation) methods with a given
configuration.

Several methods are proposed
```
  enum SolvePnpStrategy
  {
    Strategy_MySolvePnp_Epnp, //an home baked adapter of epnp using the epnp library source code
    Strategy_MySolvePnpPosit, //an home baked adapter of cvPOSIT (deprecated "OpenCV 1" pose estimation method)
    Strategy_solvePnp_P3p,    // opencv SOLVEPNP_P3P method
    Strategy_solvePnp_Iterative_InitialGuess, // opencv SOLVEPNP_ITERATIVE method with an initial guess
    Strategy_solvePnp_Epnp //opencv SOLVEPNP_EPNP method
  };
```

Based on my experimentations,


The order of the 3d points and image points *does* matter

It has to be adapted depending upon the strategy !

* With opencv's SOLVEPNP_EPNP the error can go down to 23.03 pixel.
  The order of the points does matter

* With MySolvePnpEpnp (an home baked adapter of epnp using the epnp library source code),
  the error is about 6.742 pixels, and the order *is* important
  It is strange that this "rewrite" gives different results

* With MySolvePnpPosit, the error is about 4.911 pixels
   and the order *is* important (in other cases the reprojection error is about 1278 pixels !)

* With solvePnp_P3p (cv::SOLVEPNP_P3P) the error is about 0.02961 pixels and the order does not matter much

* With solvePnp_P3p (cv::SOLVEPNP_ITERATIVE) the error can be 0 pixels
  *if a good initial extrinsic guess is given* (otherwise don't hope for any convergence).
  The order does not matter much with SOLVEPNP_ITERATIVE.
