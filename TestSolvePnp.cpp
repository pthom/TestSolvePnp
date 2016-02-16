#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <iomanip>
#include "epnp/epnp.h"


//
// Utility functions
//
void printMat(const std::string & name, const cv::Mat & mat)
{
  std::cout << name << std::endl;
  const int precision = 4;
  for(int i=0; i<mat.size().height; i++)
  {
    std::cout << "[";
    for(int j=0; j<mat.size().width; j++)
    {
      std::cout << std::setprecision(precision) << mat.at<double>(i,j);
      if(j != mat.size().width-1)
        std::cout << ", ";
      else
        std::cout << "]" << std::endl;
    }
  }
}

cv::Point2f ImageCoordsToIdealCameraCoords(const cv::Mat_<double> & cameraIntrinsicParams, const cv::Point2f & pt)
{
  return cv::Point2f(
    ( pt.x - cameraIntrinsicParams.at<double>(0,2)  ) / cameraIntrinsicParams.at<double>(0,0),
    ( pt.y - cameraIntrinsicParams.at<double>(1,2)  ) / cameraIntrinsicParams.at<double>(1,1) );
}

cv::Point2f IdealCameraCoordsToImageCoords(const cv::Mat_<double> & cameraIntrinsicParams, const cv::Point2f & pt)
{
  return cv::Point2f(
    pt.x * cameraIntrinsicParams.at<double>(0,0) + cameraIntrinsicParams.at<double>(0,2),
    pt.y * cameraIntrinsicParams.at<double>(1,1) + cameraIntrinsicParams.at<double>(1,2)  );
}


//cv::undistortPoints is quite tricky by default : it takes image coords as input and return ideal camera coords !
//(the opencv documentation does not agree with the source code here...)
cv::Point2f UnDistortPoint_ImageCoords(const cv::Point2f & pt, const cv::Mat_<double> & cameraIntrinsicParams, const std::vector<double> & distCoeffs)
{
  std::vector<cv::Point2f> src, dst;
  src.push_back(pt);
  cv::fisheye::undistortPoints(src, dst, cameraIntrinsicParams, distCoeffs);
  cv::Point2f pt_undistorted = IdealCameraCoordsToImageCoords(cameraIntrinsicParams, dst[0]);
  return pt_undistorted;
}

std::vector<cv::Point2f> UnDistortPoints_ImageCoords(const std::vector<cv::Point2f> & points, const cv::Mat_<double> & cameraIntrinsicParams, const std::vector<double> & distCoeffs)
{
  std::vector<cv::Point2f> points_ideal_undistorted(points.size());
  cv::fisheye::undistortPoints(points, points_ideal_undistorted, cameraIntrinsicParams, distCoeffs);

  std::vector<cv::Point2f> points_undistorted;
  for (const auto & pt_ideal_undistorted : points_ideal_undistorted)
    points_undistorted.push_back( IdealCameraCoordsToImageCoords(cameraIntrinsicParams, pt_ideal_undistorted) );

  return points_undistorted;
}



//
//  <MySolvePnpEpnp> : tries reproduces the expected behavior of solvePnp with the original source code of the Epnp library
//
void MySolvePnpEpnp(
  const std::vector<cv::Point3f> &objectPoints,
  const std::vector<cv::Point2f> &imagePoints,
  const cv::Mat_<double> &cameraIntrinsicParams,
  const std::vector<double> &distCoeffs,
  cv::Mat_<double> &outRotationEstimated,
  cv::Mat_<double> &outTranslationEstimated)
{
  std::vector<cv::Point2f> imagePoints_undistorted = UnDistortPoints_ImageCoords(imagePoints, cameraIntrinsicParams, distCoeffs);

  epnp epnpCaller;

  epnpCaller.set_internal_parameters(
    cameraIntrinsicParams.at<double>(0, 2),
    cameraIntrinsicParams.at<double>(1, 2),
    cameraIntrinsicParams.at<double>(0, 0),
    cameraIntrinsicParams.at<double>(1, 1)
  );
  epnpCaller.set_maximum_number_of_correspondences(objectPoints.size());
  epnpCaller.reset_correspondences();

  for (size_t i = 0; i < objectPoints.size(); i++)
  {
    epnpCaller.add_correspondence(
      objectPoints[i].x, objectPoints[i].y, objectPoints[i].z,
      imagePoints_undistorted[i].x, imagePoints_undistorted[i].y
    );
  }
  double rotationArray[3][3];
  double translationArray[3];

  epnpCaller.compute_pose(rotationArray, translationArray);

  cv::Mat_<double> rotation3x3(cv::Size(3, 3));
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      rotation3x3(j,i) = rotationArray[j][i];
  outRotationEstimated = cv::Mat_<double>(cv::Size(3, 1));

  outTranslationEstimated = cv::Mat_<double>(cv::Size(1, 3));
  for (int j = 0; j < 3; j++)
    outTranslationEstimated(j, 0) = translationArray[j];

  cv::Rodrigues(rotation3x3, outRotationEstimated);
}
//  </MySolvePnpEpnp>


//
//  MySolvePnpPosit : adapter for cvPOSIT with a C++ prototype close to solvePnp
//

namespace
{
  //cvPOSIT wrapper
  void Posit_IdealCameraCoords(const std::vector<cv::Point3f> & objectPoints, const std::vector<cv::Point2f> & imagePoints,
                               cv::Mat_<double> &outRotationEstimated, cv::Mat_<double> & outTranslationEstimated)
  {

    CvPoint2D32f * imagePoints_c = (CvPoint2D32f *) malloc(sizeof(CvPoint2D32f) * imagePoints.size());
    {
      for (size_t i = 0; i < imagePoints.size(); i++)
        imagePoints_c[i] = cvPoint2D32f(imagePoints[i].x, imagePoints[i].y);
    }

    CvPoint3D32f * objectPoints_c = (CvPoint3D32f *) malloc(sizeof(CvPoint3D32f) * objectPoints.size());
    {
      for (size_t i = 0; i < objectPoints.size(); i++)
        objectPoints_c[i] = cvPoint3D32f(objectPoints[i].x, objectPoints[i].y, objectPoints[i].z);
    }

    CvPOSITObject * positObject = cvCreatePOSITObject(objectPoints_c, objectPoints.size() );


    CvTermCriteria criteria;
    criteria.type = CV_TERMCRIT_EPS|CV_TERMCRIT_ITER;
    criteria.epsilon = 0.00000000000000010;
    criteria.max_iter = 30;
    //criteria.epsilon = 0.10;
    //criteria.max_iter = 6;

    float positTranslationArray[3];
    float positRotationArray[9];

    const double idealFocal = 1.;
    cvPOSIT(positObject, imagePoints_c,
            idealFocal, criteria,
            positRotationArray, positTranslationArray);


    cv::Mat_<double> positRotationMat1x3;
    {
      cv::Mat_<double> positRotationMat3x3(cv::Size(3, 3));
      {
        int idx = 0;
        for (int j = 0; j < 3; j++)
        {
          for (int i = 0; i < 3; i++)
          {
            positRotationMat3x3(j, i) = (double)positRotationArray[idx++];
          }
        }
      }
      cv::Rodrigues(positRotationMat3x3, positRotationMat1x3);
    }
    outRotationEstimated = positRotationMat1x3;

    outTranslationEstimated = cv::Mat_<double>(cv::Size(1, 3));
    for (int i = 0; i < 3; i++)
      outTranslationEstimated.at<double>(i, 0) = (double)positTranslationArray[i];

    cvReleasePOSITObject(&positObject);
    free(imagePoints_c);
    free(objectPoints_c);
  }
}



// MySolvePnpPosit implementation
void MySolvePnpPosit(const std::vector<cv::Point3f> &objectPoints, const std::vector<cv::Point2f> &imagePoints,
                     const cv::Mat_<double> &cameraIntrinsicParams, const std::vector<double> &distorsionCoeffs,
                     cv::Mat_<double> &outRotationEstimated, cv::Mat_<double> &outTranslationEstimated)
{
  std::vector<cv::Point2f> imagePoints_IdealCameraCoords_undistorted;
  cv::fisheye::undistortPoints(imagePoints, imagePoints_IdealCameraCoords_undistorted, cameraIntrinsicParams, distorsionCoeffs);

  Posit_IdealCameraCoords(objectPoints, imagePoints_IdealCameraCoords_undistorted, outRotationEstimated, outTranslationEstimated);
}






void TestSolvePnp()
{
  enum SolvePnpStrategy
  {
    Strategy_MySolvePnp_Epnp,
    Strategy_MySolvePnpPosit,
    Strategy_solvePnp_P3p,
    Strategy_solvePnp_Iterative_InitialGuess,
    Strategy_solvePnp_Epnp
  };
  SolvePnpStrategy strategy = Strategy_MySolvePnp_Epnp;
  //SolvePnpStrategy strategy = Strategy_MySolvePnpPosit;
  //SolvePnpStrategy strategy = Strategy_solvePnp_P3p;
  //SolvePnpStrategy  strategy = Strategy_solvePnp_Iterative_InitialGuess;
  //SolvePnpStrategy  strategy = Strategy_solvePnp_Epnp;


  std::vector<cv::Point3f> objectPoints;

  // Based on my experimentations
  // The order of the 3d points and image points *does* matter
  // It has to be adapted depending upon the strategy !

  // With opencv's SOLVEPNP_EPNP the error can go down to 23.03 pixel with the following order
  //the order of the points does matter
  if (strategy == Strategy_solvePnp_Epnp)
  {
    objectPoints.push_back(cv::Point3f(-62.1225319f, 15.7540569f, 0.819464564f));
    objectPoints.push_back(cv::Point3f(62.3174629f, 15.7940502f, 0.819983721f));
    objectPoints.push_back(cv::Point3f(-0.372639507f, 16.4230633f,  -36.5060043f));
    objectPoints.push_back(cv::Point3f(0.f, 0.f, 0.f));
  }

  // With MySolvePnpEpnp (an home baked adapter of epnp using the epnp library source code),
  // the error is about 6.742 pixels, and the order *is* important
  // It is strange that this "rewrite" gives different results
  else if (strategy == Strategy_MySolvePnp_Epnp)
  {
    objectPoints.push_back(cv::Point3f(-62.1225319f, 15.7540569f, 0.819464564f));
    objectPoints.push_back(cv::Point3f(62.3174629f, 15.7940502f, 0.819983721f));
    objectPoints.push_back(cv::Point3f(0.f, 0.f, 0.f));
    objectPoints.push_back(cv::Point3f(-0.372639507f, 16.4230633f,  -36.5060043f));
  }

  // With MySolvePnpPosit, the error is about 4.911 pixels
  // and the order *is* important (in other cases the reprojection error is about 1278 pixels !)
  else if (strategy == Strategy_MySolvePnpPosit)
  {
    objectPoints.push_back(cv::Point3f(0.f, 0.f, 0.f));
    objectPoints.push_back(cv::Point3f(-62.1225319f, 15.7540569f, 0.819464564f));
    objectPoints.push_back(cv::Point3f(62.3174629f, 15.7940502f, 0.819983721f));
    objectPoints.push_back(cv::Point3f(-0.372639507f, 16.4230633f,  -36.5060043f));
  }

  // With solvePnp_P3p (cv::SOLVEPNP_P3P) the error is about 0.02961 pixels and the order does not matter much
  else if (strategy == Strategy_solvePnp_P3p)
  {
    objectPoints.push_back(cv::Point3f(0.f, 0.f, 0.f));
    objectPoints.push_back(cv::Point3f(-62.1225319f, 15.7540569f, 0.819464564f));
    objectPoints.push_back(cv::Point3f(62.3174629f, 15.7940502f, 0.819983721f));
    objectPoints.push_back(cv::Point3f(-0.372639507f, 16.4230633f,  -36.5060043f));
  }

  // With solvePnp_P3p (cv::SOLVEPNP_ITERATIVE) the error can be 0 pixels
  // *if a good initial extrinsic guess is given* (otherwise don't hope for any convergence)
  // the order does not matter much
  else if (strategy == Strategy_solvePnp_Iterative_InitialGuess)
  {
    objectPoints.push_back(cv::Point3f(0.f, 0.f, 0.f));
    objectPoints.push_back(cv::Point3f(-62.1225319f, 15.7540569f, 0.819464564f));
    objectPoints.push_back(cv::Point3f(62.3174629f, 15.7940502f, 0.819983721f));
    objectPoints.push_back(cv::Point3f(-0.372639507f, 16.4230633f,  -36.5060043f));
  }

  cv::Mat_<double> cameraIntrinsicParams(cv::Size(3, 3));
  cameraIntrinsicParams = 0.;
  cameraIntrinsicParams(0, 0) = 3844.4400000000001f;
  cameraIntrinsicParams(1, 1) = 3841.0599999999999f;
  cameraIntrinsicParams(0, 2) = 640.f;
  cameraIntrinsicParams(1, 2) = 380.f;
  cameraIntrinsicParams(2, 2) = 1.f;

  std::vector<double> distCoeffs(4);
  distCoeffs[0] = -0.063500002026557922;
  distCoeffs[1] = -2.5915000438690186;
  distCoeffs[2] = -0.0023300000466406345;
  distCoeffs[3] = 0.0008411200251430273;


  cv::Mat_<double> rotation(cv::Size(1, 3));
  rotation(0,0) = 0.07015543380659847f;
  rotation(0,1) = 0.06922079477774973f;
  rotation(0,2) = -0.00254676088325f;

  cv::Mat_<double> translation(cv::Size(1, 3));
  translation(0,0) = -35.3236f;
  translation(0,1) = -48.1699f;
  translation(0,2) = 769.068f;

  std::vector<cv::Point2f> imagePoints;
  cv::projectPoints(objectPoints, rotation, translation, cameraIntrinsicParams, distCoeffs, imagePoints);


  cv::Mat_<double> rotation2(cv::Size(1, 3));
  cv::Mat_<double> translation2(cv::Size(1, 3));
  rotation2.setTo(0.);
  translation2.setTo(0.);
  translation2(2, 0) = 600.; //Hint for SOLVEPNP_ITERATIVE


  switch(strategy)
  {
    case Strategy_MySolvePnp_Epnp:
      MySolvePnpEpnp(
        objectPoints,
        imagePoints,
        cameraIntrinsicParams,
        distCoeffs,
        rotation2,
        translation2);
      break;
    case Strategy_MySolvePnpPosit:
      MySolvePnpPosit(
        objectPoints,
        imagePoints,
        cameraIntrinsicParams,
        distCoeffs,
        rotation2,
        translation2);
      break;
    case Strategy_solvePnp_P3p:
      cv::solvePnP(objectPoints, imagePoints,
                   cameraIntrinsicParams, distCoeffs,
                   rotation2, translation2,
                   false,//useExtrinsicGuess
                   cv::SOLVEPNP_P3P
      );
      break;
    case Strategy_solvePnp_Iterative_InitialGuess:
      cv::solvePnP(objectPoints, imagePoints,
                   cameraIntrinsicParams, distCoeffs,
                   rotation2, translation2,
                   true,//useExtrinsicGuess
                   cv::SOLVEPNP_ITERATIVE
      );
      break;
    case Strategy_solvePnp_Epnp:
      cv::solvePnP(objectPoints, imagePoints,
                   cameraIntrinsicParams, distCoeffs,
                   rotation2, translation2,
                   true,//useExtrinsicGuess
                   cv::SOLVEPNP_EPNP
      );
      break;
  }



  std::vector<cv::Point2f> imagePoints_Reproj(3);
  cv::projectPoints(objectPoints, rotation2, translation2, cameraIntrinsicParams, distCoeffs, imagePoints_Reproj);

  float sum = 0.;
  for (size_t i = 0; i < imagePoints.size(); i++)
    sum += cv::norm(imagePoints_Reproj[i] - imagePoints[i]);


  printMat("rotation", rotation);
  printMat("rotation2", rotation2);
  printMat("translation", translation);
  printMat("translation2", translation2);
  std::cout << "Reproj Error=" << sum << std::endl;
}

int main(int argc, char **argv)
{
  TestSolvePnp();
}
