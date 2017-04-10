#define _MAIN

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"
#include  "opencv2/features2d.hpp"
#include  "opencv2/text.hpp"
#include  "opencv2/photo.hpp"

#include "region.h"
#include "agglomerative_clustering.h"
#include "utils.h"

using namespace std;
using namespace cv;

/* Diversivication Configurations :                                     */
/* These are boolean values, indicating whenever to use a particular    */
/*                                   diversification strategy or not    */

#define PYRAMIDS     1 // Use spatial pyramids
#define CUE_D        1 // Use Diameter grouping cue
#define CUE_FGI      1 // Use ForeGround Intensity grouping cue
#define CUE_BGI      1 // Use BackGround Intensity grouping cue
#define CUE_G        1 // Use Gradient magnitude grouping cue
#define CUE_S        1 // Use Stroke width grouping cue
#define CHANNEL_I    0 // Use Intensity color channel
#define CHANNEL_R    1 // Use Red color channel
#define CHANNEL_G    1 // Use Green color channel
#define CHANNEL_B    1 // Use Blue color channel


void mergeOverlappingBoxes(std::vector<cv::Rect> &inputBoxes, cv::Mat &image, std::vector<cv::Rect> &outputBoxes)
{
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1); // Mask of original image
    cv::Size scaleFactor(10,10); // To expand rectangles, i.e. increase sensitivity to nearby rectangles. Doesn't have to be (10,10)--can be anything
    for (int i = 0; i < inputBoxes.size(); i++)
    {
        cv::Rect box = inputBoxes.at(i) + scaleFactor;
        cv::rectangle(mask, box, cv::Scalar(255), CV_FILLED); // Draw filled bounding boxes on mask
    }

    std::vector< std::vector <cv::Point> > contours;
    // Find contours in mask
    // If bounding boxes overlap, they will be joined by this function call
    cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    for (int j = 0; j < contours.size(); j++)
    {
        outputBoxes.push_back(cv::boundingRect(contours.at(j)));
    }
}


int main( int argc, char** argv )
{
    // Params
    float x_coord_mult              = 0.25; // a value of 1 means rotation invariant

    // Pipeline configuration
    bool conf_channels[4]={CHANNEL_R,CHANNEL_G,CHANNEL_B,CHANNEL_I};
    bool conf_cues[5]={CUE_D,CUE_FGI,CUE_BGI,CUE_G,CUE_S};

    /* initialize random seed: */
    srand (time(NULL));

    Mat src, img, grey, lab_img, gradient_magnitude;

    img = imread(argv[1]);
    img.copyTo(src);

    int delta = 13;
    int img_area = img.cols*img.rows;
    Ptr<MSER> cv_mser = MSER::create(delta,(int)(0.00002*img_area),(int)(0.11*img_area),55,0.);

    cvtColor(img, grey, CV_BGR2GRAY);
    cvtColor(img, lab_img, CV_BGR2Lab);
    gradient_magnitude = Mat_<double>(img.size());
    get_gradient_magnitude( grey, gradient_magnitude);

    vector<Mat> channels;
    split(img, channels);
    channels.push_back(grey);
    int num_channels = channels.size();

    if (PYRAMIDS)
    {
      for (int c=0; c<num_channels; c++)
      {
        Mat pyr;
        resize(channels[c],pyr,Size(channels[c].cols/2,channels[c].rows/2));
        //resize(pyr,pyr,Size(channels[c].cols,channels[c].rows));
        channels.push_back(pyr);
      }
      /*for (int c=0; c<num_channels; c++)
      {
        Mat pyr;
        resize(channels[c],pyr,Size(channels[c].cols/4,channels[c].rows/4));
        //resize(pyr,pyr,Size(channels[c].cols,channels[c].rows));
        channels.push_back(pyr);
      }*/
    }
    std::vector<Rect> mergedProposalsChannel;
    std::vector<Rect> proposalsChannel;
    for (int c=0; c<channels.size(); c++)
    {

        if (!conf_channels[c%4]) continue;

        if (channels[c].size() != grey.size()) // update sizes for smaller pyramid lvls
        {
          resize(grey,grey,Size(channels[c].cols,channels[c].rows));
          resize(lab_img,lab_img,Size(channels[c].cols,channels[c].rows));
          resize(gradient_magnitude,gradient_magnitude,Size(channels[c].cols,channels[c].rows));
        }

        // TODO you want to try single pass MSER?
        //channels[c] = 255 - channels[c];
        //cv_mser->setPass2Only(true);

        /* Initial over-segmentation using MSER algorithm */
        vector<vector<Point> > contours;
        vector<Rect>  mser_bboxes;
        //t = (double)getTickCount();
        cv_mser->detectRegions(channels[c], contours, mser_bboxes);
        //cout << " OpenCV MSER found " << contours.size() << " regions in " << ((double)getTickCount() - t)*1000/getTickFrequency() << " ms." << endl;
   

        /* Extract simple features for each region */ 
        vector<Region> regions;
        Mat mask = Mat::zeros(grey.size(), CV_8UC1);
        int max_stroke = 0;
        for (int i=contours.size()-1; i>=0; i--)
        {
            Region region;
            region.pixels_.push_back(Point(0,0)); //cannot swap an empty vector
            region.pixels_.swap(contours[i]);
            region.bbox_ = mser_bboxes[i];
            region.extract_features(lab_img, grey, gradient_magnitude, mask, conf_cues);
            max_stroke = max(max_stroke, region.stroke_mean_);
            regions.push_back(region);
        }
          
        unsigned int N = regions.size();
        if (N<3) continue;
        int dim = 3;
        t_float *data = (t_float*)malloc(dim*N * sizeof(t_float));

        /* Single Linkage Clustering for each individual cue */
        for (int cue=0; cue<5; cue++)
        {

          if (!conf_cues[cue]) continue;
    
          int count = 0;
          for (int i=0; i<regions.size(); i++)
          {
            data[count] = (t_float)(regions.at(i).bbox_.x+regions.at(i).bbox_.width/2)/channels[c].cols*x_coord_mult;
            data[count+1] = (t_float)(regions.at(i).bbox_.y+regions.at(i).bbox_.height/2)/channels[c].rows;
            switch(cue)
            {
              case 0:
                data[count+2] = (t_float)max(regions.at(i).bbox_.height, regions.at(i).bbox_.width)/max(channels[c].rows,channels[c].cols);
                break;
              case 1:
                data[count+2] = (t_float)regions.at(i).intensity_mean_/255;
                break;
              case 2:
                data[count+2] = (t_float)regions.at(i).boundary_intensity_mean_/255;
                break;
              case 3:
                data[count+2] = (t_float)regions.at(i).gradient_mean_/255;
                break;
              case 4:
                data[count+2] = (t_float)regions.at(i).stroke_mean_/max_stroke;
                break;
            }
            count = count+dim;
          }
      
          HierarchicalClustering h_clustering(regions);
          vector<HCluster> dendrogram;
          h_clustering(data, N, dim, (unsigned char)0, (unsigned char)3, dendrogram, x_coord_mult, channels[c].size());
          std::vector<Rect> proposals;
          std::vector<Rect> mergedProposals;
          for (int k=0; k<dendrogram.size(); k++)
          {
             int ml = 1;
             if (c>=num_channels) ml=2;// update sizes for smaller pyramid lvls
             if (c>=2*num_channels) ml=4;// update sizes for smaller pyramid lvls

            /* cout << dendrogram[k].rect.x*ml << " " << dendrogram[k].rect.y*ml << " "
                  << dendrogram[k].rect.width*ml << " " << dendrogram[k].rect.height*ml << " "
                  << (float)dendrogram[k].probability*-1 << endl;*/
             //     << (float)dendrogram[k].nfa << endl;
             //     << (float)(k) * ((float)rand()/RAND_MAX) << endl;
             //     << (float)dendrogram[k].nfa * ((float)rand()/RAND_MAX) << endl;
             if ((float)dendrogram[k].probability*-1 < -0.98){
               proposals.push_back(Rect(Point(dendrogram[k].rect.x*ml,dendrogram[k].rect.y*ml), Point(dendrogram[k].rect.x*ml+dendrogram[k].rect.width*ml, dendrogram[k].rect.y*ml+dendrogram[k].rect.height*ml)));
             }
             
          }
          mergeOverlappingBoxes(proposals,src,mergedProposals);
          for( int b = 0; b < mergedProposals.size();b++){

            proposalsChannel.push_back(mergedProposals.at(b));
           
          }
        }
        free(data);

    }
    mergeOverlappingBoxes(proposalsChannel,src,mergedProposalsChannel);
    for( int b = 0; b < mergedProposalsChannel.size();b++){
      rectangle(src,mergedProposalsChannel.at(b), Scalar(0,0,255));
    }

    Ptr<cv::text::OCRTesseract> tess = cv::text::OCRTesseract::create(NULL,NULL,NULL,0,11);
    
    std::string output_string;
    for( int b = 0; b < mergedProposalsChannel.size();b++){
      tess->setWhiteList("0123456789");
      Mat miniMat = src(mergedProposalsChannel.at(b)).clone();
      int coef = 200;
      miniMat = miniMat/coef;
      Mat miniMatLab;
      cv::cvtColor(miniMat, miniMatLab, CV_BGR2Lab);

        // Extract the L channel
      std::vector<cv::Mat> lab_planes(3);
      cv::split(miniMatLab, lab_planes);  // now we have the L image in lab_planes[0]

      // apply the CLAHE algorithm to the L channel
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
      clahe->setClipLimit(4);
      cv::Mat dst;
      clahe->apply(lab_planes[0], dst);

      // Merge the the color planes back into an Lab image
      dst.copyTo(lab_planes[0]);
      cv::merge(lab_planes, miniMatLab);

     // convert back to RGB
      cv::Mat image_clahe;
      cv::cvtColor(miniMatLab, image_clahe, CV_Lab2BGR);

      // display the results  (you might also want to see lab_planes[0] before and after).
      cv::imshow("image original", miniMat);
      cv::imshow("image CLAHE", image_clahe);
      cv::waitKey();
      Mat gray;
      cvtColor( miniMat, gray, CV_BGR2GRAY );
      Mat binary;
      //adaptiveThreshold( gray, binary,255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY_INV,13, 1);
      threshold( gray, binary,160,255,CV_THRESH_BINARY_INV);
      cv::imshow("binary", binary);
      imwrite("./binary.png",binary);
      waitKey(-1);
      tess->run(binary, output_string);
      cout << output_string <<endl;
    }

    
    
    imshow("",src);
    waitKey(-1);
}
